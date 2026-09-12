# JitRL Router — Session 4 优化进度

> 状态：v1.3（2026-09-12）
> 范围：优先优化论文式记忆更新回路，固定顺序为 M1 兼容重构 → M1 验证 → M2 多轮轨迹。
> 边界：本阶段仅修改独立 Python harness 与 `jitrl_core`，不修改 PilotDeck 实际 Router 和 Web Demo。
> 正本：Pdwork `docs/optimization/`；仓库 `docs/optimization/` 为工作副本。

## 检查点 0：路线确认（已完成）

- 优化基准：现有 24 个主任务 + 5 个探针；
- 第一优先级：路由质量；
- 已核查当前系统有外部盲评 Evaluator，但仅做单步 episode 的终局评分和单标量记忆写入；
- 已确认升级为“完整轨迹 → 逐决策评价 → 信用分配 → 生命周期记忆更新”；
- 已确认实施顺序：先 M1，再 M2；
- 决策正本：`docs/optimization/MEMORY-LOOP-DECISIONS.md` v1.0。

## 检查点 1：M1 设计审计（已完成）

硬兼容约束：

1. `JitRLEngine.decide()` 的数值路径和 RNG 消耗不变；
2. 旧 `learn/update/update_from_message` API 保留；
3. memory entry 六键 schema 保留；
4. `run_ablation.py` 直接替换 `engine.memory` 的鸭子类型用法继续有效；
5. 单步轨迹严格满足 `G=r`；
6. 执行或 Evaluator 失败不写 memory；
7. 仅做加法式 record schema 扩展。

发现的在制品：`harness/cpa_client.py` 在本 session 开始前已有未提交修改，本阶段不覆盖、不回退该文件。

## 检查点 2：M1 实现（已完成）

### 2.1 新增文件

- `jitrl_core/types.py`：`DecisionStep` / `RouteTrajectory`（open→finalized/abandoned）/ `StepEvaluation` / `TrajectoryEvaluation`；step = 一次真实路由决策；
- `jitrl_core/credit.py`：`CreditAssigner` Protocol + `IdentityCreditAssigner`（G=r）+ `DiscountedCreditAssigner`（G_t = r_t + η·Σγ^(k-t)·r_k，默认 γ=0.8、η=0.5，η<1 防止后步补救洗掉前步误路由）；
- `jitrl_core/updater.py`：`build_entry()`（冻结六键 entry 的唯一构造点，engine.learn 与所有 updater 共用）+ `DirectMemoryUpdater`（默认，行为与旧 learn 逐位一致）+ `LifecycleMemoryUpdater`（置信度 ≥0.80 active / ≥0.60 provisional / 否则 quarantine；None 置信度按 active 兼容；对普通 ExperienceMemory 鸭子类型回退 plain add）；
- `harness/trajectory.py`：`TrajectoryStep`/`Trajectory`/`TrajectoryReport` + `TrajectoryEvaluator`（M1 terminal-only：每 episode 一次盲评，prompt 与奖励数学不变）+ `update_memory_from_trajectory`（无 `learn_trajectory` 的引擎包装器回退旧 `learn()`）；
- 测试：`jitrl_core/tests/test_trajectory.py`、`harness/tests/test_trajectory.py`。

### 2.2 既有文件修改（加法式）

- `jitrl_core/memory.py`：`LifecycleMemory(ExperienceMemory)`，状态/元数据 sidecar（按 id(entry) 键控，add 时剪枝防 id 复用），`retrieve` 在 top-k 截断**前**过滤非 active；普通 add 默认 active；
- `jitrl_core/engine.py`：构造器可选 `credit_assigner`/`memory_updater`（默认即旧行为）；新增 `learn_trajectory()`（credit→updater 串联，失败评估零写入，不消耗 RNG）；`learn()` 改用共享 `build_entry()`（输出不变）；**修复**：注入空 memory 被 `or` 短路替换的隐患（`__len__` 使空容器 falsy），改为显式 `is not None`；
- `jitrl_core/__init__.py`：导出全部新符号；
- `harness/run_real.py`：单步 episode 构造长度 1 的 `Trajectory`，评估与 C 臂写记忆统一走 `TrajectoryEvaluator` + `update_memory_from_trajectory`；record 新增 `traj_id`/`n_steps` 加法字段；模块 docstring 更新。

### 2.3 实现中发现并修复的问题

1. `JitRLEngine.__init__` 的 `memory or ExperienceMemory(...)`：空记忆对象（定义了 `__len__`）为 falsy 会被静默替换为默认实例——任何注入空 LifecycleMemory/ExperienceMemory 的用法都会踩中；已改 `is not None`；
2. dataclass 字段缺注解（`decision=None`）不构成字段导致 `TypeError`——已补类型注解（TYPE_CHECKING 导入，无运行时依赖）。

## 检查点 3：M1 验证（已完成）

- `python -m pytest jitrl_core -q` → **71 passed**（原 47 + 新 24）；
- `python -m pytest harness jitrl_core local_judge eval -q` → **298 passed**（基线 261 + 新 37），既有测试零修改全绿；
- `python -m pytest demo -q` → **32 passed**；
- 关键等价性测试：同 seed 下 `learn_trajectory`（单步 + identity credit）与旧 `learn()` 产出的 entry 逐位相等，且 RNG 状态不变（`test_single_step_equivalent_to_legacy_learn`）；
- 折扣回报手算用例：(-0.7, 0.2, 0.8)→(-0.364, 0.52, 0.8)；补救场景 (-0.7, 0.8)→G₀=-0.38 保持负值；
- 失败语义：exec/eval 失败 → report not ok → 零记忆写入（harness 级与 engine 级双重测试）；
- run_real 轨迹路径：memory entry 严格六键、G 与 record reward 一致、entry 顺序与 record 顺序一致。

## M2 接口预留（已就位）

- 核心 `StepEvaluation` 已含 `routing_verdict` / `recommended_tier` / `failure_tags` / `feedback` / `confidence`，M2 轨迹级 Evaluator 只需填充；
- `LifecycleMemoryUpdater` 已把这些语义字段写入 `LifecycleMemory` sidecar（`meta_of()`），不污染冻结 entry schema；
- `harness/run_real.py` 的 `Trajectory`/`TrajectoryStep` 已携带 `previous_tier`/`decision`，多轮只需扩展 steps 列表与任务源；
- `DiscountedCreditAssigner` 已支持多步（None-reward 中间步贡献 0 但推进时间指数）。

## 检查点 4：M2 实现（已完成）

### 4.1 新增文件

- `harness/multiturn.py`：`group_trajectories`（traj_id 分组 + turn_index 排序，无 traj_id 行退化为单轮）+ `build_conversation`（user/assistant 交替历史，失败轮只保留 user 消息，历史截断 2000 字，首轮与旧格式逐字节一致）；
- `harness/evaluator.py` 追加：`TrajectoryLevelEvaluator`（每轨迹一次调用；盲评边界：可见档位/不可见模型名、价格、usage）+ `validate_trajectory_evaluation`（steps 数=轮数、turn_index 对位、枚举校验，任一失败 → None → 零记忆）+ `StepVerdict`/`TrajectoryLevelResult`；
- `eval/multiturn_tasks.jsonl`：8 轨迹 28 轮（M01 数据修正、M02 代码生成→扩展、M03 文档改写→双版本、M04 重构→集成、M05 闲聊→排程、M06 检索→冲突甄别→综合、M07 留存深析→分层→实验设计、M08 脚本→审查），轮型覆盖升级/修正/扩展/收尾；
- 测试：`harness/tests/test_multiturn.py`（22 项）。

### 4.2 既有文件修改（加法式）

- `harness/trajectory.py`：`TrajectoryEvaluator` 支持 `mode="terminal"|"per_step"`；`StepReport` 增加 step_verdict/trajectory_result 与 `eval_record()`；逐步 reward=同一 D3 公式作用于 local_quality + 该轮真实模型/usage；adapter 把语义字段（verdict/recommended_tier/failure_tags/feedback/certainty）映射进核心 `StepEvaluation` → LifecycleMemoryUpdater sidecar；
- `harness/run_real.py`：`--arm T1|T2|T3`（默认 T1=M1 兼容）；主循环改为轨迹分组循环（路由数学原样提取为 `_route_one`）；多轮执行携带会话历史；record 新增 turn_index/n_turns/arm/step-verdict 字段（加法）；`_summary` 新增 n_trajectories/memory_status_counts；eval 成本只在末轮记账防重复计价；
- 失败语义按轨迹：任一轮 exec 失败或轨迹评估失败 → 整条轨迹零记忆（与决策文档 7.4 一致）。

### 4.3 验证

- `python -m pytest harness jitrl_core local_judge eval -q` → **320 passed**（M1 后 298 + 新 22），既有测试零修改；
- `python -m pytest demo -q` → **32 passed**；
- 任务集分组核验：28 行 → 8 轨迹，档位分布 m/r/m/s、m/m/c/s、m/m/r、m/m/r/s、s/m/r、s/m/r、r/r/r/s、c/r/s；
- T2 折扣信用手算验证：G0 = r0 + 0.4·r1（各轮真实执行模型计价）；T3 置信度 0.9/0.65 → active/provisional；轨迹中途 exec 失败 → 该轨迹零写入、其余轨迹不受影响。

## 检查点 5：S4-M2 预注册（已完成）

- `docs/optimization/S4-M2-PREREGISTRATION.md` v1.0 冻结：三臂 × episodes=2、冒烟规则、P1–P5 判据（P1/P2 主判据：episode-2 轮级准确率不降 + 质量增益；P3 生命周期无害；P4 信息性；P5 成本护栏 ≤1.10×T1）、失败率>20% 与 $1.50 预算停止线；
- 真实运行需：本地 Judge 服务（llama/MiniCPM5-1B）在线 + CPA key；运行后出 `S4-M2-RESULTS.md`，不回写预注册。

## 检查点 6：S4-M2 真实运行（第一次尝试，已按停止规则中止）

- 前置核验：本地 Judge（llama.cpp / MiniCPM5-1B，:18080）探活通过；
- 冒烟（M08 × T2 × episodes=1，2026-09-12 11:04）：3/3 轮 exec 失败——CPA 网关 HTTP 429「Usage limit reached for 5 hour. Your limit will reset at 2026-09-12 14:27:21」（第 3 轮报 all credentials cooling down）；
- 失败率 100% > 预注册 20% 停止线 → **停止，等待限额重置**；零成本消耗（exec 全部失败未计费），Judge 路由正常工作（3 轮均完成 4-request 判档）；
- 冒烟资产：`logs/s4m2_smoke.jsonl` / `logs/s4m2_smoke_mem.jsonl`（保留作审计，不入正式数据）；
- 恢复计划：14:27 限额重置后重跑冒烟，通过即顺序执行 T1/T2/T3 正式三臂（命令与判据以预注册 v1.0 为准，不回改）。

## 下一步

1. 14:27 限额重置后：重跑冒烟（M08 × T2）→ 通过 → 正式三臂（T1→T2→T3，episodes=2）→ `S4-M2-RESULTS.md`；
2. 结果若支持 T2/T3：考虑把 per-step 语义字段引入检索侧（受控白名单，参照 T8-PM 教训）；
3. 结果若不支持：回到 Judge 校准主线（Session 4 讨论的第一优先级遗留项）。

## 测试命令

- 快速：`python -m pytest jitrl_core -q`
- 完整离线：`python -m pytest harness jitrl_core local_judge eval -q`
- Demo 回归（如需）：`python -m pytest demo -q`

## Changelog

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-09-12 | v1.0 | 建立优化成果目录与 Session 4 进度基线，记录 M1/M2 顺序和兼容约束 |
| 2026-09-12 | v1.1 | M1 完成并验证：types/credit/updater/LifecycleMemory/learn_trajectory 落地，run_real 接入统一轨迹接口（traj_id/n_steps 加法字段），修复注入空 memory 被替换隐患；测试 71/298/32 全绿（既有零修改），M2 接口预留就位 |
| 2026-09-12 | v1.2 | M2 实现+预注册完成：multiturn 分组/会话构建、TrajectoryLevelEvaluator（盲评边界）、per_step 模式、T1/T2/T3 三臂、8 轨迹 28 轮任务集；测试 320/32 全绿（既有零修改）；`S4-M2-PREREGISTRATION.md` v1.0 冻结，待真实运行 |
| 2026-09-12 | v1.3 | S4-M2 第一次运行尝试：Judge 探活通过；冒烟 3/3 exec 失败（CPA 429 五小时限额，14:27:21 重置），触发预注册 >20% 停止线中止；零成本；已安排限额重置后自动恢复（重跑冒烟 → 正式三臂） |
