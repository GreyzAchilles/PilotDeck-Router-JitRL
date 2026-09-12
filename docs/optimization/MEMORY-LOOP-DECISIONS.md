# JitRL Router 记忆更新回路优化决策

> 状态：v1.1（2026-09-12）
> 阶段：M1 已完成并验证；M2 实现完成，真实实验待预注册后运行。

## 1. 已确认目标

将当前“终局一次盲评 → 单标量 reward → 单条记忆”的近似 contextual-bandit 回路升级为：

```text
完整路由轨迹采集
→ 轨迹结束后统一评估
→ 逐路由决策奖励与反馈
→ 确定性信用分配
→ 带生命周期的记忆更新
→ 后续检索与 Judge logits 调制
```

可学习 step 定义为一次真实 Router 档位决策，不把 token、普通工具调用或模型内部推理作为 step。

## 2. 实施顺序

### M1：兼容重构

1. 引入 `DecisionStep`、`RouteTrajectory`、`StepEvaluation`、`TrajectoryEvaluation`、`StepCredit` 数据契约；
2. 引入可替换的 `CreditAssigner`；单步轨迹严格退化为 `G=r`；
3. 引入 `MemoryUpdater` 与 active/provisional/quarantine 生命周期；
4. 现有单步真实 harness 改走统一轨迹接口；
5. 保持既有 Judge、决策数学、RNG 位相、六键 memory entry、JSONL 消费方和失败语义兼容。

### M2：多轮轨迹

1. 定义并加入多轮任务格式；
2. 一条轨迹包含多个真实路由决策；
3. 外部 Evaluator 读取完整轨迹，一次输出 episode 评价及逐 step 语义评价；
4. 确定性代码计算逐步 reward 与 return；
5. 对比终局单标量、逐步标量、逐步结构化反馈三种记忆更新方案。

## 3. 信用分配基线

M1 提供两种实现：

- Identity：`G_t = r_t`，用于单步兼容；
- Discounted：`G_t = r_t + η Σ(k>t) γ^(k-t) r_k`。

M2 初始建议参数：`gamma=0.8`、`delayed_weight=0.5`。后续成功不能完全洗掉前序错误降档。

## 4. 记忆准入基线

- `confidence >= 0.80`：active；
- `0.60 <= confidence < 0.80`：provisional；
- `confidence < 0.60`、评估无效或基础设施失败：quarantine/仅审计；
- 仅 active 记忆参与在线检索和 logits 调制。

为保持冻结的六键 entry 契约，生命周期、逐步反馈、推荐档位和失败标签先保存在 memory sidecar metadata 中，不改变：

```text
intent_class, signature_tokens, tier, G, ts, episode_id
```

## 5. M1 验收标准

- 旧 `learn()`、`decide()` 和 memory JSONL 格式不破坏；
- 单步 trajectory 的 `G` 与旧 reward 数值一致；
- `decide()` 不新增随机数消耗；
- Evaluator/执行失败仍不写记忆；
- provisional/quarantine 不参与检索，promote 后可参与；
- 快速测试与完整离线测试通过。

## 6. 暂不纳入 M1

- critic/价值网络；
- 把自由文本反馈直接注入本地 Judge prompt；
- 工具调用级 action；
- 多智能体评估；
- 对既有冻结实验结果的回写。

## 7. M2 设计（v1.1 冻结）

### 7.1 三臂定义（只改 mode C 的记忆更新，路由/执行/评估输入完全一致）

| 臂 | 评估方式 | 信用分配 | 记忆写入 | 记忆密度 |
|---|---|---|---|---|
| T1 | terminal-only（旧盲评，每轨迹末轮一次） | identity（G=r） | DirectMemoryUpdater | 每轨迹 1 条（末轮） |
| T2 | per-step（轨迹级评估器，每轨迹一次调用输出逐步评价） | discounted（γ=0.8, η=0.5） | DirectMemoryUpdater | 每轮 1 条 |
| T3 | per-step | discounted（γ=0.8, η=0.5） | LifecycleMemoryUpdater（置信度准入） | 每轮 1 条（active/provisional/quarantine） |

### 7.2 轨迹级评估器盲评边界

- **可见**：初始目标（首轮消息+checklist）、每轮用户消息、路由档位、回复（每轮截断 800 字）；
- **不可见**：执行模型名、价格、usage——D8 模型盲评原则保持；档位是路由层抽象，判断 under/over-routed 必需；
- 输出：episode 评价（success/quality/summary）+ 每轮 `routing_verdict`/`local_quality`/`recommended_tier`/`failure_tags`/`feedback`/`certainty`；
- 校验严格：steps 数必须等于轮数、turn_index 对位、verdict/quality 枚举合法；任一失败 → None → 该轨迹零记忆写入。

### 7.3 逐步奖励合成（确定性代码，不让 LLM 生成数值）

每轮 reward = 同一 D3 公式（0.6·quality + 0.3·cost_saving）作用于该轮 `local_quality` 与该轮真实执行模型/usage——与 M1 终局奖励同尺度，臂间 G 可比。评估器的 `contribution` 类语义字段 v1 不进入奖励（避免尺度漂移），只入记忆 sidecar。

### 7.4 失败语义（按轨迹为单位）

- 轨迹中任一轮执行失败，或轨迹级评估失败 → 整条轨迹零记忆写入（信用分配单位是轨迹）；
- 单轮失败仍写该轮 record（exec_error/eval_error），不影响其他轨迹。

### 7.5 多轮任务集

`eval/multiturn_tasks.jsonl`：8 轨迹 28 轮（M01–M08），行带 `traj_id`/`turn_index`；轮内档位分布 simple→reasoning 覆盖升级、修正、扩展、收尾四种轮型；无 `traj_id` 行自动退化为单轮轨迹（M1 兼容）。

## Changelog

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-09-12 | v1.0 | 冻结先 M1 后 M2 的记忆回路优化路线、数据契约、信用分配与生命周期基线 |
| 2026-09-12 | v1.1 | 冻结 M2 设计：三臂定义、轨迹级评估器盲评边界（可见档位/不可见模型）、逐步奖励=同 D3 公式逐轮应用、按轨迹的失败语义、多轮任务集格式 |
