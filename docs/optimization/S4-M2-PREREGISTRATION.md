# S4-M2 轨迹级记忆更新实验预注册

> 状态：v1.1（2026-09-12，运行前修订：基础设施强制换档位映射）
> 依据：`MEMORY-LOOP-DECISIONS.md` v1.1 第 7 节；实现在本冻结版代码上运行，运行后不回改判据。

## 1. 研究问题

在多轮路由轨迹上，逐步轨迹评估 + 信用分配的记忆更新（T2/T3），相对终局单标量更新（T1），是否改善第二集的路由质量且不增加执行成本？

## 2. 设计

- 任务：`eval/multiturn_tasks.jsonl`（8 轨迹 28 轮，含升级/修正/扩展/收尾轮型）；
- 臂：T1 / T2 / T3（定义见决策文档 7.1，均为 mode C；路由 Judge、执行模型、seed=42、min_neighbors=1 三臂完全一致）；
- Judge：本地 llama（MiniCPM5-1B，`JITRL_JUDGE_ENDPOINT`，需先启动）；
- episodes=2，记忆跨集保留；`--episodes 2`；
- 评估器：CPA/gpt-5.6-sol（T1 走旧单响应盲评；T2/T3 走轨迹级盲评，一次调用输出 episode+逐步评价）；
- 命令（冻结）：

```bash
python -m harness.run_real --mode C --arm T1 --tasks eval/multiturn_tasks.jsonl \
    --judge llama --episodes 2 --out logs/s4m2_T1.jsonl --memory-out logs/s4m2_T1_mem.jsonl
# T2/T3 同参数替换 --arm
```

- 冒烟：M08 单轨迹（3 轮）× T2 × 1 episode，不入正式；冒烟失败先修基础设施再重跑冒烟，不直接进入正式。

## 3. 预注册判据（运行前冻结）

- **P1（主判据，路由质量不降）**：episode-2 的轮级 gt_tier 准确率：T2 ≥ T1 − 2pp 且 T3 ≥ T1 − 2pp；
- **P2（主判据，学习增益）**：episode-2 平均 local_quality（T2/T3 同一轨迹评估器口径）：T2 或 T3 至少一个 > T1（T1 用其终局 quality 的每轮记录口径，文档标注差异）；
- **P3（生命周期无害性）**：T3 的 active 记忆数 ≥ T2 记忆数的 60%，且 T3 episode-2 准确率 ≥ T2 − 2pp；
- **P4（信息性，不计 PASS/FAIL）**：under_routed 轮占比 episode-2 vs episode-1 的变化（T2/T3 vs T1）；
- **P5（成本护栏）**：T2/T3 的 total_exec_cost_usd ≤ 1.10 × T1。

## 4. 停止与保护规则

- 任一臂 exec/eval 失败率 > 20%（按轮计）→ 停止，先修基础设施；失败轨迹不写记忆（实现保证）；
- 总支出（exec+eval，三臂+冒烟）> $1.50 → 停止并记录；
- 不回写 S2/S3/T8 冻结结论；本实验只做臂间比较（新任务集，无历史可比基线）。

## 5. 预算估算

- exec：28 轮 × 2 集 × 3 臂 ≈ 168 次，≈ $0.15–0.50；
- eval：T1 每轨迹 1 次 ×8×2 + T2/T3 每轨迹 1 次 ×8×2×2 ≈ 48 次，≈ $0.20–0.40；
- 冒烟 ≈ $0.03；合计 ≈ $0.40–1.00，上限 $1.50。

## 6. 输出资产

- 记录：`logs/s4m2_T{1,2,3}.jsonl`（含 traj_id/turn_index/n_turns/arm/step verdict 字段）；
- 记忆：`logs/s4m2_T{1,2,3}_mem.jsonl`（六键 schema；T3 的状态在 record `_summary.memory_status_counts`）；
- 结果文档：`docs/optimization/S4-M2-RESULTS.md`（运行后新增，不回写本文件）。

## 7. 修订 A（v1.1，2026-09-12 运行前冻结）：换 provider 执行映射

**触发**（第一次冒烟后探针确认）：

1. CPA 中继 `glm-5.3-flash` / `glm-5.3` 凭据池 5 小时限额（429，14:27:21 重置）；
2. `opencode-v4-flash`（spec 中档）已从中继上游**永久消失**（HTTP 400 unknown provider）——即使限额重置，spec 映射也无法恢复；
3. 项目所有者指示切换 provider1 测试。

**新档位映射（`--tier-map switched`）**，全部经 1024 token 执行预算探针验证（2026-09-12 ~11:20）：

| 档位 | spec 映射 | switched 映射 | 状态 |
|---|---|---|---|
| simple | CPA/glm-5.3-flash | **provider1/deepseek-v4-flash-vision-exp** | ✅ 79 out |
| medium | CPA/opencode-v4-flash（已消失） | **CPA/tokendance-v4.1-flash** | ✅ 89 out |
| complex | CPA/OpenBMB-5.3 | CPA/OpenBMB-5.3（不变） | ✅ 638 out |
| reasoning | CPA/glm-5.3 | **provider1/glm-5.3** | ✅ 617 out（推理前导 ~550） |

**不变项**：评估器仍为 CPA/gpt-5.6-sol（确认存活，429 仅限执行模型凭据池）；Judge、三臂定义、任务集、episodes=2、P1–P5 判据、停止规则全部不变。三臂使用同一 switched 映射，臂间可比性保持。

**计价**：switched 模型按同档价格类代理计价（deepseek≈glm-5.3-flash 0.15/0.50；tokendance≈opencode-v4-flash 0.30/1.20；provider1/glm-5.3≈glm-5.3 1.40/4.40），已入 `eval/pricing.json`。绝对成本为近似值；三臂同表，相对比较（P5 ≤1.10×T1）有效。reasoning 档与默认模型同价 → cost_saving=0 的结构性质与 spec 一致。

**修订后命令**（冒烟产物改名 `s4m2_smoke2.*`，原 429 冒烟保留审计）：

```bash
python -m harness.run_real --mode C --arm T2 --tasks eval/multiturn_tasks.jsonl \
    --task-ids M08T1,M08T2,M08T3 --judge llama --episodes 1 --tier-map switched \
    --out logs/s4m2_smoke2.jsonl --memory-out logs/s4m2_smoke2_mem.jsonl
# 正式三臂：--arm T1/T2/T3 全量任务，其余参数同 v1.0，加 --tier-map switched
```

**新风险登记**：provider1 自身配额未知——若运行中触顶，按既有停止规则（失败率>20%）中止；tokendance-v4.1-flash 质量未知，若中档质量系统性偏低，三臂同受影响、臂间比较仍成立，但 P2 绝对质量读数需谨慎解读。

## Changelog

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-09-12 | v1.0 | 运行前冻结：设计、命令、P1–P5 判据、停止规则与预算 |
| 2026-09-12 | v1.1 | 修订 A：基础设施强制换 `--tier-map switched` 跨 provider 映射（opencode-v4-flash 永久消失 + 5h 限额），评估器/判据不变，代理计价入 pricing.json，命令更新 |
