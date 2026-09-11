# JitRL Router — Session 3 Table-8 式消融预注册

> 状态：v1.0（2026-09-11，实验运行前冻结）
> 实验名称：T8 — Prompt Memory Injection vs Logit Modulation
> 关系：本实验独立于 C0/C1；不得覆盖其日志、聚合资产、结果文档或正式判定。
> 结论强度：24 条任务、单次完整运行，仅作为方向性证据，不声称统计显著。

---

## 1. 研究问题

在冻结同一个本地 Judge、相同历史记忆状态、相同任务和执行评估条件下，比较两种让历史经验影响 Router 的机制：

1. **T8-LM — Logit Modulation**：检索历史经验，计算 `V/Q/Â`，使用 `z′ = z + β·Â` 闭式更新 Judge logits；
2. **T8-PM — Prompt Memory Injection**：检索完全相同的历史经验，把白名单化的经验摘要加入 Judge prompt，由 Judge 自身重新产生四档 logits；不再叠加 logit modulation。

实验只比较“记忆作用机制”，不重新训练 Judge，不改变执行模型、Evaluator、reward 或任务集。

---

## 2. 冻结记忆轨迹与公平性

### 2.1 来源

记忆来源固定为正式 C0 完整成功序列：

- C0 运行：`logs/s2_full_C_main24_retry2.jsonl`；
- C0 memory：`logs/s2_full_C_main24_retry2_memory.jsonl`；
- Session 3 可复现副本：`eval/ablation/frozen-c0-memory-trace.jsonl`；
- 生成副本时为每条 memory 增加 `source_task_id` 和 `source_order`，原有 `intent_class/signature_tokens/tier/G/ts/episode_id` 不修改；
- 记录源文件与副本的 SHA-256。

### 2.2 Prefix replay，防止自身结果和未来结果泄漏

不把 C0 最终 24 条记忆一次性暴露给每个任务。对冻结顺序中的第 `i` 个任务，T8-LM 与 T8-PM 都只能读取 `source_order < i` 的 C0 前缀记忆。

因此：

- 当前任务不能检索自身 C0 outcome；
- 当前任务不能看到未来任务 outcome；
- 每个任务面对的记忆状态与正式 C0 在该任务决策前的状态一致；
- 两个实验组在每个任务上使用完全相同、只读的记忆前缀；
- 两组均不在线写回 memory，避免不同决策造成学习轨迹分叉。

冻结任务顺序：

```text
T01,T05,T09,T13,T17,T21,
T02,T06,T10,T14,T18,T22,
T03,T07,T11,T15,T19,T23,
T04,T08,T12,T16,T20,T24
```

---

## 3. 两个实验组

### 3.1 T8-LM — 冻结轨迹 logit modulation

每条任务：

```text
raw user message
→ 原始 Judge prompt
→ Method B 四请求得到 z_base
→ 从冻结 C0 prefix memory 检索
→ 估计 V/Q/Â
→ min_neighbors=1 gate
→ z_mod = z_base + β·Â
→ argmax tier
→ 执行、Evaluator、reward（但不写 memory）
```

无邻居时 `z_mod=z_base`，与 C0 默认语义一致。

### 3.2 T8-PM — prompt memory injection

每条任务：

```text
raw user message
→ 从同一冻结 C0 prefix memory 做同一检索
→ 生成确定性的白名单 memory block
→ 将 block 插入 Judge prompt
→ Method B 四请求得到 z_prompt
→ argmax tier
→ 执行、Evaluator、reward（但不写 memory）
```

T8-PM 不执行 `z + β·Â`。无邻居时不注入任何 block，prompt 必须与原始 `build_prompt(message, previous_tier)` 字节一致。

为进行 influence 与 Judge drift 审计，T8-PM 额外进行一次原始 prompt 的 Method B 调用，记录 `z_base`；真正执行 tier 只由 `z_prompt` 决定。额外本地调用不产生 API 费用，但其 latency 单独报告，不能混入 JitRL 后处理延迟。

---

## 4. Prompt injection 白名单与防泄漏

### 4.1 允许注入

仅允许使用当前检索返回的以下字段：

- 邻居序号；
- `tier`；
- `G`（固定小数精度）；
- 与当前任务的 Jaccard similarity；
- 邻居数量；
- 按 tier 聚合的 count 与 mean G。

注入文本必须明确：这些是相似历史请求的匿名 outcome，Judge 应将其作为参考而非硬规则。

### 4.2 禁止注入

不得注入或派生注入：

- 当前任务或历史任务的 `gt_tier`；
- `quality_checklist`；
- Evaluator 的 quality/verdict/certainty/checklist/capability 字段；
- 原始用户消息、模型回复或 exec preview；
- task id、family 标签、未来顺序、时间戳、episode id；
- API key、endpoint 凭据或 PilotDeck 配置内容。

`G` 是 Evaluator 与真实成本共同形成的标量 reward，是 JitRL 记忆本来包含的信息；其使用必须在结果文档中披露。

### 4.3 确定性格式

- 邻居按现有 retrieval 排序输出；
- 数字精度与字段顺序固定；
- block 进行占位符转义，不得触发 `{USER_MESSAGE}` 或 `{CONTINUATION}` 的二次替换；
- 每条记录保存 block 字符数与 SHA-256，不默认保存含用户原文的完整 prompt。

---

## 5. 冻结实验条件

两组除记忆作用机制外保持一致：

- Judge：MiniCPM5-1B，经 llama.cpp `http://127.0.0.1:18080`；
- Judge logit：Method B 四请求；
- tiers：`simple / medium / complex / reasoning`；
- seed=42；
- k=10；β=5；λ=0.05；α=5；Jaccard threshold=0.5；z_min=−10；
- min_neighbors=1；
- 主任务 24 条，顺序见 §2.2；不混入 continuation probes；
- `max_tokens_exec=1024`；
- `temperature_exec=0.7`；
- `max_tokens_eval=300`；Evaluator temperature=0.2；
- tier→model 映射、CPA 配置读取、价格、Evaluator 和 reward 与 C0 相同；
- 每组一个 episode；
- memory 只读，不写回；
- previous tier 按各组上一条实际选择传递。该状态可能因处理机制不同而分叉，属于机制效果的一部分，必须在结果中标注。

---

## 6. 输出命名与不可覆盖边界

代码与资产使用独立命名：

```text
harness/run_ablation.py
eval/analyze_ablation.py
eval/ablation/frozen-c0-memory-trace.jsonl
eval/ablation/frozen-c0-memory-trace.sha256.json
logs/s3_ablation_T8_LM.jsonl
logs/s3_ablation_T8_PM.jsonl
eval/ablation/results-summary.json
eval/ablation/results-tasks.csv
eval/ablation/charts/*.svg
```

失败重跑使用 `_retry1`、`_retry2` 后缀。严禁覆盖：

- `logs/s2_full_*`；
- `eval/results-summary.json`、`eval/results-tasks.csv`、`eval/charts/`；
- `eval/c1/`；
- `results.md` 与 `S2-C1-RESULTS.md` 的冻结事实。

---

## 7. 记录字段

每条 T8 记录至少包含现有 real episode 契约，并增加：

- `ablation_arm`：`T8-LM` 或 `T8-PM`；
- `memory_frozen=true`；
- `memory_prefix_size`；
- `memory_source_sha256`；
- `memory_written=false`；
- `retrieved_memories`：只包含 tier/G/similarity 的审计副本；
- `z_base`；
- T8-LM 的 `z_mod`；
- T8-PM 的 `z_prompt`；
- `injected_block_chars` 与 `injected_block_sha256`；
- 原始 Judge latency、prompt Judge latency、JitRL/retrieval 后处理 latency分栏；
- gate reason、chosen tier、base choice、flipped/influenced；
- 执行、Evaluator、成本、reward 与错误字段。

不保存 PilotDeck key；详细 prompt 默认不落盘。

---

## 8. 预注册指标与判读

### 8.1 主要指标

1. 平均质量与逐任务质量差；
2. 执行成本总计与逐任务成本差；
3. GT tier accuracy / misgrade rate；
4. retrieval-hit 数与 memory influence 数；
5. 相对各自 `z_base` 的 tier flip 数；
6. T8-LM 与 T8-PM tier agreement；
7. Judge 调用次数与 latency；
8. JitRL/retrieval 后处理 mean/max latency；
9. `finish_reason=length` 与空响应计数；
10. 失败、重试和 Judge drift 审计。

### 8.2 方向性假设

- **H1 机制影响**：两个机制至少之一在 retrieval-hit 任务上改变原始 Judge 选择；若两者均 0 influence，则判为“弱基策略/记忆格式主导，机制差异不可观察”。
- **H2 质量非劣**：`mean_quality(LM) - mean_quality(PM) ≥ -0.3`。
- **H3 成本比较**：报告 `(cost_PM - cost_LM) / cost_PM`，不预设一定达到 D10 的 20%；若 LM 质量满足 H2 且成本低于 PM，才可表述为“本次方向性证据偏向 LM”。
- **H4 延迟归因**：T8-LM 的 JitRL 后处理 max ≤500ms；T8-PM 的额外耗时归入 Judge prompt 机制，不得与 JitRL 0.x ms 后处理混报。
- **H5 稳定性**：报告两组原始 `z_base` 的 base-choice 不一致任务数以及每 tier logit 差异，避免把 Judge 跨运行漂移归因于记忆机制。

### 8.3 结论规则

- 若 `LM−PM` 质量差 ≥−0.3 且 LM 执行成本 < PM，结论为“本次方向性证据偏向 logit modulation”；
- 若 PM 在质量或成本上更优，则如实表述为“本次方向性证据偏向 prompt injection”；
- 若质量差绝对值 <0.3 且成本差绝对值 ≤$0.001，则表述为“未观察到实质差异”；
- 不把同 tier/model 下 temperature=0.7 的质量或 usage 波动归因于机制；必须单列真正 tier 改变任务。

---

## 9. 运行可靠性与停止条件

1. 先运行单元/集成测试，不消耗 CPA；
2. 再各做 1 条 retrieval-hit 冒烟，只验证 prompt、字段、Judge 与代理链路；冒烟日志不得作为正式结果；
3. 正式组必须按冻结顺序完整运行 24 条；
4. 任一组存在 execution 或 Evaluator 失败时，该组整次运行只作故障审计，不拼接单条 retry；
5. 每组最多 3 次完整尝试，采用首个 24/24 成功的完整序列；
6. 三次仍失败则停止并报告基础设施失败；
7. 不根据本次结果修改 block 格式、参数或阈值；任何变化必须另立 T8.1 预注册；
8. 预计总 API 预算不超过 $1.00；超过即停止。

---

## 10. 实现前测试门槛

至少覆盖：

1. `memory_block=None` 时 Judge prompt 与当前版本字节一致；
2. Method B 四个请求使用完全相同的注入 prompt；
3. prefix replay 对第 i 个任务只加载 `source_order < i`；
4. 当前 task/self outcome 与未来 outcome 不可检索；
5. 两组同任务得到相同 memory prefix 和 retrieval 集合；
6. 白名单渲染不包含 GT、quality checklist、Evaluator 结构、原文或 secret；
7. 无邻居时 PM prompt 不变、LM 不调制；
8. T8-PM 不叠加 logit modulation；T8-LM 不注入 prompt；
9. 冻结 memory 源文件和副本运行前后 hash 不变；
10. `memory_written=false`，运行后 memory 条数不变；
11. 任务顺序严格等于冻结顺序；
12. 分析器拒绝顺序、配置、hash 或 24/24 完整性不匹配的正式输入；
13. 全部既有 196 项测试继续通过。

---

## Changelog

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-09-11 | v1.0 | 运行前冻结 T8-LM vs T8-PM：采用正式 C0 prefix memory replay，排除自身/未来 outcome；冻结 prompt 白名单、防泄漏边界、组间对称条件、指标、结论规则、日志命名、完整序列重试与停止条件 |
