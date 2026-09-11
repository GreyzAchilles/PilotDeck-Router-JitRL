# JitRL Router — Session 2 全量实验结果

> 状态：v1.0（2026-09-11）
> 实验范围：A/B/C × 24 条主任务，单 episode；5 条 continuation probe 单独统计。
> 结论强度：n=24，仅报告方向性证据，不声称统计显著。
> 海报边界：本文不处理海报，仅提供后续交付可引用的实验事实。

---

## 1. 实验设置

### 对照组

- **A — Static**：不调用 Router，固定以 `reasoning` tier 执行，即 `CPA/glm-5.3`。
- **B — Judge**：MiniCPM5-1B 本地 Judge 决定 tier，不使用跨轮记忆。
- **C — JitRL**：使用与 B 相同的本地 Judge logits，再根据非参数记忆检索结果做闭式 logit 调制并执行对应 tier。

B/C 均使用 llama.cpp `http://127.0.0.1:18080`、模型 alias `minicpm5-1b`、Method B 四请求 logprob 提取。执行回复由 CPA tier 映射模型生成，质量由独立 `CPA/gpt-5.6-sol` 盲评。

实际映射为 simple=`CPA/glm-5.3-flash`、medium=`CPA/opencode-v4-flash`、reasoning=`CPA/glm-5.3`、complex=`CPA/OpenBMB-5.3`。

### 冻结参数与公平性

- 主任务：T01–T24；continuation probe P01–P05 不执行生成，单独做 Judge 路由探针。
- A/B/C 使用完全相同的家族分块任务顺序：`T01,T05,T09,T13,T17,T21,...,T24`。
- `max_tokens_exec=1024`，三组全程固定；截断不事后补跑，而由 `finish_reason=length` 暴露并进入 Evaluator 质量结果。
- 执行温度沿 harness 默认值 0.7；因此相同 tier/model 的 B/C 回复仍有采样方差。
- Evaluator 成本是测量开销，与执行模型成本分开报告。

正式输入：

- A：`logs/s2_full_A_main24.jsonl`
- B：`logs/s2_full_B_family_order.jsonl`
- C：`logs/s2_full_C_main24_retry2.jsonl`

聚合资产：

- `eval/results-summary.json`
- `eval/results-tasks.csv`
- `eval/charts/*.svg`

---

## 2. 核心结果

| 指标 | A：Static | B：Judge | C：JitRL |
|---|---:|---:|---:|
| 任务 / 评估成功 | 24 / 24 | 24 / 24 | 24 / 24 |
| 平均质量（1–5） | 2.2083 | 2.6667 | **2.7917** |
| useful / neutral / harmful | 8 / 16 / 0 | 11 / 12 / 1 | **13 / 11 / 0** |
| 执行成本总计 | $0.088888 | **$0.028333** | $0.029406 |
| 平均执行成本 | $0.003704 | $0.001181 | $0.001225 |
| Evaluator 预算 | $0.147676 | $0.179900 | $0.188448 |
| GT tier 准确率 | 不适用 | 7/24 = 29.17% | 7/24 = 29.17% |
| 误档率 | 不适用 | 70.83% | 70.83% |
| C memory | — | — | 24/24 |
| C 调制 flips | — | — | 2 |
| 平均 Judge 延迟 | 0（跳过） | 见原始聚合 | 2311.0ms |
| 平均 JitRL 后处理延迟 | 0 | 0 | **0.2444ms** |
| 最大 JitRL 后处理延迟 | 0 | 0 | **0.6664ms** |

A 的 `chosen_tier=reasoning` 只是静态执行语义。A 日志中的 GT coincidence 不能解释为 Router accuracy。

---

## 3. B 与 C 的直接比较

### 3.1 质量

```text
C − B = +0.125 / 5
```

- 2 条质量提升；
- 0 条质量下降；
- 22 条持平。

C 的聚合质量没有下降，满足 D10 的 `C−B ≥ −0.3` 质量约束。但该 `+0.125` 不能直接归因于 JitRL：

- JitRL 真正改变执行 tier 的 2 条任务质量变化合计为 0；
- 1 分提升来自 B/C 之间本地 Judge 基础选择漂移的任务；
- 另外 2 分提升来自相同 tier/model 的随机重采样。

因此应表述为：**C 在本次单 episode 方向性实验中质量未劣于 B，但没有观察到可归因于 logit 调制的质量增益。**

### 3.2 成本

```text
B 执行成本 = $0.028333
C 执行成本 = $0.029406
(B − C) / B = −3.78%
```

C 相对 B 没有省钱，反而增加约 3.78%，与 D10 的“成本节省 ≥20%”目标相反。

成本变化分解：

- 2 次 JitRL flips：`+$0.008239`；
- B/C Judge 基础选择漂移：`−$0.007347`；
- 相同 tier 的生成采样/usage 差异：约 `+$0.000180`。

相对于 A 静态 reasoning 基线，B/C 仍显著更便宜；但该收益主要来自弱 Judge 大量选择 simple，而不是 C 的跨轮学习：

- B 较 A 执行成本约降低 68.1%；
- C 较 A 执行成本约降低 66.9%。

### 3.3 路由

B 与 C 都是 7/24 命中 GT，误档率均为 70.83%，所以 C 相对 B 没有恶化，满足“误档率不超过 B +5pp”。

本地 Judge 跨运行并非完全确定：T13、T22 的 B chosen tier 与 C 的 `base_choice` 不同。分析将这两条标为 Judge drift，未把它们算作 JitRL flips。

---

## 4. JitRL 检索和翻转

C 的记忆写入为 24/24。共有 6/24 任务检索到邻居：

```text
T06, T18, T07, T19, T08, T20
```

- 命中时平均检索 2.0 条；
- top similarity 均为 1.0；
- 6 次检索中 2 次导致执行 tier 翻转。

两次 flips：

| 任务 | GT | base → C | 质量 B→C | 执行成本变化 | 结果 |
|---|---|---|---:|---:|---|
| T06 code_gen | medium | simple → complex | 1 → 1 | +$0.004045 | 错→错；无质量收益 |
| T19 refactor | reasoning | simple → complex | 1 → 1 | +$0.004194 | 错→错；无质量收益 |

两条均调用 `CPA/OpenBMB-5.3`，但在 1024 tokens 达到 `finish_reason=length` 且最终 response 为空。它们合计增加约 `$0.008239` 成本，没有改善 GT 或质量。

这与 Session 1 的风险判断一致：小邻域下探索 bonus 可能主导调制，导致从 simple 直接跳到 complex。当前数据支持下一阶段做一项显式、可审计的保守改动或消融，例如 min-neighbor gate 或 bonus clip；不得悄悄修改冻结数学后重报本组结果。

---

## 5. 学习曲线

C 的检索活动随记忆累积增加：三个三分段的检索条目总数为 1 → 3 → 8。但质量为 3.625 → 3.25 → 1.50，不能解释为“越学越差”或遗忘，因为任务序列按 family 分块，后半部分集中包含更难的 reasoning/complex 任务。

同理，前半/后半质量为 3.4167 → 2.1667，受任务难度分布严重混淆。当前实验只能证明：

- 记忆确实累积并被检索；
- 检索能改变 logits 和最终 tier；
- 本次两个实际 flips 没有产生预期的质量/成本收益。

不能据此声称学习曲线显著改善。

---

## 6. Continuation probes

P01–P05 均显式传入正确的上一轮 tier，而不是把短消息作为无上下文新任务：

| Probe | previous / expected | chosen | 命中 |
|---|---|---|---|
| P01 | medium | simple | 否 |
| P02 | reasoning | simple | 否 |
| P03 | medium | simple | 否 |
| P04 | reasoning | simple | 否 |
| P05 | reasoning | simple | 否 |

结果：**0/5**，平均 Judge 延迟 901.2ms。

对照检查显示 continuation prompt 确实提高了 previous tier 的相对 logit，但提升不足以超过 MiniCPM5-1B 对 simple 的强偏置。因此这是可复现的 Judge 能力限制，而非 previous_tier 未传递的管线 bug。

原始记录：

- `logs/s2_continuation_probes.jsonl`
- `logs/s2_continuation_probes_summary.json`

---

## 7. D10 验收

| 标准 | 实测 | 结论 |
|---|---:|---|
| C 较 B 执行成本节省 ≥20% | **−3.78%**（成本增加） | **FAIL** |
| C vs B 质量差 ≥−0.3/5 | +0.125 | PASS |
| C 误档率 ≤ B +5pp | 70.83% vs 70.83%，+0pp | PASS |
| JitRL 后处理延迟 ≤500ms | max 0.6664ms；mean 0.2444ms | PASS |

**总判定：未达标，3/4 通过。**

项目仍证明了“冻结 Judge + 非参数记忆 + 闭式 logit 更新”的端到端可执行性和极低后处理开销，但当前默认探索行为没有达到相对 B 降本目标。

---

## 8. 故障与重试审计

为了降低网络不稳定和网关故障的影响，所有阶段均及时落盘，失败尝试未覆盖：

- A 首次误含 5 条 probes 的 29 条运行：非正式，仅留审计；随后正式运行 24 主任务。
- B 早期数字顺序批次：T14 medium 执行遭上游 HTTP 500 / unexpected EOF；单条 retry 成功。后续因顺序公平性问题，完整按 A/C 同序重跑 B，24/24 成功并作为正式 B。
- C attempt 1：T14、T22 执行 HTTP 500，22/24；
- C attempt 2：T20、T24 Evaluator transport/parse/validate 失败，22/24；
- C attempt 3：24/24 成功，作为正式 C。

没有把 C 的单条 retry 拼入失败批次，因为 C 的后续决策依赖前序 memory；正式采用一次完整成功的全序列，保持学习轨迹一致。

---

## 9. 已知限制

1. n=24、每任务单 episode，没有统计显著性。
2. 执行温度 0.7，同 tier/model 的 B/C 输出有采样方差，限制质量归因。
3. MiniCPM5-1B 强烈偏 simple，B/C GT accuracy 仅 29.17%，continuation 0/5。
4. 检索只在归一化签名 sim=1.0 时命中，泛化有限。
5. 1024-token 预算导致大量 `length` 和空响应；固定预算保证了组间公平，但降低了绝对质量。
6. 小邻域探索 bonus 导致两次 simple→complex 的高成本、无收益翻转。
7. llama Judge 跨运行存在基础选择漂移；T13/T22 的差异不能归因于 JitRL。
8. Evaluator 是最大 API 预算项，而且 LLM-as-judge 本身存在漂移风险。
9. 按 family 分块的序列使“前半 vs 后半”学习曲线与任务难度混淆。

---

## 10. 复现

```bash
python eval/analyze_results.py
python -m pytest harness jitrl_core local_judge eval -q
```

当前完整测试结果：

```text
150 passed
```

---

## Changelog

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-09-11 | v1.0 | 首版全量实验报告：正式同序 A/B/C ×24、continuation probes、质量/成本/误档/延迟、检索与 flips、D10 3/4 判定、网络故障审计与限制 |
