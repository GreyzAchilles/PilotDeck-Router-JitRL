# JitRL Router — C1 保守策略实验结果

> 状态：v1.0（2026-09-11）
> 预注册：`S2-C1-PREREGISTRATION.md` v1.0
> 结论：C1 min-neighbor gate 未通过预注册判据（P1–P5 为 3/5）；按停止条件不在本数据上继续调参。

---

## 1. 实验定义

C1 唯一改动：

```text
min_neighbors: 1（C0）→ 3（C1）
```

当 `n_retrieved < 3` 时，不做 logit 调制，退化为本地 Judge。其余 Judge、任务顺序、模型映射、reward、seed、检索参数、`max_tokens_exec=1024`、Evaluator 与执行温度均与 C0 冻结一致。

正式输入：

- B：`logs/s2_full_B_family_order.jsonl`
- C0：`logs/s2_full_C_main24_retry2.jsonl`
- C1：`logs/s2_full_C1_main24.jsonl`

三组均为同序 24 条、24/24 评估成功。C1 首次完整尝试成功，无需 retry。

---

## 2. 核心结果

| 指标 | B | C0 | C1 gated |
|---|---:|---:|---:|
| 平均质量 | 2.6667 | 2.7917 | 2.4167 |
| 执行成本 | $0.028333 | $0.029406 | $0.033047 |
| 误档率 | 70.83% | 70.83% | 70.83% |
| flips | 0 | 2 | 1 |
| finish=length | 16 | 16 | 17 |
| 平均 JitRL latency | 0 | 0.2444ms | 0.2233ms |
| 最大 JitRL latency | 0 | 0.6664ms | 0.4845ms |

B→C1：

- 质量差：`−0.25/5`；
- 执行成本增加：`+$0.004714`，即 `+16.64%`；
- 路由准确率不变；
- 0 条质量提升、3 条下降、21 条持平；质量差全部来自相同 tier 的生成采样，并非 C1 flip。

C0→C1：

- 质量：2.7917 → 2.4167（−0.375）；
- 成本：$0.029406 → $0.033047（+12.38%）；
- flips：2 → 1。

---

## 3. Gate 行为

C1 的 24 条任务中：

- 18 条 `no_neighbors`；
- 4 条 `below_min_neighbors`：T06(n=1)、T18(n=1)、T07(n=2)、T19(n=2)；
- 2 条允许调制：T08(n=3)、T20(n=3)；
- 1 条最终 flip：T20。

Gate 集合与预注册预测完全一致；T06、T19 的 C0 flips 确实被抑制。但预测的“flips 2→0”被证伪，因为门控改变了写入记忆的 tier/reward，产生了后续级联。

---

## 4. T20 级联机制

C1 中 T20：

```text
base simple → complex
```

其 3 个相似邻居为 T17/T18/T19。C1 将 T19 门控为 simple，并写入 simple 的负 reward；于是邻域中三条记忆全部属于 simple。未见的 complex 得到相对优势，归一化后再次饱和为 `A_norm(complex)=1`，最终压过 Judge 的 simple logit。

结果：

| 指标 | B T20 | C1 T20 |
|---|---:|---:|
| tier | simple | complex |
| quality | 1 | 1 |
| exec cost | $0.000536 | $0.004725 |
| cost delta | — | +$0.004190 |
| finish | length | length |

即约 8.8 倍执行成本没有换来质量改善。min-neighbor gate 把有害 flip 从 T06/T19 延后到 T20，而非消除根因。

---

## 5. 预注册 P1–P5 判定

| 判据 | 实测 | 结果 |
|---|---:|---|
| P1 C1成本 ≤ B+$0.001 | $0.033047 > $0.029333 | **FAIL** |
| P2 质量差 ≥−0.3 | −0.25 | PASS |
| P3 误档率 ≤B+5pp | +0pp | PASS |
| P4 max JitRL latency ≤500ms | 0.4845ms | PASS |
| P5 每次flip质量不降且成本≤B+$0.001 | T20成本超限约$0.003190 | **FAIL** |

**总判定：NOT PASSING（3/5）。**

按照预注册停止条件：停止，不在同一数据上继续调 `min_neighbors`；任何新方案必须单独建立 C2 预注册。

---

## 6. 信息性 D10

该计算不替代 C0 的正式 D10 结果：

- 成本节省：−16.64%，FAIL；
- 质量差：−0.25，PASS；
- 误档率差：0pp，PASS；
- 最大 JitRL latency：0.4845ms，PASS。

信息性判定仍为 3/4，成本项失败。

---

## 7. 结论与后续含义

单纯使用 `min_neighbors=3` 不足以解决成本问题。原因不是 gate 没有工作，而是记忆策略存在级联：被 gate 的决策仍会按 base tier 写回记忆，改变未来 Q/Â，未见 tier 在负 V 下仍可能得到归一化后的最大正优势。

若继续做 C2，候选方向应针对根因，而不是继续扫描 gate 阈值：

1. 对“未观察过的 tier”使用中性优势 0，而不是由 `Q=0` 与负 V 产生正优势；
2. 只允许有直接 tier 证据的动作参与调制；
3. 将探索限制为成本单调或相邻 tier；
4. 将调制后增量做显式 clip，但需重新定义归一化后的数学并预注册。

C1 结果应与 C0 一并纳入 Web Demo，以展示“合理的局部修复为何因记忆级联仍失败”，而非只展示成功数据。

---

## 8. 资产与复现

仓库资产：

- `eval/analyze_c1.py`
- `eval/c1/results-summary.json`
- `eval/c1/results-tasks.csv`
- `eval/c1/charts/cost-comparison.svg`
- `eval/c1/charts/quality-comparison.svg`
- `eval/c1/charts/gate-modulation.svg`
- `eval/c1/charts/bc1-paired-exec-cost.svg`

```bash
python eval/analyze_c1.py
python -m pytest harness jitrl_core local_judge eval -q
```

当前测试：`196 passed`。

---

## Changelog

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-09-11 | v1.0 | 首版 C1 结果：同序 B/C0/C1、gate与级联机制、P1–P5 3/5、信息性D10、停止结论与C2方向 |
