# JitRL Router — C1 保守策略实验预注册

> 状态：v1.0（2026-09-11，实验运行前冻结）
> 实验名称：C1 — min-neighbor gated JitRL
> 关系：C0 正式结果保持不变；C1 是针对 C0 成本目标未达标的显式后续实验，不回写或替换 `results.md` v1.0 中的 C0 数据。

---

## 1. 动机

正式 C0 中共有 6 次检索、2 次 tier flip。T06 与 T19 均在小邻域下由探索分支触发 `simple → complex`，合计增加执行成本约 `$0.008239`，质量变化为 0，导致 C0 相对 B 的成本节省为 `−3.78%`。

探索 bonus 在优势最大绝对值归一化后容易饱和到 `A_norm=1`，因此简单裁剪 `α/n` 未必能降低最终 `β·A_norm`。C1 选择更容易解释的 min-neighbor gate：记忆证据不足时完全退化为原生 Judge，不允许少量样本改写 tier。

---

## 2. 唯一策略变更

新增配置：

```text
min_neighbors = 3
```

规则：

```text
若 n_retrieved < min_neighbors：
    z_mod = z_base
    A_norm = 0
    chosen_tier = base_choice
否则：
    沿用 C0 的 V/Q/Â 与 z′ = z + β·Â
```

向后兼容默认值为 `min_neighbors=1`，用于精确保留 C0 的“有至少一条邻居即允许调制”语义。C1 运行时显式传 `--min-neighbors 3`。

`min_neighbors=3` 的先验依据：任务集每个 family 有 4 个变体，要求观察至少 3 条同签名历史经验后，才允许在第 4 条上改写 Judge。该选择与当前任务集结构绑定，外推时必须注明限制。

不修改：reward、`estimate_values`、探索 RNG、优势归一化、β、λ、α、检索、模型映射或 Judge 数学。

---

## 3. 冻结实验条件

- 任务顺序与正式 A/B/C 完全相同：

```text
T01,T05,T09,T13,T17,T21,
T02,T06,T10,T14,T18,T22,
T03,T07,T11,T15,T19,T23,
T04,T08,T12,T16,T20,T24
```

- Judge：llama.cpp `:18080`，MiniCPM5-1B，Method B 四请求；
- seed=42；k=10；β=5；λ=0.05；α=5；Jaccard threshold=0.5；z_min=−10；
- `max_tokens_exec=1024`；
- `max_tokens_eval=300`；
- `temperature_exec=0.7`；
- 执行模型映射、Evaluator、价格和 reward 公式与 C0 相同；
- 唯一实验变量：`min_neighbors: 1 → 3`。

---

## 4. 预注册判据

C1 定位为“保守、无害、接近 B 成本”的后续变体，不回溯改变 C0 的正式 D10 判定。

| # | 判据 | 通过标准 |
|---|---|---|
| P1 | 执行成本 | C1 总执行成本 ≤ B + $0.001 |
| P2 | 质量 | mean(C1) − mean(B) ≥ −0.3/5 |
| P3 | 误档率 | C1 ≤ B +5pp |
| P4 | JitRL 延迟 | max `jitrl_latency_ms` ≤500ms |
| P5 | flip 无害 | 每个 flip 的质量不低于 B 同任务，且成本不高于 B 同任务 +$0.001；0 flip 视为平凡通过 |

全部通过时，只表述为“C1 在 n=24 上提供无害平价的方向性证据”，不声称统计显著，也不声称满足 C0 原定的 ≥20% 成本节省目标。

---

## 5. 预期与证伪条件

运行前预期：

- n=1/2 的 T06、T18、T07、T19 被 gate；
- C0 的 T06/T19 flips 被抑制；
- n=3 的 T08/T20 仍允许调制；
- flips 预计由 2 降为 0；
- 质量预计基本不变，执行成本接近 B。

以上均是可证伪预测，不作为结果。若实际不符，如实报告，不在同一结果上调参。新的参数方案必须另立 C2 预注册。

---

## 6. 日志与运行可靠性

新输出不得覆盖 C0：

- `logs/s2_full_C1_main24.jsonl`
- `logs/s2_full_C1_main24_memory.jsonl`
- 若失败：使用 `_retry1`、`_retry2` 后缀。

每条记录新增或明确记录：

- `min_neighbors`；
- `modulation_active`；
- `gate_reason`；
- `n_retrieved`；
- `jitrl_latency_ms`。

由于 C1 有跨任务记忆依赖，任意执行或 Evaluator 失败都会使该次全序列仅作为故障审计；不得用单条 retry 拼接正式 C1。最多进行 3 次完整尝试，正式采用首个 24/24 成功的完整序列。

分析资产放入独立目录 `eval/c1/`，不得覆盖 C0 的 `eval/results-summary.json`、CSV 或图表。

---

## 7. 停止条件

- 三次完整尝试仍无法得到 24/24：停止并报告基础设施失败；
- C1 P1–P5 任一失败：如实报告并停止，不在本数据上继续调参；
- API 预算上限：三次尝试累计不超过 `$1.00`。

---

## Changelog

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-09-11 | v1.0 | 运行前冻结 C1 min-neighbor gate：唯一变量、完整参数、P1–P5、输出命名、全序列重试和停止条件 |
