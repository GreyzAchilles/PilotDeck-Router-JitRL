# JitRL Router — Session 3 Table-8 式消融结果

> 状态：v1.0（2026-09-11）
> 预注册：`S3-TAB8-PREREGISTRATION.md` v1.0
> 实验：T8-LM（Prompt 外闭式 logit modulation）vs T8-PM（Prompt memory injection）
> 结论强度：n=24、每组单次完整运行，仅作方向性证据，不声称统计显著。

---

## 1. 正式实验完整性

两个正式组均按冻结顺序完成 24/24：

- T8-LM：`logs/s3_ablation_T8_LM.jsonl`；
- T8-PM：`logs/s3_ablation_T8_PM.jsonl`；
- 冻结 memory trace：`eval/ablation/frozen-c0-memory-trace.jsonl`；
- trace SHA-256：`817e5d8a4726062e5316ac507734245f6394e3d191e2525225e80621b9ecc5c8`；
- 聚合：`eval/ablation/results-summary.json`、`results-tasks.csv`、4 张 SVG。

分析器硬校验全部通过：任务顺序、配置、memory hash、24/24 完整性、机制纯度、LM 数学、PM prompt 字节完整性、两组 prefix/retrieval 公平性。

两组都使用正式 C0 的冻结轨迹做 prefix replay：第 i 条任务只读取 `source_order < i` 的历史，避免当前任务自身 outcome 和未来 outcome 泄漏；两组均不写 memory。

---

## 2. 核心结果

| 指标 | T8-LM | T8-PM |
|---|---:|---:|
| 任务 / Evaluator 成功 | 24 / 24 | 24 / 24 |
| 平均质量（1–5） | 2.5417 | 2.5417 |
| useful / neutral | 10 / 14 | 10 / 14 |
| 执行成本 | $0.032653 | **$0.027993** |
| 平均 reward | 0.0672 | 0.0782 |
| GT tier accuracy | 29.17% | 29.17% |
| 误档率 | 70.83% | 70.83% |
| retrieval hits | 6 | 6 |
| memory influence / flip | 1 | 0 |
| tier agreement | 23/24 | 23/24 |
| finish=length | 15 | 16 |
| 空响应 | 14 | 14 |
| Judge 调用总数 | 24 | 48 |
| mean deciding Judge latency | 1091.2ms | 1051.5ms |
| mean/max JitRL/retrieval 后处理 | 0.1918 / 0.4434ms | 0.0107 / 0.0689ms |

LM 与 PM 平均质量完全相同。LM 执行成本比 PM 高 `$0.004660`，即相对 PM 增加约 16.65%。按预注册结论规则，本次方向性结果偏向 prompt injection。

但这个结论必须结合行为分解理解：**PM 的 prompt memory 并没有改变任何一个原始 Judge tier；成本差几乎完全来自 LM 在 T19 的一次有害升级。** 因此数据不支持“prompt injection 已有效利用记忆”，只支持“在本次冻结轨迹中，LM 的一次探索性 flip 比 PM 的零影响更贵，且没有质量收益”。

---

## 3. 唯一 tier 分歧：T19

两组 23/24 tier 一致，唯一分歧为 T19：

| 指标 | T8-LM | T8-PM |
|---|---:|---:|
| family / GT | refactor / reasoning | refactor / reasoning |
| retrieved neighbors | 2 | 2 |
| base tier | simple | simple |
| 最终 tier | **complex** | simple |
| quality | 1 | 1 |
| 执行成本 | $0.004730 | $0.000536 |
| 成本差 | +$0.004194 | — |
| finish | length | length |

T8-LM 复现了 C0 的核心失败模式：小邻域下闭式优势更新把 simple 推到 complex，成本增加约 `$0.004194`，质量和 GT 匹配均未改善。T8-PM 注入相同的两条历史摘要，但 Judge 仍选择 simple。

全组成本差 `$0.004660` 中，T19 单条占约 90%；其余同 tier 差异主要是 temperature=0.7 下 usage 采样波动，不归因于机制。

---

## 4. Prompt injection 的实际行为

T8-PM 在 6 条 retrieval-hit 任务上注入白名单 memory block：T06、T18、T07、T19、T08、T20。

结果：

- 0/6 改变原始 Judge tier；
- 24 条任务整体 0 flips；
- GT accuracy 仍为 29.17%；
- continuation 仍不属于本实验，冻结事实保持 0/5；
- prompt injection 没有修复 MiniCPM5-1B 的 simple 偏置。

这表明以 `tier + G + similarity + per-tier count/mean G` 构成的短记忆摘要，在当前 1B Judge 和 Method B 设置下影响不足。它避免了 LM 的 T19 高成本 flip，但其“更便宜”来自没有改变决策，而非观察到更好的记忆推理。

---

## 5. Judge 调用与延迟归因

T8-LM 每条任务调用一次 Method B，即四个 llama.cpp 请求；T8-PM 为审计同时调用 injected prompt 和 raw prompt，因此每条任务调用两次 Method B，共八个 llama.cpp 请求。

- T8-LM：24 次 Judge API 级调用；
- T8-PM：48 次 Judge API 级调用；
- PM injected deciding Judge mean：1051.5ms；
- PM 额外 raw Judge mean：818.5ms；
- LM JitRL 后处理 max：0.4434ms，远低于 500ms。

本地 Judge API 成本记 0，但 PM 为正式实验审计付出了约双倍本地推理请求。这个额外 raw 调用不是部署 prompt-injection 机制的必需成本；若线上只调用 injected prompt，PM 的一次决策 Judge latency 应参考 deciding Judge 的约 1051.5ms，而不是把两次调用相加后与 LM 的 JitRL 0.x ms 混报。

---

## 6. 预注册假设判定

| 假设 | 实测 | 判定 |
|---|---|---|
| H1 至少一种机制在 retrieval-hit 上产生 influence | LM=1，PM=0 | PASS |
| H2 LM 相对 PM 质量非劣，差值 ≥−0.3 | 0.0 | PASS |
| H3 成本比较 | LM $0.032653；PM $0.027993 | 方向偏向 PM |
| H4 LM JitRL max ≤500ms | 0.4434ms | PASS |
| H5 报告 Judge 稳定性 | base choice mismatch 0/24；logit 有数值漂移 | 完成审计 |

正式结论：**本次方向性证据偏向 prompt injection，但不是因为 prompt memory 改善了路由，而是因为 logit modulation 在 T19 产生了一次高成本、零质量收益的 flip。**

---

## 7. 与 C0/C1 的关系

- C0 在线学习中有 T06、T19 两次 simple→complex flips；
- C1 min-neighbor gate 抑制早期 flips，但写回级联在 T20 新增 simple→complex flip；
- T8 使用只读 C0 prefix replay，不在线写回，因此隔离了“作用机制”与“不同学习轨迹”；
- T8-LM 在当前真实 Judge 运行中只于 T19 flip；T06 未 flip，说明 Judge logits 跨运行有数值漂移；
- T8-PM 在 6 次检索中均未改变 tier，避免了有害 flip，也没有体现出学习收益。

C0/C1 的正式判定保持不变：C0 D10 3/4，C1 P1–P5 3/5；T8 不替代或回写这些结果。

---

## 8. 限制

1. n=24、单次运行，无统计显著性；
2. 执行 temperature=0.7，同 tier 质量和 usage 波动不能归因；
3. MiniCPM5-1B 强烈偏 simple，限制 prompt memory 的可观察作用；
4. 检索仍只在规则签名 sim=1.0 的离散命中上工作；
5. 1024-token 截断导致 15–16 条 `length`、14 条空响应，压低绝对质量；
6. PM block 格式只有一个预注册版本，未扫描措辞、长度或 few-shot 表达；
7. LM/PM 逐组运行，Judge logits 有数值漂移；本次 base choice 恰好 24/24 一致，但不能视为确定性保证；
8. PM 额外 raw Judge 仅用于审计，实际部署成本应与实验审计成本区分；
9. T8 冻结轨迹隔离了机制，却不测在线写回导致的长期轨迹分叉；C1 已显示该分叉可能产生级联。

---

## 9. 复现

```bash
python -m harness.run_ablation --arm T8-LM --judge llama --out logs/s3_ablation_T8_LM.jsonl
python -m harness.run_ablation --arm T8-PM --judge llama --out logs/s3_ablation_T8_PM.jsonl
python eval/analyze_ablation.py --lm logs/s3_ablation_T8_LM.jsonl --pm logs/s3_ablation_T8_PM.jsonl --out-dir eval/ablation
python -m pytest harness jitrl_core local_judge eval -q
```

当前测试：

```text
261 passed
```

---

## Changelog

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-09-11 | v1.0 | 首版 T8 正式结果：冻结 C0 prefix memory 下 LM/PM 均24/24；质量同为2.5417，PM成本低$0.004660；LM仅T19有害flip，PM 6次注入均未改变tier；完成H1–H5、延迟归因、限制与复现记录 |
