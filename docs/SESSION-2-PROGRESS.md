# JitRL Router — Session 2 当前进度

> 状态：v1.8（2026-09-11）
> 范围：Workstream A（CPA 真实执行 + Evaluator + real episode）阶段性落盘。
> 边界：llama.cpp 已在 `http://127.0.0.1:18080` 就绪并完成真实 Judge 校准；本记录不涉及海报工作。

---

## 1. 当前结论

Session 2 Workstream A 的代码实现、真实 CPA 执行、`CPA/gpt-5.6-sol` Evaluator 和本地 llama.cpp Judge 已全部连通。真实 Judge 的 3 episode 校准已通过，证明“Judge → C 路由 → CPA 执行 → Evaluator → reward → 记忆写回”端到端链路可用。

校准发现当前 llama-server `b10903-481c65f09` 会把 grammar 强制输出放在 `message.reasoning_content` 而非 `message.content`。`JudgeClient` 和独立 probe 已做兼容修复，logprob 提取数学未改变。当前具备进入全量 A/B/C × 24 主任务实验的条件。

---

## 2. 已完成实现

代码位于仓库 `harness/`：

- `cpa_client.py`：从 PilotDeck 配置读取 CPA provider，调用 OpenAI-compatible chat completions；返回回复、usage、finish reason 和延迟；包含超时、重试及 usage 缺失处理。
- `evaluator.py`：调用 `CPA/gpt-5.6-sol` 做结构化盲评；Evaluator 失败或返回非法结构时，该 episode 不写 C 记忆。
- `pricing.py`：读取 `eval/pricing.json`，按 input/cache/output usage 计算执行成本。
- `real_rewards.py`：按冻结奖励公式计算真实 episode reward：

```text
r = 0.6·quality + 0.3·cost_saving + 0.1·revision_penalty
```

- `run_real.py`：串联 Judge、A/B/C 路由、CPA 执行、Evaluator、成本/reward 与 C 记忆写回。

已落实的执行模型映射：

| tier/角色 | 模型 |
|---|---|
| simple | CPA/glm-5.3-flash |
| medium | CPA/opencode-v4-flash |
| reasoning | CPA/glm-5.3 |
| complex | CPA/OpenBMB-5.3 |
| A 组静态执行 | CPA/glm-5.3；记录 `chosen_tier=reasoning`、`route_mode=static` |
| Evaluator | CPA/gpt-5.6-sol |

配置读取、endpoint 拼接、Bearer 鉴权与 usage 字段已对照 `PilotDeck-src` 源码和本机 PilotDeck 配置 schema；未打印、复制或写入任何凭据。

---

## 3. API/日志契约

执行客户端主契约：

```text
CPAExecClient.completion(model, messages, ...)
  -> CompletionResult(
       content,
       usage={input_tokens, cache_read_tokens, cache_write_tokens,
              output_tokens, total_tokens, native_cost},
       model,
       finish_reason,
       latency_ms)
```

Evaluator：

```text
Evaluator.evaluate(task, response) -> EvaluationResult | None
```

结构化评估至少包含：

- quality：1–5；
- verdict：useful / harmful / neutral；
- certainty；
- capability_insufficient；
- checklist hits；
- 精简理由。

real episode JSONL 记录包含：

- episode/task 标识；
- chosen tier、route mode、执行模型；
- usage、执行成本、finish reason；
- Evaluator 结果；
- reward 与 breakdown；
- memory_written。

---

## 4. 测试状态

独立复验命令：

```bash
python -m pytest harness/tests jitrl_core/tests -q
```

结果（真实 Judge 兼容修复与 JitRL latency 插桩后）：

```text
119 passed in 0.81s
```

覆盖范围包括：

- CPA 成功响应；
- usage 缺失重试与耗尽；
- 非法 Evaluator JSON 拒收；
- 执行或评估失败不写 C memory；
- A 模式 route 语义；
- 成本与 reward 数学；
- secret 不进入错误和日志；
- `exec_finish_reason` 插桩。

---

## 5. 阶段性 3 episode 校准

条件：

- CPA 执行：真实；
- Evaluator：真实 `CPA/gpt-5.6-sol`；
- Judge：mock（因为 llama.cpp 尚未启动）；
- 模式：C。

结果：

| 指标 | 结果 |
|---|---:|
| Evaluator 成功 | 3/3 |
| 平均质量 | 3.67/5 |
| useful / neutral / harmful | 2 / 1 / 0 |
| reward 范围 | -0.334 ～ +0.822 |
| memory 写入 | 3/3 |
| 执行模型成本 | 约 $0.00098 |
| Evaluator 成本 | 约 $0.01895 |
| 本轮总 API 成本 | 约 $0.020 |

本地原始记录（`logs/` 已 gitignore）：

- `logs/s2_c_calib.jsonl`
- `logs/s2_c_calib_memory.jsonl`

这些数据不得作为最终 B/C 性能结论，也不得与 Session 1 routing-only stub 数据混用。

### 5.1 真实 Judge 校准（当前有效校准）

条件：真实 llama.cpp/MiniCPM5-1B Judge + 真实 CPA 执行 + 真实 `CPA/gpt-5.6-sol` Evaluator，模式 C；任务 T09、T13、T04。

| 指标 | 结果 |
|---|---:|
| Judge 健康状态 | `/health` ok；alias `minicpm5-1b` |
| Evaluator 成功 | 3/3 |
| 平均质量 | 4.667/5 |
| useful / neutral / harmful | 3 / 0 / 0 |
| reward 范围 | 0.5659 ～ 0.8660 |
| 平均 reward | 0.7659 |
| memory 写入 | 3/3 |
| GT 路由准确率 | 1/3 |
| flips / retrieval | 0 / 0（首轮无邻居） |
| 平均 Judge 延迟 | 2082.7ms |
| 执行模型成本 | $0.000987 |
| Evaluator 成本 | $0.021148 |
| 总 API 成本 | $0.022135 |

任务级结论：

- T09：GT medium → simple，质量 5，finish=`stop`，reward 0.8660；
- T13：GT reasoning → simple，质量 4，finish=`length`（1024 tokens），reward 0.5659；
- T04：GT simple → simple，质量 5，finish=`stop`，reward 0.8659。

有效日志：

- `logs/s2_c_calib_real_llama.jsonl`
- `logs/s2_c_calib_real_llama_memory.jsonl`

此前 mock Judge 校准只保留为管线开发记录，不作为后续性能基线。

---

## 6. 校准驱动修正

已做以下保守修正：

1. `harness/run_real.py` 默认执行输出预算由 512 提升至 1024 tokens。原因是 CPA gateway 上部分模型会先产生计入 completion usage 的 reasoning 内容，512 预算可能在正式 `content` 前耗尽。
2. episode 记录新增 `exec_finish_reason`，用于发现 `length` 截断。
3. `local_judge/client.py` 和 `local_judge/probe.py`：优先读取 `message.content`，为空时回退 `message.reasoning_content`，兼容 llama-server b10903 thinking chat template。logprob token 路径和 z 求和方法未改。
4. `harness/run_real.py` 新增 `jitrl_latency_ms`，只计 `engine.decide()`；summary 新增 `avg_jitrl_latency_ms`。A/B 记 0.0，不与四请求 Judge 延迟混算。

奖励数学与冻结超参数均未修改。

---

## 7. 当前风险与恢复点

1. **MiniCPM Judge 弱基策略**：真实校准 3 条全部选择 simple，GT 仅 1/3；与 Session 1 的 simple 偏置一致。正式 B/C 必须公平使用同一四请求 Method B。
2. **T13 截断**：误路由至 simple 后在 1024 tokens 达到 `finish_reason=length`。当前不因单样本静默提高预算；全量实验前需一次性预注册保持 1024 或提高至 1536–2048，运行中不得改变。
3. **llama-server 代际差异**：b10903 thinking template 使 grammar 输出进入 `reasoning_content`；代码已兼容，但跨 Session 比较原始 z 数值时必须注明 server/template 代际。
4. **Evaluator 是最贵单项**：真实校准评估成本 $0.021148，显著高于执行成本 $0.000987；继续保持每 episode 仅评一次。
5. **首轮 C 无检索增益**：这 3 条任务均无邻居，故 z 未调制、flips=0；学习效果要由按既定顺序运行的 24 主任务衡量。
6. **海报隔离**：海报由其他 session 负责，本工作流不得读取、修改或清理 `poster-a3/`。

---

## 8. 全量实验阶段进度

实验协议已冻结：仅运行 24 条主任务；5 条 continuation probe 排除并留待单独统计；`max_tokens_exec=1024`，A/B/C 全程固定，截断由 `finish_reason` 暴露并进入质量评估。

A 组静态基线已经完成（正式 24 主任务日志：`logs/s2_full_A_main24.jsonl`）：

| 指标 | A |
|---|---:|
| records / Evaluator 成功 | 24 / 24 |
| 平均质量 | 2.2083/5 |
| useful / neutral / harmful | 8 / 16 / 0 |
| 执行成本 | $0.088888 |
| Evaluator 成本 | $0.147676 |
| 平均 reward | -0.2375 |
| memory | 0 |

说明：A 固定执行 `reasoning`/`CPA/glm-5.3`；其 tier accuracy 仅为兼容字段，不作为路由准确率结论。曾先误含 5 条 probes 跑出 29 条日志 `logs/s2_full_A.jsonl`，已保留为非正式记录；正式比较只使用 `_main24` 文件。

B 组原生 Judge 基线已完成。早期数字 ID 顺序版本中 T14 遇到上游 HTTP 500，虽通过单条 retry 形成完整文件，但后续分析发现它与 A/C 的家族分块顺序不一致，不宜作正式配对基线。已及时按与 A/C 完全相同的家族分块顺序重跑 B，24/24 成功；正式 B 改为 `logs/s2_full_B_family_order.jsonl`。旧的 `s2_full_B_main24*` 与 T14 retry 仅保留故障审计。

| 指标 | B（正式同序 24） |
|---|---:|
| records / Evaluator 成功 | 24 / 24 |
| GT 路由准确率 | 7/24 = 0.2917 |
| 平均质量 | 2.6667/5 |
| useful / neutral / harmful | 11 / 12 / 1 |
| 执行成本 | $0.028333 |
| Evaluator 成本 | $0.179900 |
| 平均 reward | 0.1158 |
| memory | 0 |

1024-token 冻结预算造成的大量 `finish_reason=length` 仍是重要质量风险；为保持组间公平，不在 B/C 之间调整预算。

C 组在网络/网关波动下进行了三次独立的全序列尝试，均保留原始日志，避免用逐条拼接破坏 C 的记忆依赖：

- attempt 1：T14、T22 的 medium 执行上游 HTTP 500，22/24 评估成功；
- attempt 2：执行均成功，但 T20、T24 Evaluator transport/parse/validate 失败，22/24；
- attempt 3（正式采用）：24/24 执行与 Evaluator 成功，memory 24/24 写入。

正式 C 文件：`logs/s2_full_C_main24_retry2.jsonl` 与 `logs/s2_full_C_main24_retry2_memory.jsonl`。前两次仅作为故障审计记录，不参与最终比较。

| 指标 | C（正式 attempt 3） |
|---|---:|
| records / Evaluator 成功 | 24 / 24 |
| GT 路由准确率 | 7/24 = 0.2917 |
| flips | 2 |
| 平均质量 | 2.7917/5 |
| useful / neutral / harmful | 13 / 11 / 0 |
| 执行成本 | $0.029406 |
| Evaluator 成本 | $0.188448 |
| 平均 reward | 0.1512 |
| 平均 Judge 延迟 | 2311.0ms |
| 平均 JitRL 后处理延迟 | 0.2444ms |
| memory | 24/24 |

## 9. 分析与交付状态

已完成正式同序 A/B/C 聚合、5 条 continuation probe、JSON/CSV 与 5 张 SVG 图表；完整结论见 `results.md` v1.0。当前测试为 150 passed。

D10 最终为 3/4：质量、误档率和 JitRL 后处理延迟通过；C 相对 B 的执行成本节省为 −3.78%（即成本增加 3.78%），未达到 ≥20% 目标。

## 10. C1 保守策略状态

C1 已在 `S2-C1-PREREGISTRATION.md` v1.0 中运行前冻结，唯一变量为 `min_neighbors=3`。实现完成且默认 `min_neighbors=1` 精确保留 C0；记录新增 `min_neighbors`、`modulation_active`、`gate_reason`，summary 新增 config snapshot、modulated/gated counts。完整实现测试为 165 passed。

C1 首次真实全序列运行即 24/24 成功，无需基础设施 retry；正式日志为 `logs/s2_full_C1_main24.jsonl` 与 `logs/s2_full_C1_main24_memory.jsonl`。平均质量 2.4167、执行成本 $0.033047、Evaluator 成本 $0.175652、GT准确率 0.2917、flips=1、memory=24、modulated=2、gated=22、平均 JitRL latency=0.2233ms。

P1–P5 正式分析已完成，见 `S2-C1-RESULTS.md` v1.0：P1成本与P5 flip成本失败，其余通过，总判定3/5 NOT PASSING。T06/T19被gate抑制，但改变后的记忆使T20在n=3时新增simple→complex flip，成本+$0.004190、质量1→1。按预注册停止，不在本数据上继续调参。

## 11. 下一步

1. Web Demo 同时纳入 A/B/C0/C1、continuation probes、门控级联与故障审计数据。
2. 若继续优化，必须另立 C2 预注册；建议针对未见 tier 在负 V 下产生伪正优势的根因，而非继续扫描 gate 阈值。
3. Session 3 再做 prompt 注入 vs logit 调制消融。

---

## Changelog

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-09-11 | v1.0 | 首次落盘 Session 2 Workstream A：真实执行/Evaluator 实现、104 项测试、mock Judge 条件下 3 episode 校准、当前阻塞与 llama.cpp 恢复步骤 |
| 2026-09-11 | v1.1 | llama.cpp 就绪：完成真实 Judge + CPA + Evaluator 的 3 episode 校准；修复 b10903 `reasoning_content` 兼容；新增 JitRL 后处理延迟插桩；测试升至 119 项；登记真实成本、质量、路由和截断风险 |
| 2026-09-11 | v1.2 | 全量实验阶段检查点：冻结主任务 24 条与 `max_tokens_exec=1024`；A 组正式运行完成（24/24 评估成功，平均质量 2.2083，执行成本 $0.088888）；登记误含 probes 的非正式日志边界 |
| 2026-09-11 | v1.3 | B 组阶段检查点：24 条完整记录，平均质量 2.5833、路由准确率 0.2917、执行成本 $0.026316；登记 T14 上游 500 的独立 retry/合并审计链及 15/24 截断风险 |
| 2026-09-11 | v1.4 | C 组阶段检查点：三次全序列尝试均落盘；正式采用 attempt 3（24/24 评估成功、memory 24、平均质量 2.7917、执行成本 $0.029406、JitRL 0.2444ms）；前两次作为网关/Evaluator 故障审计 |
| 2026-09-11 | v1.5 | 公平性修正与 Session 2 实验收尾：发现早期 B 顺序不一致后按 A/C 同序重跑；正式 A/B/C 聚合、continuation probes、结果文档与图表完成；150 项测试通过；D10 3/4，成本目标未达标 |
| 2026-09-11 | v1.6 | C1 实现检查点：运行前预注册 min-neighbor gate=3；默认1保持C0；新增门控记录/summary字段与15项测试，完整测试165通过；尚未运行真实C1 |
| 2026-09-11 | v1.7 | C1运行检查点：首次完整真实全序列24/24成功，无需retry；质量2.4167、执行成本$0.033047、flips1、modulated2、gated22、memory24；等待P1–P5正式分析 |
| 2026-09-11 | v1.8 | C1收尾：独立分析与4图完成，完整测试196通过；P1/P5失败、总判定3/5；确认gate记忆级联使有害flip转移到T20，按预注册停止调参 |
