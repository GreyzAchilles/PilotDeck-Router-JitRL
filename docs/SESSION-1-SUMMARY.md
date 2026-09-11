# Session 1 收尾总结（SESSION-1-SUMMARY）

版本：v1.0（2026-09-11）

一句话结论：Session 1 核心交付（Judge 探针 / 任务集 / 核心引擎 / harness / 集成冒烟）**全部完成且集成门槛全绿**；唯一未勾选项为海报内容简报——由外部 session 处理中，不作为阻塞。会话 2（CPA 真实执行 + Evaluator + Web Demo）可直接启动。

---

## 1. 交付产物

| 产物 | 位置 | 报告 |
|---|---|---|
| Judge logprob 客户端（Method B 加固版：canonical 校验 + 重试 + clamp） | `local_judge/` | `S1-logprob-probe-report.md` v1.0 |
| 评测任务集 v0：24 主任务（6 族 × 4 变体）+ 5 续轮探针 = 29 条 | `eval/tasks.jsonl` | `TASK-taskset-report.md` v1.0 |
| JitRL 无梯度引擎（签名/检索/记忆/V-Q-Â/logit 闭式调制/JSONL 插桩） | `jitrl_core/` | `S1-core-harness-report.md` v1.0 |
| A/B/C 对照 harness + separability 工具 + 定价表 | `harness/`、`eval/pricing.json` | 同上 |
| 集成冒烟：契约对齐 + 任务集复检 + 真实 Judge B/C 单轮 | `logs/s1_{b,c}_real.jsonl` 等 | `S1-integration-smoke-report.md` v1.0 |

决策基线：`technical-decisions.md` v1.3 + `project-decisions.md` v2.5（D1–D15 冻结）；文档正本与全量清单见 `INDEX.md`。

## 2. 门槛结果（全绿）

- **测试**：48/48 通过（含 id/task_id loader 兼容回归）；
- **契约**：JudgeClient 四字段/签名/4 档完备，`jitrl_core` ↔ `local_judge` 逐字一致，真实接入零改动、29/29 无契约失败；
- **任务集复检**：29 行、唯一 id、GT 分布 simple 5 / medium 8 / reasoning 9 / complex 2，探针 GT 继承全部正确；
- **B vs C（真实 MiniCPM5-1B，单轮）**：B 7/29 → C 8/29；C 记忆 0→29、检索命中 6/29（跨变体 12 对全同族、0 跨族噪声）、logit 调制翻转 3 次（1 改善 GT / 2 中性 / 0 恶化）；**任务级 C ⊇ B**；B/C 组间判档延迟相当（2860 vs 2768 ms），对照公平；
- **判定**：通过 Session 1 集成门槛，无集成阻断、本轮零代码修复。

## 3. 当前风险（Session 2 必须携带）

1. **Judge 偏 simple**：MiniCPM5-1B 把 medium/reasoning 大多判 simple（探针 3/7、B 仅 7/29、续轮 0/5）；JitRL 有增益但无法凭单轮记忆完全修复弱基策略；
2. **Method B 判档需 4 请求**：本地判档延迟可达 ~2.8 s/任务；报告时须区分"判档开销"与"JitRL 调制开销（≈0 ms）"，B/C 用相同 z 提取通道保证公平；
3. **tokenization 路径退化**：GBNF 约束文本而非 token 路径；`local_judge` 的 canonical 校验/重试/clamp 加固逻辑不得删除；
4. **签名检索离散坍缩**：跨变体仅在 sim=1.0 命中，0.4–0.9 区间无分辨力；Demo 有利（稳定命中）但 README 必须写泛化限制；
5. **小邻域探索 bonus 过强**：λ 分支 α/|N| 在 N 小时主导翻转；最终实验需检查 reward hacking，如需修正只做一项保守变更并记录版本，不可悄悄改数学；
6. **A 模式 accuracy 恒 0**：chosen_tier 记为 `static`，不参与 tier accuracy；会话 2 报告路由正确率前应先修正该语义（记录实际执行档或显式排除 A）；
7. **stub 结果 ≠ 最终数据**：当前 reward 为 routing-only STUB（quality=±0.5 + 假想用量），成本差异 $0.00140 vs $0.00148 仅方向性参考，不得混入会话 2 真实实验结论。

## 4. Session 2 输入（P1/P2 摘要）

- **Workstream A（优先）**：CPA 执行客户端 + Evaluator——读 `~/.pilotdeck/pilotdeck.yaml`（不打印/不复制 key），tier→model 映射，返回 response + usage；Evaluator（gpt-5.6-sol）输出结构化 JSON（质量 1–5 / useful-harmful-neutral / certainty / 能力不足 / checklist），失败不写 C 记忆；先读 PilotDeck 本地 provider 协议源码再实现，不猜 schema；先做 3 episode 端到端校准；
- **Workstream B**：Web Demo（FastAPI 或 stdlib 后端 + 单页前端）——决策可视化（base logits / 检索记忆 / Â / 调制后 logits / 执行模型）、学习曲线与成本面板、A/B/C 切换、会话级隐私隔离、mock/offline 模式；依赖 A 稳定后的 episode API 契约；
- **P2 全量实验**：3 episode 校准 → A/B/C × 24 主任务 → `results.md` + 图表（正本目录）；核心指标含质量/真实成本/误档率/学习曲线/Judge latency/调制 latency；
- **预算口径**：预计 $5-8（<$20 上限），cost_saving 分母 = glm-5.3；Evaluator 为最贵单项（$4/20 $/Mtok），仅 episode 末调用一次。

## Changelog

- v1.0（2026-09-11）：首版。P0 收尾产物：会话 1 交付清单、门槛结果、7 项风险、会话 2 输入摘要（依据四份 S1 报告 v1.0 与 ORCHESTRATOR-HANDOFF.md v1.0 汇编）。
