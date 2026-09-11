# PilotDeck-Router-JitRL

> **Just-In-Time Reinforcement Learning（JitRL，面向 LLM agent 的无梯度持续学习）× PilotDeck Router 的 TokenSaver Judge**
> 冻结本地 Judge 权重，维护非参数 `(state, tier, reward)` 记忆，检索相似历史经验估计 V/Q/优势，再对 Judge 的四档 logit 做闭式更新——填补 Router 的**跨轮经验真空**，让路由器跨轮学习，而不是每轮从零开始、重复犯同样的错误。

```text
z′(tier) = z(tier) + β · Â(tier)     # 对冻结 Judge 的 simple / medium / complex / reasoning 四档 logit 做闭式更新
```

本仓库是一个**纯标准库（stdlib-only）的独立 Python harness**：不依赖 PilotDeck 启动链路，但**逐项忠实复刻 Router 的决策路径**（相同的 judge prompt 格式、四档体系、previousTier 续轮规则），在真实 CPA 模型调用与盲评 Evaluator 下完成了 A/B/C0/C1 预注册对照实验与 T8 机制消融。目标：保住回答质量的同时降低模型调用成本。

**诚实结论先行**：端到端管线完全可行，JitRL 后处理开销近零（mean 0.2444ms / max 0.6664ms）；但默认 C0 **未达到 ≥20% 的成本目标**（D10 判定 3/4），C1 保守门控也**未通过预注册判据**（3/5，且出现记忆级联），T8 消融显示当前 1B Judge 几乎不利用注入的记忆（0/6 影响）。本项目把「带机理分析的诚实负结果」当作一等公民，全部数字为冻结口径，见[实验](#5-实验)与[诚实结论](#6-诚实结论)。

---

## 目录

1. [TL;DR](#tldr)
2. [1. 动机](#1-动机)
3. [2. 架构](#2-架构)
4. [3. 仓库结构](#3-仓库结构)
5. [4. 快速开始](#4-快速开始)
6. [5. 实验](#5-实验)
7. [6. 诚实结论](#6-诚实结论)
8. [7. 已知限制](#7-已知限制)
9. [8. Web Demo](#8-web-demo)
10. [9. PilotDeck 集成（future work）](#9-pilotdeck-集成future-work)
11. [10. 可复现性](#10-可复现性)

---

## TL;DR

任务集：24 条主任务（6 族 × 4 变体）+ 5 条续轮探针；所有组同序执行、单 episode、24/24 完成；质量为 `CPA/gpt-5.6-sol` 盲评（1–5 分）。**n=24，方向性证据，不声称统计显著性。**

| Arm | Mean quality | Exec cost | GT tier accuracy | Verdict |
|---|---:|---:|---:|---|
| **A** Static（全量 `CPA/glm-5.3`） | 2.2083 | $0.088888 | — | 基线 |
| **B** Judge（仅本地 Judge 路由） | 2.6667 | **$0.028333** | 29.17% | 比 A 省 68.1%，但主要来自弱 Judge 偏 simple |
| **C0** JitRL（默认） | **2.7917** | $0.029406 | 29.17% | D10 判定 3/4：成本目标 FAIL（−3.78% vs B） |
| **C1** gated（`min_neighbors=3`） | 2.4167 | $0.033047 | 29.17% | 预注册 3/5：P1/P5 FAIL，记忆级联 |
| **T8-PM**（消融：prompt 注入） | 2.5417 | $0.027993 | —¹ | 方向性占优，但 0/6 注入改变决策 |

¹ T8 为机制消融（两臂重放同一份冻结记忆），GT 准确率不在其预注册判据内。T8-LM：quality 2.5417 / cost $0.032653，见 §5.4。

---

## 1. 动机

PilotDeck Router 的 TokenSaver 用一个本地小模型 Judge 把用户消息分为四档（simple / medium / complex / reasoning），再映射到不同价位的执行模型。精读上游源码（`RouterRuntime.ts`、`AgentLoop.ts`、`classifyAndRoute.ts`、`generateJudgePrompt.ts`，快照见 `PilotDeck-src/`）得到三个结构性事实：

1. **每个用户轮开始都会调用 `invalidateSticky()`**——sticky 只覆盖轮内的工具循环；
2. **轮与轮之间，Judge 从零重新判档**；
3. 唯一的跨轮信号是 prompt 里的 **`previousTier` 文本提示**——历史的质量/成本从不回流。

这是一个**跨轮经验真空**：上一轮学到的教训（这个任务其实不需要 reasoning 档 / 这个便宜档已经够用）全部丢失，路由器每轮都在重新犯同类错误。

JitRL 恰好补上这一块。映射关系：

| JitRL 概念 | 本项目实现 |
|---|---|
| 冻结策略（no gradient updates） | 本地 llama.cpp 部署 MiniCPM5-1B（Q4_K_M GGUF），权重全程不动 |
| 非参数记忆 | `(state, tier, reward)` 条目，跨 episode 累积 |
| 经验检索 | top-k（k=10）签名 Jaccard 检索相似历史 |
| 价值估计 | 邻域均值 V、per-tier 均值 Q、λ 探索乐观项，归一化得优势 Â |
| 策略更新 | 闭式 logit 调制 `z′ = z + β·Â`，clamp(−10) 后 argmax 选档 |

目标：**保质量、降成本、让路由器跨轮学习**，而不是每轮盲判。

## 2. 架构

![architecture](diagrams/architecture.svg)

`jitrl_core.engine` 的在线闭环：**retrieve → estimate → gate/modulate → choose → learn**——检索相似经验、估计 V/Q/Â、（C1）邻域门控、对四档 logit 做闭式调制后 argmax 选档；episode 结束把 `(state, tier, reward)` 写回记忆。

![arms](diagrams/arms.svg)

| 模块 | 内容 |
|---|---|
| `jitrl_core/` | engine（retrieve → estimate → gate/modulate → choose → learn）、memory（top-k Jaccard 检索）、state（intent_class + 规则归一化签名 + CJK bigram）、value（V/Q/Â，含 λ 探索）、policy（clamp / modulate / argmax）、config（冻结超参） |
| `local_judge/` | 面向 llama.cpp 的生产 JudgeClient：**Method B 硬化 logit 提取**（四档各一次 grammar-forced 请求、共 4 次；canonical tokenization 校验 + 重试阶梯 + `Z_MIN=−10` clamp）、GBNF grammar、逐字节复刻 PilotDeck `generateJudgePrompt` |
| `harness/` | `run.py`（routing-only A/B/C）、`run_real.py`（真实 episode A/B/C/C1：CPA 执行 + Evaluator + reward + 记忆写回）、`run_ablation.py`（T8-LM vs T8-PM）、`cpa_client.py`（运行时读取 PilotDeck CPA provider 配置）、`evaluator.py`（盲评结构化评估）、`real_rewards.py`、`pricing.py` |
| `eval/` | 任务集 `tasks.jsonl`、`tasks-meta.md`、`pricing.json`、分析器（`analyze_results.py` / `analyze_c1.py` / `analyze_ablation.py` / `run_continuation_probes.py`）与冻结聚合资产（见 §10） |
| `diagrams/` | 4 张技术 SVG：`architecture.svg`、`arms.svg`（本节）、`online-offline.svg`、`pilotdeck-integration.svg` |
| `demo/` | Web Demo（离线优先，见 [§8](#8-web-demo)） |

**关键工程细节**

- **Method B logit 提取**：llama.cpp 的 GBNF 只掩码采样、`top_logprobs` 返回的是掩码前分布，且 `reasoning` 在该词表是 2 个 token——单请求读不齐四档。生产客户端对四档**各做一次单档 grammar-forced 请求**，取档位名 canonical token 路径的 logprob 之和，附校验/重试/clamp(−10) 阶梯。每次判档 = 4 个本地请求，mean ≈2.3s。
- **b10903 `reasoning_content` 兼容**：llama-server build b10903 会把 grammar-forced 输出路由进 `message.reasoning_content` 而 `content` 为空；客户端同时处理 `content` 与 `reasoning_content`（已文档化的兼容修复，token 路径与 logprob 不变）。
- **延迟分离计量**：Judge 的 4 请求延迟（≈2.3s mean）与 JitRL 后处理延迟（mean 0.2444ms / max 0.6664ms）**分开插桩**，绝不混报。

## 3. 仓库结构

```text
PilotDeck-Router-JitRL/
├── jitrl_core/     # JitRL 引擎：state / memory / value / policy / engine / config（+ tests）
├── local_judge/    # 生产 JudgeClient（Method B）、GBNF grammar、探针（+ tests）
├── harness/        # run / run_real / run_ablation / cpa_client / evaluator / real_rewards /
│                   #   pricing / separability（+ tests，testdata/）
├── eval/           # tasks.jsonl（24 主任务 + 5 续轮探针）、tasks-meta.md、pricing.json、
│                   #   analyze_results / analyze_c1 / analyze_ablation / run_continuation_probes、
│                   #   冻结聚合：results-summary.json、results-tasks.csv、charts/（5 SVG）、
│                   #   c1/（summary + CSV + 4 SVG）、ablation/（冻结 memory trace + SHA-256 +
│                   #   summary + CSV + 4 SVG）
├── diagrams/       # architecture.svg / arms.svg / online-offline.svg / pilotdeck-integration.svg
├── demo/           # Web Demo（离线优先）
├── docs/           # 设计文档
├── logs/           # 实验日志（gitignore；正式命名见 §10）
└── PilotDeck-src/  # 上游 PilotDeck 源码快照（决策路径复刻的对照基准）
```

## 4. 快速开始

环境：Python 3.11，**仅标准库**（测试另需 pytest）。

### 4.1 本地 Judge（必需）

llama.cpp `llama-server` 部署 **MiniCPM5-1B Q4_K_M GGUF（约 683MB）**：

- 端点默认 `http://127.0.0.1:18080`，可用环境变量 `JITRL_JUDGE_ENDPOINT` 覆盖；
- 模型 alias `minicpm5-1b`（可用 `JITRL_JUDGE_MODEL` 覆盖）；
- 本地推理的边际 API 成本记 **$0**。

### 4.2 执行/评估模型（真实实验必需）

harness 在**运行时**读取 PilotDeck 配置（默认 `~/.pilotdeck/pilotdeck.yaml`，可用 `--config` 或 `PILOTDECK_CONFIG_PATH` 覆盖）中的 CPA provider 凭据。**API key 只在运行时读取，仓库从不存储任何密钥。**

| 角色 | 模型 | $/Mtok（input / output） |
|---|---|---:|
| simple 档执行 | `CPA/glm-5.3-flash` | 0.15 / 0.50 |
| medium 档执行 | `CPA/opencode-v4-flash` | 0.30 / 1.20 |
| complex 档执行 | `CPA/OpenBMB-5.3` | 1.40 / 4.40 |
| reasoning 档执行 + A 组静态路由 | `CPA/glm-5.3` | 1.40 / 4.40 |
| 盲评 Evaluator（不在档位映射内，杜绝自评） | `CPA/gpt-5.6-sol` | 4.00 / 20.00 |

成本节省的分母（cost_saving denominator）= `CPA/glm-5.3`。价格正本：`eval/pricing.json`。

### 4.3 测试与运行

```bash
# 0) 测试（无需任何服务与 key）——当前 261 passed
python -m pytest harness jitrl_core local_judge eval -q

# 1) routing-only（只跑决策循环，不执行生成；无 llama.cpp 时可改用 --judge mock）
python -m harness.run --mode B --tasks eval/tasks.jsonl --judge llama \
  --out logs/routing_B.jsonl
python -m harness.run --mode C --tasks eval/tasks.jsonl --judge llama \
  --out logs/routing_C.jsonl --memory-out logs/routing_C_memory.jsonl

# 2) 签名可分性检查
python -m harness.separability --tasks eval/tasks.jsonl

# 3) 真实 episode A/B/C0（需要 llama.cpp Judge + PilotDeck CPA 配置）
python -m harness.run_real --mode A --tasks eval/tasks.jsonl --judge llama \
  --out logs/s2_full_A_main24.jsonl
python -m harness.run_real --mode B --tasks eval/tasks.jsonl --judge llama \
  --out logs/s2_full_B_family_order.jsonl
python -m harness.run_real --mode C --tasks eval/tasks.jsonl --judge llama \
  --out logs/s2_full_C_main24_retry2.jsonl \
  --memory-out logs/s2_full_C_main24_retry2_memory.jsonl   # 默认 min_neighbors=1，即 C0

# 4) C1：同一命令加预注册门控参数
python -m harness.run_real --mode C --tasks eval/tasks.jsonl --judge llama \
  --min-neighbors 3 \
  --out logs/s2_full_C1_main24.jsonl --memory-out logs/s2_full_C1_main24_memory.jsonl

# 5) 续轮探针（仅 Judge 路由，不执行生成）
python -m eval.run_continuation_probes

# 6) T8 消融（重放冻结的 C0 prefix 记忆，只读、无在线写回）
python -m harness.run_ablation --arm T8-LM --judge llama --out logs/s3_ablation_T8_LM.jsonl
python -m harness.run_ablation --arm T8-PM --judge llama --out logs/s3_ablation_T8_PM.jsonl

# 7) 聚合分析（从正式日志重算全部聚合与图表；默认路径即正式日志）
python eval/analyze_results.py
python eval/analyze_c1.py
python eval/analyze_ablation.py --lm logs/s3_ablation_T8_LM.jsonl \
  --pm logs/s3_ablation_T8_PM.jsonl --out-dir eval/ablation
```

冻结超参（正式实验全程一致）：

```text
k=10, β=5, λ=0.05, α=5, jaccard_threshold=0.5, z_min=−10, seed=42,
min_neighbors=1（C0）/ 3（C1，预注册）,
max_tokens_exec=1024, temperature_exec=0.7,
Evaluator: temperature=0.2, max_tokens=300
```

## 5. 实验

### 5.1 实验设置与公平性

- **任务集**：24 条主任务 = 6 族 × 4 变体（`chat_qa` / `code_gen` / `doc_writing` / `data_analysis` / `refactor` / `info_retrieval`），另有 5 条续轮探针，共 29 条（`eval/tasks.jsonl`）；
- **顺序**：冻结的 family-blocked 顺序 `T01, T05, T09, …, T24`；所有组同序执行，单 episode per task；
- **完成度**：A/B/C0 各 24/24，C1 24/24，T8 两臂各 24/24；
- **质量口径**：`CPA/gpt-5.6-sol` 盲评（1–5 分，temperature 0.2，max_tokens 300）；Evaluator 为测量开销，单列、不计入路由成本；
- **证据强度**：n=24、单次运行，全部结论为方向性证据，不声称统计显著性。

### 5.2 A/B/C0 与 D10 判定

| Metric | A Static | B Judge | C0 JitRL |
|---|---:|---:|---:|
| Mean quality（盲评 1–5） | 2.2083 | 2.6667 | **2.7917** |
| Exec cost | $0.088888 | **$0.028333** | $0.029406 |
| Cost saving vs A | — | 68.1% | 66.9% |
| GT tier accuracy | — | 29.17% | 29.17% |
| Misgrade rate | — | 70.83% | 70.83% |
| 记忆改变决策（flips） | — | 0 | 2 |
| JitRL 后处理延迟 | — | — | mean 0.2444ms / max 0.6664ms |

判读：

- **B/C0 vs A 的节省主要来自弱 Judge 大量选 simple**，而不是跨轮学习；
- **C0 vs B**：质量 +0.125，但**不可归因于调制**——2 条 flip 任务的净质量变化为 0，增益来自 Judge base-choice 漂移 + temperature 0.7 重采样；成本 **−3.78%**（即 C0 比 B 贵 3.78%）；
- **C0 的 2 条 flip（T06、T19）均为有害 flip**：小邻域 λ-exploration 把 simple 推到 complex，合计 +$0.008239，质量增益为零；
- **延迟分离**：Judge（Method B，4 个 llama.cpp 请求）≈2.3s mean；JitRL 后处理 mean 0.2444ms / max 0.6664ms。

**D10 判定（预注册，总判定 3/4）**：

| Criterion | Threshold | Result | Verdict |
|---|---|---|---|
| Cost saving vs B | ≥20% | −3.78% | **FAIL** |
| Quality vs B | ≥ −0.3 | +0.125 | PASS |
| Misgrade vs B | ≤ B + 5pp | 0pp | PASS |
| JitRL latency | ≤500ms | 0.6664ms（max） | PASS |

### 5.3 C1：保守门控与记忆级联

预注册的邻域门控（`min_neighbors=3`：检索邻居数不足则不调制，直接用 base logits）：

| Metric | C1 gated |
|---|---:|
| Mean quality | 2.4167 |
| Exec cost | $0.033047（**+16.64% vs B**） |
| Misgrade rate | 70.83% |
| Flips | 1 |
| 门控统计 | gated 22 / modulated 2（T08、T20，均 n=3） |

**P1–P5 预注册判据：3/5 NOT PASSING**

| Criterion | Result | Verdict |
|---|---|---|
| P1 exec cost ≤ B + $0.001 | +16.64% | **FAIL** |
| P2 quality | −0.25 | PASS |
| P3 misgrade | 0pp | PASS |
| P4 JitRL latency | max 0.4845ms | PASS |
| P5 flip 无害 | 有害级联 | **FAIL** |

按预注册停止条件**停止，未做进一步的 gate 阈值扫描**。

**诚实的机理解读**：门控确实抑制了 C0 的 T06/T19 flip，**但被 gate 的决策仍按 base tier 写回记忆**，改变了后续的 Q/Â——记忆级联在 T20 制造了**新的** flip（simple→complex，+$0.004190，质量 1→1，约为 B 组 simple 档成本的 8.8 倍）。问题被**移动**（T06/T19 → T20），没有被解决。

### 5.4 T8 消融：logit 调制 vs prompt 注入

设置：**冻结 C0 prefix 记忆重放**（第 i 个任务只读先前记忆，杜绝自身/未来 outcome 泄漏），只读、无在线写回；两臂各 24/24，唯一差异是记忆的作用机制。

| | T8-LM（logit 调制） | T8-PM（prompt 注入） |
|---|---|---|
| 机制 | 精确复现 C0 数学（`z′ = z + β·Â`） | 白名单记忆块注入 Judge prompt，无调制 |
| Mean quality | 2.5417 | 2.5417 |
| Exec cost | $0.032653 | **$0.027993** |
| 改变原始 Judge 选择 | 1（T19：simple→complex，+$0.004194，质量 1→1） | 0（注入 6 个记忆块，0 次影响） |
| Judge 调用次数 | 24 | 48² |

两臂 tier agreement 23/24，唯一分歧 T19。预注册假设判定：H1 满足（≥1 种机制产生影响）、H2 满足（Δquality 0.0 ≥ −0.3）、H4 满足（LM max jitrl 0.4434ms）。

² PM 每任务多出的 raw-prompt 调用仅用于审计；部署成本 = 1 次决策调用（≈1051.5ms mean）。

**预注册结论：方向性证据偏向 prompt injection——但必须附带强制 caveat**：PM 更便宜是因为它**什么都没改变**（0/6 影响），而 LM 的单次有害 T19 flip 主导了 $0.004660 的成本差；PM 并**未**在 1B Judge 上展示出可用的记忆推理能力。

### 5.5 续轮探针：0/5

5 条 continuation probes 全部失败（**0/5**）：`previousTier` 确实传入 prompt 并提高了其相对 logit，但 MiniCPM5-1B 的 simple 偏置每次都压过它——这是**可复现的 Judge 能力限制，不是管线 bug**。

## 6. 诚实结论

**实现了什么**：端到端证明了「冻结 Judge + 非参数记忆 + 闭式 logit 更新」的完整可行，JitRL 后处理开销近零（0.2–0.7ms），工程资产（harness、分析器、冻结数据、261 项测试）完全可复现。

**没实现什么**：

- 默认 C0 **未达到 ≥20% 的成本目标**（D10 判定 3/4；相对 B 反而贵 3.78%）；
- C1 保守门控**也未通过自己的预注册判据**（3/5；门控把有害 flip 级联到了新任务 T20）；
- T8 消融显示当前 1B Judge 几乎不利用注入的记忆（0/6 影响）。

**贡献定位**（按我们认为的价值排序）：

1. **忠实复刻 Router 决策路径的独立 harness**——相同的 judge prompt、四档体系、previousTier 续轮规则，可在不改动 PilotDeck 的前提下做路由实验；
2. **预注册实验纪律**——C1 与 T8 先冻结判据后运行，触发停止条件即停，不在同一数据上继续调参；
3. **带机理分析的诚实负结果**——小邻域探索 bonus 的有害 flip、门控下的记忆级联，都是逐任务数据可复核的失效模式，而非笼统的「效果不好」；
4. **完全可复现的资产集**——冻结任务集、冻结记忆 trace（含 SHA-256）、逐任务 CSV、图表与分析器。

## 7. 已知限制

1. **1024-token 固定执行预算**：大量 `finish_reason=length` 与空响应（每臂 15–16 条 length、14 条空回复），压低绝对质量但保持组间公平；
2. **MiniCPM5-1B 强 simple 偏置**：B/C0 GT 档位准确率仅 29.17%，续轮探针 0/5——弱基策略封顶了任何调制机制的收益空间；
3. **检索只在 similarity=1.0 命中**：规则签名的离散坍缩——同族变体完全归一化后相同才命中，0.4–0.9 相似度区间无泛化带；
4. **执行 temperature 0.7**：同 tier/模型的质量与用量波动无法归因于算法；
5. **llama Judge 跨运行漂移**：base-choice 漂移（S2 中出现过 T13/T22；T8 运行 0 次 base-choice 不一致，但数值 logit 仍有漂移）；
6. **LLM-as-judge Evaluator 自身漂移风险**；
7. **family-blocked 顺序**把「前半 vs 后半」的学习曲线与任务难度混叠；
8. **n=24、单次运行**：全部结论仅为方向性证据，无统计显著性。

## 8. Web Demo

`demo/` 是**离线优先**的 Web Demo（在线/离线双模式路径见 `diagrams/online-offline.svg`）：

1. **实验数据面板（离线，零网络可用）**：直接读取 `eval/` 冻结聚合——A/B/C0/C1/T8 的质量、成本、误档、截断、延迟；C0 的两条有害 flip；C1 门控计数与 T20 级联；续轮探针 0/5；网络/网关故障重试审计。断网可完整演示。
2. **可选的实时决策面板（清晰标记 ONLINE）**：预设任务 + 自由输入，可切换 arm，展示完整决策 trace——base logits → 检索到的记忆 → V/Q/Â → 调制后 logits、gate 原因、所选 tier、执行模型、记忆条数、累计成本、会话重置。失败时优雅降级，不白屏。
3. **隐私**：记忆按会话内存隔离、不持久化观众输入文本；API key 永不发送到浏览器（由后端代理调用）。

运行（stdlib 后端，无需安装任何依赖；离线为默认模式，断网可完整演示数据面板）：

```bash
python -m demo.server                 # 离线模式（默认）→ http://127.0.0.1:8300
python -m demo.server --online        # 启用在线决策面板（需本地 llama.cpp Judge + PilotDeck CPA 配置）
python -m pytest demo -q              # Demo 自身 32 项测试（无需任何服务与 key）
```

## 9. PilotDeck 集成（future work）

![pilotdeck-integration](diagrams/pilotdeck-integration.svg)

深度集成是 22h 预算之外的 future work；本仓库用独立 Python harness 忠实复刻 Router 决策路径替代（相同的 judge prompt 格式、四档体系、previousTier 续轮规则）。扩展点分析已完成：

- **CustomRouter（上游已存在的扩展点）**：`decide()` 返回 provider/model 即可完全接管路由，JitRL engine 将承载于此——证明集成**不需要改动 Router 内核**；
- **缺口① logprob 透传**：Judge 调用的 logprob 目前不暴露给路由层，需上游补通道（或如本仓库在 Judge 客户端内做 Method B 四请求）；
- **缺口② episode outcome 回调**：episode 结束后的质量/成本需要回传路由层以喂给 `engine.learn()`——上游目前只存 previousTier 文本。

`jitrl_core` + `local_judge` 即为该集成准备的可移植核心；Demo 的在线/离线双模式路径另见 `diagrams/online-offline.svg`。

## 10. 可复现性

**正式数据（唯一口径）**：

| Experiment | Logs（`logs/`，gitignore） | Aggregates（随仓库分发） |
|---|---|---|
| A / B / C0 | `s2_full_A_main24.jsonl`、`s2_full_B_family_order.jsonl`、`s2_full_C_main24_retry2.jsonl`（C0 另有 `_memory`） | `eval/results-summary.json`、`eval/results-tasks.csv`、`eval/charts/`（5 SVG） |
| C1 | `s2_full_C1_main24.jsonl`（+ `_memory`） | `eval/c1/`（summary + CSV + 4 SVG） |
| T8 消融 | `s3_ablation_T8_LM.jsonl`、`s3_ablation_T8_PM.jsonl` | `eval/ablation/`（冻结 `frozen-c0-memory-trace.jsonl` + SHA-256 sidecar + summary + CSV + 4 SVG） |
| 续轮探针 | `s2_continuation_probes.jsonl` | `s2_continuation_probes_summary.json` |

- **分析器可从日志重算全部聚合与图表**（默认路径即上表正式日志；`analyze_results.py` 另含 audit-only 输入默认值）；
- **冻结资产**：任务集、定价表、聚合 JSON/CSV/SVG、冻结记忆 trace（含 SHA-256 校验）全部随仓库分发；
- **测试**：`python -m pytest harness jitrl_core local_judge eval -q` → **261 passed**（无需任何服务与 key）；
- **预注册纪律**：C1 与 T8 的判据在运行前冻结，触发停止条件即停；后续方案须另立预注册，不覆盖、不回写。
