# JitRL Router 技术决策文档

> 项目：将 JitRL（Just-In-Time Reinforcement Learning, arXiv:2506.10985）应用于 PilotDeck Router 的 TokenSaver Judge，赋予其跨轮次的测试时学习能力。
> 状态：v1.3 —— D1–D5 不变；实验协议（§5）与仓库形态（§7）按 `project-decisions.md` **v2.1**（22h 时间预算）简化执行，本文相应章节保留为完整版参考（= future work 清单）。

---

## 1. 问题定义与动机

### 1.1 现状：TokenSaver 的跨轮失忆

PilotDeck Router 的 TokenSaver 用轻量 Judge 模型判定用户消息复杂度，选择模型档位。源码精读（`PilotDeck-src/src/router/`）确认的关键事实：

- **每轮重新决策**：`AgentLoop` 在每个用户轮开始时调用 `invalidateSticky()` 清除 sticky 的 provider/model（`AgentLoop.ts:363`），随后 Judge 独立重新分类。唯一的跨轮信号是 `previousTier` 作为 prompt 内文本提示（用于防止 "go/继续" 等短续轮被误降档）。
- **决策结果不回流**：上一轮选错档位造成的成本浪费或质量下降，不参与下一轮决策。Judge 永远停留在零经验状态。
- **档位体系**：源码默认 4 档 `simple / medium / complex / reasoning`（非文档所写 high/medium/low；描述经 PinchBench 校准）。

### 1.2 JitRL 的切入点

JitRL 的机制（记忆 `(s, a, G)` 三元组 → 检索估计优势 → logit 加性调制 `z' = z + β·Â`）恰好填补"轮间经验真空"：

- **不是替换 Sticky，而是学习型 Sticky**：该保持档位时优势估计自然偏向历史选择（邻域内同档位 G 高）；该切换时自然离开。现有 Sticky 用启发式（绑死）获得连续性，JitRL 用经验（优势估计）获得连续性。
- **对弱 Judge 免疫**：JitRL 理论保证（Thm 4.2/4.3）——即使基策略弱（本地 1B Judge），价值估计收敛后策略收敛到 KL 正则化最优。弱 Judge + 强记忆 vs 弱 Judge + 无记忆，增益空间反而更大。

### 1.3 总体目标

在**同等任务质量**下降低路由成本，同时消除重复状态的重复决策错误。量化验收目标：见 `project-decisions.md` D10。

---

## 2. 概念映射：JitRL 论文 → 本项目

| JitRL 论文概念 | 论文实现（WebArena/Jericho） | 本项目对应物 |
|---|---|---|
| 状态 s | 正则化 URL / 名词摘要 | 三层状态向量（见 D2） |
| 动作 a | 候选动作（点击/命令） | **模型档位 tier**（见 D1） |
| 候选集 C | LLM 候选 ∪ 记忆动作 | tiers 配置集（封闭集） |
| 基策略 logit z | 候选动作 logprob | 本地 1B Judge 的档位 token logprob（见 D4） |
| 轨迹 τ | 一局游戏/一次网页任务 | 一轮用户任务的完整生命周期（Judge 决策 → 模型执行 → Agent 工具循环 → 最终回复） |
| Evaluator 步级奖励 | LLM 回溯打分 -3~+3 | 质量分 + 成本节省 + 修正惩罚（见 D3） |
| 折扣回报 G_t | γ=0.1/0.5 | 单步 episode 退化 G_t = r_t |
| 记忆 M | (s,a,G) 三元组动态库 | 同构，附生命周期管理 |
| Sticky（论文无此项） | — | PilotDeck 现有机制，保留为退化行为（见 D5） |

---

## 3. 已锁定的技术决策

### D1. 动作空间：两级结构，优势估计作用于 tier 级

**决策**：
```
第一级（主动作）：tier ∈ {simple, medium, complex, reasoning}  ← JitRL 优势估计作用层
第二级（解析）：  tier → (provider, model) 由配置静态映射，不参与学习
```

**理由**：
1. **收敛速度**：动作空间仅 4，`|N(s,a)|`（邻域内同动作计数）增长快，满足论文附录 C.4（action frequency →∞）假设。若在 provider/model 对（10+ 个）上做，记忆需求量翻数倍，Demo 的 episode 数不够。
2. **表达力不损失**：tier→model 映射换配置即可，记忆的档位优势仍然有效——比论文更稳健（论文换 backbone 记忆作废，我们的 tier 抽象层对配置变化免疫）。
3. **封闭集对齐**：Q̂(s,a) 要求跨请求匹配"同一个动作"，档位是天然封闭集。

**边界**：
- 不做连续动作（tier 混合概率）。JitRL 闭式解要求 softmax 重归一化，纯离散选择。
- 记忆中出现但当前配置已删除的 tier：执行时直接丢弃（无法执行的动作不进入候选）。

**关键约束——β 下界**：若 Judge 对某档位 logprob 极低（如 -30），而记忆显示其 G 高，`z' = z + β·Â`（Â∈[-1,1]）需要 β>30 才能翻转。1B 模型档位间典型 logprob 差为 2-8，**β 初始值取 5**，扫描范围 [3, 10]。这也是论文做 max-归一化（式 25）后再乘 β 的原因。

### D2. 状态表示：三层结构 + 分层检索（方法的命门）

**决策**：状态向量与检索分两层：

```
s = ( intent_class , task_signature , context_features )

检索（两阶段，模仿论文 WebArena 的 URL 粗过滤 + 精匹配）：
  第一阶段：intent_class 分桶（4-8 类：代码生成/问答/重构/数据分析/文档写作/闲聊确认/信息检索…）
  第二阶段：桶内对 task_signature 做 token 集合 Jaccard 相似度，取 top-k
```

**各层设计**：

- **task_signature（结构代理，对应论文的正则化 URL）**
  从 last user message（即 Judge 的输入）规则式抽取：文件路径→`<path>`、数字→`<num>`、引号内长串→`<str>`；保留动词短语+宾语类型骨架。设计原则沿用论文：**把功能等价的状态映射到相似表示**。允许本地 1B 顺带生成（推理已在本地，边际成本极低），但规则式兜底必须保留。

- **context_features（客户端状态，对应论文的"本地动作历史"）**
  会话内轮次号、当前上下文 token 长度、已用工具集、previousTier。

- **intent_class（粗分类）**
  第一版用关键词规则表。唯一作用是预切桶：提速 + 避免跨域误命中。

**相似度度量**：token 集合 Jaccard（阈值初始 0.8，参考论文 WebArena 设置）。**不用 embedding**——论文附录 L 消融明确显示文本规则表示 > 向量嵌入（嵌入对"描述相似但拓扑不同"的状态区分度差）。

**失败模式与体检**：状态太粗（所有请求都像"写代码"）→ 检索退化为全局平均 → JitRL 退化为"全局最优档位查找器"。Phase 1 必须做**状态区分度体检**（见 Spike S2）：30-50 条真实请求的两两 Jaccard 分布应为双峰（同类 >0.6，异类 <0.2），单峰集中即状态设计失败，回头调 signature 规则。体检脚本进仓库（可复现性卖点）。

### D3. 奖励信号：质量为主、成本为辅、修正惩罚兜底

**决策**：

```
r_t = 0.6 · quality + 0.3 · cost_saving + 0.1 · revision_penalty
G_t = r_t        （单步 episode：一轮用户任务 = 一次 Judge 决策 = 一条记忆）
```

**信号来源**：

| 信号 | 来源 | 说明 |
|---|---|---|
| quality（任务质量分） | LLM-as-judge 回溯评估 | 沿用论文 WebArena Evaluator 的 USEFUL/CERTAINTY 结构（-3~+3，含确定性分级）。**附加"能力不足检测"**：回复是否表现出截断/敷衍/遗漏要求——能力不足 + 低档位 = 降档过度的强信号（反事实近似） |
| cost_saving（成本节省） | `observeUsage` + 定价表（Router 现成） | 实际档位成本 / default 档参考成本。**必须按任务长度归一**（每千 token 节省率），否则长任务 G 系统性偏大污染优势估计 |
| revision_penalty（修正惩罚） | 用户隐性反馈 | 追问率/"重新来"类消息。默认权重仅 0.1，防"观众客气不纠错"偏置 |

**原则**：w_q > w_c 是防 reward hacking 的底线——否则 JitRL 学到"无脑全降档"（降档必省钱）。此权衡写入 README known limitations。

**记忆粒度**：第一版**每轮一条**（Judge 决策粒度）。工具循环中间态不经过 Router 决策（sticky 命中时不重新 decide），不构成可学习的决策点。MVP 跑通后再评估是否细化。

**已知局限（写入 README）**：同一请求只发生一次，永远观测不到"当时选 complex 会怎样"的严格反事实。用 Evaluator 能力不足检测做语言近似。

### D4. Logit 来源：本地部署 Q4 量化 1B 模型，取真实 token logprob

**决策**：本地部署 1B 模型作为 Judge，通过推理后端的 logprobs 接口获取档位 token 的真实对数概率，作为 JitRL 的基策略 logit z。

**理由**：绕开论文 Verbalized Logit（置信度→logit 转换公式论文未给出）的缺陷；本地部署同时服务 Demo（无外网依赖）与成本控制。

**三项工程现实（已在方案中处置）**：
1. **Q4 保真度**：Q4 对 logprob 的序（ordinal）通常稳定，绝对偏差由 β 吸收。实验期用 20-30 条人工标注请求对比 Q4 vs FP16 档位排序一致率；不一致则做温度校准。结果写入已知限制。
2. **1B 能力边界**：1B Judge 裸奔（无 JitRL）作为独立对照组（实验协议 §5 消融 2），证明增益来自记忆而非瓶颈补偿。
3. **协议通道**：PilotDeck 的 Canonical 协议未必透传 logprob 字段，可能需为本地 provider 加自定义通道——**最高风险项**，Spike S1 第一周验证。

**具体模型与后端选型**：MiniCPM5-1B（Q4 量化 GGUF）+ llama.cpp（用户并行会话部署），详见 `project-decisions.md` D7。

### D5. 与 Sticky 的关系：保留单轮 Sticky 作为退化行为，JitRL 主导轮间

**决策**：
- **轮间（每轮重新决策时）**：JitRL 完全主导——检索记忆、估计优势、调制 Judge logit。
- **单轮内（工具循环多次调用）**：保留现有 sticky 行为。记忆为空或检索无命中时，单轮 sticky 是 JitRL 的自然退化行为（JitRL 在冷启动时 ≈ 原生 Judge）。
- **优先级**：记忆有证据（|N(s,a)|>0）或探索触发时，JitRL 决策胜出；否则回落基策略。

**论证价值**（写进 README motivation）：现有 Sticky 的存在证明系统设计者知道"重复决策需要连续性"，而 Sticky 的轮间失效（`invalidateSticky`）证明现有方案只能用启发式而非学习型方式获得连续性。JitRL 提供的是"软 Sticky"。

---

## 4. 适配后的算法流程

```
【每轮用户任务开始】
1. invalidateSticky() 清除轮间 sticky（沿用现有 AgentLoop 行为）
2. 抽象状态 s ← AbstractState(last_user_message, context_features)
3. 检索：N(s) ← intent_class 桶内 Jaccard top-k（k=10 初始）
4. 基线：V̂(s) ← mean(G_i), i∈N(s)
5. 获取基策略 logit：本地 1B Judge 对 4 个档位 token 的 logprob → z(s, tier)
   （Judge prompt 沿用现有 <tier>NAME</tier> 格式，温度/规则不变）
6. 动作价值：
   Q̂(s,tier) = mean(G_i) over N(s,tier)          # 有历史证据
   Q̂(s,tier) = V̂(s) + α/|N(s)|  以概率 λ         # 无证据：乐观探索（UCB）
   Q̂(s,tier) = 0                    以概率 1-λ    # 防过度探索
7. 优势归一化：Ã = (Q̂-V̂) / (max|Q̂-V̂| + ε)
8. 闭式更新：z'(s,tier) = z(s,tier) + β·Ã(s,tier)
9. 选择 tier* = argmax softmax(z')，经 tier→model 映射执行
10. 正常执行 Agent 工具循环（单轮 sticky 照旧）

【episode 结束（最终回复产生）】
11. Evaluator 回溯打分：quality（LLM-as-judge，隔离部署）
12. 采集客观信号：cost_saving（observeUsage/定价表）、revision_penalty（可选）
13. r = 0.6·q + 0.3·c + 0.1·rp；记忆 M ← M ∪ {(s, tier*, r)}
```

**初始超参**（论文值起步，扫描计划见 §5.4）：

| 参数 | 初始值 | 论文参照 |
|---|---|---|
| k（检索邻居数） | 10 | 10（两 benchmark 一致；8-14 稳健区间） |
| β（logit 温度） | 5 | 论文未披露（超参表缺失），自行扫描 [3,10] |
| λ（探索率） | 0.05 | WebArena 0.05（结构化动作空间）；Jericho 0.65（大动作空间）。Router 档位空间小，取 WebArena 侧 |
| α（UCB bonus） | 5 | 5（两 benchmark 最优） |
| Jaccard 阈值 | 0.8 | WebArena 0.8 |
| γ | 退化（单步） | WebArena 0.1 / Jericho 0.5 |

---

## 5. 实验协议（与开发并行，非事后补充）

> **v1.2 注**：22h 预算下按 `project-decisions.md` v2.0 简化执行——任务集 24 条、单轮运行、默认超参（不做扫描）、消融保留 Table-8 式一组。本节其余内容（多 run、扫描、B′ 对照等）转为 future work 参考。

### 5.1 三层对照组

| 组 | 配置 | 回答的问题 |
|---|---|---|
| A. 静态路由 | 固定 `agent.model`（不做档位判定） | 学习增益总量（地板） |
| B. 原生 TokenSaver | 现有 Judge（远程或本地同款 1B），无记忆 | **"Before" 基线 = PilotDeck 现状** |
| C. JitRL Router | 本地 1B Judge + 记忆 + 优势调制 | "After" |

> 只拿 A 对比 C 会被质疑"比 PilotDeck 自带能力强多少"；B 才是诚实基线。

### 5.2 消融实验

1. **Logit 调制 vs Prompt 注入**（复刻论文 Table 8）：同一份记忆，对照组把检索到的优势信息以文本写入 Judge prompt（"历史上该状态 medium 档表现好"），实验组做 logit 调制。**直接验证核心主张，最硬的一张表**。
2. **1B Judge 裸奔**：B 组若用远程强 Judge，则加 B'（本地 1B 无 JitRL）——证明 1B 不是瓶颈外挂，增益来自记忆。
3. **状态表示**：三层状态 vs 仅 intent_class vs 仅原始消息（验证 D2 必要性）。
4. **记忆规模曲线**：性能随记忆条目增长的变化（对应论文 Table 23，兼作检索延迟测量）。

### 5.3 指标

- **成本**：每任务平均花费（$）、相对 B 组节省率
- **质量**：Evaluator 质量分（与奖励信号同源需注意循环论证风险，另用独立人工抽检子集校准）
- **路由正确性**：误档率（vs 人工标注档位的偏离，标注集 50 条起）
- **学习性**：episode 序列上的成本/质量曲线（Avg vs Final 结构，对齐论文多 episode 协议）
- **开销**：决策延迟增量（检索+调制 vs 单次 Judge 调用）、检索延迟随记忆规模变化

### 5.4 参数扫描计划

- β ∈ [3, 10]（首要，D1 约束下界）
- λ ∈ {0, 0.05, 0.25, 0.5}（对齐论文 Table 20 网格）
- k ∈ {6, 8, 10, 12, 14}（对齐论文 Figure 4）

### 5.5 插桩（当功能做，Day 1 起）

- 每次决策落盘：state、检索命中（k、相似度分布）、各档位 z/z'/Â、最终选择、G 来源分解
- 事件总线对接：`pilotdeck_router_decision` / `token_saver_failed` 等现有事件全量记录
- Dashboard 学习曲线/成本曲线实时可视化——同时服务 Demo 展示与实验数据采集，一物两用

---

## 6. Demo 策略（游园会）

1. **任务域**：选定有重复结构的域（同类编码问答/文档生成/固定套路信息检索），使观众输入自然落在相似状态上，检索有信号。
2. **冷启动叙事**：前 3-5 次请求 JitRL ≈ 无增益（≈B 组），之后学习曲线上升——**把"冷启动→起效"过程做成实时可视化卖点**（学习曲线上升是最抓眼的演示）。
3. **可靠性兜底**：Router 自带 fallback 链作为现场断网/限流预案；另备完整预录制运行。
4. **隐私**：观众输入进记忆需脱敏（策略见 `project-decisions.md` D12）。
5. **API 预算**：$150 封顶（`project-decisions.md` D11）。

---

## 7. 仓库形态与模块架构（方案 A）

> **v1.2 注**：22h 预算下实际采用 `project-decisions.md` D14 的简化结构（独立 Python harness，无 pnpm workspace）。本节保留为完整版参考。

```
PilotDeck-Router-JitRL/            # 本仓库（公开）
├── docs/                          # 技术文档（本文档、实验报告）
├── packages/
│   ├── jitrl-core/                # JitRL 引擎（框架无关）
│   │   ├── memory/                #   (s,a,G) 记忆库 + 生命周期管理
│   │   ├── state/                 #   三层状态抽象 + Jaccard 检索
│   │   ├── value/                 #   V̂/Q̂/Â 估计 + 探索 bonus
│   │   └── policy/                #   logit 调制（闭式更新）
│   ├── jitrl-router-plugin/       # PilotDeck 集成层
│   │   └── RouterContribution 实现 # CustomRouter 接口对接 decide()
│   ├── jitrl-evaluator/           # episode 后回溯评估（隔离部署）
│   └── local-judge/               # 本地 1B Judge 服务 + logprob 通道
├── eval/                          # 评测 harness（A/B/C 对照、消融、扫描）
├── dashboard/                     # 实时可视化（Demo + 数据采集）
└── .gitignore                     # 已忽略 PilotDeck-src/（上游依赖不 vendored）
```

**上游依赖策略**：PilotDeck 以 fork/extension 形式引用。已知缺口——插件形态下 episode 结束后的 outcome 信号（Evaluator 需要最终回复与用量回流），需要对上游加回调通道（小改动）；策略：先 fork patch、实验定稿后提 upstream PR（详见 `project-decisions.md` D13）。

---

## 8. 第一周 Spike（三项并行，互不依赖）

| Spike | 内容 | 验证目标 | 风险等级 |
|---|---|---|---|
| S1 | 本地 Q4 1B 服务 → 50 行客户端取 4 档位 token logprob | logprob 通道可用、排序合理、延迟可接受 | **最高**（协议通道可能缺失） |
| S2 | 状态区分度体检：30-50 条真实请求两两 Jaccard | 双峰分布（同类 >0.6 / 异类 <0.2） | 中（失败则调 signature 规则） |
| S3 | 奖励管道最小验证：一条真实 episode 走完 usage→Evaluator→r→落库 | 信号采集完整、r 计算正确 | 中（Evaluator 隔离部署） |

S1 失败的降级路径：退回 Token-level Logit 的远程变体（Judge 输出档位 token，取 API logprob——需 provider 支持），或最坏情况 Verbalized Logit（置信度→logit，转换公式自行定义并写入已知限制）。

---

## 9. 已知风险与缓解

| 风险 | 缓解 |
|---|---|
| 本地 provider 无 logprob 通道（S1 失败） | 降级路径见 §8；或对上游小改 |
| Q4 logprob 序不稳定 | Q4 vs FP16 一致率校准（20-30 条）；温度校准 |
| 状态区分度不足（D2 失败模式） | Spike S2 前置体检；signature 规则迭代 |
| Reward hacking（无脑降档） | w_q>w_c 原则；质量分独立抽检 |
| 冷启动无增益 | Demo 叙事化（学习过程即卖点）；预热记忆备选 |
| 反事实缺失 | Evaluator 能力不足检测近似；README 已知限制明示 |
| 1B Judge 太弱 | 消融 B' 组隔离验证；必要时升级 3B/7B（成本仍可控） |
| 记忆污染（错误经验放大） | 置信度加权（Evaluator certainty 分级已在设计内）；记忆老化淘汰（生命周期管理） |

---

## 10. 未决问题清单

> **v1.1 更新（2026-09-11）**：原七项未决问题已经项目所有者授权，按推荐默认值全部解决，决策编号 D6–D13 续接本文档，详见 **`project-decisions.md`**。
> 仍开放的唯一事项：游园会具体日期（仅影响 D6 时间轴的绝对日期换算，不阻塞任何开发工作）。

---

*本文档基于 JitRL 论文（arXiv:2506.10985v3）与 PilotDeck 源码（main 分支，2026-09 快照）撰写。源码精读笔记：`PilotDeck-src/src/router/`（本地克隆，不入库）。*
