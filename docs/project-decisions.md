# JitRL Router 项目决策与规划文档（D6–D15）

> 本文续接 `technical-decisions.md`（D1–D5，技术架构决策，同目录）。
> 状态：**v2.9 已冻结**（2026-09-11 修订：Web Demo 从 Session 2 合并至 Session 3，并排在消融实验与 README 之后、演示视频之前；Session 3 执行顺序重新冻结）。
> 开放事项：仅剩游园会日期（不阻塞）。

---

## 0. 决策速览

| # | 领域 | 决策 |
|---|---|---|
| D6 | 时间线 | 22h 三会话（8+8+6）；会话 1 出口 = 海报前全部事宜；**v2.9：Web Demo 合并至 Session 3，最终功能冻结点顺延至 Demo 完成** |
| D7 | Judge 模型/后端 | **v2.1**：MiniCPM5-1B（Q4 GGUF）+ llama.cpp（用户并行会话部署中），GBNF 约束输出 |
| D8 | Evaluator | **v2.1**：CPA/gpt-5.6-sol（不在 tier 映射内，杜绝执行模型自评）；备选 CPA/OpenBMB-5.3 |
| D9 | 评测任务集 | 24 条（6 族 × 4 变体）+ 续轮探针；单轮运行；GT 档位标注 |
| D10 | 验收目标 | 成本 -20%、质量差 ≤0.3/5、误档率 +≤5pp、延迟 +≤500ms（单轮 n=24，方向性证据） |
| D11 | API 预算 | 预计实际支出 $5-8（<$20 上限）；**v2.3**：CPA 实际价格入表（`Price of TestModel.md`），cost_saving 分母 = glm-5.3 |
| D12 | 记忆隐私 | 现场会话级隔离不落盘；公开仓库带合成示例记忆；落盘仅存签名 + G |
| D13 | 上游策略 | 深度集成与 upstream PR → future work；README 给出集成路径说明 |
| D14 | 架构简化 | 独立 Python harness 忠实复刻 Router 决策路径 + Web Demo；**v2.1**：执行层运行时读取 `~/.pilotdeck/pilotdeck.yaml`（tier→model 映射见 D14.1） |
| D15 | 文档管理 | **v2.1 新增**：全部文档正本存于 `D:\GithubPlay\WorkSpace\Pdwork\docs`，助手自动写入；版本号 + INDEX 清单 |

---

## D6. 里程碑与时间线（22 小时预算）

**决策**：总开发预算 **22 小时**，分三次会话（8h + 8h + 6h）。

### 会话 1（8h）——"海报前"会话（硬性出口：海报设计所需全部材料）

| # | 时长 | 任务 | 出口判据 |
|---|---|---|---|
| 1 | 0.5h | v2.1 决策文档冻结（本次修订） | 决策基线更新完毕 |
| 2 | 0.75h | **S1**：对接并行会话部署的 llama.cpp/MiniCPM5-1B——GBNF 约束 + 4 档位 logprob 探针 | logprob 可取、序合理、亚秒延迟；失败走 D7 降级链 |
| 3 | 0.75h | 任务集 v0：6 族 × 4 变体 + 续轮探针（含 GT 档位标注） | 24+ 条就绪，签名可区分 |
| 4 | 2.5h | `jitrl_core` MVP（Python）：signature / intent / Jaccard 检索 / 记忆 / V-Q-Â / logit 调制 + JSONL 插桩 | 单元冒烟通过 |
| 5 | 1h | harness：A/B/C 模式切换决策循环（routing-only）+ 执行层配置（读 pilotdeck.yaml、tier→model 映射、定价表初版） | routing-only 模式可跑 |
| 6 | 1.25h | 冒烟实验：24 任务序列 B vs C（routing-only），验证记忆命中与调制翻转 | 学习曲线雏形可见 |
| 7 | 0.5h | **海报内容简报**：故事线、3 个核心主张、架构图规格、数据占位；结构对齐 design system 定制区（hero_label / hero_numbers / bg_atmosphere / annotations / slogan） | 简报可直接填充定制区 → lint → 生成海报 |
| 8 | 0.75h | 缓冲 | — |

> **v2.1 注**：llama.cpp + MiniCPM5-1B 部署已由项目所有者并行会话承担，S1 由"部署+验证"（1.5h）缩减为"对接验证"（0.75h），释放时间转为显式缓冲。

### 会话 2（8h）——"真实实验"会话（出口：实验数据与分析资产）

| # | 时长 | 任务 |
|---|---|---|
| 1 | 1h | 接通真实执行（tier→model 映射，经 CPA）与 Evaluator（gpt-5.6-sol） |
| 2 | 1h | 端到端校准冒烟：3 episode 走完 reward 落库 |
| 3 | 2h | 全量运行 A/B/C × 24 任务（~72 episode，含盯跑修 bug） |
| 4 | 1h | 结果分析：成本/质量/误档率/学习曲线 → `results.md` + 图表（文档正本目录） |
| 5 | 2.5h | 成本未达标后的显式 C1 保守策略：预注册 → 实现 → 真实运行 → 分析 |
| 6 | 0.5h | 缓冲 |

### 会话 3（6h）——"消融 + 文档 + Demo + 包装"会话（出口：五项交付物齐备）

| # | 顺序 | 任务 |
|---|---:|---|
| 1 | 第一 | 消融 D：prompt 注入 vs logit 调制（复刻论文 Table 8） |
| 2 | 第二 | 技术设计图（架构图/决策流程图，供 README 与 Demo 共用） |
| 3 | 第三 | README 共撰（改进点/架构/运行方式/完整实验数据/已知限制） |
| 4 | 第四 | **Web Demo**：在消融与 README 的稳定叙事/数据契约上实现，囊括 A/B/C0/C1、continuation probes、故障审计与实时决策可视化 |
| 5 | 第五 | 演示视频（2–3 分钟屏录，基于完成后的 Web Demo，先写脚本） |
| 6 | 第六 | 仓库终检：LICENSE、清理、文档正本入库、最终提交/推送、彩排清单 |
| — | 外部 | 海报由其他 session 负责，本线路不处理 |

**功能冻结点（v2.9）**：Session 3 的 Web Demo 完成后冻结功能；随后只修阻塞性 bug、制作视频并做仓库包装。

**关键风险前置**：S1（logprob 通道）仍是第一风险，置于会话 1 最前段；llama.cpp/模型部署依赖并行会话进度，会话 1 开始前确认服务就绪。

**海报工作流（v2.3，design system 已提前到位）**：`D:\GithubPlay\Skills\naked-eye-3d-design-system`（裸眼3D 模板 v2：互补色对 + 巨号字 + 黑色远景压强 + 硬边无渐变 = 伪景深）。会话 1 末简报按其**定制区**字段结构产出，数据占位在会话 2/3 回填真实数字；生成走模板定制指南（复制模板 → 只改定制区 → lint → AI 工具生成 → grep 上线自检）。主题建议 **blue-magenta**（两套已验证主题中"工具感 + 人味"最契合本项目），终选留待海报阶段。

**海报印刷参数（v2.4，项目所有者指定，硬约束）**：A3 竖版（297×420 mm）｜300 DPI｜**3508×4961 px**｜颜色模式 **CMYK**（印刷必须，避免色差）｜四边出血 3 mm（≈35 px；背景色延伸至出血边，文字一律不进出血区）｜正文最小字号 **≥10 pt**（≈42 px）｜标题 **≥18 pt**（≈75 px）｜重要文字与 LOGO 距成品边 **≥5 mm**（≈59 px 安全区）。

**画布适配注记（v2.4）**：design system 模板按 1080 px 宽数字画布设计，生成时须将模板提示词中的"9:16 竖版"**替换为上述 A3 印刷参数**；全套字号按 3508/1080 ≈ **3.25×** 等比放大后：caption 14px→45.5px≈10.9pt（恰过 10 pt 下限）、body-md 24px→78px≈18.7pt——等比缩放天然满足印刷字号约束，但仍以 42px/75px 为硬下限逐项校验。**CMYK 色域风险**：模板高饱和 RGB 专色（#FF1B6B / #1F1FFF 等）转 CMYK 存在色域压缩，定稿前须在 CMYK 模式下复核互补色对的前后分离感，必要时替换为印刷友好近似色（记入海报阶段已知限制）。

---

## D7. Judge 模型与推理后端（v2.1 修订）

**决策**：`MiniCPM5-1B`（Q4 量化 GGUF）经 llama.cpp 部署（项目所有者并行会话进行中），作为本地 Judge 与 logit 来源。

**选型理由**：
1. **生态协同**：PilotDeck 由 THUNLP / ModelBest / OpenBMB 联合开发，MiniCPM 同属 OpenBMB 系——"PilotDeck + 同家族 MiniCPM 本地 Judge"是海报叙事的自然组成部分；
2. **中文能力**：MiniCPM 系列同尺寸中文能力领先（Demo 观众主要输入中文）；
3. **部署外部化**：llama.cpp + MiniCPM5-1B Q4 由项目所有者另一会话部署，S1 缩减为对接验证（见 D6 注）；
4. **logprob 通道**：llama-server 的 OpenAI 兼容 `/v1/chat/completions` 支持 `logprobs` + `top_logprobs`（S1 第一验证项）。

**GBNF 约束输出**：

```
root ::= "<tier>" ("simple" | "medium" | "complex" | "reasoning") "</tier>"
```

三重收益：**延迟**（约束生成压缩到约 5 token，CPU 亚秒级）、**解析**（不可能产出非法格式）、**logprob 语义清晰**（四档位首字符 s/m/c/r 互异，档位名首 token 的 logprob 即基策略分布 z）。

**S1 必须验证的技术细节**（不确定性明示）：
1. GBNF 约束下 `top_logprobs` 返回的是**语法掩码后**还是原始分布的 logprob——需要的是约束集（4 档位）上的相对分布；若返回原始分布，取 top-N 中出现的档位 token、缺失档位 clamp 到低值（如 z_min = -10）；
2. MiniCPM5 chat template 细节：thinking 模式开关、系统提示行为（对齐现有 Judge 请求的 `thinking: {enabled: false}`）；
3. 4 档位首 token 可区分性（s/m/c/r）。

**能力不确定性（明示）**：MiniCPM5-1B 判档质量无独立先验——B 组（同 Judge、无记忆）即其"裸奔"基线，实验本身完成验证。若不足，升级路径：改用 CPA 远程 flash 模型作 Judge（换 Judge 不换管线与记忆，D1 的 tier 抽象层保证记忆对 Judge 换代免疫）。

**降级链**（v2.1 调整）：
1. 后端异常 → 与部署会话协调参数（上下文长度 / 并发 / 量化档）；
2. logprob 通道失败 → Verbalized Logit（置信度→logit，转换公式自行定义并写入已知限制）；
3. 判档质量不足 → CPA 远程 Judge（见升级路径）。

---

## D8. Evaluator 选型与隔离（v2.1 修订）

**决策**：episode 后回溯评估用 **CPA/gpt-5.6-sol**（经 `~/.pilotdeck/pilotdeck.yaml` 的 CPA provider 调用）。

**理由**：
- **隔离原则**：gpt-5.6-sol 不在 tier→model 映射（D14.1）内——不存在执行模型自评；与本地 Judge（MiniCPM5-1B）不同模型、不同部署；
- **凭据现成**：CPA provider 已配置于 pilotdeck.yaml，零额外配置（原 deepseek-chat 外部依赖取消）；
- **备选**：CPA/OpenBMB-5.3（注：其同时是 complex 档执行模型——但对 B/C 两组对称施加，偏差方向一致，可接受并明示）。

**评分协议**：沿用论文 WebArena Evaluator 的 USEFUL / CERTAINTY 双轴结构（-3 ~ +3），附加"能力不足检测"字段（见 technical-decisions.md D3）。输出用 JSON Schema 约束。

**失败兜底**：Evaluator 超时/失败时，该 episode **不写记忆**（宁缺毋滥，防错误经验入库），仅记插桩日志。

---

## D9. 评测任务集

**决策**：**24 条任务 = 6 族 × 4 变体**，另加 4-6 条续轮探针。单轮运行（无多 seed）。

**构成**：
- **6 个任务族**（D2 intent_class 的合并版）：代码生成、闲聊/简单问答、文档写作、数据分析、重构、信息检索——每族 4 条**同签名变体**（同结构、不同实例，如"给这段 Python 函数写单测"换 4 个不同函数）；
- **变体即重复**：族内变体在序列中间隔出现——后出现的变体从前面的记忆受益，**这是学习曲线信号的来源**（原多轮组设计的替代方案，成本减半）；
- **每条任务三要素**：用户消息、GT 档位标注（助手起草、用户抽检）、质量要点清单（Evaluator 评分依据，压低 LLM-as-judge 漂移）；
- **续轮探针**："继续/好的"类消息 4-6 条，验证 previousTier 继承与短续轮不降档；
- **取舍**：n=24 单轮 → 统计功效有限，结果按**方向性证据**报告（写入已知限制）；任务集仍脱敏入库作为可复现性资产。

---

## D10. 验收量化目标

**决策**（写入 README 目标声明；单轮 n=24，按下表判方向性达标，非统计显著性检验）：

| 指标 | 目标 | 测法 |
|---|---|---|
| 成本节省 | C 组较 B 组 **≥ 20%** | $/任务（计费口径；A/B/C 同任务集） |
| 质量持平 | C vs B 质量分差 **≤ 0.3**（5 分制） | Evaluator 盲评（不告知组别）+ 人工抽检校准 |
| 路由正确性 | 误档率 ≤ B 组 **+ 5pp** | vs D9 人工标注档位 |
| 学习性 | C 组后半程成本曲线显著低于前半程 | episode 序列分段比较 |
| 决策开销 | 端到端决策延迟增量 **≤ 500ms** | 检索+调制耗时插桩 |

**判定原则**：若质量/误档率达标但成本节省 <20%——项目仍成立（同等成本下更准的路由），README 如实报告；反向（省钱但掉质量）判定失败，回炉调 w_q/w_c 权重。

---

## D11. API 预算与计费口径（v2.3：实际价格到位）

**决策**：**预计实际支出 $5-8**（<$20 上限）；计费采用项目所有者提供的 CPA 实际价格（`Price of TestModel.md`，正本目录），会话 1 写入 `eval/pricing.json` 供 harness 计费。

**定价表（$/Mtok）**：

| # | 模型 | 实验角色 | 输入 | cache 输入 | 输出 |
|---|---|---|---|---|---|
| 1 | CPA/glm-5.3-flash | simple 档执行 | 0.15 | 0.03 | 0.50 |
| 2 | CPA/opencode-v4-flash | medium 档执行 | 0.30 | 0.006 | 1.20 |
| 3 | CPA/glm-5.3 | reasoning 档执行 + **A 组静态路由** | 1.40 | 0.26 | 4.40 |
| 4 | CPA/OpenBMB-5.3 | complex 档执行 | 1.40 | 0.26 | 4.40 |
| 5 | CPA/gpt-5.6-sol | Evaluator（三组对称调用） | 4.00 | 0.40 | 20.00 |

- **MiniCPM5-1B 本地 Judge 成本记 0**（本地推理无边际计费）；
- **cost_saving 分母 = glm-5.3**（A 组静态路由 = `scenarios.default`，与 D3 的"default 档参考成本"定义一致）；
- **梯度印证**：simple→reasoning 输入价差 **9.3×**、输出价差 **8.8×**——路由降档的节省空间充足，D10 的 20% 总体节省目标量级合理；
- Evaluator 成本**不计入 episode 路由成本**（对三组对称），计入总预算核算；
- v2.1 简化 harness 不做 cache-aware 切换，cache 列仅留字段。

**量级核算（按实际价格）**：设 episode 平均 2K 输入 + 1K 输出——A 组全量 72 episode ≈ 72 × ($0.0028 + $0.0044) ≈ **$0.52**；Evaluator 每次 ≈ 3K 输入 + 0.3K 输出 ≈ $0.018，~100 次 ≈ **$1.8**；消融 + Demo + 缓冲 ≈ $2-3。**合计 ≈ $5-8**。

**结构注记**：gpt-5.6-sol 的评估开销（$1.8）超过全部执行模型之和（$0.52）——**评估是本项目最贵单项**。D3 的"单步 episode、Evaluator 仅在 episode 结束调用一次（不逐步打分）"设计正是对这一价差的结构性回应；如需进一步控费，备选 Evaluator OpenBMB-5.3（价 1.4/4.4，约省 75% 评估费，代价是引入 complex 档自评面，见 D8 备选注记）。

---

## D12. 记忆与数据隐私

**决策**：
1. **游园会现场**：记忆**会话级隔离**——每次演示会话独立记忆、仅驻内存、结束即清，不落盘；观众输入不进任何持久存储。
2. **公开仓库**：附带一份**合成示例记忆**（脚本化场景生成，无真实用户数据），供 onboarding 与离线 Demo。
3. **实验/评测落盘**：只存 `task_signature`（D2 抽象层天然是脱敏层——路径/数字/长串已替换为占位符）+ 档位 + G 值 + 时间戳；**不存**原始用户消息与模型输出（质量只存数值）。
4. **插桩日志**：含原文的详细日志仅存本地 `logs/`（已 gitignore），对外报告只出现聚合统计。

---

## D13. 上游贡献策略

**决策**：fork patch 与 upstream PR 均转为 **future work**。22h 预算内不做 PilotDeck 运行时深度集成。

**替代交付**：
- README 专节"集成路径"：说明 CustomRouter 扩展点（`decide()` 返回 provider/model 即完全接管路由）+ 需要上游补齐的两个通道（logprob 透传、episode outcome 回调），配代码级指引——证明集成路径真实可行，只是时间预算外；
- 会话有富余时（stretch）：提供 TS 版 CustomRouter 适配器骨架（未接入真实运行时，标注 PoC）。

**保留的叙事**：低侵入性论证从"已实现的 diff"改为"设计上的扩展点分析"——CustomRouter 接口本身就证明该改造不需要动 Router 内核。

---

## D14. 架构简化：独立 harness 优先

**决策**：以**独立 Python harness 忠实复刻 Router 决策路径**为主交付形态，而非 PilotDeck fork/插件深度集成。

**复刻保真清单**（从 `PilotDeck-src/src/router/` 逐项对齐）：
- Judge prompt：沿用 `generateJudgePrompt` 的 `<tier>NAME</tier>` 格式、tier 描述、previousTier 续轮规则；
- 档位体系：simple / medium / complex / reasoning 四档 + tier→model 静态映射；
- 语义：每轮重新决策（无轮间 sticky）、短续轮继承逻辑；
- 事件语义：决策/失败事件对应 `pilotdeck_router_*` 系列字段。

**理由**：
1. 22h 内完成"能跑实验 + 能现场演示"的最短路径——不依赖 PilotDeck 运行时启动链路；
2. 游园会现场可靠性：独立进程 + Web 页面，故障面最小；
3. 语言选择 Python：迭代速度快（harness / eval / demo 后端一体），算法核心与运行时本就解耦（D1 的 tier 抽象层）。

**仓库结构（v2.1）**：

```
PilotDeck-Router-JitRL/
├── docs/                  # 会话 3 包装阶段从文档正本目录（D15）复制终版入库
├── jitrl_core/            # Python 包：state / memory / value / policy + 插桩
├── harness/               # A/B/C 决策循环 runner（复刻 Router 语义）
├── local_judge/           # GBNF grammar + logprob 客户端 + endpoint 配置（llama.cpp 服务由并行会话部署）
├── eval/                  # 任务集（JSONL）+ 定价表 + 分析脚本 + 图表生成
├── demo/                  # Web Demo：FastAPI + 单页前端
└── adapter-poc/           # stretch：TS CustomRouter 适配器骨架
```

### D14.1 执行层模型映射（实验默认配置，v2.1 新增）

经 `~/.pilotdeck/pilotdeck.yaml` 确认，CPA provider 下 8 个模型可直接调用：`glm-5.3`、`glm-5.3-flash`、`tokendance-v4.1-flash`、`opencode-v4-pro`、`opencode-v4-flash`、`opencode-v4-flash-vision-exp`、`OpenBMB-5.3`、`gpt-5.6-sol`。

| 角色 | 模型 | 推断依据 |
|---|---|---|
| simple 档 | CPA/glm-5.3-flash | flash 系最轻，与默认主模型同家族 |
| medium 档 | CPA/opencode-v4-flash | 中档 flash（与用户现配置 Judge 同系） |
| reasoning 档 | CPA/glm-5.3 | 默认主模型（agent.model = scenarios.default） |
| complex 档 | CPA/OpenBMB-5.3 | 旗舰，承载 autoOrchestrate 编排 |
| A 组（静态路由） | CPA/glm-5.3 | "无路由"现状 = scenarios.default |
| Evaluator | CPA/gpt-5.6-sol | 不在 tier 映射内（隔离），见 D8 |

- 映射基于模型命名惯例推断（flash < pro / 旗舰），**已获实际价格印证**（D11 定价表：flash 系 < glm-5.3/OpenBMB-5.3，梯度与推断一致），实验配置文件中一行可替换；
- 用户当前 router 配置为四档同模型（opencode-v4-flash-vision-exp）且 `router.enabled: false`——本映射为实验统一设定，A/B/C 三组共用，保证组间公平；
- harness 运行时直接读取 `~/.pilotdeck/pilotdeck.yaml` 的 CPA provider 凭据调用上述模型（单一凭据源，不复制 key）。

**明确砍掉清单**：参数扫描（用论文默认值 k=10 / β=5 / λ=0.05 / α=5）、多 seed / 多 run、B′ 远程强 Judge 对照（B 组直接用本地 Judge 无记忆——同 Judge 对比才能把差异归因于记忆）、Q4 vs FP16 系统校准（保留机会性抽查）、记忆老化淘汰（简单容量上限即可）、upstream PR。

**保留清单**：三组对照（A/B/C）、Table-8 式消融（prompt 注入 vs logit 调制，会话 3 执行）、全量 JSONL 插桩、学习曲线、D12 隐私边界。

---

## D15. 文档管理（v2.1 新增）

**决策**：
1. 本项目**全部文档正本**统一存放于 `D:\GithubPlay\WorkSpace\Pdwork\docs`，由助手自动写入与更新，项目所有者无需手动保存；
2. **版本规范**：每份文档头部状态行标注版本号；文内 Changelog 记录历次变更；`INDEX.md` 维护全量清单（文件 / 标题 / 版本 / 日期）；
3. 文件名保持稳定（不带版本后缀），版本以头部状态行 + INDEX 为准；
4. 公开仓库（交付物 #1）在会话 3 包装阶段从正本目录复制终版入库；此前仓库不维护文档副本，避免双源漂移。

---

## 11. 变更控制

- 本文档与 `technical-decisions.md` 共同构成完整决策基线（D1–D15）；
- 冻结后的变更：在本文档追加 changelog 条目，记录**变更原因与影响面**，不静默改写；
- 实验数据推翻决策（如 S1 降级、冒烟显示状态表示需调整）属**合法变更**，走变更记录流程。

### Changelog

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-09-11 | v1.0 | 初版：D6–D13 按项目所有者授权采用推荐默认值（六周计划），冻结 |
| 2026-09-11 | v2.0 | 时间预算收缩为 22h（8+8+6）：D6 重排为三会话；D9 任务集缩至 24；D10 加方向性证据注；D11 预算重估 <$20；D13 转 future work；新增 D14 架构简化（独立 harness） |
| 2026-09-11 | v2.1 | 三项前置输入落地：D7 Judge 换型 MiniCPM5-1B（并行会话部署 llama.cpp）；D8 Evaluator 改 CPA/gpt-5.6-sol（弃外部 deepseek 依赖）；D14.1 执行层模型映射（CPA 8 模型，harness 直读 pilotdeck.yaml 凭据）；D11 计费口径注记；新增 D15 文档管理（正本目录 + 版本规范） |
| 2026-09-11 | v2.2 | D11 增补待计价模型清单（5 模型 + MiniCPM5-1B 记 0 说明 + 所需字段），前置输入 #6 转为进行中；design system 确认延迟中（不阻塞会话 1 其余工作） |
| 2026-09-11 | v2.3 | 前置输入全闭环：D11 定价表落定（5 模型实际价格 + 量级核算 $5-8 + cost_saving 分母 = glm-5.3 + 评估为最贵单项注记）；D14.1 映射获价格梯度印证；D6 增海报工作流（design system 提前到位，blue-magenta 建议）；§13 #5/#6 解决 |
| 2026-09-11 | v2.4 | D6 新增海报印刷参数硬约束（A3 竖版 / 300 DPI / 3508×4961 px / CMYK / 3mm 出血 / 正文 ≥10pt / 标题 ≥18pt / 安全区 ≥5mm）与画布适配注记（3.25× 等比缩放校验 + CMYK 色域风险）；llama.cpp 服务就绪，会话 1 启动（S1 探针 / 任务集 / 核心引擎三线并行） |
| 2026-09-11 | v2.5 | §12 会话 1 checklist 收尾（无决策内容变更）：决策冻结 / S1 logprob 探针 / 任务集 v0 / jitrl_core / harness / 集成冒烟 六项完成并勾选，各附交付报告索引；海报内容简报项标注"外部 session 处理中，不作为阻塞"；会话 1 出口达成（除海报线），摘要见 `SESSION-1-SUMMARY.md` v1.0 |
| 2026-09-11 | v2.6 | §12 Session 2 状态同步（无冻结决策变更）：CPA 真实执行客户端、Evaluator、真实 usage/定价/reward 链路及失败不写 memory 已完成，104 项测试通过；3 episode 已以真实 CPA + mock Judge 校准，真实 Judge 校准等待外部 session 启动 llama.cpp；详见 `SESSION-2-PROGRESS.md` v1.0 |
| 2026-09-11 | v2.7 | §12 Session 2 校准完成（无冻结数学变更）：llama.cpp `:18080` 已就绪；真实 Judge + CPA 执行 + Evaluator 的 3 episode 全链路成功；修复 b10903 `reasoning_content` 响应兼容，新增独立 JitRL 后处理延迟插桩，119 项测试通过；详见 `SESSION-2-PROGRESS.md` v1.1 |
| 2026-09-11 | v2.8 | §12 Session 2 实验状态同步（无静默改数学）：同序 A/B/C ×24、5 条 continuation probes、聚合 JSON/CSV 与 5 张 SVG 图表完成，150 项测试通过；D10 质量/误档/延迟达标，C 相对 B 成本节省 −3.78% 未达 ≥20% 目标；详见 `results.md` v1.0 |
| 2026-09-11 | v2.9 | D6/§12 时间线重排：Web Demo 从 Session 2 合并进 Session 3，固定排在消融实验与 README 之后、演示视频之前；Session 2 出口改为实验与分析资产，功能冻结点顺延至 Demo 完成；海报仍由外部 session 负责 |

---

## 12. 三会话执行清单

### 会话 1（8h）——出口：海报前全部事宜
- [x] v2.1 决策文档冻结（本次修订）（v2.5 注：决策基线此后演进至 v2.5，当前冻结）
- [x] S1：对接并行会话部署的 llama.cpp/MiniCPM5-1B → GBNF → 4 档位 logprob 探针（含掩码后分布语义、chat template、首 token 可区分性验证）——已交付 `S1-logprob-probe-report.md` v1.0（Method B 加固版客户端 `local_judge/`）
- [x] 任务集 v0：6 族 × 4 变体 + 续轮探针 + GT 标注（助手起草，用户抽检）——已交付 `TASK-taskset-report.md` v1.0（24 主任务 + 5 续轮探针 = 29 条；GT simple 5 / medium 8 / reasoning 9 / complex 2）
- [x] `jitrl_core` MVP + JSONL 插桩——已交付 `S1-core-harness-report.md` v1.0（引擎规格逐字实现 + 测试，当前 48/48）
- [x] harness routing-only 模式（A/B/C 切换）+ 执行层配置（pilotdeck.yaml 读取、tier→model 映射、定价表初版）——同上报告（A/B/C CLI、mock/llama 双 judge、separability 工具、`eval/pricing.json`）
- [x] 冒烟：24 任务序列 B vs C，记忆命中 + 调制翻转可见——已交付 `S1-integration-smoke-report.md` v1.0（集成门槛全绿：48/48 测试、任务集复检一致、真实 Judge B/C 全通；C 记忆 0→29、检索命中 6/29、跨变体 12 对全同族、翻转 3 次（1 改善 / 2 中性 / 0 恶化）、准确率 7/29→8/29、任务级 C ⊇ B）
- [ ] **海报内容简报**（故事线 / 3 个核心主张 / 架构图规格 / 数据占位，对齐 design system 定制区字段）——外部 session 处理中，不作为阻塞

### 会话 2（8h）——出口：实验数据与分析资产
- [x] 接通真实执行（tier→model 映射，经 CPA）与 Evaluator（gpt-5.6-sol）——客户端、结构化盲评、真实 usage/定价/reward 与失败不写 memory 已完成；见 `SESSION-2-PROGRESS.md`
- [x] 端到端校准：3 episode reward 落库——真实 llama Judge + CPA 执行 + Evaluator 全链路成功
- [x] 全量 A/B/C × 24（~72 episode）——正式 A/B/C 均 24/24；B 已按 A/C 同序重跑保证公平性
- [x] `results.md` + 图表——`results.md` v1.0；聚合 JSON/CSV 与 5 张 SVG 完成；D10 3/4
- [x] C1 成本修复实验——min-neighbor gate 已预注册、实现、真实 24/24 运行与分析；P1–P5 为 3/5，按规则停止，见 `S2-C1-RESULTS.md` v1.0
- [→] Web Demo——**移入 Session 3**，排在消融与 README 之后、演示视频之前

### 会话 3（6h）——出口：五项交付物齐备
- [ ] **1. 消融**：prompt 注入 vs logit 调制
- [ ] **2. 技术设计图**（README + Demo 共用）
- [ ] **3. README 共撰**（改进点 / 架构 / 运行方式 / A/B/C0/C1/消融结果 / 已知限制）
- [ ] **4. Web Demo**（表单 + 实时决策可视化 + A/B/C0/C1/消融/probes/故障审计 + 学习曲线/成本面板）
- [ ] **5. 演示视频**（2–3 分钟屏录，基于完成后的 Demo，先写脚本）
- [ ] **6. 仓库终检** + LICENSE + 文档正本入库 + 最终提交/推送 + 彩排清单
- [ ] 海报由外部 session 负责，本线路不处理、不作为 Session 3 阻塞项

---

## 13. 会话前置输入清单（v2.1 更新）

| # | 事项 | 状态 |
|---|---|---|
| 1 | API key | ✅ **已解决**：harness 运行时读取 `~/.pilotdeck/pilotdeck.yaml`（CPA provider，8 模型可直接调用，凭据不复制） |
| 2 | GPU / llama.cpp 部署 | ✅ **已解决**：llama.cpp + MiniCPM5-1B Q4 由项目所有者并行会话部署 |
| 3 | 模型预下载 | ✅ **已解决**：同上 |
| 4 | 游园会日期 | 开放，不阻塞 |
| 5 | design system | ✅ **已解决（提前到位）**：`D:\GithubPlay\Skills\naked-eye-3d-design-system`（裸眼3D 模板 v2）；海报工作流见 D6 附注 |
| 6 | CPA 计费口径 | ✅ **已解决**：`Price of TestModel.md`（正本目录）——5 模型 input/cache/output 单价已入 D11 定价表 |

---

*本文档的决策均基于 `technical-decisions.md`（D1–D5）的技术框架与 PilotDeck 源码精读结论。上游源码：`PilotDeck-src/`（本地克隆，不入库）。*
