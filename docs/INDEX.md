# JitRL Router 项目文档索引（INDEX）

> 状态：v2.0（2026-09-12）
> 本目录（`D:\GithubPlay\WorkSpace\Pdwork\docs`）为项目**全部文档正本**存放处（决策 D15）。由助手自动写入与更新，项目所有者无需手动保存。版本以各文档头部状态行为准，本清单同步维护。

## 文档清单

| 文件 | 标题 | 版本 | 日期 | 说明 |
|---|---|---|---|---|
| `technical-decisions.md` | 技术决策文档（D1–D5） | v1.3 | 2026-09-11 | 问题定义、JitRL↔Router 概念映射、五项技术决策（动作空间/状态表示/奖励信号/logit 来源/Sticky 关系）、算法流程与初始超参、完整版实验协议（future work 参考） |
| `project-decisions.md` | 项目决策与规划文档（D6–D15） | v2.9 | 2026-09-11 | Web Demo 从 Session 2 合并至 Session 3；固定顺序为消融→技术图→README→Demo→视频→发布，功能冻结点顺延至 Demo 完成 |
| `SESSION-2-PROGRESS.md` | Session 2 当前进度 | v1.8 | 2026-09-11 | C1 min-neighbor gate 已预注册、实现、真实运行和分析收尾；196项测试，P1–P5为3/5并按规则停止 |
| `results.md` | Session 2 全量实验结果 | v1.0 | 2026-09-11 | 正式同序 A/B/C ×24、continuation 0/5、质量/成本/路由/延迟/检索与 flips；D10 总判定 3/4、成本目标未达标 |
| `S2-C1-PREREGISTRATION.md` | C1 保守策略实验预注册 | v1.0 | 2026-09-11 | 运行前冻结 min-neighbor gate（n=3）唯一变量、完整参数、P1–P5 判据、全序列重试和停止条件；C0 结果不变 |
| `S2-C1-RESULTS.md` | C1 保守策略实验结果 | v1.0 | 2026-09-11 | B/C0/C1对比；gate抑制早期flip但级联新增T20 flip，P1/P5失败、总判定3/5，停止继续调参 |
| `SESSION-3-ORCHESTRATOR-HANDOFF.md` | Session 3 Orchestrator 交接文档 | v1.0 | 2026-09-11 | Session 3 完整接续：固定顺序、正式数据源、消融/技术图/README/Demo/视频/发布要求、风险与首轮动作 |
| `S3-TAB8-PREREGISTRATION.md` | Session 3 Table-8 式消融预注册 | v1.0 | 2026-09-11 | T8-LM vs T8-PM；正式 C0 prefix memory replay、防自身/未来 outcome 泄漏、prompt 白名单、公平条件、指标与停止规则在运行前冻结 |
| `S3-TAB8-RESULTS.md` | Session 3 Table-8 式消融结果 | v1.0 | 2026-09-11 | LM/PM 均24/24且质量同为2.5417；PM成本低$0.004660，但6次注入均未改tier，差异主要来自LM在T19的有害flip |
| `S3-DEMO-VIDEO-SCRIPT.md` | 演示视频脚本 | v1.0 | 2026-09-11 | 11 分镜 ≈2:00：动机/架构/六组结果/判据/C0 flips/C1 级联/T8 caveat/延迟归因/在线降级/隐私/开源结尾；成片 demo/demo-video.mp4 |
| `SESSION-3-PROGRESS.md` | Session 3 进度 | v1.7 | 2026-09-11 | 六项检查点完成记录 + 并行协调裁决 + 彩排清单 + provider 表述修正（检查点7，README 按用户指示精简）+ 集成 MVP 登记（检查点8） |
| `SESSION-3-PROGRESS.md` | Session 3 进度 | v1.2 | 2026-09-11 | 检查点制进度：T8消融、4张SVG、README完成并锁定Demo契约（demo.server :8300）；下一项 Web Demo |
| `S4-INTEGRATION-REPORT.md` | Session 4 · PilotDeck 集成 MVP 交付报告 | v1.0 | 2026-09-12 | D13 future work 首次执行：TS 原生移植经官方 CustomRouter/RouterContribution 接入（builtin jitrl 插件，零内核决策链改动）；缺口①打分式 Judge 保守等价、缺口② onTurnOutcome 回调；reward=0.6·quality+0.3·costSaving 学习闭环与记忆持久化；交付物 patch+overlay 入库（commit ce470e3），README §9 改写推送（7b10fd6）；tsc 通过、59/59、Python 261 不变、签名差分 21/21；冻结结论不受影响 |
| `Price of TestModel.md` | CPA 实际计费口径 | v1.0 | 2026-09-11 | 项目所有者提供：5 模型 input/cache/output 单价（$/Mtok）；消费方为 D11 与 `eval/pricing.json` |
| `S1-logprob-probe-report.md` | S1 Logprob 探针报告 | v1.0 | 2026-09-11 | llama.cpp/MiniCPM5-1B logprob 通道验证；Method B 加固版、tokenization 风险、延迟与 3/7 Judge 命中结果 |
| `TASK-taskset-report.md` | 评测任务集 v0 执行报告 | v1.0 | 2026-09-11 | Session1-T3 汇报：族与 GT 分布、29 行呈现序列、误导面设计、校验输出及文件边界确认 |
| `S1-core-harness-report.md` | S1 核心引擎 + A/B/C Harness 交付报告（TASK-core-harness） | v1.0 | 2026-09-11 | Session1-T4/T5 交付：jitrl_core 引擎 + harness 垂直切片、原报告 47/0 测试（集成兼容修复后当前 48/48）、A/B/C 与 separability 命令及 mock 实测、JudgeClient 契约确认、偏离及限制 |
| `S1-integration-smoke-report.md` | S1 集成冒烟核验报告 | v1.0 | 2026-09-11 | 三产物集成门槛全绿：48/48、任务集复检、真实 MiniCPM Judge B/C 单轮冒烟；C 记忆 29、6 次检索命中、3 次翻转、准确率 7/29→8/29；含关键限制 |
| `SESSION-1-SUMMARY.md` | Session 1 收尾总结 | v1.0 | 2026-09-11 | P0 收尾：会话 1 交付产物清单、集成门槛全绿结果（48/48、C ⊇ B、翻转 1 改善 0 恶化）、7 项当前风险、Session 2 输入（Workstream A/B + P2 实验协议） |
| `ORCHESTRATOR-HANDOFF.md` | Orchestrator 交接文档 | v1.0 | 2026-09-11 | 当前全量上下文快照：项目边界、D1–D15、Session 1 产物与集成数据、海报隔离边界、Session 2/3 待办及关键风险；下一 orchestrator 必读 |
| `POSTER-A3-copy-sheet.md` | 裸眼3D 海报 · A3 文案定稿与验收记录 | v1.5 | 2026-09-11 | 交付物 #3 定稿记录：终稿文案（hero `PD-JitRL` / headline `越用越牛批` / 无 eyebrow_cn / 空注脚 / 远景 `ROUTER·114514·JitRL` / 插图 `高松灯.jpg`）、主视物字号实测自适应（681px，与字符数解耦）、九项实测验收（印刷参数 / 安全边 9.9~19.1mm / 字号 163.4~11.0pt / 墨迹分层 / 颜文字字形 / 插图版位与 300DPI / 色彩实测 / CMYK 色差 / **远景模糊代价**）、两处有意偏离设计系统（纵深比下限 0.6→0.35、远景放开零模糊禁令）、交付位置与自足性验证、待确认项（slogan 声明 / 插图破 4 色 / hero 与远景贴压） |

## 外部资产（不在本目录，已登记引用）

| 资产 | 路径 | 用途 |
|---|---|---|
| 裸眼3D 海报 design system（模板 v2） | `D:\GithubPlay\Skills\naked-eye-3d-design-system`（TEMPLATE + EXAMPLE 两文件） | 交付物 #3 海报：会话 1 末简报对齐其定制区字段，生成走"复制模板 → 只改定制区 → lint → AI 工具 → grep 自检"流程（工作流详见 project-decisions.md D6 附注） |

## 版本历史

| 文档 | 版本链 |
|---|---|
| `technical-decisions.md` | v1.0 → v1.1（D6–D13 落盘并引用）→ v1.2（22h 简化注记）→ **v1.3**（Judge 换型 MiniCPM5-1B；引用路径改为同目录） |
| `project-decisions.md` | v1.0 → v2.0（22h）→ v2.1（Judge/执行映射/D15）→ v2.2 → v2.3（价格）→ v2.4 → v2.5（S1）→ v2.6 → v2.7 → v2.8（全量实验）→ **v2.9**（Web Demo并入Session 3并重排） |
| `SESSION-2-PROGRESS.md` | v1.0 → v1.1（真实校准）→ v1.2（A）→ v1.3（早期B）→ v1.4（C）→ v1.5（B同序）→ v1.6（C1实现）→ v1.7（C1运行）→ **v1.8**（C1分析收尾） |
| `results.md` | **v1.0**（正式全量实验结果首版） |
| `S2-C1-PREREGISTRATION.md` | **v1.0**（C1 min-neighbor gate 运行前预注册） |
| `S2-C1-RESULTS.md` | **v1.0**（C1实验结果首版） |
| `SESSION-3-ORCHESTRATOR-HANDOFF.md` | **v1.0**（Session 3首次完整交接） |
| `S3-TAB8-PREREGISTRATION.md` | **v1.0**（运行前冻结 T8-LM vs T8-PM 的 C0 prefix replay、公平性、防泄漏与判读规则） |
| `S3-TAB8-RESULTS.md` | **v1.0**（T8正式结果：质量持平，PM成本更低但0次memory influence；LM的T19有害flip主导成本差） |
| `S3-DEMO-VIDEO-SCRIPT.md` | **v1.0**（11 分镜脚本 + 配音逐字稿；成片与产线说明） |
| `SESSION-3-PROGRESS.md` | v1.0 → v1.1（设计图）→ v1.2（README+Demo契约）→ v1.3（Web Demo 完成并记录并行协调裁决）→ v1.4（演示视频完成）→ v1.5（终检发布完成，Session 3 收官）→ v1.6（provider 表述修正 + 集成 MVP 登记）→ **v1.7**（按用户指示精简 README 的 CPA 解释） |
| `SESSION-3-PROGRESS.md` | v1.0（T8消融全记录）→ v1.1（技术设计图完成）→ **v1.2**（README完成，锁定Demo契约） |
| `S4-INTEGRATION-REPORT.md` | **v1.0**（D13 future work 第一阶段：PilotDeck 集成 MVP 交付全记录） |
| `Price of TestModel.md` | **v1.0**（项目所有者提供，头部补版本标注） |
| `S1-logprob-probe-report.md` | **v1.0**（Session1-T2 首次探针报告） |
| `TASK-taskset-report.md` | **v1.0**（Session1-T3 首次执行报告落盘） |
| `S1-core-harness-report.md` | **v1.0**（Session1-T4/T5 交付报告首版；集成修复后当前测试数 48） |
| `S1-integration-smoke-report.md` | **v1.0**（Session 1 集成验收首版） |
| `SESSION-1-SUMMARY.md` | **v1.0**（P0 收尾首版：产物 / 门槛结果 / 风险 / Session 2 输入） |
| `ORCHESTRATOR-HANDOFF.md` | **v1.0**（跨 orchestrator 首次完整交接快照） |
| `POSTER-A3-copy-sheet.md` | v1.0（交付物 #3：AI 侧完成版式与印刷链路，文案字段交由项目所有者填写）→ v1.1（文案定稿回填 + eyebrow_cn 移除 + 纵深比双向钳制不变量 + 五项实测验收）→ v1.2（远景 `JitRL`/`114514` 对调版位，四项复测无变化）→ v1.3（接入插图层 + 修复 36px 坐标系偏移 + 色彩校验改为实测报数 + CMYK 色差实测）→ v1.4（交付就位 `Pdwork\Poster\`，自足性验证通过）→ **v1.5**（主视物字号改实测自适应 681px；放开远景零模糊禁令 atmoBlur:20 并记账代价；纵深比下限 0.6→0.35） |

## 约定

- 文件名保持稳定（不带版本后缀）；历次版本见各文档 Changelog；
- 公开仓库（交付物 #1）在会话 3 包装阶段从本目录复制终版入库，此前不维护仓库副本；
- 后续文档（`results.md` 实验报告、海报内容简报等）产生后即时登记至本清单。

## 项目状态速览（2026-09-12）

- **决策基线**：D1–D15 全部冻结（technical v1.3 + project v2.9）；
- **Session 1**：已收尾——核心交付与集成冒烟全绿（48/48；任务集 29 条；真实 Judge B/C 全通；任务级 C ⊇ B），checklist 除海报外全部勾选，收尾摘要见 `SESSION-1-SUMMARY.md` v1.0；
- **代码卫生**：仓库根 `nul` 误产物已删除；`.gitignore` 已补 `__pycache__/`、`*.py[cod]`、`.tmp/`；
- **海报**：由独立 session 负责，外部 session 处理中，不作为阻塞；
- **Session 2**：正式 A/B/C 与 C1 后续实验均完成；C1 首次24/24成功并独立分析，完整测试196通过；详见 `SESSION-2-PROGRESS.md` v1.8、`results.md` v1.0、`S2-C1-RESULTS.md` v1.0；
- **验收结论**：C0 D10为3/4，成本节省−3.78%；C1预注册P1–P5为3/5，成本和flip无害性失败，信息性成本节省−16.64%；
- **Session 3 固定顺序**：消融实验 → 技术设计图 → README → Web Demo → 演示视频 → 仓库终检/发布；Web Demo 已从 Session 2 合并，并必须囊括 A/B/C0/C1/消融/probes/故障审计；
- **Session 3 当前进度**：**六项检查点全部完成**（`SESSION-3-PROGRESS.md` v1.7）——消融、技术设计图、README、Web Demo、演示视频、仓库终检与发布；commit `d419ab7`（104 文件）已推送；后续追加检查点 7（provider 表述修正 + README 精简，`JITRL_PROVIDER_ID` 覆盖，296 tests）与检查点 8（集成 MVP 登记）；现场彩排清单就绪；
- **Session 4（2026-09-12）**：**D13 future work 第一阶段完成**——PilotDeck 集成 MVP（TS 原生移植 + builtin jitrl 插件 + onTurnOutcome 学习闭环 + 记忆持久化）落地并推送 GitHub（commit `ce470e3` 集成交付物、`7b10fd6` README §9 改写）；tsc 通过、59/59 集成测试、Python 261 不变、TS↔Python 签名差分 21/21；在线学习效果对照评估未做（须预注册），D10/C1/T8 冻结结论不变；正本见 `S4-INTEGRATION-REPORT.md` v1.0；
- **交接入口**：下一 orchestrator 先完整阅读 `SESSION-3-ORCHESTRATOR-HANDOFF.md` v1.0 与 `S4-INTEGRATION-REPORT.md` v1.0。

## Changelog

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-09-11 | v1.1 | 登记 `S3-TAB8-PREREGISTRATION.md` v1.0；更新 Session 3 状态，明确消融已预注册并进入实现阶段 |
| 2026-09-11 | v1.2 | 登记 `S3-TAB8-RESULTS.md` v1.0；同步 T8 正式结果、261项测试与 Session 3 转入技术设计图阶段 |
| 2026-09-11 | v1.3 | 登记 `SESSION-3-PROGRESS.md`（v1.0→v1.1）：技术设计图完成，4张SVG入库仓库 `docs/diagrams/`，进入 README 阶段 |
| 2026-09-11 | v1.4 | `SESSION-3-PROGRESS.md` 升 v1.2：README 完成并锁定 Demo 契约；进入 Web Demo 阶段 |
| 2026-09-11 | v1.5 | `SESSION-3-PROGRESS.md` 升 v1.3：Web Demo 完成（32+261 tests、11/11 冒烟）；登记并行线路协调裁决；进入演示视频阶段 |
| 2026-09-11 | v1.6 | 登记 `S3-DEMO-VIDEO-SCRIPT.md` v1.0；`SESSION-3-PROGRESS.md` 升 v1.4：视频完成（demo/demo-video.mp4 2:00）；Session 3 进入最后一项仓库终检与发布 |
| 2026-09-11 | v1.7 | `SESSION-3-PROGRESS.md` 升 v1.5：终检发布完成（清理/正本入库/LICENSE/secret扫描/293 tests/commit d419ab7 推送成功/彩排清单）；**Session 3 收官** |
| 2026-09-12 | v1.8 | 登记 Session 4 集成线路：`S4-INTEGRATION-REPORT.md` v1.0（PilotDeck 集成 MVP，commit `ce470e3`/`7b10fd6`）；状态速览新增 Session 4 条目 |
| 2026-09-12 | v1.9 | `SESSION-3-PROGRESS.md` 升 v1.6：**provider 表述修正**（README/demo 全仓 15+ 处「CPA provider」误导表述改为通用 provider 描述；`harness/cpa_client.py` 新增 `JITRL_PROVIDER_ID` 覆盖，默认 CPA 向后兼容；新增 3 测试 → 核心 264 + demo 32 = 296 passed；冻结模型 ID 不改写）；登记外部集成 MVP 检查点 8 |
| 2026-09-12 | v2.0 | 按用户指示精简 README：删除 CPA 前缀解释块与冻结 ID 注记（避免过度解释），仅保留一句通用 provider 描述；`SESSION-3-PROGRESS.md` 升 v1.7 |
| 2026-09-12 | v1.8 | 登记 `S4-INTEGRATION-REPORT.md` v1.0：Session 4 完成 D13 future work 第一阶段（PilotDeck 集成 MVP，commit ce470e3 + 7b10fd6 已推送）；速览与交接入口同步更新 |
