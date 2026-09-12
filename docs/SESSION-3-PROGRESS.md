# JitRL Router — Session 3 进度

> 状态：v1.7（2026-09-11，含 provider 表述修正与集成 MVP 登记）
> 范围：Session 3 固定顺序 = 消融 → 技术设计图 → README → Web Demo → 演示视频 → 仓库终检/发布。
> 边界：海报由外部 session 负责；`poster-a3/` 不读取不修改。文档正本在 Pdwork docs，修改必须升版并同步 INDEX。

---

## 检查点 0：接续核验（已完成）

- 阅读交接文档与前置（project-decisions v2.9、results v1.0、C1 prereg/results v1.0）；
- 只读测试基线：`python -m pytest harness jitrl_core local_judge eval -q` → **196 passed**，与交接一致；
- 并行委派两个只读 subagent：消融协议设计 + prompt/memory/API 契约审计，父 agent 合并后落预注册。

## 检查点 1：Table-8 式消融（已完成）

### 1.1 预注册（运行前冻结）

- 正本：`S3-TAB8-PREREGISTRATION.md` v1.0；
- 两组：T8-LM（logit modulation）vs T8-PM（prompt memory injection）；
- 关键设计：**正式 C0 memory 的 prefix replay**（第 i 任务只读 `source_order < i`），排除自身/未来 outcome 泄漏；两组同 prefix、同检索、只读不写回；
- prompt 注入白名单（tier/G/similarity/count/per-tier 聚合），禁 GT/checklist/Evaluator 结构/原文/secret；
- 与 C0/C1 完全隔离：新命名 `s3_ablation_*`、`eval/ablation/`，不得覆盖冻结资产。

### 1.2 实现

- `eval/ablation/frozen-c0-memory-trace.jsonl` + `.sha256.json`（copy sha `817e5d8a…`，source sha `7f4e2d82…`）；
- `harness/run_ablation.py`：T8-LM 用未改动 `engine.decide()`（bit-for-bit 复现 C0 数学，测试验证 24/24）；T8-PM 注入白名单块、不调制、另做 raw prompt 审计调用；latency 四栏分离；
- `eval/analyze_ablation.py`：顺序/配置/hash/24-of-24/机制纯度/数学/prompt 字节完整性/组间公平性硬校验；输出仅写 `eval/ablation/`；
- `local_judge/client.py` 最小增量：`build_prompt(..., memory_block=None)`，None 时字节不变（测试锁定）；
- 测试：新增 65 项，全套 **261 passed**。

### 1.3 正式运行（均为首次尝试 24/24，无 retry）

- 冒烟：T06 各 1 条（不入正式）；
- T8-LM：`logs/s3_ablation_T8_LM.jsonl`（exec $0.032653 + eval $0.170352）；
- T8-PM：`logs/s3_ablation_T8_PM.jsonl`（exec $0.027993 + eval $0.167228）；
- 消融总支出约 $0.60（含冒烟），低于预注册 $1.00 上限。

### 1.4 结果与结论（诚实口径）

- 正本：`S3-TAB8-RESULTS.md` v1.0；聚合 `eval/ablation/results-summary.json`、`results-tasks.csv`、4 张 SVG；
- 质量：LM=PM=2.5417；执行成本 PM 低 $0.004660；
- 结论规则判定：**方向性偏向 prompt injection**——但机理上 PM 的 6 次注入 0 次改变 tier，成本差约 90% 来自 LM 在 T19 的一次 simple→complex 有害 flip（+$0.004194，质量 1→1）；
- H1 PASS、H2 PASS、H3 方向偏 PM、H4 PASS（LM max 0.4434ms）、H5 完成审计（base choice mismatch 0/24，logit 存在数值漂移）；
- C0 D10 3/4 与 C1 P1–P5 3/5 的冻结判定不变，T8 不回写。

## 检查点 2：技术设计图（已完成）

- 生成脚本入库：`docs/diagrams/make_diagrams.py`（stdlib only，可复现，防文字溢出自检）；
- 四张 SVG（README 与 Demo 共用，不依赖海报资产）：
  - `docs/diagrams/architecture-decision-path.svg` — 在线决策+学习闭环（User→Judge→retrieve→V/Q/Â→z′→tier→执行→Evaluator→memory，延迟归因标注）；
  - `docs/diagrams/arms-comparison.svg` — A/B/C0/C1/T8-LM/T8-PM 六组机制差异矩阵+关键结果；
  - `docs/diagrams/online-offline-paths.svg` — 离线冻结数据面板（默认）与在线实时决策双路径、降级与隐私边界；
  - `docs/diagrams/pilotdeck-integration.svg` — 上游集成点：CustomRouter 接管、①logprob 透传、②outcome 回调（D13 future work）；
- 全部通过 XML 解析与文字宽度校验，无溢出警告。

## 检查点 3：README（已完成）

- 仓库 `README.md` 落盘：动机与 JitRL→Router 映射、四图嵌入、结构、无 secret 配置说明、全套复现命令（与实际 CLI 逐一核对）、六组结果汇总表、失败分析四条（C0 有害 flips / C1 级联 / T8 双弱 / B 偏置封顶）、T8 消融、九条已知限制、Demo 在线/离线契约（`python -m demo.server [--online]`，:8300）、PilotDeck 集成 future work、正式/审计资产索引、预注册纪律、MIT 计划；
- 自检：数字与冻结文档一致（修正一处 C1 准确率笔误 29.83→29.17）；路径引用检查仅缺 `docs/S3-TAB8-PREREGISTRATION.md`（按计划包装阶段随正本入库）；
- README 定义了 Demo 契约（命令/端口/模式），Demo 实现必须遵循。

## 检查点 4：Web Demo（已完成）

- 契约落地：`python -m demo.server`（:8300 默认，`--port/--host/--online/--max-session-usd`），**离线为默认模式**；README §8 已补运行命令，与早期 README 草稿锁定的契约一致；
- `demo/server.py`（stdlib `http.server`，零依赖）：`/api/data` 逐字读取冻结聚合（main/c1/ablation/continuation/tasks_meta/sources，前端不重算口径）、`/api/tasks`（24+5）、`/api/charts/*` 白名单 SVG 服务（含 diagrams/，路径穿越 404）、`/api/health`、在线 `POST /api/decide`（A/B/C0/C1/PM 五臂）、`POST /api/reset`；
- 离线数据面板：TL;DR 六组表、D10 3/4 与 P1–P5 3/5 判据表（PASS/FAIL 着色）、C0 T06/T19 有害 flip 卡、C1 门控与 T20 级联卡、T19 唯一分歧与 PM 0/6 influence 强制 caveat、continuation 0/5、延迟归因（Judge ≈1–3s vs JitRL 0.2–0.7ms）、截断统计、故障/重试审计叙述、16 张既有 SVG 图表嵌入；
- 在线决策面板（清晰 ONLINE 标记）：预设任务 + 自由输入、z_base 与 z_mod/z_prompt 对比条、检索记忆/V/Q/Â/gate_reason、tier+执行模型徽章、回复预览与单次成本、真实 Evaluator 盲评→reward→会话记忆写回（C 臂）、PM 臂注入演示（会话 C0 记忆为源，不调制不写回）、会话面板（记忆数/累计成本/迷你曲线/重置）、预算上限默认 $1.00；
- 可靠性与隐私：会话仅内存、请求体不落日志、key 只在后端进程、离线/失败返回结构化 JSON + 前端横幅降级不白屏、路径穿越防护；
- 验证：`python -m pytest demo -q` → **32 passed**（fake 注入，无需服务）；核心套件 **261 passed** 不变；orchestrator 进程内冒烟 `demo/smoke_check.py` **11/11 ALL PASS**（含三种穿越攻击 404 与离线 503 结构化降级）。

### 并行工作流协调记录（重要）

Session 3 存在两条并行 orchestrator 线路先后落盘，产物有重叠，裁决如下：

1. **技术设计图双份**：并行线路产出 `docs/diagrams/`（4 SVG + 脚本），本线路产出根目录 `diagrams/`（4 SVG + 脚本 + README）。**正本裁决定为根 `diagrams/`**——当前 README 引用口径、内容经本 orchestrator 逐项验证（XML 良构 + 事实核对）；`docs/diagrams/` 标记为待清理重复集，**仓库终检阶段删除**（届时 `docs/` 由本线路按 D15 复制文档正本接管）；
2. **README 双版**：并行线路早期 README（六组表/:8300 契约/MIT 计划）已被本线路验证版覆盖（命令逐条比对 argparse、数字全部冻结口径）；其有价值要素已回收：:8300 契约由 Demo 实现并回写 README §8，MIT 由发布阶段 LICENSE 落地；
3. 本文档即并行线路的 SESSION-3-PROGRESS 正本，后续仅由当前活跃 orchestrator 单线升版，避免再次分叉。

## 检查点 5：演示视频（已完成）

- 脚本正本：`S3-DEMO-VIDEO-SCRIPT.md` v1.0（11 分镜 ≈2:00，旁白逐字稿 + 烧录字幕要点，数字全部冻结口径）；
- 成片：`demo/demo-video.mp4`（2:00.00，1600×900，H.264 yuv420p，~2.2MB，静音 + 卡片字幕；含 4 段**真实 Demo 截图**——离线首屏/长屏数据段/#live 在线面板离线降级态——与 2 张技术图实拍、5 张叙事卡）；
- 产线：`demo/make_video.py`（无头 Edge 截图真实 UI + 程序化卡片 + ffmpeg 合成；截图时服务端日志确认全部聚合/图表请求 200，画面含真实数据；断网可录——全程离线模式）；
- 辅助改动：`demo/static/app.js` 增加 `#live` hash 深链（无头截取与分享链接两用）；`demo/smoke_check.py` 进程内 11 项冒烟；
- 配音可后补：字幕已覆盖全部信息；任何分镜可替换为实时操作录屏（离线段断网可录）。

## 检查点 6：仓库终检与发布（已完成）

- **清理**：`docs/diagrams/` 重复集已删除（按协调裁决）；`nul` 误产物已删除；`__pycache__/`、`.pytest_cache/`、`demo/.video/`、`logs/`、`agent-tasks/`、`reports/`、`poster-a3/`（不修改内容、仅排除出库）全部 gitignore；
- **文档正本入库**：`docs/` 复制 12 份正本（technical/project-decisions、Price、results、S2-C1 与 S3-TAB8 的 prereg+results、视频脚本、三个 SESSION 进度/总结）；内部交接文档与海报 copy-sheet 不入库；
- **LICENSE**：MIT（GreyzAchilles, 2026）；
- **secret 扫描**：模式扫描（sk-/api_key=/Bearer）零命中；`pilotdeck.yaml` 相关仅存在路径/配置说明引用；
- **测试**：`python -m pytest harness jitrl_core local_judge eval demo -q` → **293 passed**（261 核心 + 32 demo）；
- **首次提交并推送**：commit `d419ab7`，104 文件，分支 `main` → `https://github.com/GreyzAchilles/PilotDeck-Router-JitRL` 推送成功（首次超时为凭据交互，非交互重试通过）；
- **彩排与离线降级验证**：`python demo/smoke_check.py` 11/11 ALL PASS（离线面板/图表/路径穿越防护/503 结构化降级）；视频产线全程离线模式录制成功，断网演示可行。

### 现场彩排清单（游园会）

1. `python -m demo.server`（默认离线，:8300）→ 打开 http://127.0.0.1:8300，走一遍数据面板（TL;DR → 判据 → flips/级联/T8 → 0/5 → 延迟归因）；
2. 切到「实时决策」tab 展示离线降级横幅（断网态预期行为）；
3. 有网络 + llama.cpp 时：`python -m demo.server --online`，选 C0 跑 1–2 条预设任务展示完整 trace，再演示 PM 臂与 reset；
4. 备播：`demo/demo-video.mp4`（2:00，全程可静音播放）。

## 检查点 7：provider 表述修正（已完成，用户反馈驱动）

**问题**：README §4.2（快速开始·执行/评估模型）把 `--config` 中的 provider 凭据写成"CPA provider 凭据"，暗示 CPA 是必需/外部的 provider 类型。事实：`CPA` 只是作者设备上一个暴露本地端口的中转站（OpenAI 兼容代理）在 `pilotdeck.yaml` 里的 provider 条目名，**不是云端厂商，也不是必需的 provider 名**；模型 ID `PROVIDER/model` 的前缀只是该 provider 名。

**修正范围（描述 + 最小代码增强）**：

1. **README §4.2**：改为通用描述——运行时读取配置中 provider 条目的 `url` + `apiKey`，按 OpenAI 兼容协议调用（provider 名默认 `CPA`，可用 `JITRL_PROVIDER_ID` 覆盖）。**用户后续指示：不在 README 中过度解释 CPA**，故已删除初版的 `CPA/` 前缀说明块与"冻结原始 ID"注记，仅保留这一句通用描述；
2. **README 其它 5 处**：开篇"真实 CPA 模型调用"→"真实执行模型调用（经 PilotDeck provider 配置）"；模块表 `cpa_client.py` 描述；§4.3 命令注释；§8 Demo 运行注释；§9 配置示例注记（已回退为原简版）；
3. **demo/**：`README.md` 3 处（在线模式说明、隐私条款、故障排查项）、`server.py` 文档字符串 2 处、`index.html` 2 处可见文案（状态点 `CPA`→`provider`、执行开关改"真实模型调用"）、`app.js` 3 处横幅文案；
4. **代码增强**：`harness/cpa_client.py` 新增 `JITRL_PROVIDER_ID` 环境变量覆盖 provider 名（默认 `CPA`，完全向后兼容）；文档字符串澄清 CPA 是本机中转站；传输错误信息改用实际 provider 名；`load_cpa_provider` 增加 `provider_id` 参数；
5. **测试**：新增 3 项（env 覆盖、默认值、模型前缀与 provider 名无关）→ 核心 **264 passed** + demo 32 = **296 passed**；
6. **不改写冻结事实**：所有 A/B/C0/C1/T8 表格、聚合 JSON、日志与文档中的 `CPA/xxx` 模型 ID 均为实验原始记录，保持原样。

## 检查点 8（外部推进）：PilotDeck 集成 MVP（由用户/另一线路完成，已并入仓库）

仓库新增提交 `ce470e3`（feat: add PilotDeck JitRL integration）与 `7b10fd6`（README §9 更新）：

- 交付物 `integrations/pilotdeck-jitrl/`（patch + overlay + verify.py + INTEGRATION.md），基线上游 `v2026.09.10`（commit `cfc4d17`）；
- 走上游官方扩展点 `PilotDeckCustomRouter` / `RouterContribution`，`router.customRouter.extensionId: jitrl` 即接管路由，零内核改动；
- TS 原生移植 `jitrl_core`（10 模块），签名与 Python 参考实现逐 token 差分 21/21 一致；新增可选 `onTurnOutcome` 钩子回传 usage/响应/错误；
- 记忆持久化 `<pilotHome>/router/jitrl-memory.json`（原子写 + 节流 + 容量 5000）；
- 验证：tsc 通过、node:test 59 passed、根仓库 Python 侧不受影响；
- 已知限制：TS RNG 为 seeded mulberry32（与 Python MT19937 探索分支不逐位一致）；在线学习效果尚未做对照评估（须按预注册纪律另立实验）。

> 注：本检查点由外部线路落地，本次读取核对后登记；JitRL 决策/数学与 Python harness 冻结实验结论不变。

## 下一步

Session 3 六项检查点 + provider 表述修正均已完成；集成 MVP 已由外部线路并入。项目出口达成：公开仓库 + Demo + 视频 + README + 技术图 + 消融 + 集成 MVP（海报由外部 session 负责）。

---

## 风险跟踪（Session 3）

- 消融公平性：已由 prefix replay + 只读记忆 + 分析器硬校验解决；
- Judge 数值漂移：T8 内 base choice 0/24 不一致，但 logit 有漂移，报告已单列；
- Demo 范围膨胀：已按 offline 优先收敛交付（数据面板零网络可用，在线面板后置且降级完备）；
- 并行线路冲突：图表/README 双份已裁决（根 `diagrams/` 为正，README 为验证版），详见「并行工作流协调记录」；
- 文档双源：正本仅在 Pdwork docs，包装阶段才复制入库。

## Changelog

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-09-11 | v1.0 | 首版：接续核验196基线、T8预注册/实现/正式运行/分析全记录（261 tests）、诚实结论、风险跟踪；进入技术设计图阶段 |
| 2026-09-11 | v1.1 | 技术设计图完成：make_diagrams.py + 4 张 SVG（决策闭环/组别矩阵/在线离线/集成点）入库 docs/diagrams/；进入 README 阶段 |
| 2026-09-11 | v1.2 | README 完成并锁定 Demo 契约（demo.server :8300 offline 默认）；数字自检通过；进入 Web Demo 阶段 |
| 2026-09-11 | v1.3 | Web Demo 完成（demo/ 全量，:8300 契约、离线面板/在线五臂/隐私降级，32+261 tests、11/11 冒烟）；记录并行工作流协调裁决（根 diagrams/ 为正、docs/diagrams/ 待终检清理、README 验证版回收 :8300/MIT 要素）；进入演示视频阶段 |
| 2026-09-11 | v1.4 | 演示视频完成：`S3-DEMO-VIDEO-SCRIPT.md` v1.0 + `demo/demo-video.mp4`（2:00.00，4 段真实 Demo 截图 + 2 图实拍 + 5 叙事卡，断网可录产线 `demo/make_video.py`）；`#live` 深链与 `demo/smoke_check.py` 辅助入库；进入仓库终检/发布 |
| 2026-09-11 | v1.5 | 仓库终检与发布完成：清理/正本入库12份/MIT LICENSE/secret扫描零命中/293 tests/commit d419ab7 104文件推送成功/彩排清单落盘；Session 3 六项检查点全部完成 |
| 2026-09-11 | v1.6 | 检查点 7（provider 表述修正，用户反馈驱动）：README §4.2 及全仓 15+ 处"CPA provider"类误导表述改为通用 provider 描述 + `CPA/` 前缀说明；`harness/cpa_client.py` 新增 `JITRL_PROVIDER_ID` 覆盖（默认 CPA，向后兼容）；新增 3 测试 → 核心 264 + demo 32 = 296 passed；冻结模型 ID 不改写。检查点 8：登记外部线路的 PilotDeck 集成 MVP（ce470e3/7b10fd6，integrations/ 交付物 + 59 node tests） |
| 2026-09-11 | v1.7 | 按用户后续指示精简 README：删除 `CPA/` 前缀解释块与冻结 ID 注记（避免对 CPA 过度解释），§4.2 仅保留一句通用描述（provider 条目 url+apiKey、`JITRL_PROVIDER_ID` 可覆盖）；§9 示例注记回退简版；demo/README 排查项同步精简；代码增强与测试不变 |
