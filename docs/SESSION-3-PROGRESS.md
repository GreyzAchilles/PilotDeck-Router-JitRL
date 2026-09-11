# JitRL Router — Session 3 进度

> 状态：v1.4（2026-09-11）
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

## 下一步

第 6 项仓库终检/发布：docs/diagrams/ 清理 → 文档正本入库 → LICENSE → secret 扫描 → 全量测试 → README 自检 → 首次 commit & push → 彩排清单。

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
