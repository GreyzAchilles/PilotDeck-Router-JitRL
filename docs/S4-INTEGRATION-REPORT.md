# S4 · PilotDeck 集成 MVP 交付报告

> 状态：v1.0（2026-09-12）
> 范围：Session 3 收官（commit `d419ab7` 推送）之后的追加会话——首次执行 D13 future work：在 PilotDeck 官方扩展点上落地可运行的 JitRL 自定义路由器 MVP。
> 边界：Python 研究代码（`jitrl_core/`、`local_judge/`、`harness/`、`eval/`、`demo/`）零改动；D10（3/4）、C1（3/5）、T8 的冻结结论不变；本报告不构成新的实验结论。

---

## 0. 一句话结论

D13 标记为 future work 的 PilotDeck 深度集成，第一阶段**可运行 MVP 已完成**：`jitrl_core` 核心语义 TypeScript 原生移植，经官方 `PilotDeckCustomRouter` / `RouterContribution` 扩展点以内置 `jitrl` 插件接入（**零 Router 内核决策链改动**）；两个上游缺口各给出落地通道（缺口① 打分制 Judge 保守等价、缺口② `onTurnOutcome` 回调）；学习闭环（reward = 0.6·quality + 0.3·costSaving）与记忆持久化打通。交付物以 patch + overlay 双形式落盘并推送 GitHub（commit `ce470e3`、`7b10fd6`）；验证 tsc 通过、node:test **59/59**、Python **261** 不受影响、TS↔Python 任务签名差分 **21/21 一致**。

## 1. 背景与任务范围

### 1.1 决策回溯

- **D13**（project-decisions v2.9）：深度集成与 upstream PR → future work，22h 预算内不做；替代交付为 README「集成路径」专节 + CustomRouter 扩展点论证；
- **D14**：独立 Python harness 忠实复刻 Router 决策路径——本会话不推翻该架构，MVP 是其之上的**增量**（研究代码不动，集成代码以补丁形态独立交付）；
- 本会话即 D13 future work 的首次执行，属冻结基线之外的合法新工作，不回写任何冻结判定。

### 1.2 需求确认（会话内问答，项目所有者裁定）

| 问题 | 选择 |
|---|---|
| 第一阶段做到哪一层 | **可运行 MVP**（优先不改 Router 内核） |
| JitRL 核心接入形态 | **TypeScript 原生移植**（非 Python sidecar / 混合模式） |
| reward/outcome 来源 | **成本 + 模型评审**（复用 harness 的 reward 口径） |
| 提交目标 | **外层仓库**（patch/overlay 形式随 `GreyzAchilles/PilotDeck-Router-JitRL` 分发，不直接推 OpenBMB） |

### 1.3 参考基线

- 官方架构文档：PilotDeck 模块开发指南（Router 模块结构、`PilotDeckCustomRouter` 接口、Canonical Protocol）；
- 上游源码：`PilotDeck-src/`（OpenBMB/PilotDeck，tag `v2026.09.10`，commit `cfc4d1779228f91fececc5d6705c14dab5b7ef2f`，部分检出快照）；
- 本仓参考实现：`jitrl_core`（算法语义正本）、`local_judge`（Method B）、`harness/run_real.py`（reward 口径 0.6/0.3、失败不学习纪律）。

## 2. 接入设计

### 2.1 官方扩展点接入（零内核决策链改动）

- builtin 插件 `src/extension/plugins/builtin/jitrl/`（`plugin.json` + 程序化 `RouterContribution`，id = `jitrl`）；
- 用户配置 `router.customRouter.extensionId: jitrl` 即接管路由：`RouterRuntime.resolveCustom()` 命中 custom 时优先于 scenario/tokenSaver，`decide()` 返回 `{provider, model, tokenSaverTier}`；
- 交付形态兑现 D13 叙事升级：从「设计上的扩展点分析」升为「已实现、可 `git apply` 的补丁」。

### 2.2 缺口①（logprob 透传）→ 打分制保守等价

- Canonical 协议响应无 logprobs（上游全文检索确认），Method B 单请求路径无法在插件内复现；
- MVP 采用**4 次并行每档打分请求**：judge 模型对 simple/medium/complex/reasoning 各评 0–10 拟合分，`scoreToLogit = logit((round(s)+0.5)/11)`（p 钳至 [0.02, 0.98]）；
- 任一档失败/未配置 judge/解析失败 → **整体**退化为确定性启发式 Judge（`mock_judge` 的 TS 移植：medium 偏置 +1、代码线索 +8→complex、文档线索 +8→simple、长消息 +8→reasoning、continuation 回声 previousTier +6、种子噪声 ±0.05），保证始终返回完整四档数值（不混用两种量纲）；
- judge/evaluator 均直连注入的 `ModelRuntime.complete()`，**不经过 Router，无递归路由**。

### 2.3 缺口②（outcome 回调）→ 可选 `onTurnOutcome` 钩子

- `PilotDeckCustomRouter` 新增可选 `onTurnOutcome?(outcome)`；`RouterRuntime.execute()` 在 custom 路由的**成功、最终失败、媒体能力不符错误**三条路径回传：sessionId/turnId/projectPath/userMessage/decision/usage/response 文本/error + 依赖注入 context；
- 钩子异常仅告警，绝不破坏主流；`resolvedFrom !== "custom"` 不触发；
- 这是对上游协议的最小增量（可选字段，存量实现零影响）。

### 2.4 学习闭环与纪律（对齐 harness 口径）

- reward = `0.6·quality + 0.3·costSaving`：
  - quality：配置的 evaluator 模型（缺省回落 judge）评审 (userMessage, response)，0–10 分归一化 [0,1]，超时/解析失败 → **不写记忆**；
  - costSaving：`lookupModelPricing` 计算候选集中各档 episode 价（usage 缺省按 1500/800 token 假设），对最贵候选归一化并 clamp [0,1]；
- **执行失败不写记忆**（无可靠质量信号）；两者与 `run_real.py` 的失败策略一致；
- episode 以 `jitrl:{sessionId}:{turnId}` 为 id 写回记忆。

### 2.5 记忆持久化与实例抖动

- 落盘 `<pilotHome>/router/jitrl-memory.json`（`{version:1, entries:[…]}`，entries 为 `(intent_class, signature_tokens, tier, G, ts, episode_id)` 标准条目）；
- 原子写（tmp + rename）+ 500ms 节流合并突发写；容量 5000 drop-oldest；
- **关键问题与解法**：`PluginRuntime.lookupRouter()` 每次 decide 都 `createCustomRouter()` 新实例——学习状态必须进程级共享。按 memory path 建共享态缓存（engine + memory + store + seed RNG）；超参变更时重建 engine 但保留 episodes（已文档化的 RNG 流重置偏差）。

### 2.6 配置扩展（兼容旧格式）

`router.customRouter.*` 新增可选字段：`judge` / `evaluator`（provider/model ref，须已存在于模型配置）/ `tiers`（档位→候选模型，缺省回落 tokenSaver tiers）/ `hyperparams`（k、beta、lam、alpha、jaccardThreshold、zMin、memoryCap、seed、minNeighbors，缺省即 Python 冻结超参）/ `memoryPath` / `judgeTimeoutMs` / `evalTimeoutMs`。旧的仅 `{ extensionId }` 配置解析保持有效（回归测试锁定）。

### 2.7 配套上游修复（集成发现的既有 bug）

`PluginRuntime.refreshWithReport()` 磁盘重载 builtin 插件时会丢弃程序化 `RouterContribution`（函数无法序列化进 plugin.json），导致插件刷新后自定义路由静默失效——已修复为重载后 re-attach；不影响既有 builtin。

## 3. 交付物与 GitHub 提交

### 3.1 集成交付物（commit `ce470e3`，30 文件 +6233 行）

```text
integrations/pilotdeck-jitrl/
├── INTEGRATION.md          # 集成说明（基线/行为/配置/应用/验证/限制）
├── pilotdeck-jitrl.patch   # 对上游 v2026.09.10（cfc4d17）的完整补丁（git apply 即用）
├── verify.py               # overlay↔本地检出逐文件 sha256 校验 + git apply --check
└── overlay/                # 26 个结果文件（src 19 + tests 7），按上游相对路径归档
```

- patch：129469 bytes，SHA-256 `139aa4fe4dfd9a06e3dab9b3045679cde5dcf0a00e39285e1ccbd974e3c5a720`；
- 上游侧变更：修改 7 文件（`customRouter.ts`、`RouterRuntime.ts`、`schema.ts`、`parseRouterConfig.ts`、`router/index.ts`、`loadBuiltinPlugins.ts`、`PluginRuntime.ts`）+ 新增 `src/router/jitrl/` 10 模块 + builtin `jitrl` 插件 2 文件 + `tests/router/jitrl/` 7 个测试文件；
- 选择外层仓库分发的原因：内层 `PilotDeck-src` 远端为官方 `OpenBMB/PilotDeck`（无推送权限/未走 PR 流程），项目所有者裁定以 patch + overlay 随外层仓库发布。

### 3.2 README 更新（commit `7b10fd6`，+60/−7）

仓库 `README.md` §9 从「PilotDeck 集成（future work）」改写为「**PilotDeck 集成（MVP 已完成）**」：9.1 旧版分析兑现表（CustomRouter/缺口①/缺口②三行对照）、9.2 实现要点 + YAML 配置示例、9.3 应用/校验/验证结果（含 patch SHA-256）、9.4 七条已知限制；目录锚点与 §3 仓库结构树同步更新。修改前独立复验 §9 事实声明（签名差分 21/21 复跑确认）。

### 3.3 文档副本状态

本报告与 INDEX v1.8 正本在 Pdwork docs；GitHub 仓库 `docs/` 副本（Session 3 包装时复制的 12 份）**暂未同步**本报告——按 D15 约定正本为准，下次包装/发布动作时复制入库。

## 4. 验证结果

| 项 | 命令 | 结果 |
|---|---|---|
| TypeScript 类型检查 | `npx tsc -p tsconfig.json`（PilotDeck-src） | **PASS** |
| 集成单测 | `node --test dist/tests/router/jitrl/*.test.js` | **59 passed / 0 failed** |
| Python 侧回归 | `python -m pytest harness jitrl_core local_judge eval -q` | **261 passed**（研究代码零改动） |
| TS↔Python 签名差分 | `.tmp/diff-state.mjs`（21 条消息逐 token 对照） | **ALL 21 MESSAGES MATCH** |
| 交付物一致性 | `verify.py PilotDeck-src`（26 文件 sha256） | **PASS** |
| patch 干净度 | `git diff --check` | PASS（仅 CRLF 提示） |

node:test 覆盖面：核心算法等价（V/Q/Â 手算值、tie-break、z_min clamp、门控 RNG 流不变量）、Judge 四请求/退化路径、reward 合成与 evaluator 失败不学习、执行失败不学习、记忆原子持久化/节流/容量/共享态、配置解析 + 旧格式兼容、插件注册与刷新保留、RouterRuntime outcome 全路径（fake registry 端到端）。

## 5. 已知限制与后续工作

1. **RNG 不逐位一致**：TS 默认 seeded mulberry32 vs Python MT19937——语义等价但探索分支随机序列不同（测试注入 rng 锁数值）；
2. **判档成本**：4 次并行打分请求（每档 `judgeTimeoutMs` 上限）+ 每成功轮 1 次 evaluator 调用，延迟/费用高于单请求路径；
3. **执行失败不学习**：无可靠质量信号（有意为之的保守纪律）；
4. **上游快照缺口**：部分检出缺 `scripts/check-node-runtime.mjs`，标准 `npm test` 停在既有 prebuild；已直接运行等价 build + test 主体（非集成引入的问题）；
5. **上游 `.gitignore` 忽略 `*.test.ts`**：集成测试随外层仓库以 overlay/patch 分发，不受影响；若进上游需 `git add -f`；
6. **在线学习效果未做对照评估**：§5 冻结结论（A/B/C0/C1/T8）均来自 Python harness；真实 PilotDeck 部署下的学习效果**须按预注册纪律（对齐 S2-C1/S3-TAB8 模式）另立实验**，本 MVP 只交付工程管线；
7. **后续**：canonical 协议暴露 logprobs 后，打分式 Judge 可切回真正 Method B 单请求路径；memory 落盘可加压缩与并发锁（当前单进程假设）。

## 6. 与既有冻结口径的关系

- D13「低侵入性论证」叙事升级为「已实现可应用补丁」；D14 独立 harness 架构不变（Python 侧零改动）；
- D10（3/4）、C1 P1–P5（3/5）、T8（LM/PM）冻结判定**不受本会话任何影响**，不回写、不覆盖；
- 记忆隐私（D12）边界延续：落盘记忆仅含签名 + tier + G + ts + id，不含用户原文（userMessage 仅在内存态用于签名计算）；
- 预注册纪律延续：集成效果的任何实验声明都必须先冻结判据再运行（见 §5 第 6 条）。

## Changelog

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-09-12 | v1.0 | 首版：D13 future work 第一阶段（PilotDeck 集成 MVP）交付全记录——设计、实现清单、GitHub 提交（ce470e3 / 7b10fd6）、验证结果、限制与后续工作 |
