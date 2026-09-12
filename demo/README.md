# JitRL Router · Web Demo（Session 3）

离线优先的 Web Demo：实验数据面板断网可完整演示；在线模式提供实时决策 trace。
**纯标准库**（`http.server`，无 pip 依赖、无 FastAPI、无前端构建）。

## 运行

```bash
# 离线模式（默认，零网络）：只有本机文件读取，绝无出站请求
python -m demo.server
# → http://127.0.0.1:8300

# 在线模式：启用实时决策 API（本地 llama.cpp Judge + PilotDeck 配置中的 provider 凭据）
python -m demo.server --online

# 可选 flag
python -m demo.server --port 8300 --online --max-session-usd 1.00
```

| Flag | 默认 | 说明 |
|---|---|---|
| `--port` | 8300 | 监听端口 |
| `--host` | 127.0.0.1 | 监听地址 |
| `--online` | 关 | 开启实时决策 API；离线模式下 `POST /api/decide` 返回结构化 503 |
| `--max-session-usd` | 1.00 | 每会话**执行**累计花费上限（美元）；超限返回结构化 429 |

测试：`python -m pytest demo -q`（无需任何服务，注入 fake judge/exec/evaluator）。

## 两种模式

### 1. 实验数据面板（离线，默认可用）

`GET /api/data` 把冻结聚合**原样**读出（绝不重算指标）：
`eval/results-summary.json`（A/B/C0 + D10）、`eval/c1/results-summary.json`（C1 + P1–P5）、
`eval/ablation/results-summary.json`（T8-LM/T8-PM）、`logs/s2_continuation_probes_summary.json`
（续轮探针，缺失时 fail-soft 为 null）+ `eval/tasks.jsonl` 任务集元信息。
前端渲染 TL;DR、D10/P1–P5 判据表、C0 有害 flip（T06/T19）、C1 记忆级联（T20）、
T8 消融 caveat、续轮 0/5、延迟归因、截断统计、故障重试审计与全部冻结 SVG 图表。
**所有实验数字来自 `/api/data`，前端不重算。**

### 2. 实时决策面板（在线，需 `--online`）

`POST /api/decide`，body `{arm, message, execute, evaluate, session_id}`：

| Arm | 语义 |
|---|---|
| **A** | 静态路由：固定 `reasoning`（CPA/glm-5.3），不调 Judge |
| **B** | 仅本地 Judge：`local_judge.JudgeClient`（Method B，四档各一次请求）→ argmax |
| **C0** | JitRL：judge → `engine.decide`（min_neighbors=1，seed 42）→ z′ = z + β·Â |
| **C1** | 门控 JitRL：min_neighbors=3，邻居不足退回纯 Judge |
| **PM** | prompt 注入（T8-PM 语义）：从本会话 **C0 记忆**检索邻居 → `harness.run_ablation.render_memory_block` 渲染白名单块 → 注入 Judge prompt 得 z_prompt；**无调制、不写记忆**（记忆仍随同会话 C0/C1 决策增长） |

- `execute=true`：`harness.cpa_client.CPAExecClient` 真实生成（max_tokens 1024，temp 0.7）；
  失败时决策 trace 仍返回，带结构化 `exec_error`。
- `evaluate=true`（需 execute 成功）：`harness.evaluator.Evaluator`（CPA/gpt-5.6-sol 盲评）；
  失败 → `evaluation=null` 且**不写记忆**（fail-safe，与 harness 语义一致）；成功 →
  `harness.real_rewards.RealQualityReward` 计算 reward，C0/C1 执行 `engine.learn` 写会话记忆。
- 判档 Judge 不可用 → 结构化 502；预算耗尽 → 结构化 429；离线 → 结构化 503。页面永不白屏。

### 其他端点

- `GET /api/tasks` — 24 条主任务 + 5 条续轮探针（id / family / gt_tier / 完整 message）
- `GET /api/charts/<main|c1|ablation|diagrams>/<name>.svg` — 白名单 SVG（硬路径穿越防护）
- `GET /api/health` — `{online_mode, judge_online, cpa_configured}`（judge 探测 ≤1.5s 超时；
  离线模式跳过探测，保持零网络）
- `POST /api/reset` `{session_id}` — 清空该会话的内存状态与计数器

## 隐私与安全

- **会话记忆只在服务器 RAM**（`JitRLEngine` 实例 + 成本计数 + 历史）；`/api/reset` 清空；
  进程退出即消失，绝不落盘。
- **观众自由输入文本不落盘、不进日志**：访问日志只记 `方法 路径 状态码`，从不记录请求体。
- **PilotDeck provider apiKey 只被后端读取**（复用 `harness.cpa_client` 的配置解析），只出现在
  出站请求的 Authorization 头里，**永不发送到浏览器**；`/api/health` 只报配置文件是否存在。
- 预算护栏：每会话累计**执行**花费 ≤ `--max-session-usd`（默认 $1.00）；判档（本地 llama）
  免费，不计入预算；盲评开销为测量开销，同样不计入执行预算口径。

## 故障排查

| 症状 | 排查 |
|---|---|
| 在线模式横幅「Judge 不可达」 | 先启动 llama-server：`llama-server -m minicpm5-1b … --port 18080`；端点可用 `JITRL_JUDGE_ENDPOINT` 覆盖（默认 `http://127.0.0.1:18080`），模型用 `JITRL_JUDGE_MODEL`，超时 `JITRL_JUDGE_TIMEOUT_S` |
| 横幅「provider 配置缺失」 | PilotDeck 配置解析顺序：`PILOTDECK_CONFIG_PATH` → `PILOT_HOME/pilotdeck.yaml` → `~/.pilotdeck/pilotdeck.yaml`；需含 `model.providers.CPA`（protocol=openai、url、apiKey）——provider 名可用 `JITRL_PROVIDER_ID` 改成你自己的（默认 `CPA`，实验时是作者设备上的本地中转 provider）。Demo 不读取也不展示其内容 |
| 决策返回 `budget_exceeded` | 会话执行花费已达上限：点「重置会话」或调大 `--max-session-usd` 重启 |
| 想断网演示 | 直接默认离线模式启动即可——实验面板完全本地，`/api/decide` 返回 503 JSON 并显示黄色横幅 |
| 端口占用 | `--port` 换端口 |

## 目录

```
demo/
├── server.py          # ThreadingHTTPServer + DemoApp（judge/exec/evaluator 可注入）
├── static/            # index.html / style.css / app.js（原生，无构建）
├── tests/test_server.py
└── README.md
```
