/* JitRL Router Web Demo — vanilla JS, no build, no external assets.
   All experiment numbers are read verbatim from /api/data; the frontend
   never recomputes metrics. */
"use strict";

const TIERS = ["simple", "medium", "complex", "reasoning"];
const TIER_ZH = { simple: "简单", medium: "中等", complex: "复杂", reasoning: "推理" };

const ARM_DESC = {
  A: "静态路由：永远 reasoning（执行模型 CPA/glm-5.3），不调用 Judge —— D14 静态默认。",
  B: "仅本地 Judge：MiniCPM5-1B 四档 logit（Method B，每档一次 grammar-forced 请求，共 4 次）取 argmax。",
  C0: "JitRL 默认：z′ = z + β·Â（β=5，min_neighbors=1）；盲评成功后把经验写回本会话记忆（仅内存）。",
  C1: "门控 JitRL：min_neighbors=3 —— 检索邻居不足 3 条时退回纯 Judge（z_mod = z_base）。",
  PM: "prompt 注入（T8-PM 语义）：从本会话 C0 记忆检索邻居，渲染白名单记忆块注入 Judge prompt；无调制、不写记忆。",
};

const CHARTS = {
  chartsMain: { group: "main", cap: "A/B/C0 主实验", items: [
    ["bc-paired-exec-cost", "B↔C0 配对执行成本"],
    ["bc-paired-quality", "B↔C0 配对质量"],
    ["cost-summary", "成本汇总"],
    ["learning-curve", "学习曲线（family-blocked 顺序混叠，仅方向性）"],
    ["quality-distribution", "质量分布"],
  ]},
  chartsC1: { group: "c1", cap: "C1 门控实验", items: [
    ["bc1-paired-exec-cost", "B↔C1 配对执行成本"],
    ["cost-comparison", "成本对比"],
    ["gate-modulation", "门控与调制计数"],
    ["quality-comparison", "质量对比"],
  ]},
  chartsAblation: { group: "ablation", cap: "T8 消融", items: [
    ["latency-attribution", "延迟归因"],
    ["per-task-exec-cost", "逐任务执行成本"],
    ["per-task-quality", "逐任务质量"],
    ["quality-distribution", "质量分布"],
  ]},
  chartsDiagrams: { group: "diagrams", cap: "架构图", items: [
    ["architecture", "整体架构"],
    ["arms", "A/B/C 实验路径"],
    ["online-offline", "本 Demo 的在线/离线双模式"],
  ]},
};

const state = {
  data: null, tasks: [], health: null,
  arm: "C0", selectedTask: null,
  sessionId: null, onlineReady: false,
};

/* ------------------------------------------------------------- helpers */
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g,
  (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const f6 = (x) => (x === null || x === undefined) ? "—" : "$" + Number(x).toFixed(6);
const f4 = (x) => (x === null || x === undefined) ? "—" : Number(x).toFixed(4);
const pct = (r) => (r === null || r === undefined) ? "—" : (r * 100).toFixed(2) + "%";
const signed = (x) => (x > 0 ? "+" : "") + x;

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> HTTP ${r.status}`);
  return r.json();
}

/* =========================================================== TAB 1 render */
function renderOffline(data) {
  const main = data.main, c1 = data.c1, abl = data.ablation;
  const gA = main.groups.A, gB = main.groups.B, gC = main.groups.C;
  const gC1 = c1.groups.C1;
  const t8lm = abl.arms["T8-LM"], t8pm = abl.arms["T8-PM"];

  /* ---- TL;DR ---- */
  const d10c0 = main.d10.criteria[0];
  const rows = [
    ["A", "静态路由（全量 CPA/glm-5.3）", f4(gA.quality.mean), f6(gA.exec_cost_usd.total),
     pct(gA.static_tier_gt_coincidence.match_rate) + "（静态巧合）", "基线（D14 静态默认）", ""],
    ["B", "仅本地 Judge 路由", f4(gB.quality.mean), f6(gB.exec_cost_usd.total),
     pct(gB.tier_vs_gt.match_rate), "便宜但主要来自弱 Judge 偏 simple", ""],
    ["C0", "JitRL（默认，β=5）", f4(gC.quality.mean), f6(gC.exec_cost_usd.total),
     pct(gC.tier_vs_gt.match_rate),
     `D10 判定 ${main.d10.n_criteria_met}/4：${main.d10.verdict}` +
     `（成本目标 FAIL，savings ${d10c0.savings_pct}%）`, ""],
    ["C1", "gated（min_neighbors=3）", f4(gC1.quality.mean), f6(gC1.exec_cost_usd.total),
     pct(gC1.tier_vs_gt.match_rate),
     `预注册 ${c1.p1_p5.n_criteria_met}/5：${c1.p1_p5.verdict}（P1/P5 FAIL，记忆级联）`, ""],
    ["T8-LM", "消融：logit 调制", f4(t8lm.quality.mean), f6(t8lm.exec_cost_usd.total),
     pct(t8lm.tier_vs_gt.match_rate) + "¹", "机制消融（重放冻结 C0 记忆）", ""],
    ["T8-PM", "消融：prompt 注入", f4(t8pm.quality.mean), f6(t8pm.exec_cost_usd.total),
     pct(t8pm.tier_vs_gt.match_rate) + "¹",
     `方向性占优（Δ$${Number(abl.paired.exec_cost.total_delta_LM_minus_PM_usd).toFixed(6)}），` +
     `但 ${t8pm.influenced_records}/${t8pm.retrieval_hits} 注入改变决策`, "warn"],
  ];
  $("tldrTable").querySelector("tbody").innerHTML = rows.map((r) => `
    <tr><td class="mono"><b>${r[0]}</b></td><td>${esc(r[1])}</td>
    <td class="num">${r[2]}</td><td class="num">${r[3]}</td><td class="num">${r[4]}</td>
    <td>${esc(r[5])}</td></tr>`).join("");
  $("tldrFoot").textContent =
    "¹ T8 为机制消融（两臂重放同一份冻结记忆），GT 准确率不在其预注册判据内；" +
    "A 的准确率是静态 reasoning 档与 GT 的巧合重合，非路由准确率。" +
    "质量为 CPA/gpt-5.6-sol 盲评（1–5 分）；成本为执行成本口径（盲评预算为测量开销）。" +
    "n=24、单次运行，方向性证据。";

  /* ---- D10 / P1–P5 ---- */
  $("d10Verdict").textContent = `${main.d10.n_criteria_met}/4 · ${main.d10.verdict}`;
  $("d10Verdict").className = "pill " + (main.d10.verdict === "MET" ? "pill-pass" : "pill-fail");
  $("d10Table").querySelector("tbody").innerHTML = main.d10.criteria.map((c) => `
    <tr><td>${esc(c.criterion)}</td><td class="num">${esc(d10Measured(c))}</td>
    <td><span class="pill ${c.met ? "pill-pass" : "pill-fail"}">${c.met ? "PASS" : "FAIL"}</span></td></tr>`).join("");

  const p15 = c1.p1_p5;
  $("p15Verdict").textContent = `${p15.n_criteria_met}/5 · ${p15.verdict}`;
  $("p15Verdict").className = "pill " + (p15.failed.length === 0 ? "pill-pass" : "pill-fail");
  $("p15Table").querySelector("tbody").innerHTML = ["P1", "P2", "P3", "P4", "P5"].map((k) => {
    const p = p15[k];
    return `<tr><td>${esc(p.criterion)}</td><td class="num">${esc(pMeasured(p, k))}</td>
      <td><span class="pill ${p.met ? "pill-pass" : "pill-fail"}">${p.met ? "PASS" : "FAIL"}</span></td></tr>`;
  }).join("");

  /* ---- C0 flips ---- */
  $("c0FlipList").innerHTML = main.flips.flips.map((f) => `
    <div class="flip-item">
      <div class="flip-title"><span class="tid">${f.task_id}</span>
        <span>${esc(f.family)} · GT ${f.gt_tier}</span>
        <span class="tier-arrow tier-${f.base_choice}">${f.base_choice}</span>→
        <span class="tier-arrow tier-${f.chosen_tier}">${f.chosen_tier}</span></div>
      <div>成本 <span class="delta-cost">+$${Number(f.exec_cost_delta_usd).toFixed(6)}</span>
        （B ${f6(f.exec_cost_B_usd)} → C0 ${f6(f.exec_cost_C_usd)}）·
        质量 ${f.quality_B} → ${f.quality_C}</div>
      <div class="meta-line">检索 ${f.n_retrieved} 条（top sim ${f.top_similarity}）·
        finish=${f.exec_finish_reason}${f.response_empty ? " · 空回复" : ""} ·
        GT 正误 ${f.gt_correct_before} → ${f.gt_correct_after}</div>
    </div>`).join("");

  /* ---- C1 cascade ---- */
  const gc = c1.c1_gate.counts;
  $("c1GateChips").innerHTML = [
    `total <b>${gc.total}</b>`, `被 gate <b>${gc.gated}</b>`,
    `无邻居 ${gc.gated_no_neighbors}`, `低于阈值 ${gc.gated_below_min_neighbors}`,
    `调制 ${gc.modulated}`, `调制且 flip ${gc.modulated_and_flipped}`,
  ].map((t) => `<span class="chip">${t}</span>`).join("");
  const c1f = c1.c1_flips.flips[0];
  const nb = c1f.mechanism.retrieved_neighbors_C1
    .map((n) => `${n.task_id}（${n.stored_tier}，G ${n.stored_G}）`).join("、");
  $("c1CascadeList").innerHTML = `
    <div class="flip-item">
      <div class="flip-title"><span class="tid">${c1f.task_id}</span>
        <span>${esc(c1f.family)} · GT ${c1f.gt_tier}</span>
        <span class="tier-arrow tier-${c1f.base_choice}">${c1f.base_choice}</span>→
        <span class="tier-arrow tier-${c1f.chosen_tier}">${c1f.chosen_tier}</span></div>
      <div>成本 <span class="delta-cost">+$${Number(c1f.exec_cost_delta_usd).toFixed(6)}</span>
        （B ${f6(c1f.exec_cost_B_usd)} → C1 ${f6(c1f.exec_cost_C1_usd)}）·
        质量 ${c1f.quality_B} → ${c1f.quality_C}</div>
      <div class="meta-line">机制：被 gate 的决策按 base tier 写回 → 后续检索邻居 ${nb}
        → Â 推高 medium/complex → 新 flip。</div>
    </div>
    <p class="footnote">C0 的 T06/T19 flip 在 C1 中被 gate 抑制，但问题被移动（T06/T19 → T20），没有被解决。</p>`;

  /* ---- T8 ---- */
  const ta = abl.paired.tier_agreement;
  $("t8Body").innerHTML = `
    <div class="chip-row">
      <span class="chip">tier 一致 <b>${ta.agree}/${abl.paired.n_pairs}</b></span>
      <span class="chip">唯一分歧 <b>${ta.tier_changed_task_ids.join("、")}</b></span>
      <span class="chip">LM 注入影响 ${t8lm.influenced_records}/${t8lm.retrieval_hits}
        （${t8lm.influenced_task_ids.join("、") || "无"}）</span>
      <span class="chip">PM 注入影响 ${t8pm.influenced_records}/${t8pm.retrieval_hits}</span>
    </div>
    <div class="chip-row">
      <span class="chip">质量 LM ${f4(t8lm.quality.mean)} vs PM ${f4(t8pm.quality.mean)}
        （Δ ${abl.paired.quality.mean_delta_LM_minus_PM}）</span>
      <span class="chip">成本 LM ${f6(t8lm.exec_cost_usd.total)} vs PM ${f6(t8pm.exec_cost_usd.total)}
        （Δ $${Number(abl.paired.exec_cost.total_delta_LM_minus_PM_usd).toFixed(6)}）</span>
    </div>
    <p class="card-sub">预注册结论：${esc(abl.conclusion.statement)}（${esc(abl.conclusion.strength)}）</p>`;
  $("t8Caveat").textContent =
    "强制 caveat：PM 更便宜是因为它什么都没改变——0/6 注入改变决策；LM 的单次有害 T19 flip " +
    "主导了 $0.004660 的成本差。PM 并未在 1B Judge 上展示出可用的记忆推理能力。";

  /* ---- continuation ---- */
  const cont = data.continuation;
  if (cont) {
    $("contSummary").textContent =
      `5 条续轮探针全部未命中（accuracy ${cont.accuracy}，${cont.n_judged} 条判定、` +
      `${cont.n_errors} 条错误）；prev tier 均已传入 prompt。`;
    $("contTable").querySelector("tbody").innerHTML =
      Object.entries(cont.per_probe).map(([pid, p]) => `
        <tr><td class="mono">${pid}</td><td class="mono">${p.previous_tier}</td>
        <td class="mono">${p.expected_tier}</td><td class="mono tier-${p.chosen_tier}">${p.chosen_tier}</td>
        <td><span class="pill ${p.hit ? "pill-pass" : "pill-fail"}">${p.hit ? "HIT" : "MISS"}</span></td>
        <td class="num">${p.judge_latency_ms} ms</td></tr>`).join("");
  } else {
    $("contSummary").textContent = "续轮探针摘要不可用（logs/s2_continuation_probes_summary.json 缺失）。";
    $("contTable").querySelector("tbody").innerHTML = "";
  }

  /* ---- latency / truncation ---- */
  const d10lat = main.d10.criteria[3];
  const h4 = abl.hypotheses.H4_latency_attribution;
  $("latencyStats").innerHTML = [
    [`Judge 平均延迟（B 组，每次判档 4 请求）`, `${main.groups.B.latency_ms.judge_mean} ms`],
    [`JitRL 后处理（C0 vs B 增量）`, `mean ${d10lat.increment_mean_ms} ms / max ${d10lat.increment_max_ms} ms`],
    [`C1 门控下 JitRL 最大延迟（P4）`, `max ${c1.p1_p5.P4.max_jitrl_ms} ms`],
    [`T8-LM JitRL 最大延迟`, `max ${t8lm.latency_ms.jitrl_max} ms`],
    [`T8-PM 注入后判档调用（额外）`, `mean ${h4.PM_extra_judge_mean_ms} ms`],
    [`执行模型平均延迟（B 组）`, `${main.groups.B.latency_ms.exec_mean} ms`],
  ].map(([k, v]) => `<div class="stat"><span>${esc(k)}</span><span class="v">${esc(v)}</span></div>`).join("");

  const tr = c1.truncation_empty_response;
  const trRows = [
    ["B", tr.B.finish_reason_length, tr.B.empty_response_preview],
    ["C0", tr.C0.finish_reason_length, tr.C0.empty_response_preview],
    ["C1", tr.C1.finish_reason_length, tr.C1.empty_response_preview],
    ["T8-LM", t8lm.finish_reason_length, t8lm.empty_responses],
    ["T8-PM", t8pm.finish_reason_length, t8pm.empty_responses],
  ];
  $("truncTable").querySelector("tbody").innerHTML = trRows.map((r) =>
    `<tr><td class="mono"><b>${r[0]}</b></td><td class="num">${r[1]}</td><td class="num">${r[2]}</td></tr>`).join("");

  /* ---- failure & retry audit ---- */
  $("retryNarrative").innerHTML = `
    <p><b>B 早期顺序批次重跑</b>：最早的 B 批次按数字顺序执行；正式 B 组为 family-blocked
    顺序的<b>完整重跑</b>（<code>s2_full_B_family_order.jsonl</code>，24/24），与 A/C 同序——旧批次不进指标。</p>
    <p><b>C attempt1/2 失败留审计</b>：第一次 C 运行中 <code>T14/T22</code> 遇到 CPA 网关
    HTTP 500（unexpected EOF；cpa_client 已按 1s/2s/4s 指数退避重试共 4 次），eval 22/24；
    失败 episode <b>不写记忆</b>（fail-safe）。第二次重跑仍有失败；正式 C 组为第三次完整重跑
    （<code>retry2</code>，24/24）。前两次只进 failure_audit，不进任何指标。</p>
    <p><b>完整序列重试策略</b>：失败任务不单独补跑——单独补跑会破坏 family-blocked 顺序与
    C0 记忆前缀的可比性，因此每次失败都是<b>完整 24 任务序列重跑</b>。</p>`;
  const audit = main.failure_audit || [];
  $("auditTable").querySelector("tbody").innerHTML = audit.map((a) => {
    const fails = (a.failures || []).map((f) =>
      `${f.task_id}：${esc(String(f.exec_error || f.eval_error || "?").slice(0, 90))}…`).join("<br>") || "—";
    return `<tr><td class="mono">${esc(a.file)}</td><td class="num">${a.n_records}</td>
      <td class="num">${esc(a.eval_success)}</td><td>${fails}</td></tr>`;
  }).join("");

  /* ---- charts ---- */
  for (const [id, cfg] of Object.entries(CHARTS)) {
    $(id).innerHTML = cfg.items.map(([name, cap]) => `
      <div class="chart-item"><img src="/api/charts/${cfg.group}/${name}.svg"
        alt="${esc(cap)}" loading="lazy"><div class="cap">${esc(cap)}</div></div>`).join("");
  }

  const tm = data.tasks_meta;
  $("sources").textContent =
    `数据来源（只读、冻结）：${Object.values(data.sources).join("、")}；` +
    `任务集 ${tm.n_total} 条（主任务 ${tm.n_main} + 续轮探针 ${tm.n_probes}，` +
    `GT 分布 ${Object.entries(tm.gt_distribution).map(([k, v]) => `${k}×${v}`).join(" / ")}）。`;
}

function d10Measured(c) {
  if (c.savings_pct !== undefined)
    return `savings ${c.savings_pct}%（B ${f6(c.B_total_usd)} → C ${f6(c.C_total_usd)}）`;
  if (c.delta !== undefined && c.B_mean !== undefined)
    return `Δ ${signed(c.delta)}（B ${f4(c.B_mean)} → C ${f4(c.C_mean)}）`;
  if (c.diff_pp !== undefined)
    return `${c.diff_pp}pp（B ${pct(c.B_misgrade_rate)} → C ${pct(c.C_misgrade_rate)}）`;
  if (c.increment_max_ms !== undefined)
    return `mean ${c.increment_mean_ms}ms / max ${c.increment_max_ms}ms`;
  return "—";
}

function pMeasured(p, id) {
  if (id === "P1") return `C1 ${f6(p.C1_total_usd)} > 上限 ${f6(p.limit_usd)}（Δ +$${Number(p.delta_usd).toFixed(6)}）`;
  if (id === "P2") return `Δ ${signed(p.delta)}（B ${f4(p.B_mean)} → C1 ${f4(p.C1_mean)}）`;
  if (id === "P3") return `${p.diff_pp}pp（B ${pct(p.B_misgrade_rate)} → C1 ${pct(p.C1_misgrade_rate)}）`;
  if (id === "P4") return `max ${p.max_jitrl_ms}ms`;
  if (id === "P5") return p.flip_checks.map((f) =>
    `${f.task_id}：成本 ${f6(f.exec_cost_C1_usd)} > 上限 ${f6(f.cost_limit_usd)}`).join("；") || "0 flips";
  return "—";
}

/* =========================================================== TAB 2 logic */
function renderArmRow() {
  $("armRow").innerHTML = ["A", "B", "C0", "C1", "PM"].map((a) =>
    `<button class="arm-btn ${a === state.arm ? "active" : ""}" data-arm="${a}">${a}</button>`).join("");
  $("armRow").querySelectorAll(".arm-btn").forEach((b) =>
    b.addEventListener("click", () => { state.arm = b.dataset.arm; renderArmRow(); }));
  $("armDesc").textContent = ARM_DESC[state.arm];
}

function renderTaskSelect() {
  const sel = $("taskSelect");
  sel.innerHTML = `<option value="">— 自由输入 —</option>` + state.tasks.map((t) =>
    `<option value="${esc(t.id)}">${esc(t.id)} · ${esc(t.family)} · GT ${esc(t.gt_tier)}</option>`).join("");
  sel.addEventListener("change", () => {
    const t = state.tasks.find((x) => x.id === sel.value);
    state.selectedTask = t || null;
    $("messageInput").value = t ? t.message : "";
  });
}

function renderHealth(h) {
  state.health = h;
  const badge = $("modeBadge");
  badge.textContent = h.online_mode ? "ONLINE" : "OFFLINE";
  badge.className = "badge " + (h.online_mode ? "badge-on" : "badge-off");
  $("dotJudge").className = "dot " + (h.judge_online ? "dot-on" : "dot-off");
  $("dotCpa").className = "dot " + (h.cpa_configured ? "dot-on" : "dot-off");

  const banner = $("onlineBanner");
  const btn = $("decideBtn");
  if (!h.online_mode) {
    banner.classList.remove("hidden");
    banner.innerHTML = `当前为 <b>离线模式</b>（零网络）：实验数据面板完整可用；实时决策 API 未启用。
      启动在线模式：<code>python -m demo.server --online</code>（需本地 llama-server 与 PilotDeck provider 配置，
      见 demo/README.md）。此处表单已禁用，不会白屏。`;
    banner.className = "banner banner-warn";
    btn.disabled = true;
    ["taskSelect", "messageInput", "toggleExec", "toggleEval"].forEach((id) => $(id).disabled = true);
    state.onlineReady = false;
    return;
  }
  banner.classList.remove("hidden");
  if (!h.judge_online && !h.cpa_configured) {
    banner.innerHTML = `本地 llama.cpp Judge <b>不可达</b>，且 PilotDeck provider 配置缺失——
      在线模式下决策将返回结构化错误。排查指引见 <code>demo/README.md</code>（llama-server 端口、
      <code>PILOTDECK_CONFIG_PATH</code>）。`;
  } else if (!h.judge_online) {
    banner.innerHTML = `本地 llama.cpp Judge <b>不可达</b>：B/C0/C1/PM 的决策会失败（结构化 502）。
      请先启动 llama-server（默认 <code>http://127.0.0.1:18080</code>，可用
      <code>JITRL_JUDGE_ENDPOINT</code> 覆盖）。A（静态路由）不依赖 Judge。`;
  } else if (!h.cpa_configured) {
    banner.innerHTML = `PilotDeck provider 配置缺失：执行生成与盲评会失败（决策 trace 仍可查看，
      可关闭「执行生成」）。配置路径见 <code>demo/README.md</code>。`;
  } else {
    banner.classList.add("hidden");
  }
  banner.className = "banner banner-warn";
  btn.disabled = false;
  ["taskSelect", "messageInput", "toggleExec", "toggleEval"].forEach((id) => $(id).disabled = false);
  state.onlineReady = true;
}

function showError(j, status) {
  const panel = $("errorPanel");
  panel.classList.remove("hidden");
  panel.className = "banner banner-err";
  const known = {
    offline_mode: "离线模式：实时决策 API 未启用。使用 python -m demo.server --online 重启。",
    judge_unavailable: "本地 Judge 不可用：" + (j.detail || ""),
    budget_exceeded: `会话执行预算已耗尽（$${(j.spent_usd ?? 0)} / $${(j.cap_usd ?? 0)}）。可点击「重置会话」后继续。`,
  };
  const msg = known[j.error] || `${j.error || "请求失败"}${j.message ? "：" + j.message : ""}`;
  panel.innerHTML = `<b>HTTP ${status}</b> · <code>${esc(j.error || "error")}</code>（fallback:
    ${esc(j.fallback || "—")}）<br>${esc(msg)}`;
}

function renderResult(r) {
  $("errorPanel").classList.add("hidden");
  $("resultCard").classList.remove("hidden");
  $("resultArm").textContent = r.arm;

  /* chosen / model / gate */
  $("chosenTier").textContent = r.chosen_tier;
  $("chosenTier").className = "tier-badge tier-" + r.chosen_tier;
  $("flipMark").classList.toggle("hidden", !r.flipped);
  $("execModel").textContent = r.exec_model;
  const gate = $("gateBadge");
  if (r.arm === "A") { gate.textContent = "静态（无 gate）"; gate.className = "pill pill-plain"; }
  else if (r.gate_reason === "no_neighbors") {
    gate.textContent = "无检索记忆 → 纯基线"; gate.className = "pill pill-plain"; }
  else if (r.gate_reason === "below_min_neighbors") {
    gate.textContent = "邻居数 < min_neighbors → 门控抑制"; gate.className = "pill pill-warn"; }
  else if (r.arm === "PM") {
    gate.textContent = r.memory_block_chars > 0 ? "已注入记忆块" : "无邻居，未注入";
    gate.className = "pill " + (r.memory_block_chars > 0 ? "pill-pass" : "pill-plain"); }
  else { gate.textContent = "调制生效"; gate.className = "pill pill-pass"; }

  /* logit bars */
  const mode = $("logitModeLabel");
  const zBase = r.z_base || {};
  const zSecond = r.arm === "PM" ? r.z_prompt : r.z_mod;
  if (r.arm === "A") {
    mode.textContent = "（静态路由：无 Judge 调用）";
    $("logitBars").innerHTML =
      `<p class="card-sub">A 为静态路由，不产生 logit；固定 chosen tier = reasoning。</p>`;
  } else {
    const secondClass = r.arm === "PM" ? "prompt" : "mod";
    const secondName = r.arm === "PM" ? "z_prompt（注入后）" :
      (r.arm === "B" ? null : "z_mod（调制后）");
    mode.textContent = r.arm === "B" ? "（z_base，B 无调制）" :
      r.arm === "PM" ? "（两态对比：base vs 注入后）" : "（z_base vs z_mod）";
    const vals = [];
    TIERS.forEach((t) => { vals.push(zBase[t]); if (zSecond) vals.push(zSecond[t]); });
    const lo = Math.min(...vals), hi = Math.max(...vals);
    const span = (hi - lo) || 1;
    const bar = (v) => Math.max(1.5, ((v - lo) / span) * 100);
    $("logitBars").innerHTML = TIERS.map((t) => `
      <div class="logit-row"><div class="logit-tier-line">
        <span class="logit-tier-name tier-${t}">${t}</span>
        <div class="logit-bars">
          <div class="logit-bar-track"><div class="logit-bar base" style="width:${bar(zBase[t])}%"></div></div>
          ${zSecond ? `<div class="logit-bar-track"><div class="logit-bar ${secondClass}" style="width:${bar(zSecond[t])}%"></div></div>` : ""}
        </div>
        <span class="logit-val">${Number(zBase[t]).toFixed(3)}${zSecond ? " / " + Number(zSecond[t]).toFixed(3) : ""}</span>
      </div></div>`).join("") + `
      <div class="logit-legend">
        <span><span class="swatch" style="background:var(--gray-bar)"></span>z_base（原始）</span>
        ${zSecond ? `<span><span class="swatch" style="background:#7c3aed"></span>${secondName}</span>` : ""}
      </div>`;
  }

  /* retrieved */
  $("nRetrieved").textContent = r.n_retrieved;
  $("retrievedList").innerHTML = r.retrieved.length
    ? r.retrieved.map((m) => `<span class="chip">tier <b class="tier-${m.tier}">${m.tier}</b>
        · G ${Number(m.G).toFixed(4)} · sim ${Number(m.similarity).toFixed(4)}</span>`).join("")
    : `<span class="chip">无检索记忆（N(s) 为空）</span>`;
  const pmMeta = $("pmMeta");
  if (r.arm === "PM") {
    pmMeta.classList.remove("hidden");
    pmMeta.textContent = `白名单记忆块 ${r.memory_block_chars} 字符（来源：本会话 C0 记忆；` +
      `PM 无调制、不写记忆——镜像 T8-PM 语义）`;
  } else pmMeta.classList.add("hidden");

  /* V / Q / Â */
  $("vVal").textContent = Number(r.V).toFixed(4);
  $("qVal").textContent = TIERS.map((t) => `${t} ${Number(r.Q[t]).toFixed(3)}`).join("  ");
  $("aVal").textContent = TIERS.map((t) => `${t} ${Number(r.A_norm[t]).toFixed(3)}`).join("  ");

  $("judgeLat").textContent = r.judge_latency_ms + " ms";
  $("jitrlLat").textContent = r.jitrl_latency_ms + " ms";
  const mod = $("modBadge");
  mod.textContent = r.modulation_active ? "modulation ON" : "modulation OFF";
  mod.className = "pill " + (r.modulation_active ? "pill-pass" : "pill-plain");

  /* exec */
  const execBlock = $("execBlock");
  if (r.exec_error) {
    execBlock.innerHTML = `<p class="caveat">执行失败（决策 trace 仍完整返回；本条不写记忆）：
      ${esc(r.exec_error)}</p>`;
  } else if (r.exec) {
    execBlock.innerHTML = `<div class="exec-preview">${esc(r.exec.content_preview)}</div>
      <div class="exec-meta">
        <span>成本 <b>${f6(r.exec.cost_usd)}</b></span>
        <span>finish <b>${esc(r.exec.finish_reason ?? "—")}</b></span>
        <span>tokens in/out <b>${r.exec.usage ? r.exec.usage.input_tokens + "/" + r.exec.usage.output_tokens : "—"}</b></span>
        <span>预览 ≤500 字符</span>
      </div>`;
  } else {
    execBlock.innerHTML = `<p class="card-sub">（本次未执行生成）</p>`;
  }

  /* evaluation */
  const evalBlock = $("evalBlock");
  if (r.evaluation) {
    evalBlock.innerHTML = `<div class="exec-meta">
        <span>quality_score <b>${r.evaluation.quality_score} / 5</b></span>
        <span>verdict <b>${esc(r.evaluation.verdict)}</b></span>
        <span>reward <b>${r.reward === null ? "—" : Number(r.reward).toFixed(4)}</b></span>
        ${r.arm === "C0" || r.arm === "C1" ? `<span>已写回本会话记忆（仅内存）</span>` : ""}
      </div>`;
  } else if (r.eval_error) {
    evalBlock.innerHTML = `<p class="caveat">盲评失败（fail-safe：不写记忆）：${esc(r.eval_error)}</p>`;
  } else {
    evalBlock.innerHTML = `<p class="card-sub">（本次未盲评${r.exec_error ? "——执行已失败" : ""}）</p>`;
  }

  renderSession(r.session);
}

function renderSession(s) {
  $("memC0").textContent = s.memory_counts.C0 ?? 0;
  $("memC1").textContent = s.memory_counts.C1 ?? 0;
  const spent = s.cumulative_cost_usd, cap = s.budget_cap_usd;
  $("budgetText").textContent = `$${Number(spent).toFixed(4)} / $${Number(cap).toFixed(2)}`;
  const fill = $("budgetFill");
  const ratio = cap > 0 ? Math.min(1, spent / cap) : 0;
  fill.style.width = (ratio * 100).toFixed(1) + "%";
  fill.className = "budget-fill" + (ratio >= 1 ? " over" : "");
  $("capNote").textContent = "$" + Number(cap).toFixed(2);
  drawSparkline(s.history);
  const list = $("historyList");
  list.innerHTML = s.history.length ? s.history.slice(-12).reverse().map((h, i, arr) => {
    const idx = s.history.length - arr.length + i + 1;
    return `<div class="h-row"><span>#${idx} ${h.arm} → ${h.chosen_tier}</span>
      <span>${h.cost_usd === null ? "—" : "$" + Number(h.cost_usd).toFixed(6)}
      · q${h.quality === null ? "—" : h.quality}</span></div>`;
  }).join("") : `<div class="h-row"><span>（暂无决策历史）</span></div>`;
}

function drawSparkline(history) {
  const el = $("sparkline");
  if (!history.length) { el.innerHTML = `<p class="card-sub">（暂无数据）</p>`; return; }
  const W = 280, H = 70, pad = 6;
  const n = history.length;
  const x = (i) => pad + (n === 1 ? (W - 2 * pad) / 2 : (i / (n - 1)) * (W - 2 * pad));
  const costs = history.map((h) => h.cost_usd ?? 0);
  const maxCost = Math.max(...costs, 1e-9);
  const yCost = (v) => H - pad - (v / maxCost) * (H - 2 * pad);
  const yQ = (v) => H - pad - ((v - 1) / 4) * (H - 2 * pad);
  const costPts = history.map((h, i) => `${x(i).toFixed(1)},${yCost(h.cost_usd ?? 0).toFixed(1)}`).join(" ");
  const qPts = history.map((h, i) =>
    h.quality === null ? null : `${x(i).toFixed(1)},${yQ(h.quality).toFixed(1)}`)
    .filter(Boolean).join(" ");
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">
    <line x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}" stroke="#e4e7ee"/>
    <polyline points="${costPts}" fill="none" stroke="#10b981" stroke-width="2"/>
    ${qPts ? `<polyline points="${qPts}" fill="none" stroke="#6366f1" stroke-width="1.6" stroke-dasharray="4 3"/>` : ""}
  </svg>
  <div class="logit-legend"><span><span class="swatch" style="background:#10b981"></span>单次成本（≤max）</span>
  <span><span class="swatch" style="background:#6366f1"></span>质量（1–5）</span></div>`;
}

/* ------------------------------------------------------------- actions */
async function decide() {
  if (!state.onlineReady) return;
  const message = $("messageInput").value;
  if (!message.trim()) {
    showError({ error: "invalid_message", message: "请输入或选择一条任务消息" }, 400);
    return;
  }
  const preset = state.selectedTask;
  const taskId = preset && preset.message === message ? preset.id : null;
  const btn = $("decideBtn"), spin = $("decideSpin");
  btn.disabled = true; spin.classList.remove("hidden");
  try {
    const r = await fetch("/api/decide", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        arm: state.arm, message,
        execute: $("toggleExec").checked,
        evaluate: $("toggleEval").checked,
        session_id: state.sessionId,
        task_id: taskId,
      }),
    });
    const j = await r.json();
    if (r.status === 200) renderResult(j);
    else showError(j, r.status);
  } catch (e) {
    showError({ error: "network_error", message: String(e) }, 0);
  } finally {
    btn.disabled = false; spin.classList.add("hidden");
  }
}

async function resetSession() {
  try {
    const r = await fetch("/api/reset", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId }),
    });
    const j = await r.json();
    if (j.ok) {
      $("resultCard").classList.add("hidden");
      $("errorPanel").classList.add("hidden");
      renderSession({ memory_counts: {}, cumulative_cost_usd: 0,
                      budget_cap_usd: 1.0, history: [] });
    }
  } catch (e) { /* ignore */ }
}

/* ----------------------------------------------------------------- init */
function initTabs() {
  $("tabs").querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
    $("tabs").querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    $("tab-" + t.dataset.tab).classList.add("active");
    if (t.dataset.tab === "online") refreshHealth();
  }));
  // Deep link: index.html#live opens the live-decision tab on load
  // (also used by the headless screenshot pipeline for the demo video).
  const want = (location.hash || "").replace("#", "");
  const target = $("tabs").querySelector(`.tab[data-tab="${want}"]`);
  if (target) target.click();
}

async function refreshHealth() {
  try { renderHealth(await getJSON("/api/health")); }
  catch (e) { /* server unreachable: keep last state */ }
}

async function init() {
  state.sessionId = sessionStorage.getItem("jitrl-demo-session");
  if (!state.sessionId) {
    state.sessionId = (crypto.randomUUID ? crypto.randomUUID() : String(Date.now())) ;
    sessionStorage.setItem("jitrl-demo-session", state.sessionId);
  }
  initTabs();
  renderArmRow();

  try {
    const [data, tasks, health] = await Promise.all([
      getJSON("/api/data"), getJSON("/api/tasks"), getJSON("/api/health")]);
    state.data = data; state.tasks = tasks.tasks; renderHealth(health);
    renderOffline(data);
    renderTaskSelect();
  } catch (e) {
    document.querySelector(".page-note").innerHTML =
      `无法加载 /api/data（${esc(e)}）——请确认 demo server 正在运行。`;
  }

  $("decideBtn").addEventListener("click", decide);
  $("resetBtn").addEventListener("click", resetSession);
}

document.addEventListener("DOMContentLoaded", init);
