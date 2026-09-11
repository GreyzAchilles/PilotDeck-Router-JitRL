"""demo.server tests — pytest-compatible, stdlib only, NO services needed.

Two layers:
  * HTTP-level tests spin a short-lived ThreadingHTTPServer on an ephemeral
    port inside the test (thread is shut down before the test ends; nothing
    is left running). The offline app makes zero network calls.
  * Unit tests drive DemoApp.handle_decide directly with FAKE judge /
    exec / evaluator injected via the constructor factories (no network):
    z_mod math, PM memory_block rendering, C1 gate, budget cap, session
    isolation, fail-safe semantics.
"""
from __future__ import annotations

import contextlib
import json
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from demo.server import ARMS, TIERS, DemoApp, DemoHTTPServer

REPO_ROOT_UART = None  # (placeholder; repo layout is resolved by demo.server)


# --------------------------------------------------------------- fakes
class FakeJudge:
    """Fake local judge: fixed 4-tier logits, records every call."""

    def __init__(self, logits=None):
        self.logits = logits or {"simple": -0.5, "medium": -2.0,
                                 "complex": -3.0, "reasoning": -4.0}
        self.calls = []

    def judge(self, message, previous_tier=None, memory_block=None):
        self.calls.append({"message": message, "previous_tier": previous_tier,
                           "memory_block": memory_block})
        return {"tier_logits": dict(self.logits), "chosen_tier": "simple",
                "raw_output": "<tier>simple</tier>", "latency_ms": 4.0}


class FakeExecClient:
    """Fake CPA exec client: returns a fixed completion + real usage dict."""

    def __init__(self, content="这是执行模型生成的示例回复，用于测试。"):
        self.content = content
        self.calls = []

    def completion(self, model, messages, *, max_tokens=512, temperature=0.7,
                   **_kw):
        self.calls.append({"model": model, "messages": messages,
                           "max_tokens": max_tokens,
                           "temperature": temperature})
        usage = {"input_tokens": 1200, "cache_read_tokens": 0,
                 "cache_write_tokens": 0, "output_tokens": 300,
                 "total_tokens": 1500}
        return SimpleNamespace(content=self.content, usage=usage, model=model,
                               finish_reason="stop", latency_ms=12.0,
                               native_cost=None)


class FakeEvaluator:
    """Fake blind evaluator: always quality 4 / useful; records calls."""

    def __init__(self, quality=4, verdict="useful"):
        self.quality = quality
        self.verdict = verdict
        self.calls = []

    def evaluate(self, task, response):
        self.calls.append({"task": task, "response": response})
        return SimpleNamespace(quality_score=self.quality, verdict=self.verdict)


class NoneEvaluator:
    """Evaluator whose evaluation always fails (returns None)."""

    def evaluate(self, task, response):
        return None


def make_app(online=True, max_session_usd=1.0, judge=None, exec_client=None,
             evaluator=None) -> DemoApp:
    judge = judge or FakeJudge()
    exec_client = exec_client or FakeExecClient()
    evaluator = evaluator if evaluator is not None else FakeEvaluator()
    app = DemoApp(
        online=online, max_session_usd=max_session_usd,
        judge_factory=lambda: judge,
        cpa_client_factory=lambda: exec_client,
        evaluator_factory=lambda _client: evaluator,
    )
    return app


MSG = ("给下面这个函数写单元测试：\n```python\ndef add(a, b):\n"
       "    return a + b\n```")


def decide(app, *, arm="C0", message=MSG, session_id="s1",
           execute=True, evaluate=True, task_id=None):
    return app.handle_decide({
        "arm": arm, "message": message, "execute": execute,
        "evaluate": evaluate, "session_id": session_id, "task_id": task_id,
    })


# ------------------------------------------------- short-lived live server
@contextlib.contextmanager
def live_server(app: DemoApp):
    httpd = DemoHTTPServer(("127.0.0.1", 0), app)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def http_get(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()


def http_post_json(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


# =========================================================== /api/data etc.
class TestReadOnlyEndpoints:
    def test_api_data_merged_json(self):
        with live_server(DemoApp()) as base:
            status, ctype, body = http_get(base + "/api/data")
        assert status == 200 and "application/json" in ctype
        data = json.loads(body)
        # merged frozen aggregates
        assert {"main", "c1", "ablation", "continuation",
                "tasks_meta", "sources"} <= set(data)
        # arms A/B/C0 present (main: A/B/C == C0; c1: B/C0/C1)
        assert {"A", "B", "C"} <= set(data["main"]["groups"])
        assert {"B", "C0", "C1"} <= set(data["c1"]["groups"])
        assert {"T8-LM", "T8-PM"} <= set(data["ablation"]["arms"])
        # task meta counts: 24 main / 5 probes / 29 total
        tm = data["tasks_meta"]
        assert tm["n_main"] == 24 and tm["n_probes"] == 5 and tm["n_total"] == 29
        assert sum(tm["gt_distribution"].values()) == 29
        assert tm["gt_distribution"] == {"complex": 2, "medium": 10,
                                         "reasoning": 12, "simple": 5}
        assert sum(tm["main_gt_distribution"].values()) == 24
        # per-task rows already present in the frozen summaries
        assert data["main"]["d10"]["criteria"]
        assert data["main"]["paired_bc"]["per_task"]
        assert data["c1"]["c1_gate"]["per_task"]
        assert data["c1"]["p1_p5"]["P1"]
        assert data["ablation"]["paired"]["per_task"]
        assert data["main"]["flips"]["flips"]
        # continuation summary exists (fail-soft would give None)
        assert data["continuation"] is not None
        assert len(data["continuation"]["per_probe"]) == 5

    def test_api_tasks(self):
        with live_server(DemoApp()) as base:
            status, _, body = http_get(base + "/api/tasks")
        assert status == 200
        tasks = json.loads(body)
        assert tasks["n_main"] == 24 and tasks["n_probes"] == 5
        assert len(tasks["tasks"]) == 24 and len(tasks["probes"]) == 5
        for t in tasks["tasks"]:
            assert {"id", "family", "gt_tier", "message"} <= set(t)
            assert t["message"]  # full message, needed for live decisions

    def test_root_serves_html_and_static(self):
        with live_server(DemoApp()) as base:
            status, ctype, body = http_get(base + "/")
            assert status == 200 and "text/html" in ctype
            assert "JitRL" in body.decode("utf-8")
            status, ctype, body = http_get(base + "/static/app.js")
            assert status == 200 and "javascript" in ctype
            status, ctype, _ = http_get(base + "/static/style.css")
            assert status == 200 and "text/css" in ctype

    def test_health_offline_mode_zero_network(self):
        with live_server(DemoApp(online=False)) as base:
            status, _, body = http_get(base + "/api/health")
        assert status == 200
        h = json.loads(body)
        assert set(h) == {"online_mode", "judge_online", "cpa_configured"}
        assert h["online_mode"] is False
        # offline mode skips the probe entirely (zero network)
        assert h["judge_online"] is False
        assert isinstance(h["cpa_configured"], bool)

    def test_unknown_route_404(self):
        with live_server(DemoApp()) as base:
            status, _, body = http_get(base + "/api/nope")
        assert status == 404
        assert json.loads(body)["error"] == "not_found"


# ============================================================== /api/charts
class TestCharts:
    def test_serves_real_svg_from_eval_charts(self):
        with live_server(DemoApp()) as base:
            status, ctype, body = http_get(
                base + "/api/charts/main/cost-summary.svg")
        assert status == 200
        assert "image/svg+xml" in ctype
        assert body.lstrip().startswith(b"<?xml")
        assert b"<svg" in body[:600]

    def test_all_groups_whitelisted(self):
        cases = [("main", "quality-distribution.svg"),
                 ("c1", "gate-modulation.svg"),
                 ("ablation", "latency-attribution.svg"),
                 ("diagrams", "architecture.svg")]
        with live_server(DemoApp()) as base:
            for group, name in cases:
                status, ctype, _ = http_get(
                    f"{base}/api/charts/{group}/{name}")
                assert status == 200, (group, name)
                assert "image/svg+xml" in ctype

    def test_blocks_path_traversal(self):
        paths = [
            "/api/charts/main/..%2f..%2fREADME.md",
            "/api/charts/main/%2e%2e%2f%2e%2e%2fserver.py",
            "/api/charts/main/../../harness/run.py",
            "/api/charts/main/..",
            "/api/charts/main/./cost-summary.svg",
            "/api/charts/main/tasks.jsonl",        # not an SVG
            "/api/charts/eval/results-summary.json",  # unknown group
            "/api/charts/main/sub/cost-summary.svg",  # extra segment
            "/static/..%2fserver.py",
        ]
        with live_server(DemoApp()) as base:
            for p in paths:
                status, _, body = http_get(base + p)
                assert status == 404, p
                assert json.loads(body)["error"] in ("not_found",
                                                     "unknown_chart_group"), p


# ============================================================ offline /api/decide
class TestOfflineDecide:
    def test_decide_offline_returns_structured_503(self):
        with live_server(DemoApp(online=False)) as base:
            status, payload = http_post_json(base + "/api/decide", {
                "arm": "C0", "message": "你好", "execute": True,
                "evaluate": True, "session_id": "x"})
        assert status == 503
        assert payload["fallback"] == "offline"
        assert payload["error"]

    def test_decide_offline_unit(self):
        app = DemoApp(online=False)
        status, payload = app.handle_decide({"arm": "B", "message": "hi"})
        assert status == 503 and payload["fallback"] == "offline"


# ============================================== live decide with injected fakes
class TestLiveDecideViaHTTP:
    def test_decide_online_full_roundtrip_and_reset(self):
        app = make_app()
        with live_server(app) as base:
            status, payload = http_post_json(base + "/api/decide", {
                "arm": "C0", "message": MSG, "execute": True,
                "evaluate": True, "session_id": "demo-1"})
            assert status == 200
            assert payload["arm"] == "C0"
            assert payload["session"]["memory_counts"]["C0"] == 1
            assert payload["evaluation"]["quality_score"] == 4
            assert payload["exec"]["cost_usd"] > 0
            assert len(payload["exec"]["content_preview"]) <= 500

            # reset clears the session RAM state
            status, r = http_post_json(base + "/api/reset",
                                       {"session_id": "demo-1"})
            assert status == 200 and r["ok"] is True and r["cleared"] is True
            assert "demo-1" not in app._sessions

            # a fresh decide on the same session_id starts from empty memory
            status, payload = http_post_json(base + "/api/decide", {
                "arm": "C0", "message": MSG, "execute": True,
                "evaluate": True, "session_id": "demo-1"})
            assert status == 200
            assert payload["session"]["memory_counts"]["C0"] == 1
            assert payload["gate_reason"] == "no_neighbors"

    def test_reset_unknown_session_ok(self):
        with live_server(DemoApp()) as base:
            status, r = http_post_json(base + "/api/reset",
                                       {"session_id": "never-existed"})
        assert status == 200 and r["ok"] is True and r["cleared"] is False


# ================================================================ unit: arms
class TestArms:
    def test_arm_A_static_no_judge(self):
        app = make_app()
        judge = app._judge_factory()
        status, p = decide(app, arm="A")
        assert status == 200
        assert p["chosen_tier"] == "reasoning"
        assert p["exec_model"] == "CPA/glm-5.3"
        assert p["z_base"] == {} and p["base_choice"] == "reasoning"
        assert p["flipped"] is False
        assert judge.calls == []  # A never calls the judge

    def test_arm_B_judge_only(self):
        app = make_app()
        judge = app._judge_factory()
        status, p = decide(app, arm="B", execute=False, evaluate=False)
        assert status == 200
        assert len(judge.calls) == 1
        assert judge.calls[0]["memory_block"] is None
        assert p["chosen_tier"] == "simple"  # argmax of the fake logits
        assert "z_mod" not in p and "z_prompt" not in p
        assert all(v >= -10.0 for v in p["z_base"].values())
        assert set(p["z_base"]) == set(TIERS)

    def test_arm_C0_first_decision_no_neighbors(self):
        app = make_app()
        status, p = decide(app, arm="C0")
        assert status == 200
        assert p["gate_reason"] == "no_neighbors"
        assert p["modulation_active"] is False
        assert p["z_mod"] == p["z_base"]
        assert p["n_retrieved"] == 0 and p["retrieved"] == []
        assert p["z_base"]["simple"] == pytest.approx(-0.5)
        # successful evaluation -> memory written (RAM only)
        assert p["session"]["memory_counts"]["C0"] == 1
        assert p["session"]["memory_count"] == 1
        assert p["reward"] is not None

    def test_arm_C0_z_mod_math_path(self):
        """z' = z + beta * A_norm via the engine (retrieval hit on re-decide)."""
        app = make_app()
        decide(app, arm="C0")                      # learn one episode
        status, p = decide(app, arm="C0")          # same message -> sim 1.0
        assert status == 200
        assert p["n_retrieved"] >= 1
        assert p["gate_reason"] is None            # min_neighbors=1 satisfied
        assert p["modulation_active"] is True
        assert p["top_similarity"] == pytest.approx(1.0)
        for t in TIERS:
            expected = max(p["z_base"][t] + 5.0 * p["A_norm"][t], -10.0)
            assert p["z_mod"][t] == pytest.approx(expected), t
        assert p["retrieved"][0]["tier"] in TIERS
        assert set(p["retrieved"][0]) == {"tier", "G", "similarity"}

    def test_arm_C1_gate_below_min_neighbors(self):
        """1 neighbor < min_neighbors=3 -> gated: z_mod == z_base."""
        app = make_app()
        decide(app, arm="C1")                      # writes 1 memory entry
        status, p = decide(app, arm="C1")
        assert status == 200
        assert p["n_retrieved"] == 1
        assert p["gate_reason"] == "below_min_neighbors"
        assert p["modulation_active"] is False
        assert p["z_mod"] == p["z_base"]
        assert all(v == 0.0 for v in p["A_norm"].values())
        # the gated decision still learns after successful evaluation
        assert p["session"]["memory_counts"]["C1"] == 2

    def test_arm_C1_three_neighbors_modulates(self):
        app = make_app()
        for _ in range(3):
            decide(app, arm="C1")                  # 3 identical episodes
        status, p = decide(app, arm="C1")
        assert status == 200
        assert p["n_retrieved"] == 3
        assert p["gate_reason"] is None
        assert p["modulation_active"] is True

    def test_arm_PM_renders_block_and_is_write_free(self):
        """PM: judge receives memory_block; no modulation; NO memory write."""
        app = make_app()
        judge = app._judge_factory()
        decide(app, arm="C0")                      # seed one C0 memory entry
        n_calls_before = len(judge.calls)

        status, p = decide(app, arm="PM")
        assert status == 200
        # two judge calls: raw base + injected prompt
        assert len(judge.calls) == n_calls_before + 2
        base_call = judge.calls[-2]
        prompt_call = judge.calls[-1]
        assert base_call["memory_block"] is None
        assert prompt_call["memory_block"] is not None
        assert "neighbor_count:" in prompt_call["memory_block"]
        assert "neighbor_1:" in prompt_call["memory_block"]
        assert "tier=" in prompt_call["memory_block"]
        assert p["memory_block_chars"] == len(prompt_call["memory_block"]) > 0
        assert p["z_prompt"] is not None and set(p["z_prompt"]) == set(TIERS)
        assert "z_mod" not in p                    # PM never modulates
        assert p["modulation_active"] is False
        assert p["chosen_tier"] == "simple"        # argmax of z_prompt
        # PM is write-free even with successful evaluation
        assert p["session"]["memory_counts"]["C0"] == 1
        assert p["evaluation"] is not None         # evaluation still allowed

    def test_arm_PM_empty_memory_no_block(self):
        app = make_app()
        judge = app._judge_factory()
        status, p = decide(app, arm="PM", execute=False, evaluate=False)
        assert status == 200
        # no neighbors -> single raw call only, no injection
        assert all(c["memory_block"] is None for c in judge.calls)
        assert p["memory_block_chars"] == 0
        assert p["z_prompt"] == p["z_base"]
        assert p["gate_reason"] == "no_neighbors"

    def test_arm_validation_errors(self):
        app = make_app()
        assert decide(app, arm="X")[0] == 400
        assert decide(app, arm="B", message="")[0] == 400
        assert decide(app, arm="B", message="   ")[0] == 400
        s, p = decide(app, arm="B", execute=False, evaluate=True)
        assert s == 400 and p["error"] == "evaluate_requires_execute"


# ====================================================== unit: fail-safe paths
class TestFailSafe:
    def test_exec_failure_returns_trace_with_error(self):
        class ExplodingExec:
            def completion(self, *a, **kw):
                raise RuntimeError("CPA request failed (HTTP 500) boom")
        app = make_app(exec_client=ExplodingExec())
        status, p = decide(app, arm="C0")
        assert status == 200                       # trace still returned
        assert p["exec"] is None
        assert p["exec_error"]
        assert p["evaluation"] is None             # no evaluation w/o exec
        assert p["reward"] is None
        # NO memory write (engine exists but stays empty)
        assert p["session"]["memory_counts"].get("C0", 0) == 0

    def test_eval_failure_no_memory_write(self):
        app = make_app(evaluator=NoneEvaluator())
        status, p = decide(app, arm="C0")
        assert status == 200
        assert p["evaluation"] is None
        assert p["eval_error"]
        assert p["session"]["memory_counts"].get("C0", 0) == 0  # no write

    def test_execute_without_evaluate_no_write(self):
        app = make_app()
        status, p = decide(app, arm="C0", evaluate=False)
        assert status == 200
        assert p["exec"] is not None and p["evaluation"] is None
        assert p["session"]["memory_counts"].get("C0", 0) == 0


# ============================================================ unit: sessions
class TestSessions:
    def test_session_isolation(self):
        app = make_app()
        # session A: two C0 episodes -> 2 memories
        decide(app, arm="C0", session_id="alice")
        _, pa = decide(app, arm="C0", session_id="alice")
        assert pa["session"]["memory_counts"]["C0"] == 2
        assert pa["session"]["cumulative_cost_usd"] > 0
        # session B: one C0 episode -> 1 memory, independent cost counter
        _, pb = decide(app, arm="C0", session_id="bob")
        assert pb["session"]["memory_counts"]["C0"] == 1
        assert pb["session"]["cumulative_cost_usd"] < \
            pa["session"]["cumulative_cost_usd"]
        # alice's second decision retrieved her own memory only (sim 1.0)
        assert pa["n_retrieved"] == 1
        assert pb["n_retrieved"] == 0
        assert len(pa["session"]["history"]) == 2
        assert len(pb["session"]["history"]) == 1

    def test_reset_clears_fabricated_session(self):
        app = make_app()
        decide(app, arm="C0", session_id="s-x")
        assert "s-x" in app._sessions
        status, r = app.handle_reset({"session_id": "s-x"})
        assert status == 200 and r["ok"] and r["cleared"]
        assert "s-x" not in app._sessions
        # reset of an unknown session is a no-op ok
        status, r = app.handle_reset({"session_id": "ghost"})
        assert status == 200 and r["cleared"] is False
        # invalid session id
        assert app.handle_reset({"session_id": ""})[0] == 400


# ============================================================= unit: budget
class TestBudgetGuard:
    def test_budget_cap_triggers_structured_error(self):
        app = make_app(max_session_usd=1.0)
        # first two executed decisions are allowed (pre-check blocks only
        # when the cap is already exhausted)
        status, p1 = decide(app, arm="C0")
        assert status == 200
        spent = p1["session"]["cumulative_cost_usd"]
        assert spent > 0
        app.max_session_usd = spent                # cap now exactly exhausted
        status, p2 = decide(app, arm="C0")
        assert status == 429
        assert p2["error"] == "budget_exceeded"
        assert p2["fallback"] == "budget"
        assert p2["cap_usd"] == pytest.approx(spent)
        assert p2["spent_usd"] == pytest.approx(spent, abs=1e-6)
        # judge-only decisions (execute=false) stay free
        status, p3 = decide(app, arm="C0", execute=False, evaluate=False)
        assert status == 200
        # reset re-opens the budget
        app.handle_reset({"session_id": "s1"})
        status, _ = decide(app, arm="C0")
        assert status == 200

    def test_budget_fabricated_session(self):
        app = make_app(max_session_usd=1.0)
        decide(app, arm="C0", session_id="rich")   # spends something
        app._sessions["rich"].cumulative_cost_usd = 5.0
        status, p = decide(app, arm="C0", session_id="rich")
        assert status == 429 and p["error"] == "budget_exceeded"


# ============================================================ misc contracts
class TestMisc:
    def test_arms_and_tiers_constants(self):
        assert ARMS == ("A", "B", "C0", "C1", "PM")
        assert TIERS == ("simple", "medium", "complex", "reasoning")

    def test_decide_generates_session_id_when_absent(self):
        app = make_app()
        status, p = app.handle_decide(
            {"arm": "A", "message": "hi", "execute": False})
        assert status == 200
        assert p["session_id"]

    def test_invalid_json_flags(self):
        app = make_app()
        status, p = app.handle_decide(
            {"arm": "C0", "message": "hi", "execute": "yes"})
        assert status == 400 and p["error"] == "invalid_flags"

    def test_judge_failure_structured_502(self):
        class ExplodingJudge:
            def judge(self, *a, **kw):
                raise RuntimeError("judge request failed after 3 attempts")
        app = make_app(judge=ExplodingJudge())
        status, p = decide(app, arm="B", execute=False, evaluate=False)
        assert status == 502
        assert p["error"] == "judge_unavailable"
        assert p["fallback"] == "judge"
