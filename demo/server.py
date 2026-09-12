"""JitRL Router Web Demo server (Session 3 deliverable).

Offline-first, stdlib-only HTTP server (http.server, ThreadingHTTPServer).

Contract (frozen by README.md + progress doc):
    python -m demo.server [--port N] [--online] [--max-session-usd USD]
      DEFAULT = offline mode (zero network): only the frozen-aggregate
      experiment panels work; /api/decide returns a structured 503 JSON.
      --online enables the live decision API (local llama.cpp judge +
      PilotDeck provider execution/evaluation, proxied by this backend).

Routes:
    GET  /                      -> demo/static/index.html
    GET  /static/<file>         -> css/js/svg from demo/static/
    GET  /api/data              -> frozen aggregates read VERBATIM (never
                                   recomputed): eval/results-summary.json,
                                   eval/c1/results-summary.json,
                                   eval/ablation/results-summary.json,
                                   logs/s2_continuation_probes_summary.json
                                   (fail-soft), plus task-set meta counts.
    GET  /api/tasks             -> preset tasks (id, family, gt_tier, message)
    GET  /api/charts/<g>/<name> -> whitelisted SVGs: main|c1|ablation|diagrams
    GET  /api/health            -> {online_mode, judge_online, cpa_configured}
    POST /api/decide            -> live decision (online only; offline -> 503)
    POST /api/reset             -> clear one session's RAM state

Privacy:
    - Session memory (JitRL engines, cost counters, history) lives in this
      process's RAM only; /api/reset clears it; nothing is persisted.
    - Visitor free-input text is NEVER written to disk or logged (the access
      log records method + path + status only, never bodies).
    - The PilotDeck provider apiKey is read by the backend only (via
      harness.cpa_client) and NEVER sent to the browser. The provider id
      defaults to "CPA" (the experiment author's local-relay provider name)
      and follows the JITRL_PROVIDER_ID env var.

Budget guard:
    - Cumulative live EXECUTION spend per session is capped
      (--max-session-usd, default $1.00). A request whose execution would
      exceed an already-exhausted budget returns a structured 429 error.

Arms (POST /api/decide, body {arm, message, execute, evaluate, session_id}):
    A  static   chosen_tier="reasoning" (exec model CPA/glm-5.3), no judge
    B  judge    local JudgeClient (Method B) -> z_base -> argmax
    C0 JitRL    judge -> engine.decide (min_neighbors=1, seed 42) -> z_mod
    C1 gated    same with min_neighbors=3
    PM inject   retrieve from the session C0 memory, render the whitelist
                block via harness.run_ablation.render_memory_block, judge
                with memory_block -> z_prompt decides. NO modulation and NO
                memory write from this arm (mirrors T8-PM semantics; memory
                still grows via C0/C1 decisions in the same session).

Standard library only. Read-only imports from jitrl_core / harness /
local_judge; this module never modifies anything outside demo/.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:  # allow `python -m demo.server` from anywhere
    sys.path.insert(0, str(_REPO_ROOT))

from jitrl_core import JitRLConfig, JitRLEngine, choose_tier  # noqa: E402
from jitrl_core.policy import clamp_logits  # noqa: E402
from jitrl_core.state import intent_class, task_signature  # noqa: E402

# ------------------------------------------------------------------ constants
TIERS: tuple[str, ...] = ("simple", "medium", "complex", "reasoning")
ARMS = ("A", "B", "C0", "C1", "PM")
C_MIN_NEIGHBORS = {"C0": 1, "C1": 3}
ENGINE_SEED = 42
GATE_NO_NEIGHBORS = "no_neighbors"
GATE_BELOW_MIN_NEIGHBORS = "below_min_neighbors"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8300
DEFAULT_MAX_SESSION_USD = 1.00
MAX_MESSAGE_CHARS = 20_000
MAX_BODY_BYTES = 1_000_000
PREVIEW_CHARS = 500
EXEC_MAX_TOKENS = 1024        # mirrors harness.run_real calibration finding
EXEC_TEMPERATURE = 0.7
HISTORY_CAP = 500             # RAM-only guard; the demo never gets here
HEALTH_CACHE_S = 5.0
JUDGE_PROBE_TIMEOUT_S = 1.5

STATIC_DIR = Path(__file__).resolve().parent / "static"

#: whitelisted chart groups -> on-disk directories (read-only)
CHART_GROUPS: dict[str, Path] = {
    "main": _REPO_ROOT / "eval" / "charts",
    "c1": _REPO_ROOT / "eval" / "c1" / "charts",
    "ablation": _REPO_ROOT / "eval" / "ablation" / "charts",
    "diagrams": _REPO_ROOT / "diagrams",
}

#: frozen aggregates served verbatim by /api/data
DATA_FILES: dict[str, Path] = {
    "main": _REPO_ROOT / "eval" / "results-summary.json",
    "c1": _REPO_ROOT / "eval" / "c1" / "results-summary.json",
    "ablation": _REPO_ROOT / "eval" / "ablation" / "results-summary.json",
    "continuation": _REPO_ROOT / "logs" / "s2_continuation_probes_summary.json",
}
DATA_SOURCES = {
    "main": "eval/results-summary.json",
    "c1": "eval/c1/results-summary.json",
    "ablation": "eval/ablation/results-summary.json",
    "continuation": "logs/s2_continuation_probes_summary.json",
    "tasks": "eval/tasks.jsonl",
}
TASKS_FILE = _REPO_ROOT / "eval" / "tasks.jsonl"

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}


def _clip_error(e: BaseException, limit: int = 200) -> str:
    """Compact one-line error text (never contains secrets: cpa_client and
    local_judge exceptions are already key-free by construction)."""
    s = str(e).replace("\n", " ").replace("\r", " ").strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _as_bool(v: Any, field: str) -> bool:
    if isinstance(v, bool):
        return v
    if v == 0:
        return False
    if v == 1:
        return True
    raise ValueError(f"{field} must be a boolean")


class _JudgeError(RuntimeError):
    """Local judge endpoint unavailable (surfaced as a structured 502)."""


# ------------------------------------------------------------------ sessions
class SessionState:
    """Per-visitor RAM-only state. Never persisted, never logged."""

    __slots__ = ("lock", "engines", "episode_counter",
                 "cumulative_cost_usd", "history")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.engines: dict[str, JitRLEngine] = {}
        self.episode_counter = 0
        self.cumulative_cost_usd = 0.0
        self.history: list[dict] = []


# --------------------------------------------------------------------- app
class DemoApp:
    """Framework-free demo core (all JSON API logic; no http.server types).

    Dependency injection: judge_factory / cpa_client_factory /
    evaluator_factory are constructible-callables used to build the live
    backends; tests inject fakes (no network). Defaults build the real
    local_judge.JudgeClient / harness.cpa_client.CPAExecClient /
    harness.evaluator.Evaluator lazily, once.
    """

    def __init__(
        self,
        *,
        online: bool = False,
        max_session_usd: float = DEFAULT_MAX_SESSION_USD,
        judge_factory=None,
        cpa_client_factory=None,
        evaluator_factory=None,
    ) -> None:
        self.online = bool(online)
        self.max_session_usd = float(max_session_usd)
        self._judge_factory = judge_factory
        self._cpa_client_factory = cpa_client_factory
        self._evaluator_factory = evaluator_factory
        self._judge = None
        self._cpa_client = None
        self._evaluator = None
        self._backend_lock = threading.RLock()
        self._sessions: dict[str, SessionState] = {}
        self._sessions_lock = threading.Lock()
        self._data_cache: tuple[Any, dict] | None = None
        self._tasks_cache: tuple[Any, dict] | None = None
        self._health_cache: tuple[float, dict | None] = (0.0, None)

    # ------------------------------------------------------------ backends
    def _get_judge(self):
        with self._backend_lock:
            if self._judge is None:
                if self._judge_factory is not None:
                    self._judge = self._judge_factory()
                else:
                    from local_judge.client import JudgeClient
                    self._judge = JudgeClient()
            return self._judge

    def _get_cpa_client(self):
        with self._backend_lock:
            if self._cpa_client is None:
                if self._cpa_client_factory is not None:
                    self._cpa_client = self._cpa_client_factory()
                else:
                    from harness.cpa_client import CPAExecClient
                    self._cpa_client = CPAExecClient()
            return self._cpa_client

    def _get_evaluator(self):
        with self._backend_lock:
            if self._evaluator is None:
                if self._evaluator_factory is not None:
                    self._evaluator = self._evaluator_factory(self._cpa_client)
                else:
                    from harness.evaluator import Evaluator
                    from harness.pricing import cost_from_usage
                    client = self._get_cpa_client()
                    self._evaluator = Evaluator(
                        client=client,
                        price_of_call=lambda m, u: cost_from_usage(m, u),
                    )
            return self._evaluator

    def _judge_call(self, message: str, memory_block: str | None = None) -> dict:
        judge = self._get_judge()
        try:
            return judge.judge(message, memory_block=memory_block)
        except Exception as e:  # noqa: BLE001 - surfaced as structured 502
            raise _JudgeError(f"judge call failed: {_clip_error(e)}") from None

    # ------------------------------------------------------------- sessions
    def _get_session(self, session_id: str) -> SessionState:
        with self._sessions_lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                sess = SessionState()
                self._sessions[session_id] = sess
            return sess

    def _engine_for(self, session: SessionState, arm: str) -> JitRLEngine:
        """Fresh engine per (session, arm); config seed 42 (C0: min_neighbors=1,
        C1: min_neighbors=3)."""
        eng = session.engines.get(arm)
        if eng is None:
            eng = JitRLEngine(config=JitRLConfig(
                seed=ENGINE_SEED, min_neighbors=C_MIN_NEIGHBORS[arm]))
            session.engines[arm] = eng
        return eng

    def handle_reset(self, body: dict) -> tuple[int, dict]:
        session_id = body.get("session_id") if isinstance(body, dict) else None
        if not isinstance(session_id, str) or not (1 <= len(session_id) <= 128):
            return 400, {"error": "invalid_session_id"}
        with self._sessions_lock:
            existed = self._sessions.pop(session_id, None) is not None
        return 200, {"ok": True, "session_id": session_id, "cleared": existed}

    # ----------------------------------------------------------- read-only API
    @staticmethod
    def _load_json_file(path: Path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def _load_task_rows(self) -> list[dict]:
        rows: list[dict] = []
        try:
            with open(TASKS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        except OSError:
            pass
        return rows

    @staticmethod
    def _task_files_stamp() -> tuple:
        stamp = []
        for p in list(DATA_FILES.values()) + [TASKS_FILE]:
            try:
                stamp.append((str(p), p.stat().st_mtime_ns))
            except OSError:
                stamp.append((str(p), None))
        return tuple(stamp)

    def get_data(self) -> dict:
        """Merged frozen aggregates, read VERBATIM (metrics never recomputed)."""
        stamp = self._task_files_stamp()
        if self._data_cache is not None and self._data_cache[0] == stamp:
            return self._data_cache[1]
        payload = {
            "main": self._load_json_file(DATA_FILES["main"]),
            "c1": self._load_json_file(DATA_FILES["c1"]),
            "ablation": self._load_json_file(DATA_FILES["ablation"]),
            # fail-soft: absent continuation summary -> null
            "continuation": self._load_json_file(DATA_FILES["continuation"]),
            "tasks_meta": self._tasks_meta(),
            "sources": dict(DATA_SOURCES),
        }
        self._data_cache = (stamp, payload)
        return payload

    def _tasks_meta(self) -> dict:
        rows = self._load_task_rows()
        main_rows = [r for r in rows if r.get("family") != "continuation"]
        probes = [r for r in rows if r.get("family") == "continuation"]

        def gt(rs):
            return dict(sorted(Counter(r.get("gt_tier") for r in rs).items()))

        return {
            "n_total": len(rows),
            "n_main": len(main_rows),
            "n_probes": len(probes),
            "families": dict(sorted(Counter(r.get("family") for r in main_rows).items())),
            "gt_distribution": gt(rows),
            "main_gt_distribution": gt(main_rows),
            "probe_gt_distribution": gt(probes),
        }

    def get_tasks(self) -> dict:
        stamp = self._task_files_stamp()
        if self._tasks_cache is not None and self._tasks_cache[0] == stamp:
            return self._tasks_cache[1]
        rows = self._load_task_rows()

        def slim(r: dict) -> dict:
            return {
                "id": r.get("id"),
                "family": r.get("family"),
                "gt_tier": r.get("gt_tier"),
                "variant": r.get("variant"),
                "misleading_surface": r.get("misleading_surface"),
                "quality_checklist": r.get("quality_checklist") or [],
                "message": r.get("message", ""),
            }

        payload = {
            "tasks": [slim(r) for r in rows if r.get("family") != "continuation"],
            "probes": [slim(r) for r in rows if r.get("family") == "continuation"],
            "n_main": sum(1 for r in rows if r.get("family") != "continuation"),
            "n_probes": sum(1 for r in rows if r.get("family") == "continuation"),
        }
        self._tasks_cache = (stamp, payload)
        return payload

    # ------------------------------------------------------------- health
    def get_health(self) -> dict:
        now = time.monotonic()
        cached = self._health_cache[1]
        if cached is not None and now - self._health_cache[0] < HEALTH_CACHE_S:
            return cached
        # offline mode = zero network: the judge probe is skipped entirely
        judge_online = self._probe_judge() if self.online else False
        payload = {
            "online_mode": self.online,
            "judge_online": judge_online,
            "cpa_configured": self._cpa_configured(),
        }
        self._health_cache = (now, payload)
        return payload

    @staticmethod
    def _probe_judge() -> bool:
        """GET llama endpoint /v1/models (fallback /) with a 1.5s timeout.
        Endpoint comes from local_judge.config (JITRL_JUDGE_ENDPOINT respected)."""
        try:
            from local_judge import config as judge_config
        except Exception:  # noqa: BLE001
            return False
        base = judge_config.ENDPOINT.rstrip("/")
        for sub in ("/v1/models", "/"):
            try:
                req = urllib.request.Request(
                    base + sub, method="GET",
                    headers={"User-Agent": "jitrl-demo-health/1"})
                with urllib.request.urlopen(req, timeout=JUDGE_PROBE_TIMEOUT_S) as r:
                    r.read(64)
                    return True
            except Exception:  # noqa: BLE001
                continue
        return False

    @staticmethod
    def _cpa_configured() -> bool:
        """PilotDeck config file EXISTS check only (no contents, no keys)."""
        try:
            from harness.cpa_client import resolve_config_path
            return bool(resolve_config_path().is_file())
        except Exception:  # noqa: BLE001
            return False

    # ------------------------------------------------------------- decide
    def handle_decide(self, body: dict) -> tuple[int, dict]:
        if not isinstance(body, dict):
            return 400, {"error": "invalid_json_object"}
        if not self.online:
            return 503, {
                "error": "offline_mode",
                "fallback": "offline",
                "message": "demo server is running in offline mode; "
                           "restart with --online to enable live decisions",
            }
        arm = body.get("arm")
        if arm not in ARMS:
            return 400, {"error": "invalid_arm", "valid_arms": list(ARMS)}
        message = body.get("message")
        if (not isinstance(message, str) or not message.strip()
                or len(message) > MAX_MESSAGE_CHARS):
            return 400, {
                "error": "invalid_message",
                "message": f"message must be a non-empty string "
                           f"(<= {MAX_MESSAGE_CHARS} chars)",
            }
        try:
            execute = _as_bool(body.get("execute", False), "execute")
            evaluate = _as_bool(body.get("evaluate", False), "evaluate")
        except ValueError:
            return 400, {"error": "invalid_flags",
                         "message": "execute/evaluate must be booleans"}
        if evaluate and not execute:
            return 400, {
                "error": "evaluate_requires_execute",
                "message": "blind evaluation runs on the executed response; "
                           "set execute=true",
            }
        session_id = body.get("session_id")
        if session_id is None:
            session_id = uuid.uuid4().hex
        if not isinstance(session_id, str) or not (1 <= len(session_id) <= 128):
            return 400, {"error": "invalid_session_id"}
        task_id = body.get("task_id")
        if task_id is not None and not isinstance(task_id, str):
            return 400, {"error": "invalid_task_id"}

        session = self._get_session(session_id)
        with session.lock:
            if execute and session.cumulative_cost_usd >= self.max_session_usd:
                return 429, {
                    "error": "budget_exceeded",
                    "fallback": "budget",
                    "spent_usd": round(session.cumulative_cost_usd, 6),
                    "cap_usd": self.max_session_usd,
                    "message": "per-session live execution budget exhausted; "
                               "POST /api/reset to clear this session",
                }
            try:
                return self._decide_locked(
                    arm=arm, message=message, execute=execute,
                    evaluate=evaluate, session=session,
                    session_id=session_id, task_id=task_id)
            except _JudgeError as e:
                return 502, {
                    "error": "judge_unavailable",
                    "fallback": "judge",
                    "detail": str(e),
                    "message": "local llama.cpp judge endpoint unreachable; "
                               "start llama-server (see demo/README.md)",
                }

    # All arm logic below runs under the per-session lock.
    def _decide_locked(self, *, arm: str, message: str, execute: bool,
                       evaluate: bool, session: SessionState,
                       session_id: str, task_id: str | None) -> tuple[int, dict]:
        from harness.pricing import DEFAULT_MODEL, TIER_TO_MODEL, cost_from_usage

        z_base: dict[str, float] = {}
        base_choice: str | None = None
        chosen: str | None = None
        n_retrieved = 0
        top_similarity = 0.0
        retrieved: list[dict] = []
        V = 0.0
        Q: dict[str, float] = {t: 0.0 for t in TIERS}
        A_norm: dict[str, float] = {t: 0.0 for t in TIERS}
        z_mod: dict[str, float] | None = None
        z_prompt: dict[str, float] | None = None
        memory_block_chars: int | None = None
        gate_reason: str | None = None
        modulation_active = False
        judge_latency_ms = 0.0
        jitrl_latency_ms = 0.0
        engine: JitRLEngine | None = None
        pm_memory_source: str | None = None

        if arm == "A":
            # static routing: always reasoning (exec model CPA/glm-5.3)
            base_choice = chosen = "reasoning"

        elif arm == "B":
            jr = self._judge_call(message)
            judge_latency_ms = float(jr.get("latency_ms", 0.0) or 0.0)
            z_base = clamp_logits(jr["tier_logits"], -10.0, TIERS)
            base_choice = chosen = choose_tier(jr["tier_logits"])

        elif arm in ("C0", "C1"):
            engine = self._engine_for(session, arm)
            jr = self._judge_call(message)
            judge_latency_ms = float(jr.get("latency_ms", 0.0) or 0.0)
            t0 = time.perf_counter()
            decision = engine.decide(message, jr["tier_logits"])
            jitrl_latency_ms = (time.perf_counter() - t0) * 1000.0
            z_base = dict(decision.z_base)
            z_mod = dict(decision.z_mod)
            base_choice = decision.base_choice
            chosen = decision.chosen_tier
            n_retrieved = decision.n_retrieved
            top_similarity = decision.top_similarity
            V = decision.V
            Q = dict(decision.Q)
            A_norm = dict(decision.A_norm)
            gate_reason = decision.gate_reason
            modulation_active = decision.modulation_active
            # audit copy: the exact retrieval engine.decide() performed
            retrieved = self._audit_retrieval(engine, message)

        else:  # PM — live prompt-injection (mirrors T8-PM, write-free)
            from harness.run_ablation import render_memory_block
            c0_engine = self._engine_for(session, "C0")
            cfg = c0_engine.config
            pm_memory_source = "C0"
            t0 = time.perf_counter()
            matches = c0_engine.memory.retrieve(
                intent_class(message),
                frozenset(task_signature(message)),
                k=cfg.k, threshold=cfg.jaccard_threshold,
            )
            n_retrieved = len(matches)
            top_similarity = matches[0][1] if matches else 0.0
            retrieved = [{"tier": e["tier"], "G": float(e["G"]),
                          "similarity": float(sim)} for e, sim in matches]
            block = (render_memory_block(matches)
                     if n_retrieved >= cfg.min_neighbors and n_retrieved > 0
                     else None)
            jitrl_latency_ms = (time.perf_counter() - t0) * 1000.0

            jr_base = self._judge_call(message)
            z_base = clamp_logits(jr_base["tier_logits"], cfg.z_min, TIERS)
            base_choice = choose_tier(z_base)
            if block is not None:
                jr_prompt = self._judge_call(message, memory_block=block)
                z_prompt = clamp_logits(jr_prompt["tier_logits"], cfg.z_min, TIERS)
                chosen = choose_tier(z_prompt)
                judge_latency_ms = float(jr_prompt.get("latency_ms", 0.0) or 0.0)
            else:
                z_prompt = dict(z_base)
                chosen = base_choice
                judge_latency_ms = float(jr_base.get("latency_ms", 0.0) or 0.0)
            memory_block_chars = len(block) if block is not None else 0
            if n_retrieved == 0:
                gate_reason = GATE_NO_NEIGHBORS
            elif n_retrieved < cfg.min_neighbors:
                gate_reason = GATE_BELOW_MIN_NEIGHBORS

        # ------------------------------------------------------- execution
        exec_model = TIER_TO_MODEL.get(chosen, DEFAULT_MODEL)
        exec_payload: dict | None = None
        exec_error: str | None = None
        exec_cost_usd: float | None = None
        exec_usage: dict | None = None
        content: str | None = None
        if execute:
            try:
                client = self._get_cpa_client()
                comp = client.completion(
                    exec_model,
                    [{"role": "user", "content": message}],
                    max_tokens=EXEC_MAX_TOKENS,
                    temperature=EXEC_TEMPERATURE,
                )
                content = comp.content
                exec_usage = dict(comp.usage) if comp.usage is not None else None
                if exec_usage is not None:
                    exec_cost_usd = cost_from_usage(exec_model, exec_usage)
                    session.cumulative_cost_usd += exec_cost_usd
                exec_payload = {
                    "content_preview": content[:PREVIEW_CHARS],
                    "usage": exec_usage,
                    "cost_usd": (round(exec_cost_usd, 8)
                                 if exec_cost_usd is not None else None),
                    "finish_reason": comp.finish_reason,
                }
            except Exception as e:  # noqa: BLE001 - trace still returned
                exec_error = _clip_error(e)
                exec_payload = None

        # ------------------------------------------------------ evaluation
        evaluation: dict | None = None
        eval_error: str | None = None
        reward: float | None = None
        if evaluate and exec_error is None and content is not None:
            try:
                ev = self._get_evaluator()
                checklist = self._checklist_for(message, task_id)
                res = ev.evaluate(
                    {"message": message, "quality_checklist": checklist}, content)
            except Exception as e:  # noqa: BLE001 - fail-safe, no memory write
                res = None
                eval_error = _clip_error(e)
            if res is None:
                # fail-safe (mirrors harness semantics): evaluation=null and
                # NO memory write. Only synthesize an error string when the
                # evaluator itself did not raise.
                if eval_error is None:
                    eval_error = "evaluator returned no usable result"
            else:
                evaluation = {
                    "quality_score": int(res.quality_score),
                    "verdict": str(res.verdict),
                }
                if exec_usage is not None:
                    from harness.real_rewards import RealQualityReward
                    reward, _breakdown = RealQualityReward().reward(
                        res.quality_score, exec_model, exec_usage)
                    # C0/C1 only: PM is write-free, A/B have no engine
                    if arm in ("C0", "C1") and engine is not None:
                        session.episode_counter += 1
                        engine.update_from_message(
                            message, chosen, reward, session.episode_counter)

        # ------------------------------------------------- session payload
        session.history.append({
            "arm": arm,
            "chosen_tier": chosen,
            "cost_usd": (round(exec_cost_usd, 8)
                         if exec_cost_usd is not None else None),
            "quality": evaluation["quality_score"] if evaluation else None,
        })
        if len(session.history) > HISTORY_CAP:
            del session.history[:len(session.history) - HISTORY_CAP]

        memory_counts = {
            a: len(session.engines[a].memory)
            for a in ("C0", "C1") if a in session.engines
        }
        if arm in ("C0", "C1"):
            memory_count = memory_counts.get(arm, 0)
        elif arm == "PM":
            memory_count = memory_counts.get("C0", 0)
        else:
            memory_count = 0

        payload: dict[str, Any] = {
            "arm": arm,
            "session_id": session_id,
            "z_base": z_base,
            "base_choice": base_choice,
            "chosen_tier": chosen,
            "n_retrieved": n_retrieved,
            "retrieved": retrieved,
            "top_similarity": top_similarity,
            "V": V,
            "Q": Q,
            "A_norm": A_norm,
            "gate_reason": gate_reason,
            "modulation_active": modulation_active,
            "flipped": chosen != base_choice,
            "exec_model": exec_model,
            "judge_latency_ms": round(judge_latency_ms, 1),
            "jitrl_latency_ms": round(jitrl_latency_ms, 3),
            "exec": exec_payload,
            "exec_error": exec_error,
            "evaluation": evaluation,
            "eval_error": eval_error,
            "reward": reward,
            "session": {
                "memory_count": memory_count,
                "memory_counts": memory_counts,
                "cumulative_cost_usd": round(session.cumulative_cost_usd, 8),
                "budget_cap_usd": self.max_session_usd,
                "history": list(session.history),
            },
        }
        if z_mod is not None:
            payload["z_mod"] = z_mod
        if z_prompt is not None:
            payload["z_prompt"] = z_prompt
        if memory_block_chars is not None:
            payload["memory_block_chars"] = memory_block_chars
            payload["pm_memory_source"] = pm_memory_source
        return 200, payload

    @staticmethod
    def _audit_retrieval(engine: JitRLEngine, message: str) -> list[dict]:
        """The exact retrieval engine.decide() performed (same params, same
        memory state) — for display only."""
        cfg = engine.config
        matches = engine.memory.retrieve(
            intent_class(message),
            frozenset(task_signature(message)),
            k=cfg.k, threshold=cfg.jaccard_threshold,
        )
        return [{"tier": e["tier"], "G": float(e["G"]), "similarity": float(sim)}
                for e, sim in matches]

    def _checklist_for(self, message: str, task_id: str | None) -> list:
        rows = self._load_task_rows()
        if task_id:
            for r in rows:
                if r.get("id") == task_id:
                    return r.get("quality_checklist") or []
        for r in rows:
            if r.get("message") == message:
                return r.get("quality_checklist") or []
        return []


# ------------------------------------------------------------ HTTP plumbing
class DemoRequestHandler(BaseHTTPRequestHandler):
    server_version = "JitRLDemo/1.0"
    protocol_version = "HTTP/1.1"

    @property
    def app(self) -> DemoApp:
        return self.server.app  # type: ignore[attr-defined,no-any-return]

    # -- minimal logging: method + path + status ONLY (never bodies) -------
    def log_request(self, code="-", size="-"):
        try:
            path = self.path.split("?", 1)[0]
            sys.stderr.write(f"{self.command} {path} {code}\n")
        except Exception:  # noqa: BLE001
            pass

    def log_error(self, format, *args):  # noqa: A002 - BaseHTTPRequestHandler API
        try:
            sys.stderr.write("ERROR " + (format % args)[:160] + "\n")
        except Exception:  # noqa: BLE001
            pass

    def log_message(self, format, *args):  # noqa: A002
        try:
            sys.stderr.write((format % args)[:160] + "\n")
        except Exception:  # noqa: BLE001
            pass

    # ---------------------------------------------------------------- GET
    def do_GET(self):  # noqa: N802 - http.server API
        try:
            self._route_get()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001
            self._safe_send_json(500, {"error": "internal_error",
                                       "detail": _clip_error(e)})

    def _route_get(self):
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._serve_static_file("index.html")
        elif path.startswith("/static/"):
            self._serve_static_file(urllib.parse.unquote(path[len("/static/"):]))
        elif path == "/api/data":
            self._send_json(200, self.app.get_data())
        elif path == "/api/tasks":
            self._send_json(200, self.app.get_tasks())
        elif path == "/api/health":
            self._send_json(200, self.app.get_health())
        elif path.startswith("/api/charts/"):
            self._serve_chart(path)
        else:
            self._send_json(404, {"error": "not_found", "path": path})

    # --------------------------------------------------------------- POST
    def do_POST(self):  # noqa: N802 - http.server API
        try:
            path = urllib.parse.urlparse(self.path).path
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if length > MAX_BODY_BYTES:
                self.close_connection = True  # unread body would corrupt keep-alive
                return self._send_json(413, {"error": "payload_too_large"})
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                body = json.loads(raw.decode("utf-8")) if raw else {}
            except (UnicodeDecodeError, json.JSONDecodeError):
                return self._send_json(400, {"error": "invalid_json"})
            if not isinstance(body, dict):
                return self._send_json(400, {"error": "invalid_json_object"})

            if path == "/api/decide":
                status, payload = self.app.handle_decide(body)
            elif path == "/api/reset":
                status, payload = self.app.handle_reset(body)
            else:
                status, payload = 404, {"error": "not_found", "path": path}
            self._send_json(status, payload)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001
            self._safe_send_json(500, {"error": "internal_error",
                                       "detail": _clip_error(e)})

    # ------------------------------------------------------------ static
    @staticmethod
    def _safe_name(name: str) -> bool:
        return (bool(_SAFE_NAME_RE.match(name)) and "/" not in name
                and "\\" not in name and ".." not in name)

    def _serve_static_file(self, name: str):
        if not self._safe_name(name):
            return self._send_json(404, {"error": "not_found"})
        base = STATIC_DIR.resolve()
        target = (STATIC_DIR / name).resolve()
        if not target.is_file() or not target.is_relative_to(base):
            return self._send_json(404, {"error": "not_found"})
        ctype = _STATIC_TYPES.get(target.suffix.lower(),
                                  "application/octet-stream")
        self._send_bytes(200, target.read_bytes(), ctype, cacheable=True)

    def _serve_chart(self, path: str):
        parts = [urllib.parse.unquote(p) for p in path.split("/") if p]
        # expected: ["api", "charts", <group>, <name>.svg]
        if len(parts) != 4:
            return self._send_json(404, {"error": "not_found"})
        _, _, group, name = parts
        base = CHART_GROUPS.get(group)
        if base is None:
            return self._send_json(
                404, {"error": "unknown_chart_group",
                      "valid_groups": sorted(CHART_GROUPS)})
        if (not self._safe_name(name) or not name.lower().endswith(".svg")):
            return self._send_json(404, {"error": "not_found"})
        base_resolved = base.resolve()
        target = (base / name).resolve()
        if not target.is_file() or not target.is_relative_to(base_resolved):
            return self._send_json(404, {"error": "not_found"})
        self._send_bytes(200, target.read_bytes(),
                         "image/svg+xml; charset=utf-8", cacheable=True)

    # ------------------------------------------------------------ sending
    def _send_json(self, status: int, payload: dict):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, data, "application/json; charset=utf-8")

    def _send_bytes(self, status: int, data: bytes, ctype: str,
                    cacheable: bool = False):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control",
                         "max-age=60" if cacheable else "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _safe_send_json(self, status: int, payload: dict):
        try:
            self._send_json(status, payload)
        except Exception:  # noqa: BLE001 - socket already broken
            pass


class DemoHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, app: DemoApp):
        super().__init__(address, DemoRequestHandler)
        self.app = app


# ------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(
        prog="python -m demo.server",
        description="JitRL Router Web Demo (offline-first, stdlib only)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help=f"TCP port (default {DEFAULT_PORT})")
    ap.add_argument("--host", default=DEFAULT_HOST,
                    help=f"bind address (default {DEFAULT_HOST})")
    ap.add_argument("--online", action="store_true",
                    help="enable the live decision API (default: offline, "
                         "zero network)")
    ap.add_argument("--max-session-usd", type=float, default=DEFAULT_MAX_SESSION_USD,
                    metavar="USD",
                    help="per-session cumulative live execution spend cap "
                         f"(default {DEFAULT_MAX_SESSION_USD:.2f})")
    args = ap.parse_args(argv)

    app = DemoApp(online=args.online, max_session_usd=args.max_session_usd)
    httpd = DemoHTTPServer((args.host, args.port), app)
    mode = ("ONLINE  live decision API enabled"
            if args.online else "OFFLINE zero network; experiment panels "
                                "fully functional, /api/decide returns 503")
    print("JitRL Router Web Demo", flush=True)
    print(f"  mode    : {mode}", flush=True)
    print(f"  address : http://{args.host}:{args.port}", flush=True)
    print(f"  budget  : ${args.max_session_usd:.2f} per session "
          "(live execution spend cap)", flush=True)
    print("  data    : frozen aggregates under eval/ (read-only, verbatim)",
          flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
