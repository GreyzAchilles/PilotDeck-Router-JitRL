"""Real-episode A/B/C harness (Session 2 / Workstream A).

Pipeline per (episode, task):
    user task -> local Judge (A skips) -> tier decision (A/B/C)
    -> CPA execution (tier -> D14 model) -> blind Evaluator (gpt-5.6-sol)
    -> real reward (0.6*quality + 0.3*cost_saving) -> C memory write

M1 unified trajectory interface (docs/optimization/MEMORY-LOOP-DECISIONS.md):
every episode is a length-1 Trajectory; TrajectoryEvaluator does the
terminal-only blind evaluation and update_memory_from_trajectory routes
the C-arm write through engine.learn_trajectory (identity credit, G=r).
Behavior is identical to the legacy inline chain; records additionally
carry additive traj_id / n_steps fields.

M2 multi-turn + arms: task rows sharing `traj_id` (ordered by
turn_index) form one trajectory — each turn is routed and executed with
the accumulated conversation (harness/multiturn.py). --arm selects the
mode-C memory update: T1 (default, M1-compatible terminal-only),
T2 (per-step trajectory review + discounted credit gamma=0.8 eta=0.5),
T3 (T2 + LifecycleMemory active/provisional/quarantine gating). Arms
never change routing, execution, or the failure policy.

Decision semantics (fixes the Session-1 risk "A accuracy always 0"):
    mode A: chosen_tier = "reasoning" (the tier actually executed via the
            D14 static default CPA/glm-5.3), route_mode = "static"
    mode B: chosen_tier = argmax(judge logits),           route_mode = "judge"
    mode C: chosen_tier = argmax(modulated logits),       route_mode = "jitrl"

Failure policy (directive):
    - Evaluator failure (transport / unparseable / invalid JSON) -> the
      record is written with eval_error set, reward=null, and C does NOT
      write memory for that episode.
    - Execution failure -> same policy (no evaluation, no memory).
    - Judge failure -> the run aborts (infrastructure problem, fail fast).

JitRL latency instrumentation: every record carries `jitrl_latency_ms`, the
wall time (perf_counter) of engine.decide() — the JitRL post-processing
(retrieval + value estimate + logit modulation) — measured in mode C only.
Modes A/B record 0.0 (documented convention mirroring judge_latency_ms=0.0
in mode A) because no JitRL post-processing runs there. It never includes
the 4-request Judge latency, which is reported separately as
judge_latency_ms. The summary exposes avg_jitrl_latency_ms.

C1 min-neighbor gate (S2-C1-PREREGISTRATION.md v1.0): when
n_retrieved < min_neighbors, mode C degrades to the native Judge
(z_mod = z_base, A_norm = 0, chosen = base_choice). Every record carries
min_neighbors, modulation_active and gate_reason ("no_neighbors" /
"below_min_neighbors" / null — A/B record the null sentinel since no
engine runs there); the summary carries config_snapshot,
modulated_records and gated_records. Default --min-neighbors 1 preserves
C0 semantics exactly; the C1 run passes 3 explicitly.

Security: records contain no credentials. The CPA key is read from the
PilotDeck config and only ever placed in the Authorization header.

CLI:
    python -m harness.run_real --mode C --tasks eval/tasks.jsonl \
        --task-ids T01,T05,T13 --judge llama --episodes 3 \
        --out logs/s2_c_calib.jsonl --memory-out logs/s2_c_calib_memory.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from jitrl_core import (
    DiscountedCreditAssigner,
    JitRLConfig,
    JitRLEngine,
    LifecycleMemory,
    LifecycleMemoryUpdater,
)
from jitrl_core.policy import choose_tier
from jitrl_core.state import intent_class, task_signature
from harness import pricing
from harness.cpa_client import CPAExecClient
from harness.evaluator import Evaluator, TrajectoryLevelEvaluator
from harness.multiturn import build_conversation, group_trajectories
from harness.pricing import (
    DEFAULT_MODEL,
    TIER_TO_MODEL,
    cost_from_usage,
    get_tier_to_model,
)
from harness.real_rewards import RealQualityReward
from harness.run import build_judge, load_tasks
from harness.trajectory import (
    Trajectory,
    TrajectoryEvaluator,
    TrajectoryStep,
    update_memory_from_trajectory,
)

JUDGE_ENDPOINT_ENV = "JITRL_JUDGE_ENDPOINT"
JUDGE_ENDPOINT_DEFAULT = "http://127.0.0.1:18080"
SIGNATURE_PREVIEW_TOKENS = 12
RESPONSE_PREVIEW_CHARS = 600
STATIC_TIER = "reasoning"   # D14: static default model CPA/glm-5.3 == reasoning tier

# Calibration finding (s2, 2026-09-11): CPA gateway models such as
# glm-5.3-flash emit `reasoning_content` (400+ tokens) BEFORE the visible
# `content`; at max_tokens=512 the reasoning preamble can exhaust the budget
# (finish_reason=length, content="") — observed on T09. 1024 leaves headroom
# for the answer after the preamble while keeping exec cost ~$0.001-0.003.
DEFAULT_MAX_TOKENS_EXEC = 1024
DEFAULT_MAX_TOKENS_EVAL = 300

# ------------------------------------------------------------- M2 arms
#: Memory-update arms (docs/optimization/MEMORY-LOOP-DECISIONS.md v1.1).
#: T1 is the M1-compatible default: terminal-only evaluation, single
#: scalar, identity credit — one memory entry per trajectory (the terminal
#: turn). T2/T3 run the M2 per-step trajectory review with discounted
#: credit (gamma=0.8, delayed weight 0.5); T3 additionally gates writes
#: through LifecycleMemory active/provisional/quarantine by evaluator
#: confidence. Arms change ONLY mode-C memory updates; routing, execution,
#: and evaluation inputs are identical across arms.
M2_GAMMA = 0.8
M2_DELAYED_WEIGHT = 0.5
ARM_SPECS = {
    "T1": {"eval_mode": "terminal", "credit": "identity", "lifecycle": False},
    "T2": {"eval_mode": "per_step", "credit": "discounted", "lifecycle": False},
    "T3": {"eval_mode": "per_step", "credit": "discounted", "lifecycle": True},
}


def build_engine(config: JitRLConfig, arm: str) -> JitRLEngine:
    """Mode-C engine per arm. T1 constructs JitRLEngine(config=config)
    exactly as before (bit-compatible, also for engine-mocking tests)."""
    spec = ARM_SPECS[arm]
    credit = (DiscountedCreditAssigner(M2_GAMMA, M2_DELAYED_WEIGHT)
              if spec["credit"] == "discounted" else None)
    if spec["lifecycle"]:
        return JitRLEngine(
            config=config,
            memory=LifecycleMemory(cap=config.memory_cap),
            credit_assigner=credit,
            memory_updater=LifecycleMemoryUpdater(),
        )
    if credit is not None:
        return JitRLEngine(config=config, credit_assigner=credit)
    return JitRLEngine(config=config)


def _route_one(mode, config, judge, engine, message, previous_tier) -> dict:
    """Route one turn (A/B/C). M1 routing math, extracted verbatim for the
    M2 multi-turn loop; single-turn behavior unchanged."""
    decision = None  # jitrl DecisionResult (mode C only)
    if mode == "A":
        chosen = STATIC_TIER
        route_mode = "static"
        z_base: dict = {}
        z_mod: dict = {}
        n_retrieved, top_sim, V = 0, 0.0, 0.0
        Q = {t: 0.0 for t in ("simple", "medium", "complex", "reasoning")}
        A_norm = dict(Q)
        base_choice, flipped = chosen, False
        latency = 0.0
        jitrl_latency_ms = 0.0  # no JitRL post-processing in A
        # no engine: gate never evaluated (schema-stable sentinels)
        min_neighbors, modulation_active, gate_reason = (
            config.min_neighbors, False, None)
    else:
        jr = judge.judge(message, previous_tier=previous_tier)
        logits = jr["tier_logits"]
        latency = float(jr["latency_ms"])
        if mode == "B":
            chosen = choose_tier(logits)
            route_mode = "judge"
            z_base = {t: max(float(logits.get(t, 0.0)), config.z_min) for t in
                      ("simple", "medium", "complex", "reasoning")}
            z_mod = dict(z_base)
            n_retrieved, top_sim, V = 0, 0.0, 0.0
            Q = {t: 0.0 for t in z_base}
            A_norm = dict(Q)
            base_choice, flipped = chosen, False
            jitrl_latency_ms = 0.0  # no JitRL post-processing in B
            # no engine: gate never evaluated (schema-stable sentinels)
            min_neighbors, modulation_active, gate_reason = (
                config.min_neighbors, False, None)
        else:  # C
            # time ONLY the JitRL post-processing; the 4-request judge
            # latency is already captured in `latency` above
            t_decide = time.perf_counter()
            decision = engine.decide(message, logits)
            jitrl_latency_ms = (time.perf_counter() - t_decide) * 1000.0
            chosen = decision.chosen_tier
            route_mode = "jitrl"
            z_base = decision.z_base
            z_mod = decision.z_mod
            n_retrieved = decision.n_retrieved
            top_sim = decision.top_similarity
            min_neighbors = decision.min_neighbors
            modulation_active = decision.modulation_active
            gate_reason = decision.gate_reason
            V = decision.V
            Q = decision.Q
            A_norm = decision.A_norm
            base_choice = decision.base_choice
            flipped = decision.flipped
    return {
        "chosen": chosen, "route_mode": route_mode,
        "z_base": z_base, "z_mod": z_mod,
        "n_retrieved": n_retrieved, "top_similarity": top_sim,
        "min_neighbors": min_neighbors,
        "modulation_active": modulation_active, "gate_reason": gate_reason,
        "V": V, "Q": Q, "A_norm": A_norm,
        "base_choice": base_choice, "flipped": flipped,
        "latency": latency, "jitrl_latency_ms": jitrl_latency_ms,
        "decision": decision,
    }


def _exec_one(cpa_client, exec_model, messages, max_tokens, temperature):
    """Execute one turn. Any failure is recorded, never fatal."""
    exec_result = None
    exec_error: str | None = None
    exec_cost_usd: float | None = None
    try:
        exec_result = cpa_client.completion(
            exec_model, messages,
            max_tokens=max_tokens, temperature=temperature,
        )
        exec_cost_usd = cost_from_usage(exec_model, exec_result.usage)
    except Exception as e:  # noqa: BLE001 - recorded, never fatal
        exec_error = f"{type(e).__name__}: {e}"[:300]
    return exec_result, exec_error, exec_cost_usd


def api_model(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def build_cpa_client(config_path: str | None = None) -> CPAExecClient:
    return CPAExecClient(config_path=config_path or None)


def _build_client_for(provider_id: str, config_path: str | None):
    from harness.cpa_client import CPAExecClient, load_cpa_provider
    return CPAExecClient(
        provider=load_cpa_provider(provider_id=provider_id,
                                   config_path=config_path))


class ExecClientRouter:
    """Route exec calls by model provider prefix (S4-M2 switched map).

    The base client serves its own provider; models carrying a DIFFERENT
    "provider/model" prefix go to a client for that provider (built lazily
    from the PilotDeck config via _build_client_for, or injected for tests
    through `extra`). A base client without a provider_id attribute (test
    fakes) is treated as universal and serves everything. The default spec
    map is single-provider and never routes away from the base, so legacy
    behavior is unchanged. The evaluator stays on the base client — the
    switched map therefore requires the base to be the CPA provider
    (gpt-5.6-sol is CPA-only; do not set JITRL_PROVIDER_ID with it)."""

    def __init__(self, base, config_path: str | None = None,
                 extra: dict | None = None):
        self._base = base
        self._base_pid = getattr(base, "provider_id", None)
        self._config_path = config_path
        self._extra: dict = dict(extra) if extra else {}

    def for_model(self, model: str):
        pid = model.split("/", 1)[0] if "/" in model else None
        if self._base_pid is None or pid is None or pid == self._base_pid:
            return self._base
        if pid not in self._extra:
            self._extra[pid] = _build_client_for(pid, self._config_path)
        return self._extra[pid]


def make_record(
    *,
    episode_id: int,
    task: dict,
    intent: str,
    signature_tokens: list[str],
    n_retrieved: int,
    top_similarity: float,
    min_neighbors: int,
    modulation_active: bool,
    gate_reason: str | None,
    V: float,
    Q: dict,
    A_norm: dict,
    z_base: dict,
    z_mod: dict,
    chosen_tier: str,
    route_mode: str,
    base_choice: str,
    flipped: bool,
    exec_model: str,
    exec_result=None,
    exec_error: str | None,
    exec_cost_usd: float | None,
    eval_result=None,
    eval_error: str | None,
    reward: float | None,
    reward_breakdown: dict | None,
    judge_latency_ms: float,
    jitrl_latency_ms: float,
    memory_written: bool,
    traj_id: str | None = None,
    n_steps: int = 1,
    turn_index: int | None = None,
    n_turns: int | None = None,
    arm: str | None = None,
    eval_record: dict | None = None,
    eval_cost_usd: float | None = None,
) -> dict:
    rec = {
        "episode_id": episode_id,
        "task_id": task["task_id"],
        # M1 trajectory fields (additive): a single-turn episode is a
        # length-1 trajectory; records stay schema-compatible.
        "traj_id": traj_id,
        "n_steps": n_steps,
        # M2 multi-turn fields (additive): position within the trajectory
        # and the memory-update arm that ran (records exist for A/B too).
        "turn_index": turn_index,
        "n_turns": n_turns,
        "arm": arm,
        "family": task.get("family"),
        "gt_tier": task.get("gt_tier"),
        "intent_class": intent,
        "signature_preview": signature_tokens[:SIGNATURE_PREVIEW_TOKENS],
        "n_retrieved": n_retrieved,
        "top_similarity": top_similarity,
        # C1 min-neighbor gate fields (present in ALL modes for schema
        # stability). A/B: modulation never runs -> modulation_active=False,
        # gate_reason=None; min_neighbors records the run's configured
        # threshold. C: gate_reason is "no_neighbors" / "below_min_neighbors"
        # when the gate suppressed modulation, else None.
        "min_neighbors": min_neighbors,
        "modulation_active": modulation_active,
        "gate_reason": gate_reason,
        "V": V,
        "Q": Q,
        "A_norm": A_norm,
        "z_base": z_base,
        "z_mod": z_mod,
        "chosen_tier": chosen_tier,
        "route_mode": route_mode,
        "base_choice": base_choice,
        "flipped": flipped,
        "exec_model": exec_model,
        "exec_response_preview": (exec_result.content[:RESPONSE_PREVIEW_CHARS]
                                  if exec_result is not None else None),
        "exec_finish_reason": (exec_result.finish_reason
                               if exec_result is not None else None),
        "exec_usage": (dict(exec_result.usage) if exec_result is not None else None),
        "exec_cost_usd": exec_cost_usd,
        "exec_latency_ms": (round(exec_result.latency_ms, 1)
                            if exec_result is not None else None),
        "exec_error": exec_error,
        # terminal mode: per-response blind evaluation (eval_record=None);
        # per-step mode (T2/T3): per-turn step verdict merged with the
        # episode summary (eval_result=None), quality_score = local_quality.
        "evaluator": (eval_record if eval_record is not None
                      else (eval_result.to_record()
                            if eval_result is not None else None)),
        "eval_cost_usd": (eval_cost_usd if eval_cost_usd is not None
                          else (eval_result.cost_usd
                                if eval_result is not None else None)),
        "eval_error": eval_error,
        "reward": reward,
        "reward_breakdown": reward_breakdown,
        "judge_latency_ms": judge_latency_ms,
        # engine.decide() wall time; mode C only — A/B record the documented
        # 0.0 sentinel (no JitRL post-processing runs there). Never includes
        # the 4-request judge latency (that is judge_latency_ms above).
        "jitrl_latency_ms": jitrl_latency_ms,
        "memory_written": memory_written,
    }
    return rec


def summarize(records: list[dict], *, mode: str, judge_kind: str, episodes: int,
              memory_size: int, config: JitRLConfig | None = None,
              memory=None) -> dict:
    labeled = [r for r in records if r.get("gt_tier")]
    # tier accuracy: A participates with its ACTUAL executed tier (reasoning)
    accuracy = (
        sum(1 for r in labeled if r["chosen_tier"] == r["gt_tier"]) / len(labeled)
        if labeled else None
    )
    eval_ok = [r for r in records if r["evaluator"] is not None]
    qualities = [r["evaluator"]["quality_score"] for r in eval_ok
                 if "quality_score" in r["evaluator"]]
    exec_costs = [r["exec_cost_usd"] for r in records if r["exec_cost_usd"] is not None]
    eval_costs = [r["eval_cost_usd"] for r in records if r["eval_cost_usd"] is not None]
    rewards = [r["reward"] for r in records if r["reward"] is not None]
    return {
        "mode": mode,
        "judge": judge_kind,
        "episodes": episodes,
        "n_records": len(records),
        # M2 observability (additive): distinct trajectories and, for the
        # T3 lifecycle arm, active/provisional/quarantine entry counts.
        "n_trajectories": len({r.get("traj_id") for r in records}),
        "memory_status_counts": (memory.status_counts()
                                 if hasattr(memory, "status_counts") else None),
        # C1 gate observability: full hyperparameter snapshot (includes
        # min_neighbors), plus how many records actually modulated vs were
        # gated by the min-neighbor rule (gate_reason set) per mode.
        "config_snapshot": (config.snapshot() if config is not None else None),
        "modulated_records": sum(1 for r in records if r.get("modulation_active")),
        "gated_records": sum(1 for r in records
                             if r.get("gate_reason") is not None),
        "accuracy_vs_gt": accuracy,
        "flips": sum(1 for r in records if r["flipped"]),
        "memory_size": memory_size,
        "memory_written": sum(1 for r in records if r["memory_written"]),
        "eval_success": f"{len(eval_ok)}/{len(records)}",
        "eval_success_rate": (len(eval_ok) / len(records)) if records else None,
        "avg_quality_score": (sum(qualities) / len(qualities)) if qualities else None,
        "verdicts": _count_verdicts(eval_ok),
        "avg_exec_cost_usd": (sum(exec_costs) / len(exec_costs)) if exec_costs else None,
        "total_exec_cost_usd": sum(exec_costs) if exec_costs else 0.0,
        "total_eval_cost_usd": sum(eval_costs) if eval_costs else 0.0,
        "avg_reward_real": (sum(rewards) / len(rewards)) if rewards else None,
        "avg_judge_latency_ms": (
            sum(r["judge_latency_ms"] for r in records) / len(records)
            if records else None),
        # engine.decide() wall time, mode C only; 0.0 for A/B runs
        # (documented sentinel — no JitRL post-processing runs there).
        "avg_jitrl_latency_ms": (
            sum(r["jitrl_latency_ms"] for r in records) / len(records)
            if records else None),
    }


def _count_verdicts(eval_ok: list[dict]) -> dict:
    out = {"useful": 0, "harmful": 0, "neutral": 0}
    for r in eval_ok:
        v = r["evaluator"].get("verdict")   # absent in per-step records
        if v:
            out[v] = out.get(v, 0) + 1
    return out


def run(
    *,
    mode: str,
    tasks_path: str | Path,
    judge_kind: str = "mock",
    out_path: str | Path | None = None,
    memory_out_path: str | Path | None = None,
    threshold: float | None = None,
    seed: int | None = None,
    min_neighbors: int | None = None,
    episodes: int = 1,
    task_ids: list[str] | None = None,
    config_path: str | None = None,
    max_tokens_exec: int = DEFAULT_MAX_TOKENS_EXEC,
    max_tokens_eval: int = DEFAULT_MAX_TOKENS_EVAL,
    temperature_exec: float = 0.7,
    cpa_client: CPAExecClient | None = None,
    evaluator: Evaluator | None = None,
    arm: str = "T1",
    tier_map: str = "spec",
    exec_clients: dict | None = None,
) -> dict:
    """Run real episodes. cpa_client/evaluator injectable for tests.

    arm (M2): T1 = M1-compatible terminal-only update (default);
    T2 = per-step trajectory review + discounted credit;
    T3 = T2 + LifecycleMemory confidence gating.

    tier_map: "spec" (default, frozen single-provider map) or "switched"
    (S4-M2 amendment v1.1 cross-provider map; exec routed per model
    prefix via ExecClientRouter, evaluator stays on the base client).
    exec_clients: injectable {provider_id: client} extras for tests."""
    if arm not in ARM_SPECS:
        raise ValueError(f"unknown arm: {arm!r}")
    tier_to_model = get_tier_to_model(tier_map)
    tasks = load_tasks(tasks_path)
    if task_ids:
        wanted = set(task_ids)
        tasks = [t for t in tasks if t["task_id"] in wanted]
        missing = wanted - {t["task_id"] for t in tasks}
        if missing:
            raise SystemExit(f"task ids not found: {sorted(missing)}")
    if not tasks:
        raise SystemExit(f"no tasks loaded from {tasks_path}")

    cfg_kwargs: dict = {"seed": seed if seed is not None else 42}
    if threshold is not None:
        cfg_kwargs["jaccard_threshold"] = threshold
    if min_neighbors is not None:
        # None -> config default 1 (exact C0 semantics); C1 passes 3.
        cfg_kwargs["min_neighbors"] = min_neighbors
    config = JitRLConfig(**cfg_kwargs)

    engine: JitRLEngine | None = None
    judge = None
    if mode == "C":
        engine = build_engine(config, arm)
    if mode in ("B", "C"):
        judge = build_judge(judge_kind, config.seed)

    if cpa_client is None:
        cpa_client = build_cpa_client(config_path)
    if evaluator is None:
        evaluator = Evaluator(
            client=cpa_client, max_tokens=max_tokens_eval,
            price_of_call=lambda m, u: cost_from_usage(m, u),
        )
    else:
        evaluator.max_tokens = max_tokens_eval

    reward_fn = RealQualityReward()
    traj_evaluator = TrajectoryEvaluator(
        evaluator=evaluator, reward_fn=reward_fn,
        mode=ARM_SPECS[arm]["eval_mode"],
        traj_evaluator=TrajectoryLevelEvaluator(
            client=cpa_client,
            price_of_call=lambda m, u: cost_from_usage(m, u),
        ),
    )
    exec_router = ExecClientRouter(cpa_client, config_path=config_path,
                                   extra=exec_clients)

    records: list[dict] = []
    previous_tier: str | None = None
    groups = group_trajectories(tasks)

    for episode_id in range(1, episodes + 1):
        for group in groups:
            traj_steps: list[TrajectoryStep] = []
            turn_ctx: list[dict] = []
            for turn_index, task in enumerate(group):
                message = task["message"]
                intent = intent_class(message)
                signature_tokens = sorted(task_signature(message))

                r = _route_one(mode, config, judge, engine, message,
                               previous_tier)
                chosen = r["chosen"]
                exec_model = tier_to_model.get(chosen, DEFAULT_MODEL)
                exec_result, exec_error, exec_cost_usd = _exec_one(
                    exec_router.for_model(exec_model), exec_model,
                    build_conversation(traj_steps, message),
                    max_tokens_exec, temperature_exec)

                step = TrajectoryStep(
                    task=task, message=message, intent=intent,
                    signature_tokens=signature_tokens, chosen_tier=chosen,
                    route_mode=r["route_mode"], exec_model=exec_model,
                    previous_tier=previous_tier, decision=r["decision"],
                    exec_result=exec_result, exec_error=exec_error,
                    exec_cost_usd=exec_cost_usd,
                    judge_latency_ms=r["latency"],
                    jitrl_latency_ms=r["jitrl_latency_ms"],
                )
                traj_steps.append(step)
                turn_ctx.append({**r, "task": task, "intent": intent,
                                 "signature_tokens": signature_tokens,
                                 "exec_model": exec_model})
                previous_tier = chosen

            # --- unified trajectory interface ---------------------------
            # Single-turn episodes are length-1 groups (M1 behavior); rows
            # sharing a traj_id form one multi-turn trajectory (M2). The
            # arm-configured TrajectoryEvaluator runs once per trajectory;
            # failure policy unchanged: exec/eval failure -> no memory.
            group_traj_key = group[0].get("traj_id") or group[0]["task_id"]
            traj = Trajectory(traj_id=f"{episode_id}:{group_traj_key}",
                              episode_id=episode_id, steps=traj_steps)
            report = traj_evaluator.evaluate(traj)
            if mode == "C" and report.ok:
                update_memory_from_trajectory(engine, traj, report)

            for i, ctx in enumerate(turn_ctx):
                sr = report.steps[i]
                # trajectory-level eval cost is booked once, on the last
                # turn, so per-record sums never double-count it
                eval_cost = (sr.trajectory_result.cost_usd
                             if (sr.trajectory_result is not None
                                 and i == len(turn_ctx) - 1) else None)
                records.append(make_record(
                    episode_id=episode_id, task=ctx["task"],
                    intent=ctx["intent"],
                    signature_tokens=ctx["signature_tokens"],
                    n_retrieved=ctx["n_retrieved"],
                    top_similarity=ctx["top_similarity"],
                    min_neighbors=ctx["min_neighbors"],
                    modulation_active=ctx["modulation_active"],
                    gate_reason=ctx["gate_reason"],
                    V=ctx["V"], Q=ctx["Q"], A_norm=ctx["A_norm"],
                    z_base=ctx["z_base"], z_mod=ctx["z_mod"],
                    chosen_tier=ctx["chosen"], route_mode=ctx["route_mode"],
                    base_choice=ctx["base_choice"], flipped=ctx["flipped"],
                    exec_model=ctx["exec_model"],
                    exec_result=traj_steps[i].exec_result,
                    exec_error=traj_steps[i].exec_error,
                    exec_cost_usd=traj_steps[i].exec_cost_usd,
                    eval_result=sr.eval_result, eval_error=sr.eval_error,
                    reward=sr.reward, reward_breakdown=sr.breakdown,
                    judge_latency_ms=ctx["latency"],
                    jitrl_latency_ms=ctx["jitrl_latency_ms"],
                    memory_written=(mode == "C" and report.ok
                                    and sr.reward is not None),
                    traj_id=traj.traj_id, n_steps=len(traj.steps),
                    turn_index=i, n_turns=len(group), arm=arm,
                    eval_record=sr.eval_record(), eval_cost_usd=eval_cost,
                ))

    memory_size = len(engine.memory) if engine is not None else 0
    summary = summarize(records, mode=mode, judge_kind=judge_kind,
                        episodes=episodes, memory_size=memory_size,
                        config=config,
                        memory=(engine.memory if engine is not None else None))

    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.write(json.dumps({"_summary": summary}, ensure_ascii=False) + "\n")

    if memory_out_path:
        mout = Path(memory_out_path)
        mout.parent.mkdir(parents=True, exist_ok=True)
        entries = engine.memory.entries() if engine is not None else []
        with open(mout, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(
        f"[run_real] mode={mode} route={summary['mode']} judge={judge_kind} "
        f"episodes={episodes} tasks={len(tasks)} records={len(records)}"
    )
    print(f"[run_real] accuracy_vs_gt={summary['accuracy_vs_gt']} "
          f"flips={summary['flips']} memory={summary['memory_size']} "
          f"(written {summary['memory_written']})")
    print(f"[run_real] min_neighbors={summary['config_snapshot']['min_neighbors']} "
          f"modulated={summary['modulated_records']} "
          f"gated={summary['gated_records']}")
    print(f"[run_real] eval_success={summary['eval_success']} "
          f"avg_quality={summary['avg_quality_score']} verdicts={summary['verdicts']}")
    print(f"[run_real] exec_cost=${summary['total_exec_cost_usd']:.6f} "
          f"eval_cost=${summary['total_eval_cost_usd']:.6f} "
          f"avg_reward={summary['avg_reward_real']} "
          f"avg_jitrl_latency_ms={summary['avg_jitrl_latency_ms']}")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="harness.run_real")
    ap.add_argument("--mode", required=True, choices=["A", "B", "C"])
    ap.add_argument("--tasks", default="eval/tasks.jsonl")
    ap.add_argument("--task-ids", default=None,
                    help="comma-separated subset of task ids (e.g. T01,T05,T13)")
    ap.add_argument("--judge", default="llama", choices=["mock", "llama"])
    ap.add_argument("--out", default=None, help="episode record JSONL output")
    ap.add_argument("--memory-out", default=None, help="C memory JSONL output")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--min-neighbors", type=int, default=None,
                    help="C1 min-neighbor gate: minimum retrieved neighbors "
                         "required before logit modulation (default 1 = exact "
                         "C0 semantics; the C1 run passes 3)")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--arm", default="T1", choices=["T1", "T2", "T3"],
                    help="memory-update arm (M2): T1 terminal-only default, "
                         "T2 per-step + discounted credit, "
                         "T3 = T2 + lifecycle confidence gating")
    ap.add_argument("--tier-map", default="spec", choices=["spec", "switched"],
                    help="tier->model map: spec (frozen single-provider) or "
                         "switched (S4-M2 amendment v1.1 cross-provider: "
                         "simple/reasoning on provider1, medium/complex on "
                         "CPA; evaluator stays on CPA/gpt-5.6-sol — do not "
                         "set JITRL_PROVIDER_ID with this map)")
    ap.add_argument("--config", default=None,
                    help="pilotdeck.yaml path (default: PILOTDECK_CONFIG_PATH / "
                         "PILOT_HOME / ~/.pilotdeck/pilotdeck.yaml)")
    ap.add_argument("--max-tokens-exec", type=int, default=DEFAULT_MAX_TOKENS_EXEC)
    ap.add_argument("--max-tokens-eval", type=int, default=DEFAULT_MAX_TOKENS_EVAL)
    ap.add_argument("--temperature-exec", type=float, default=0.7)
    args = ap.parse_args(argv)

    task_ids = [t.strip() for t in args.task_ids.split(",")] if args.task_ids else None
    run(
        mode=args.mode,
        tasks_path=args.tasks,
        judge_kind=args.judge,
        out_path=args.out,
        memory_out_path=args.memory_out,
        threshold=args.threshold,
        seed=args.seed,
        min_neighbors=args.min_neighbors,
        episodes=args.episodes,
        task_ids=task_ids,
        config_path=args.config,
        max_tokens_exec=args.max_tokens_exec,
        max_tokens_eval=args.max_tokens_eval,
        temperature_exec=args.temperature_exec,
        arm=args.arm,
        tier_map=args.tier_map,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
