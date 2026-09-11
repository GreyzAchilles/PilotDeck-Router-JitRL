"""Real-episode A/B/C harness (Session 2 / Workstream A).

Pipeline per (episode, task):
    user task -> local Judge (A skips) -> tier decision (A/B/C)
    -> CPA execution (tier -> D14 model) -> blind Evaluator (gpt-5.6-sol)
    -> real reward (0.6*quality + 0.3*cost_saving) -> C memory write

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

from jitrl_core import JitRLConfig, JitRLEngine
from jitrl_core.policy import choose_tier
from jitrl_core.state import intent_class, task_signature
from harness import pricing
from harness.cpa_client import CPAExecClient
from harness.evaluator import Evaluator
from harness.pricing import DEFAULT_MODEL, TIER_TO_MODEL, cost_from_usage
from harness.real_rewards import RealQualityReward
from harness.run import build_judge, load_tasks

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


def api_model(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def build_cpa_client(config_path: str | None = None) -> CPAExecClient:
    return CPAExecClient(config_path=config_path or None)


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
) -> dict:
    rec = {
        "episode_id": episode_id,
        "task_id": task["task_id"],
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
        "evaluator": (eval_result.to_record() if eval_result is not None else None),
        "eval_cost_usd": (eval_result.cost_usd if eval_result is not None else None),
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
              memory_size: int, config: JitRLConfig | None = None) -> dict:
    labeled = [r for r in records if r.get("gt_tier")]
    # tier accuracy: A participates with its ACTUAL executed tier (reasoning)
    accuracy = (
        sum(1 for r in labeled if r["chosen_tier"] == r["gt_tier"]) / len(labeled)
        if labeled else None
    )
    eval_ok = [r for r in records if r["evaluator"] is not None]
    qualities = [r["evaluator"]["quality_score"] for r in eval_ok]
    exec_costs = [r["exec_cost_usd"] for r in records if r["exec_cost_usd"] is not None]
    eval_costs = [r["eval_cost_usd"] for r in records if r["eval_cost_usd"] is not None]
    rewards = [r["reward"] for r in records if r["reward"] is not None]
    return {
        "mode": mode,
        "judge": judge_kind,
        "episodes": episodes,
        "n_records": len(records),
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
        v = r["evaluator"]["verdict"]
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
) -> dict:
    """Run real episodes. cpa_client/evaluator injectable for tests."""
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
        engine = JitRLEngine(config=config)
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

    records: list[dict] = []
    previous_tier: str | None = None

    for episode_id in range(1, episodes + 1):
        for task in tasks:
            message = task["message"]
            intent = intent_class(message)
            signature_tokens = sorted(task_signature(message))

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

            exec_model = TIER_TO_MODEL.get(chosen, DEFAULT_MODEL)
            exec_result = None
            exec_error: str | None = None
            eval_result = None
            eval_error: str | None = None
            reward = None
            breakdown = None
            exec_cost_usd = None
            memory_written = False

            try:
                exec_result = cpa_client.completion(
                    exec_model,
                    [{"role": "user", "content": message}],
                    max_tokens=max_tokens_exec,
                    temperature=temperature_exec,
                )
                exec_cost_usd = cost_from_usage(exec_model, exec_result.usage)
            except Exception as e:  # noqa: BLE001 - recorded, never fatal
                exec_error = f"{type(e).__name__}: {e}"[:300]

            if exec_result is not None:
                eval_result = evaluator.evaluate(task, exec_result.content)
                if eval_result is None:
                    eval_error = "evaluator failed (transport/parse/validate)"

            if eval_result is not None:
                reward, breakdown = reward_fn.reward(
                    eval_result.quality_score, exec_model, exec_result.usage)
                if mode == "C":
                    engine.learn(
                        intent=intent,
                        signature_tokens=signature_tokens,
                        tier=chosen,
                        reward=reward,
                        episode_id=episode_id,
                    )
                    memory_written = True

            records.append(make_record(
                episode_id=episode_id, task=task, intent=intent,
                signature_tokens=signature_tokens,
                n_retrieved=n_retrieved, top_similarity=top_sim,
                min_neighbors=min_neighbors,
                modulation_active=modulation_active, gate_reason=gate_reason,
                V=V, Q=Q, A_norm=A_norm, z_base=z_base, z_mod=z_mod,
                chosen_tier=chosen, route_mode=route_mode,
                base_choice=base_choice, flipped=flipped,
                exec_model=exec_model, exec_result=exec_result,
                exec_error=exec_error, exec_cost_usd=exec_cost_usd,
                eval_result=eval_result, eval_error=eval_error,
                reward=reward, reward_breakdown=breakdown,
                judge_latency_ms=latency, jitrl_latency_ms=jitrl_latency_ms,
                memory_written=memory_written,
            ))
            previous_tier = chosen

    memory_size = len(engine.memory) if engine is not None else 0
    summary = summarize(records, mode=mode, judge_kind=judge_kind,
                        episodes=episodes, memory_size=memory_size,
                        config=config)

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
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
