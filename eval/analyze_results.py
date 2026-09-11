"""Aggregate analysis of Session-2 formal full-run results (groups A/B/C).

Formal inputs (directive):
    A = logs/s2_full_A_main24.jsonl                    (static reasoning tier)
    B = logs/s2_full_B_family_order.jsonl              (judge-only routing;
        latest 24/24 baseline executed in the SAME family-blocked task
        order as A and C)
    C = logs/s2_full_C_main24_retry2.jsonl             (JitRL modulation;
        THIRD complete full-sequence run, 24/24 eval success)
    The first two C runs (s2_full_C_main24.jsonl, s2_full_C_main24_rerun.jsonl)
    are audited for failures ONLY and never enter metrics.

Hard consistency assertion (directive):
    A, B and C must contain the same task_ids in EXACTLY the same order
    (the shared family-blocked sequence). The analysis refuses to run
    (SystemExit) if any order differs.

Scope rules enforced by design:
    - reads ONLY the given JSONL logs (+ optional C memory JSONL);
      no credentials, no CPA calls, no docs writes.
    - A's tier-vs-GT match rate is reported as "static_tier_gt_coincidence",
      NOT as a routing accuracy (A is a static baseline; the number is just
      the fraction of tasks whose GT tier happens to be the static tier).
    - B/C pairing joins by task_id; order equality is asserted up front,
      and judge base-decision drift between the B and C runs is reported,
      never silently assumed away.

Outputs (under --out-dir, default eval/):
    results-summary.json   full machine-readable aggregate
    results-tasks.csv      one row per (group, task)
    charts/*.svg           hand-rolled SVG charts (stdlib only, no deps)

CLI:
    python eval/analyze_results.py \
        --a logs/s2_full_A_main24.jsonl \
        --b logs/s2_full_B_family_order.jsonl \
        --c logs/s2_full_C_main24_retry2.jsonl \
        --audit-c logs/s2_full_C_main24.jsonl logs/s2_full_C_main24_rerun.jsonl \
        --c-memory logs/s2_full_C_main24_retry2_memory.jsonl \
        --out-dir eval
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_A = "logs/s2_full_A_main24.jsonl"
DEFAULT_B = "logs/s2_full_B_family_order.jsonl"
DEFAULT_C = "logs/s2_full_C_main24_retry2.jsonl"
DEFAULT_AUDIT_C = ["logs/s2_full_C_main24.jsonl", "logs/s2_full_C_main24_rerun.jsonl"]
DEFAULT_C_MEMORY = "logs/s2_full_C_main24_retry2_memory.jsonl"

TIERS = ("simple", "medium", "complex", "reasoning")
# Ordinal used ONLY for over/under-provision classification. Caveat:
# "complex" (sub-agent orchestration) and "reasoning" (deep single-agent)
# are equally priced and not strictly ordered by capability; this follows
# the fixed tuple order used throughout the codebase.
TIER_ORDINAL = {t: i for i, t in enumerate(TIERS)}

NOTES = [
    "A's tier-vs-GT rate is a static-tier coincidence (share of tasks whose "
    "GT tier equals the static 'reasoning' tier), NOT a routing accuracy.",
    "B formal = s2_full_B_family_order.jsonl: judge-only routing, 24/24 "
    "eval success, executed in the same family-blocked task order as A "
    "and C (supersedes the older numeric-order B runs).",
    "C formal = third full run (retry2, 24/24 eval success). The first two "
    "C runs appear only in failure_audit, never in metrics.",
    "C learning-curve segments are confounded by the family-blocked task "
    "order (harder reasoning-heavy families run later); use directional only.",
    "B->C executed-tier changes occur on 4 tasks: T06/T19 are JitRL "
    "modulation flips (judge base simple -> complex); T13/T22 are judge "
    "base-decision drift between the B and C runs (the local llama judge "
    "is not fully deterministic across runs), NOT modulation. Remaining "
    "B->C deltas are same-tier execution sampling variance (same model, "
    "temperature 0.7).",
    "Judge (local llama, 127.0.0.1:18080) has $0 API cost; only its latency "
    "is reported. A/B jitrl_latency_ms=0.0 is the documented sentinel for "
    "'no JitRL post-processing', not a measurement.",
    "Evaluator budget (eval_cost_usd, blind gpt-5.6-sol) is measurement "
    "overhead reported separately from execution cost.",
]


# ---------------------------------------------------------------- loading

def load_group(path: str | Path) -> tuple[list[dict], dict | None]:
    """Load one run JSONL -> (records, embedded _summary or None)."""
    p = Path(path)
    records: list[dict] = []
    file_summary = None
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "_summary" in obj:
            file_summary = obj["_summary"]
        else:
            records.append(obj)
    return records, file_summary


def load_memory_size(path: str | Path | None) -> int | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return sum(1 for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip())


# ---------------------------------------------------------------- metrics

def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _sd(xs: list[float]) -> float | None:
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def _r(x: float | None, nd: int) -> float | None:
    return None if x is None else round(x, nd)


def group_metrics(records: list[dict]) -> dict:
    """Per-group aggregate metrics (works for any of A/B/C)."""
    ok = [r for r in records if r.get("evaluator") is not None]
    failed = [r for r in records if r.get("evaluator") is None]
    q = [float(r["evaluator"]["quality_score"]) for r in ok]
    exec_costs = [float(r["exec_cost_usd"]) for r in records
                  if r.get("exec_cost_usd") is not None]
    eval_costs = [float(r["eval_cost_usd"]) for r in records
                  if r.get("eval_cost_usd") is not None]
    verdicts = {"useful": 0, "neutral": 0, "harmful": 0}
    for r in ok:
        v = r["evaluator"].get("verdict")
        verdicts[v] = verdicts.get(v, 0) + 1

    matches = [r for r in records if r.get("chosen_tier") == r.get("gt_tier")]
    over = [r for r in records
            if r.get("chosen_tier") and r.get("gt_tier")
            and TIER_ORDINAL[r["chosen_tier"]] > TIER_ORDINAL[r["gt_tier"]]]
    under = [r for r in records
             if r.get("chosen_tier") and r.get("gt_tier")
             and TIER_ORDINAL[r["chosen_tier"]] < TIER_ORDINAL[r["gt_tier"]]]
    n_labeled = sum(1 for r in records if r.get("gt_tier"))

    return {
        "n_records": len(records),
        "eval_success": f"{len(ok)}/{len(records)}",
        "eval_success_rate": round(len(ok) / len(records), 4) if records else None,
        "eval_failures": [r.get("task_id") for r in failed],
        "quality": {
            "mean": _r(_mean(q), 4),
            "sd": _r(_sd(q), 4),
            "min": min(q) if q else None,
            "max": max(q) if q else None,
            "distribution": {str(s): q.count(float(s))
                             for s in (1, 2, 3, 4, 5)},
            "verdicts": verdicts,
            "capability_insufficient": sum(
                1 for r in ok if r["evaluator"].get("capability_insufficient")),
        },
        "exec_cost_usd": {
            "total": _r(sum(exec_costs), 6),
            "mean": _r(_mean(exec_costs), 6),
            "min": _r(min(exec_costs), 6) if exec_costs else None,
            "max": _r(max(exec_costs), 6) if exec_costs else None,
        },
        "eval_cost_usd": {  # Evaluator budget, separate from execution cost
            "total": _r(sum(eval_costs), 6),
            "mean": _r(_mean(eval_costs), 6),
        },
        "tier_vs_gt": {
            "matches": len(matches),
            "match_rate": _r(len(matches) / n_labeled, 4) if n_labeled else None,
            "over_provisioned": len(over),
            "under_provisioned": len(under),
            "misgrade_rate": _r(1 - len(matches) / n_labeled, 4) if n_labeled else None,
        },
        "finish_reason_length": sum(
            1 for r in records if r.get("exec_finish_reason") == "length"),
        "empty_response": sum(
            1 for r in records
            if r.get("exec_response_preview") is not None
            and not str(r["exec_response_preview"]).strip()),
        "flips": sum(1 for r in records if r.get("flipped")),
        "memory_written": sum(1 for r in records if r.get("memory_written")),
        "latency_ms": {
            "judge_mean": _r(_mean([float(r["judge_latency_ms"])
                                    for r in records]), 1),
            "jitrl_mean": _r(_mean([float(r["jitrl_latency_ms"])
                                     for r in records]), 4),
            "jitrl_max": _r(max((float(r["jitrl_latency_ms"])
                                 for r in records), default=0.0), 4),
            "exec_mean": _r(_mean([float(r["exec_latency_ms"])
                                   for r in records
                                   if r.get("exec_latency_ms") is not None]), 1),
            "evaluator_mean": _r(_mean([float(r["evaluator"]["evaluator_latency_ms"])
                                        for r in ok]), 1),
        },
    }


# ---------------------------------------------------- consistency checks

def consistency_checks(a: list[dict], b: list[dict], c: list[dict]) -> dict:
    ids = lambda rs: [r["task_id"] for r in rs]
    a_ids, b_ids, c_ids = ids(a), ids(b), ids(c)
    b_by_id = {r["task_id"]: r for r in b}
    base_mismatches = [
        {"task_id": cr["task_id"],
         "b_chosen_tier": b_by_id[cr["task_id"]]["chosen_tier"],
         "c_base_choice": cr.get("base_choice")}
        for cr in c if cr["task_id"] in b_by_id
        and b_by_id[cr["task_id"]]["chosen_tier"] != cr.get("base_choice")
    ]
    modes = lambda rs: sorted({r.get("route_mode") for r in rs})
    order_style = lambda ids_, ref: (
        "numeric_ascending" if ids_ == sorted(ids_)
        else "family_blocked" if ref is not None and ids_ == ref
        else "other")
    return {
        "a_ids": a_ids,
        "b_ids": b_ids,
        "c_ids": c_ids,
        "abc_same_id_set": set(a_ids) == set(b_ids) == set(c_ids),
        "abc_same_order": a_ids == b_ids == c_ids,
        "bc_same_id_set": set(b_ids) == set(c_ids),
        "bc_same_id_count": len(set(b_ids)) == len(b_ids) == len(c_ids) == len(set(c_ids)),
        "bc_same_order": b_ids == c_ids,
        "b_order_style": order_style(b_ids, a_ids),
        "c_order_style": order_style(c_ids, a_ids),
        "c_order_equals_a_order": c_ids == a_ids,
        "all_episode_id_1": all(r.get("episode_id") == 1 for r in a + b + c),
        "route_modes": {"A": modes(a), "B": modes(b), "C": modes(c)},
        "bc_judge_base_choice_consistent": not base_mismatches,
        "bc_judge_base_choice_mismatches": base_mismatches,
    }


# ------------------------------------------------------------- paired B/C

def paired_bc(b: list[dict], c: list[dict]) -> dict:
    """Task-paired B vs C comparison (join by task_id)."""
    b_by_id = {r["task_id"]: r for r in b}
    if set(b_by_id) != {r["task_id"] for r in c}:
        raise SystemExit("B/C task id sets differ; refusing to pair")

    per_task = []
    q_deltas, c_deltas = [], []
    q_improved = q_worsened = q_unchanged = 0
    for cr in c:
        br = b_by_id[cr["task_id"]]
        qd = cr["evaluator"]["quality_score"] - br["evaluator"]["quality_score"]
        cd = cr["exec_cost_usd"] - br["exec_cost_usd"]
        q_deltas.append(qd)
        c_deltas.append(cd)
        q_improved += qd > 0
        q_worsened += qd < 0
        q_unchanged += qd == 0
        per_task.append({
            "task_id": cr["task_id"],
            "quality_B": br["evaluator"]["quality_score"],
            "quality_C": cr["evaluator"]["quality_score"],
            "quality_delta": qd,
            "exec_cost_B_usd": _r(br["exec_cost_usd"], 6),
            "exec_cost_C_usd": _r(cr["exec_cost_usd"], 6),
            "exec_cost_delta_usd": _r(cd, 6),
            "tier_B": br["chosen_tier"], "tier_C": cr["chosen_tier"],
            "flipped": cr.get("flipped", False),
        })

    total_b = sum(r["exec_cost_usd"] for r in b)
    total_c = sum(r["exec_cost_usd"] for r in c)
    flip_ids = {d["task_id"] for d in per_task if d["flipped"]}
    drift_ids = {d["task_id"] for d in per_task
                 if not d["flipped"] and d["tier_B"] != d["tier_C"]}
    tier_changed_ids = {d["task_id"] for d in per_task
                        if d["tier_B"] != d["tier_C"]}
    flip_cost = sum(d["exec_cost_delta_usd"] for d in per_task
                    if d["task_id"] in flip_ids)
    drift_cost = sum(d["exec_cost_delta_usd"] for d in per_task
                     if d["task_id"] in drift_ids)
    flip_q = sum(d["quality_delta"] for d in per_task if d["task_id"] in flip_ids)
    drift_q = sum(d["quality_delta"] for d in per_task
                  if d["task_id"] in drift_ids)
    return {
        "n_pairs": len(per_task),
        "quality": {
            "mean_B": _r(_mean([d["quality_B"] for d in per_task]), 4),
            "mean_C": _r(_mean([d["quality_C"] for d in per_task]), 4),
            "mean_delta": _r(_mean(q_deltas), 4),
            "improved_tasks": q_improved,
            "worsened_tasks": q_worsened,
            "unchanged_tasks": q_unchanged,
        },
        "exec_cost": {
            "total_B_usd": _r(total_b, 6),
            "total_C_usd": _r(total_c, 6),
            "total_delta_usd": _r(total_c - total_b, 6),
            "total_change_pct": _r(100.0 * (total_c - total_b) / total_b, 2)
                if total_b else None,
            "savings_pct_bc": _r(100.0 * (total_b - total_c) / total_b, 2)
                if total_b else None,
            "mean_delta_usd": _r(_mean(c_deltas), 6),
            "flip_tasks_delta_usd": _r(flip_cost, 6),
            "judge_drift_tasks_delta_usd": _r(drift_cost, 6),
            "nonflip_tasks_delta_usd": _r(sum(c_deltas) - flip_cost, 6),
            "same_tier_tasks_delta_usd": _r(sum(c_deltas) - flip_cost
                                            - drift_cost, 6),
            "note": "flip_tasks = C modulation changed the executed tier vs "
                    "the judge base choice; judge_drift_tasks = not flipped "
                    "but the judge issued a different base decision in the "
                    "B run than in the C run (local llama judge "
                    "nondeterminism); same_tier deltas are execution "
                    "sampling variance (temperature 0.7).",
        },
        "tier_changes": {
            "n_total": len(tier_changed_ids),
            "task_ids": sorted(tier_changed_ids),
            "modulation_flip_task_ids": sorted(flip_ids),
            "judge_base_drift_task_ids": sorted(drift_ids),
            "note": "executed tier differs between B and C on these tasks; "
                    "only modulation flips are attributable to JitRL, "
                    "judge base drift is not.",
        },
        "quality_attribution": {
            "flip_tasks_quality_delta": flip_q,
            "judge_drift_tasks_quality_delta": drift_q,
            "same_tier_tasks_quality_delta": sum(q_deltas) - flip_q - drift_q,
            "note": "B->C quality delta decomposes into JitRL-flip tasks, "
                    "judge-base-drift tasks (different executed tier "
                    "because the judge decided differently across runs) and "
                    "same-tier tasks (pure execution sampling variance, "
                    "temperature 0.7); only the flip part is attributable "
                    "to JitRL modulation.",
        },
        "per_task": per_task,
    }


# ----------------------------------------------------------------- flips

def flip_analysis(c: list[dict], b: list[dict]) -> dict:
    b_by_id = {r["task_id"]: r for r in b}
    flips = []
    for cr in c:
        if not cr.get("flipped"):
            continue
        br = b_by_id[cr["task_id"]]
        flips.append({
            "task_id": cr["task_id"],
            "family": cr.get("family"),
            "gt_tier": cr.get("gt_tier"),
            "base_choice": cr.get("base_choice"),
            "chosen_tier": cr.get("chosen_tier"),
            "exec_model": cr.get("exec_model"),
            "n_retrieved": cr.get("n_retrieved"),
            "top_similarity": cr.get("top_similarity"),
            "V": _r(cr.get("V"), 4),
            "exec_finish_reason": cr.get("exec_finish_reason"),
            "response_empty": not str(cr.get("exec_response_preview") or "").strip(),
            "gt_correct_before": br["chosen_tier"] == br["gt_tier"],
            "gt_correct_after": cr["chosen_tier"] == cr["gt_tier"],
            "quality_B": br["evaluator"]["quality_score"],
            "quality_C": cr["evaluator"]["quality_score"],
            "quality_delta": cr["evaluator"]["quality_score"]
                             - br["evaluator"]["quality_score"],
            "exec_cost_B_usd": _r(br["exec_cost_usd"], 6),
            "exec_cost_C_usd": _r(cr["exec_cost_usd"], 6),
            "exec_cost_delta_usd": _r(cr["exec_cost_usd"] - br["exec_cost_usd"], 6),
        })
    return {
        "n_flips": len(flips),
        "flips": flips,
        "quality_delta_sum": sum(f["quality_delta"] for f in flips),
        "exec_cost_delta_sum_usd": _r(sum(f["exec_cost_delta_usd"] for f in flips), 6),
        "gt_correctness_changes": sum(
            1 for f in flips if f["gt_correct_before"] != f["gt_correct_after"]),
        "improved": sum(1 for f in flips if f["quality_delta"] > 0),
        "worsened": sum(1 for f in flips if f["quality_delta"] < 0),
    }


# --------------------------------------------------------- learning curve

def learning_curve(c: list[dict]) -> dict:
    """Positional segments of the C sequence (tertiles = pre/mid/post,
    plus halves). Confounded by family-blocked task difficulty order."""
    n = len(c)
    segs = {}
    defs = ([("pre_tertile", 0, n // 3), ("mid_tertile", n // 3, 2 * n // 3),
             ("post_tertile", 2 * n // 3, n)]
            if n >= 3 else [])
    defs += [("first_half", 0, (n + 1) // 2), ("second_half", (n + 1) // 2, n)]
    for name, lo, hi in defs:
        seg = c[lo:hi]
        if not seg:
            continue
        segs[name] = {
            "positions": [lo + 1, hi],
            "quality_mean": _r(_mean([r["evaluator"]["quality_score"] for r in seg]), 4),
            "reward_mean": _r(_mean([float(r["reward"]) for r in seg
                                     if r.get("reward") is not None]), 4),
            "gt_match_rate": _r(sum(1 for r in seg
                                    if r["chosen_tier"] == r["gt_tier"]) / len(seg), 4),
            "n_retrieved_sum": sum(r.get("n_retrieved", 0) for r in seg),
        }
    return {
        "segments": segs,
        "per_position_quality": [r["evaluator"]["quality_score"] for r in c],
        "per_position_task_ids": [r["task_id"] for r in c],
        "confound": "family-blocked order: reasoning-heavy families run in "
                    "positions 13-24, so late-segment quality drops are task "
                    "difficulty, not necessarily forgetting/decay.",
    }


# ------------------------------------------------------ retrieval/memory

def retrieval_memory(c: list[dict], memory_path: str | Path | None) -> dict:
    retrieved = [r for r in c if r.get("n_retrieved", 0) > 0]
    return {
        "tasks_with_retrieval": len(retrieved),
        "retrieval_task_ids": [r["task_id"] for r in retrieved],
        "avg_n_retrieved_when_hit": _r(_mean([r["n_retrieved"] for r in retrieved]), 2)
            if retrieved else 0.0,
        "top_similarity_mean_when_hit": _r(_mean([r["top_similarity"] for r in retrieved]), 2)
            if retrieved else None,
        "flips_among_retrieval_hits": sum(1 for r in retrieved if r.get("flipped")),
        "memory_written": sum(1 for r in c if r.get("memory_written")),
        "memory_entries_final": load_memory_size(memory_path),
    }


# ----------------------------------------------------------- audit (C 1/2)

def audit_failures(paths: list[str | Path]) -> list[dict]:
    """Failure audit of the first two C runs (never used in metrics)."""
    out = []
    for p in paths:
        pp = Path(p)
        if not pp.exists():
            continue
        records, _ = load_group(pp)
        failures = []
        for r in records:
            if r.get("evaluator") is None or r.get("exec_error") or r.get("eval_error"):
                failures.append({
                    "task_id": r.get("task_id"),
                    "chosen_tier": r.get("chosen_tier"),
                    "exec_error": r.get("exec_error"),
                    "eval_error": r.get("eval_error"),
                    "exec_finish_reason": r.get("exec_finish_reason"),
                    "memory_written": r.get("memory_written"),
                })
        out.append({
            "file": pp.name,
            "n_records": len(records),
            "eval_success": f"{len(records) - len(failures)}/{len(records)}",
            "failures": failures,
        })
    return out


# ---------------------------------------------------------------- D10

def d10_judgment(summary: dict) -> dict:
    """D10 gate check against the FROZEN criteria (project-decisions).

    Frozen criteria, judged exactly as written on the formal n=24 runs:
      1. C vs B execution-cost savings >= 20%: (B - C) / B >= 20%.
      2. Quality: C not lower than B by more than 0.3/5:
         mean_quality_C - mean_quality_B >= -0.3.
      3. C misgrade rate <= B misgrade rate + 5 percentage points.
      4. JitRL modulation post-processing latency increment <= 500 ms.
    n=24 is directional evidence only; no statistical significance is
    claimed. No additional thresholds (e.g. cost-increase tolerance or
    "flips cause no worsening") are applied.
    """
    b = summary["groups"]["B"]
    c = summary["groups"]["C"]
    paired = summary["paired_bc"]
    total_b = paired["exec_cost"]["total_B_usd"]
    total_c = paired["exec_cost"]["total_C_usd"]
    # prefer the paired savings computed from unrounded totals
    savings_pct = paired["exec_cost"].get("savings_pct_bc")
    if savings_pct is None:
        savings_pct = (100.0 * (total_b - total_c) / total_b
                       if total_b else None)
    q_delta = c["quality"]["mean"] - b["quality"]["mean"]
    mis_b = b["tier_vs_gt"]["misgrade_rate"]
    mis_c = c["tier_vs_gt"]["misgrade_rate"]
    mis_diff_pp = ((mis_c - mis_b) * 100.0
                   if mis_b is not None and mis_c is not None else None)
    jm_b, jx_b = b["latency_ms"]["jitrl_mean"], b["latency_ms"]["jitrl_max"]
    jm_c, jx_c = c["latency_ms"]["jitrl_mean"], c["latency_ms"]["jitrl_max"]
    inc_mean = _r(jm_c - (jm_b or 0.0), 4) if jm_c is not None else None
    inc_max = _r(jx_c - (jx_b or 0.0), 4) if jx_c is not None else None
    criteria = [
        {"criterion": "C vs B exec cost savings >= 20% ((B-C)/B)",
         "B_total_usd": total_b, "C_total_usd": total_c,
         "savings_pct": _r(savings_pct, 2),
         # round to 6dp so an exact boundary is not lost to float error
         "met": savings_pct is not None and round(savings_pct, 6) >= 20.0},
        {"criterion": "quality: C - B >= -0.3 (C at most 0.3/5 below B)",
         "B_mean": b["quality"]["mean"], "C_mean": c["quality"]["mean"],
         "delta": _r(q_delta, 4), "met": round(q_delta, 6) >= -0.3},
        {"criterion": "C misgrade rate <= B misgrade rate + 5pp",
         "B_misgrade_rate": mis_b, "C_misgrade_rate": mis_c,
         "diff_pp": _r(mis_diff_pp, 2),
         "met": mis_diff_pp is not None and round(mis_diff_pp, 6) <= 5.0},
        {"criterion": "JitRL modulation post-processing latency increment "
                      "<= 500 ms",
         "increment_mean_ms": inc_mean, "increment_max_ms": inc_max,
         "basis": "C jitrl_latency_ms minus the A/B 0.0 'no JitRL' "
                  "sentinel; judged on the worst per-task increment",
         "met": inc_max is not None and round(inc_max, 6) <= 500.0},
    ]
    met = all(cr["met"] for cr in criteria)
    return {
        "definition_status": "Frozen D10 criteria from project-decisions, "
                             "relayed verbatim in the analysis directive; "
                             "judged as-is on the formal runs.",
        "criteria": criteria,
        "n_criteria_met": sum(1 for cr in criteria if cr["met"]),
        "verdict": "MET (directional evidence only, n=24)" if met
                   else "NOT MET",
        "caveats": [
            "n=24, one episode per task: directional evidence only; no "
            "statistical significance claimed.",
            "Cost is judged on execution-cost totals ((B-C)/B); the blind "
            "evaluator budget is measurement overhead and is excluded.",
            "JitRL latency is judged on the max per-task post-processing "
            "increment; the mean increment is reported alongside.",
        ],
    }


# ----------------------------------------------------------------- charts

_COLORS = {"A": "#6b7280", "B": "#3b82f6", "C": "#f59e0b",
           "exec": "#3b82f6", "eval": "#ef4444"}


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _nice_max(v: float) -> float:
    if v <= 0:
        return 1.0
    mag = 10 ** int(f"{v:e}".split("e")[1])
    for m in (1, 2, 2.5, 5, 10):
        if v <= m * mag:
            return m * mag
    return 10 * mag


def svg_grouped_bars(title: str, y_label: str, categories: list[str],
                     series: dict[str, list[float]], out_path: Path,
                     value_fmt: str = "{:.2f}") -> None:
    """Grouped vertical bar chart, one cluster per category."""
    n_cat, n_ser = len(categories), len(series)
    w, h = 860, 480
    ml, mr, mt, mb = 78, 24, 56, 74
    iw, ih = w - ml - mr, h - mt - mb
    vmax = _nice_max(max(v for vs in series.values() for v in vs) or 1.0)
    x0 = ml
    cluster = iw / max(n_cat, 1)
    bar_w = cluster * 0.72 / n_ser
    parts = [
        f'<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="Helvetica,Arial,sans-serif">',
        f'<rect width="{w}" height="{h}" fill="white"/>',
        f'<text x="{w / 2}" y="26" font-size="17" font-weight="bold" '
        f'text-anchor="middle">{_esc(title)}</text>',
    ]
    for g in range(6):
        yv = vmax * g / 5
        y = mt + ih - ih * g / 5
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{w - mr}" y2="{y:.1f}" '
                     f'stroke="#e5e7eb"/>')
        parts.append(f'<text x="{ml - 8}" y="{y + 4:.1f}" font-size="11" '
                     f'text-anchor="end">{yv:.3g}</text>')
    parts.append(f'<text x="16" y="{mt + ih / 2}" font-size="12" '
                 f'text-anchor="middle" transform="rotate(-90 16 {mt + ih / 2})">'
                 f'{_esc(y_label)}</text>')
    for ci, cat in enumerate(categories):
        cx = x0 + cluster * ci + cluster * 0.14
        for si, (name, vals) in enumerate(series.items()):
            v = vals[ci]
            bh = ih * v / vmax
            x = cx + si * bar_w
            y = mt + ih - bh
            color = _COLORS.get(name, "#10b981")
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                         f'height="{bh:.1f}" fill="{color}" rx="2"/>')
            if v > 0:
                parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 4:.1f}" '
                             f'font-size="9" text-anchor="middle">'
                             f'{value_fmt.format(v)}</text>')
        parts.append(f'<text x="{x0 + cluster * (ci + 0.5):.1f}" y="{h - mb + 18}" '
                     f'font-size="12" text-anchor="middle">{_esc(cat)}</text>')
    lx = ml
    for name in series:
        color = _COLORS.get(name, "#10b981")
        parts.append(f'<rect x="{lx}" y="{h - mb + 36}" width="12" height="12" '
                     f'fill="{color}" rx="2"/>')
        parts.append(f'<text x="{lx + 17}" y="{h - mb + 46}" font-size="12">'
                     f'{_esc(name)}</text>')
        lx += 34 + 9 * len(str(name))
    parts.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")


def svg_line(title: str, y_label: str, x_labels: list[str],
             series: dict[str, list[float]], out_path: Path,
             y_min: float | None = None) -> None:
    """Multi-series line chart with markers (learning curve)."""
    w, h = 1020, 480
    ml, mr, mt, mb = 70, 150, 56, 64
    iw, ih = w - ml - mr, h - mt - mb
    all_v = [v for vs in series.values() for v in vs]
    lo = y_min if y_min is not None else min(all_v)
    hi = max(all_v) + 0.001
    span = (hi - lo) or 1.0
    n = len(x_labels)
    xs = [ml + iw * i / max(n - 1, 1) for i in range(n)]
    ys = lambda v: mt + ih - ih * (v - lo) / span
    parts = [
        f'<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="Helvetica,Arial,sans-serif">',
        f'<rect width="{w}" height="{h}" fill="white"/>',
        f'<text x="{w / 2}" y="26" font-size="17" font-weight="bold" '
        f'text-anchor="middle">{_esc(title)}</text>',
    ]
    for g in range(6):
        yv = lo + span * g / 5
        y = ys(yv)
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{w - mr}" y2="{y:.1f}" '
                     f'stroke="#e5e7eb"/>')
        parts.append(f'<text x="{ml - 8}" y="{y + 4:.1f}" font-size="11" '
                     f'text-anchor="end">{yv:.2f}</text>')
    parts.append(f'<text x="16" y="{mt + ih / 2}" font-size="12" '
                 f'text-anchor="middle" transform="rotate(-90 16 {mt + ih / 2})">'
                 f'{_esc(y_label)}</text>')
    palette = ["#f59e0b", "#ef4444", "#3b82f6"]
    for si, (name, vals) in enumerate(series.items()):
        color = palette[si % len(palette)]
        dash = ' stroke-dasharray="6 4"' if "mean" in name.lower() else ""
        pts = " ".join(f"{x:.1f},{ys(v):.1f}" for x, v in zip(xs, vals))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                     f'stroke-width="2"{dash}/>')
        if "mean" not in name.lower():
            for x, v in zip(xs, vals):
                parts.append(f'<circle cx="{x:.1f}" cy="{ys(v):.1f}" r="3.5" '
                             f'fill="{color}"/>')
        ly = mt + 20 + si * 20
        parts.append(f'<line x1="{w - mr + 16}" y1="{ly}" x2="{w - mr + 46}" '
                     f'y2="{ly}" stroke="{color}" stroke-width="2"{dash}/>')
        parts.append(f'<text x="{w - mr + 52}" y="{ly + 4}" font-size="12">'
                     f'{_esc(name)}</text>')
    for i, lbl in enumerate(x_labels):
        if n > 14 and i % 2 == 1:
            continue
        parts.append(f'<text x="{xs[i]:.1f}" y="{h - mb + 18}" font-size="10" '
                     f'text-anchor="middle" transform="rotate(-45 {xs[i]:.1f} '
                     f'{h - mb + 18})">{_esc(lbl)}</text>')
    parts.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_charts(summary: dict, out_dir: Path) -> list[str]:
    charts_dir = out_dir / "charts"
    written = []
    g = summary["groups"]
    # 1. quality distribution
    svg_grouped_bars(
        "Quality score distribution (formal runs, n=24 each)",
        "tasks",
        ["1", "2", "3", "4", "5"],
        {k: [g[k]["quality"]["distribution"][s] for s in ("1", "2", "3", "4", "5")]
         for k in ("A", "B", "C")},
        charts_dir / "quality-distribution.svg", value_fmt="{:.0f}")
    # 2/3. paired B/C per task
    pt = summary["paired_bc"]["per_task"]
    labels = [d["task_id"] for d in pt]
    svg_grouped_bars("B vs C quality per task (paired by task_id)",
                     "quality score", labels,
                     {"B": [d["quality_B"] for d in pt],
                      "C": [d["quality_C"] for d in pt]},
                     charts_dir / "bc-paired-quality.svg", value_fmt="{:.0f}")
    svg_grouped_bars("B vs C execution cost per task (USD)",
                     "exec cost (USD)", labels,
                     {"B": [d["exec_cost_B_usd"] for d in pt],
                      "C": [d["exec_cost_C_usd"] for d in pt]},
                     charts_dir / "bc-paired-exec-cost.svg", value_fmt="{:.3f}")
    # 4. learning curve
    lc = summary["learning_curve"]
    n = len(lc["per_position_quality"])
    tert = []
    for name in ("pre_tertile", "mid_tertile", "post_tertile"):
        seg = lc["segments"].get(name)
        if seg:
            tert += [seg["quality_mean"]] * (
                seg["positions"][1] - seg["positions"][0] + 1)
    if len(tert) == n:
        svg_line("C learning curve: quality by sequence position "
                 "(order confounded by task difficulty)",
                 "quality score",
                 [f"{i}:{tid}" for i, tid in
                  enumerate(lc["per_position_task_ids"], 1)],
                 {"quality per task": lc["per_position_quality"],
                  "tertile mean": tert},
                 charts_dir / "learning-curve.svg", y_min=0)
    # 5. cost summary: execution cost vs Evaluator budget per group
    svg_grouped_bars(
        "Execution cost vs Evaluator budget per group (totals, USD)",
        "USD", ["A", "B", "C"],
        {"exec": [g[k]["exec_cost_usd"]["total"] for k in ("A", "B", "C")],
         "eval": [g[k]["eval_cost_usd"]["total"] for k in ("A", "B", "C")]},
        charts_dir / "cost-summary.svg", value_fmt="{:.4f}")
    for f in sorted(charts_dir.glob("*.svg")):
        written.append(str(f))
    return written


# ------------------------------------------------------------------ CSV

CSV_FIELDS = ["group", "position", "task_id", "family", "gt_tier",
              "chosen_tier", "base_choice", "flipped", "quality_score",
              "verdict", "capability_insufficient", "exec_model",
              "exec_finish_reason", "exec_cost_usd", "exec_latency_ms",
              "eval_cost_usd", "judge_latency_ms", "jitrl_latency_ms",
              "n_retrieved", "top_similarity", "reward"]


def write_tasks_csv(groups: dict[str, list[dict]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        wcsv.writeheader()
        for name in ("A", "B", "C"):
            for pos, r in enumerate(groups[name], 1):
                ev = r.get("evaluator") or {}
                wcsv.writerow({
                    "group": name, "position": pos,
                    "task_id": r.get("task_id"), "family": r.get("family"),
                    "gt_tier": r.get("gt_tier"),
                    "chosen_tier": r.get("chosen_tier"),
                    "base_choice": r.get("base_choice"),
                    "flipped": r.get("flipped"),
                    "quality_score": ev.get("quality_score"),
                    "verdict": ev.get("verdict"),
                    "capability_insufficient": ev.get("capability_insufficient"),
                    "exec_model": r.get("exec_model"),
                    "exec_finish_reason": r.get("exec_finish_reason"),
                    "exec_cost_usd": r.get("exec_cost_usd"),
                    "exec_latency_ms": r.get("exec_latency_ms"),
                    "eval_cost_usd": r.get("eval_cost_usd"),
                    "judge_latency_ms": r.get("judge_latency_ms"),
                    "jitrl_latency_ms": r.get("jitrl_latency_ms"),
                    "n_retrieved": r.get("n_retrieved"),
                    "top_similarity": r.get("top_similarity"),
                    "reward": r.get("reward"),
                })


# ------------------------------------------------------------------ main

def build_summary(a_recs, b_recs, c_recs, *, a_summ, b_summ, c_summ,
                  audit_paths, c_memory_path,
                  formal_paths: dict[str, str] | None = None) -> dict:
    # Hard consistency assertion (directive): the formal A/B/C runs must
    # contain the same task_ids in EXACTLY the same (family-blocked) order.
    a_ids = [r["task_id"] for r in a_recs]
    b_ids = [r["task_id"] for r in b_recs]
    c_ids = [r["task_id"] for r in c_recs]
    if not (a_ids == b_ids == c_ids):
        first_diff = next(
            (i for i, xyz in enumerate(zip(a_ids, b_ids, c_ids))
             if len({xyz[0], xyz[1], xyz[2]}) > 1), None)
        raise SystemExit(
            "consistency assertion failed: A/B/C must contain the same "
            f"task_ids in exactly the same order (len A={len(a_ids)}, "
            f"B={len(b_ids)}, C={len(c_ids)}, first divergence at position "
            f"{first_diff if first_diff is None else first_diff + 1}); "
            "refusing to analyze")
    groups_raw = {"A": a_recs, "B": b_recs, "C": c_recs}
    metrics = {k: group_metrics(v) for k, v in groups_raw.items()}
    # A: reframe the tier-vs-GT rate — NOT a routing accuracy.
    a_rate = metrics["A"].pop("tier_vs_gt")
    metrics["A"]["static_tier_gt_coincidence"] = {
        **a_rate,
        "note": "static tier 'reasoning' vs GT coincidence; NOT routing "
                "accuracy (A runs no router).",
    }
    summary = {
        "inputs": {
            "formal": formal_paths or {"A": DEFAULT_A, "B": DEFAULT_B,
                                       "C": DEFAULT_C},
            "audit_only": [str(p) for p in audit_paths],
        },
        "outputs": {},
        "groups": metrics,
        "file_summaries": {"A": a_summ, "B": b_summ, "C": c_summ},
        "consistency": consistency_checks(a_recs, b_recs, c_recs),
        "paired_bc": paired_bc(b_recs, c_recs),
        "flips": flip_analysis(c_recs, b_recs),
        "learning_curve": learning_curve(c_recs),
        "retrieval_memory": retrieval_memory(c_recs, c_memory_path),
        "failure_audit": audit_failures(audit_paths),
        "notes": NOTES,
    }
    summary["d10"] = d10_judgment(summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="eval.analyze_results",
        description="Aggregate Session-2 A/B/C full-run analysis "
                    "(reads logs only; no CPA calls, no credentials).")
    ap.add_argument("--a", default=DEFAULT_A, help="formal A run JSONL")
    ap.add_argument("--b", default=DEFAULT_B, help="formal B run JSONL")
    ap.add_argument("--c", default=DEFAULT_C, help="formal C run JSONL")
    ap.add_argument("--audit-c", nargs="*", default=DEFAULT_AUDIT_C,
                    help="failed C runs for failure audit only (excluded "
                         "from metrics)")
    ap.add_argument("--c-memory", default=DEFAULT_C_MEMORY,
                    help="C memory JSONL (counts final entries)")
    ap.add_argument("--out-dir", default="eval",
                    help="output directory (results-summary.json, "
                         "results-tasks.csv, charts/)")
    args = ap.parse_args(argv)

    a_recs, a_summ = load_group(args.a)
    b_recs, b_summ = load_group(args.b)
    c_recs, c_summ = load_group(args.c)
    for name, recs, path in (("A", a_recs, args.a), ("B", b_recs, args.b),
                             ("C", c_recs, args.c)):
        if not recs:
            raise SystemExit(f"group {name} empty: {path}")
        n_ok = sum(1 for r in recs if r.get("evaluator") is not None)
        if n_ok != len(recs):
            print(f"[warn] group {name}: {len(recs) - n_ok} record(s) without "
                  f"evaluator result will be excluded from quality stats")

    summary = build_summary(a_recs, b_recs, c_recs, a_summ=a_summ,
                            b_summ=b_summ, c_summ=c_summ,
                            audit_paths=args.audit_c,
                            c_memory_path=args.c_memory,
                            formal_paths={"A": args.a, "B": args.b,
                                          "C": args.c})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary["outputs"]["charts"] = write_charts(summary, out_dir)
    with open(out_dir / "results-summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_tasks_csv({"A": a_recs, "B": b_recs, "C": c_recs},
                    out_dir / "results-tasks.csv")

    m = summary["groups"]
    p = summary["paired_bc"]
    print(f"[analyze] A q={m['A']['quality']['mean']} "
          f"exec=${m['A']['exec_cost_usd']['total']:.6f} "
          f"eval=${m['A']['eval_cost_usd']['total']:.6f}")
    print(f"[analyze] B q={m['B']['quality']['mean']} "
          f"exec=${m['B']['exec_cost_usd']['total']:.6f} "
          f"eval=${m['B']['eval_cost_usd']['total']:.6f} "
          f"acc={m['B']['tier_vs_gt']['match_rate']}")
    print(f"[analyze] C q={m['C']['quality']['mean']} "
          f"exec=${m['C']['exec_cost_usd']['total']:.6f} "
          f"eval=${m['C']['eval_cost_usd']['total']:.6f} "
          f"acc={m['C']['tier_vs_gt']['match_rate']}")
    print(f"[analyze] B->C quality delta={p['quality']['mean_delta']} "
          f"(+{p['quality']['improved_tasks']}/-{p['quality']['worsened_tasks']}) "
          f"exec cost {p['exec_cost']['total_change_pct']}% "
          f"(savings (B-C)/B {p['exec_cost']['savings_pct_bc']}%; "
          f"flips {p['exec_cost']['flip_tasks_delta_usd']:+.6f}, "
          f"judge drift {p['exec_cost']['judge_drift_tasks_delta_usd']:+.6f}, "
          f"same-tier {p['exec_cost']['same_tier_tasks_delta_usd']:+.6f})")
    print(f"[analyze] tier changes B->C: {p['tier_changes']['n_total']} "
          f"(modulation flips {p['tier_changes']['modulation_flip_task_ids']}, "
          f"judge base drift {p['tier_changes']['judge_base_drift_task_ids']})")
    print(f"[analyze] flips={summary['flips']['n_flips']} "
          f"improved={summary['flips']['improved']} "
          f"worsened={summary['flips']['worsened']}")
    print(f"[analyze] D10: {summary['d10']['verdict']} "
          f"({summary['d10']['n_criteria_met']}/{len(summary['d10']['criteria'])} "
          f"criteria met)")
    for cr in summary["d10"]["criteria"]:
        print(f"[analyze]   {'PASS' if cr['met'] else 'FAIL'}: "
              f"{cr['criterion']}")
    print(f"[analyze] wrote {out_dir / 'results-summary.json'}, "
          f"{out_dir / 'results-tasks.csv'}, "
          f"{len(summary['outputs']['charts'])} charts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
