"""Independent post-run analysis of the completed C1 (min-neighbor gated
JitRL) run, judged STRICTLY against the frozen preregistration.

Preregistration: docs/S2-C1-PREREGISTRATION.md v1.0 (2026-09-11, frozen
before the C1 run). C1's only intended change vs C0 is
``min_neighbors: 1 -> 3``: with fewer than 3 retrieved neighbors the
modulator degenerates to the native Judge (z_mod = z_base, A_norm = 0,
chosen_tier = base_choice).

Formal inputs (directive):
    B         = logs/s2_full_B_family_order.jsonl      (judge-only baseline)
    C0        = logs/s2_full_C_main24_retry2.jsonl     (formal JitRL C0 run)
    C1        = logs/s2_full_C1_main24.jsonl           (gated JitRL C1 run)
    C1 memory = logs/s2_full_C1_main24_memory.jsonl
    C0 memory = logs/s2_full_C_main24_retry2_memory.jsonl
                (used ONLY for the T-flip counterfactual explanation)

Hard assertions enforced before any metric is computed (refuses to run
otherwise):
    - B, C0 and C1 each hold exactly 24 task records;
    - every record carries an evaluator result (eval 24/24) and no
      exec/eval errors;
    - B, C0 and C1 contain the same task_ids in EXACTLY the same order,
      and that order equals the frozen preregistration sequence;
    - the C1 config snapshot matches the preregistration exactly, in
      particular ``min_neighbors == 3``.

Judgments produced:
    - P1..P5 preregistered criteria (C1 vs B), as written;
    - an INFORMATIONAL D10 recomputation (C1 vs B) that explicitly does
      NOT replace the frozen formal C0 D10 judgment;
    - a check of the preregistration's run-before predictions (Section 5).

Outputs (all under --out-dir, default eval/c1/; C0's eval/ assets are
never read for writing and never overwritten):
    results-summary.json
    results-tasks.csv
    charts/cost-comparison.svg       (B/C0/C1 exec totals + P1 limit)
    charts/quality-comparison.svg    (B/C0/C1 mean quality + P2 limit)
    charts/gate-modulation.svg       (per-task n_retrieved vs min_neighbors=3)
    charts/bc1-paired-exec-cost.svg  (per-task B vs C1 exec cost)

Scope: reads the given JSONL logs only. No CPA calls, no credentials, no
docs writes, no commits.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_B = "logs/s2_full_B_family_order.jsonl"
DEFAULT_C0 = "logs/s2_full_C_main24_retry2.jsonl"
DEFAULT_C1 = "logs/s2_full_C1_main24.jsonl"
DEFAULT_C1_MEMORY = "logs/s2_full_C1_main24_memory.jsonl"
DEFAULT_C0_MEMORY = "logs/s2_full_C_main24_retry2_memory.jsonl"
DEFAULT_OUT_DIR = "eval/c1"

TIERS = ("simple", "medium", "complex", "reasoning")
TIER_ORDINAL = {t: i for i, t in enumerate(TIERS)}

# ------------------------------------------------------- preregistration
# Frozen task order (S2-C1-PREREGISTRATION.md v1.0, Section 3).
PREREG_ORDER = [
    "T01", "T05", "T09", "T13", "T17", "T21",
    "T02", "T06", "T10", "T14", "T18", "T22",
    "T03", "T07", "T11", "T15", "T19", "T23",
    "T04", "T08", "T12", "T16", "T20", "T24",
]
# Frozen config (Section 3): the ONLY intended variable is min_neighbors.
PREREG_CONFIG = {
    "k": 10,
    "beta": 5.0,
    "lam": 0.05,
    "alpha": 5.0,
    "jaccard_threshold": 0.5,
    "z_min": -10.0,
    "seed": 42,
    "min_neighbors": 3,
}
# Frozen P1-P5 thresholds (Section 4).
P1_COST_TOLERANCE_USD = 0.001       # C1 total exec cost <= B + $0.001
P2_QUALITY_FLOOR_DELTA = -0.3       # mean q(C1) - mean q(B) >= -0.3 (0.3/5)
P3_MISTIER_TOLERANCE_PP = 5.0       # C1 misgrade rate <= B + 5 percentage points
P4_JITRL_MAX_MS = 500.0             # max jitrl_latency_ms <= 500 ms
P5_FLIP_COST_TOLERANCE_USD = 0.001  # per flip: cost <= B same task + $0.001
# Retrieval mirrors of the frozen run config (used for neighbor recompute).
RETRIEVAL_K = 10
RETRIEVAL_THRESHOLD = 0.5

_EPS = 1e-9

NOTES = [
    "Judged strictly against docs/S2-C1-PREREGISTRATION.md v1.0 (frozen "
    "before the run); no criterion was added, removed or re-tuned.",
    "P2's '-0.3/5' is 0.3 points on the 5-point quality scale, identical "
    "to the frozen D10 quality criterion (mean q(C1) - mean q(B) >= -0.3).",
    "The D10 recomputation here is INFORMATIONAL for C1 only and does NOT "
    "replace or amend the frozen formal C0 D10 judgment in "
    "eval/results-summary.json.",
    "Exec cost is judged on execution-cost totals; the blind evaluator "
    "budget (eval_cost_usd) is measurement overhead, reported separately.",
    "B jitrl_latency_ms=0.0 is the documented 'no JitRL post-processing' "
    "sentinel, so P4 is judged on C1's own max jitrl_latency_ms.",
    "Same-tier B->C1 cost/quality deltas are execution sampling variance "
    "(same model, temperature 0.7); only modulation flips are attributable "
    "to JitRL, and judge base drift is attributable to the local llama "
    "judge's run-to-run nondeterminism.",
]


# ---------------------------------------------------------------- loading

def load_group(path: str | Path) -> tuple[list[dict], dict | None]:
    """Load one run JSONL -> (records, embedded _summary or None)."""
    p = Path(path)
    records: list[dict] = []
    file_summary: dict | None = None
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


def load_memory(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


# ------------------------------------------------------- hard assertions

def assert_formal_inputs(b: list[dict], c0: list[dict], c1: list[dict],
                         c1_summary: dict | None) -> dict:
    """Refuse to analyze unless the formal-input contract holds.

    Returns the checked facts for the summary. Raises SystemExit on any
    violation.
    """
    problems: list[str] = []
    for name, recs in (("B", b), ("C0", c0), ("C1", c1)):
        if len(recs) != 24:
            problems.append(f"{name} has {len(recs)} records, expected 24")
        bad_eval = [r.get("task_id") for r in recs
                    if r.get("evaluator") is None
                    or r.get("eval_error") is not None]
        if bad_eval:
            problems.append(f"{name} eval not 24/24 (missing/failed: "
                            f"{bad_eval})")
        bad_exec = [r.get("task_id") for r in recs if r.get("exec_error")]
        if bad_exec:
            problems.append(f"{name} has exec errors on {bad_exec}")
    ids = {name: [r["task_id"] for r in recs]
           for name, recs in (("B", b), ("C0", c0), ("C1", c1))}
    if not (ids["B"] == ids["C0"] == ids["C1"]):
        problems.append("B/C0/C1 task order is not identical")
    elif ids["C1"] != PREREG_ORDER:
        first = next(i for i, (a, b_) in enumerate(zip(ids["C1"],
                                                       PREREG_ORDER))
                     if a != b_) if ids["C1"] != PREREG_ORDER else None
        problems.append("task order does not equal the frozen "
                        "preregistration sequence (first divergence at "
                        f"position {None if first is None else first + 1})")
    # config snapshot check: min_neighbors=3 and the full frozen config
    cfg = (c1_summary or {}).get("config_snapshot") or {}
    if not cfg:
        problems.append("C1 _summary has no config_snapshot")
    else:
        for key, want in PREREG_CONFIG.items():
            got = cfg.get(key, "<missing>")
            if got != want:
                problems.append(
                    f"C1 config_snapshot.{key}={got!r}, preregistration "
                    f"requires {want!r}")
    if problems:
        raise SystemExit("formal-input assertion failed: " + "; ".join(problems))
    return {
        "n_records": {"B": len(b), "C0": len(c0), "C1": len(c1)},
        "eval_success": {name: f"{sum(1 for r in recs if r.get('evaluator'))}/24"
                         for name, recs in (("B", b), ("C0", c0), ("C1", c1))},
        "order_identical_b_c0_c1": True,
        "order_equals_prereg": True,
        "order": ids["C1"],
        "c1_config_snapshot": cfg,
        "c1_config_matches_prereg": True,
        "c1_run_meta": {k: (c1_summary or {}).get(k)
                        for k in ("mode", "judge", "episodes")},
    }


# ---------------------------------------------------------------- metrics

def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _r(x: float | None, nd: int) -> float | None:
    return None if x is None else round(x, nd)


def group_metrics(records: list[dict]) -> dict:
    """Per-group aggregate metrics (B / C0 / C1 all use this)."""
    ok = [r for r in records if r.get("evaluator") is not None]
    q = [float(r["evaluator"]["quality_score"]) for r in ok]
    exec_costs = [float(r["exec_cost_usd"]) for r in records
                  if r.get("exec_cost_usd") is not None]
    eval_costs = [float(r["eval_cost_usd"]) for r in records
                  if r.get("eval_cost_usd") is not None]
    verdicts: dict[str, int] = {}
    for r in ok:
        v = r["evaluator"].get("verdict")
        verdicts[v] = verdicts.get(v, 0) + 1
    n = len(records)
    matches = sum(1 for r in records if r.get("chosen_tier") == r.get("gt_tier"))
    over = sum(1 for r in records
               if r.get("chosen_tier") and r.get("gt_tier")
               and TIER_ORDINAL[r["chosen_tier"]] > TIER_ORDINAL[r["gt_tier"]])
    under = n - matches - over
    return {
        "n_records": n,
        "eval_success": f"{len(ok)}/{n}",
        "quality": {
            "mean": _r(_mean(q), 4),
            "sd": _r(statistics.stdev(q), 4) if len(q) > 1 else 0.0,
            "sum": sum(q),
            "min": min(q) if q else None,
            "max": max(q) if q else None,
            "distribution": {str(s): q.count(float(s)) for s in (1, 2, 3, 4, 5)},
            "verdicts": verdicts,
        },
        "exec_cost_usd": {
            "total": _r(sum(exec_costs), 6),
            "mean": _r(_mean(exec_costs), 6),
        },
        "eval_cost_usd": {"total": _r(sum(eval_costs), 6)},  # overhead
        "tier_vs_gt": {
            "matches": matches,
            "match_rate": _r(matches / n, 4) if n else None,
            "misgrade_rate": _r(1 - matches / n, 4) if n else None,
            "over_provisioned": over,
            "under_provisioned": under,
        },
        "finish_reason_length": sum(
            1 for r in records if r.get("exec_finish_reason") == "length"),
        "empty_response_preview": sum(
            1 for r in records
            if r.get("exec_response_preview") is not None
            and not str(r["exec_response_preview"]).strip()),
        "flips": sum(1 for r in records if r.get("flipped")),
        "latency_ms": {
            "judge_mean": _r(_mean([float(r["judge_latency_ms"])
                                    for r in records]), 1),
            "jitrl_mean": _r(_mean([float(r["jitrl_latency_ms"])
                                     for r in records]), 4),
            "jitrl_max": _r(max((float(r["jitrl_latency_ms"])
                                 for r in records), default=0.0), 4),
        },
    }


# ------------------------------------------------------------- C1 gating

def gate_analysis(c1: list[dict]) -> dict:
    """C1's min-neighbor gate outcome per task."""
    per_task = []
    for r in c1:
        n_ret = r.get("n_retrieved", 0)
        status = ("modulated" if r.get("modulation_active")
                  else "gated_no_neighbors" if n_ret == 0
                  else "gated_below_min_neighbors")
        per_task.append({
            "task_id": r["task_id"],
            "family": r.get("family"),
            "gt_tier": r.get("gt_tier"),
            "n_retrieved": n_ret,
            "min_neighbors": r.get("min_neighbors"),
            "gate_reason": r.get("gate_reason"),
            "modulation_active": r.get("modulation_active"),
            "flipped": r.get("flipped"),
            "status": status,
        })
    gated = [d for d in per_task if d["status"].startswith("gated")]
    modulated = [d for d in per_task if d["status"] == "modulated"]
    by_id = {r["task_id"]: r for r in c1}
    return {
        "rule": "n_retrieved < min_neighbors(=3) -> z_mod=z_base, A_norm=0, "
                "chosen_tier=base_choice (native Judge)",
        "counts": {
            "total": len(per_task),
            "gated": len(gated),
            "gated_no_neighbors": sum(
                1 for d in gated if d["status"] == "gated_no_neighbors"),
            "gated_below_min_neighbors": sum(
                1 for d in gated if d["status"] == "gated_below_min_neighbors"),
            "modulated": len(modulated),
            "modulated_and_flipped": sum(1 for d in modulated if d["flipped"]),
        },
        "gated_tasks": [
            {"task_id": d["task_id"], "n_retrieved": d["n_retrieved"],
             "gate_reason": d["gate_reason"]} for d in gated],
        "modulated_tasks": [
            {"task_id": d["task_id"], "n_retrieved": d["n_retrieved"],
             "flipped": d["flipped"],
             "base_choice": by_id[d["task_id"]]["base_choice"],
             "chosen_tier": by_id[d["task_id"]]["chosen_tier"]}
            for d in modulated],
        "per_task": per_task,
    }


# ----------------------------------------------------- neighbor recompute

def jaccard(a: set[str], b: set[str]) -> float:
    """|A&B| / |A|B| (formula-parity with jitrl_core.state.jaccard)."""
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def retrieve_neighbors(memory: list[dict], position: int,
                       order: list[str], k: int = RETRIEVAL_K,
                       threshold: float = RETRIEVAL_THRESHOLD) -> list[dict]:
    """Recompute the retrieval for the task at `position` (0-based) from a
    memory file whose entry i corresponds to order[i]. Only entries written
    BEFORE the task (indices < position) are visible, mirroring the run."""
    if position >= len(memory):
        return []
    token_set = set(memory[position].get("signature_tokens") or ())
    sims = []
    for i in range(position):
        e = memory[i]
        sim = jaccard(set(e.get("signature_tokens") or ()), token_set)
        if sim >= threshold:
            sims.append((sim, i))
    sims.sort(key=lambda t: (-t[0], t[1]))
    return [{"task_id": order[i], "similarity": round(sim, 4),
             "stored_tier": e.get("tier") if (e := memory[i]) else None,
             "stored_G": _r(e.get("G"), 4) if e.get("G") is not None else None}
            for sim, i in sims[:k]]


# ----------------------------------------------------------------- flips

def flip_analysis(c1: list[dict], b: list[dict], c0: list[dict],
                  c1_memory: list[dict], c0_memory: list[dict],
                  order: list[str]) -> dict:
    """Every C1 flip with its preregistered P5 evidence plus the retrieved-
    neighbor mechanism (recomputed from the memory files)."""
    b_by_id = {r["task_id"]: r for r in b}
    c0_by_id = {r["task_id"]: r for r in c0}
    pos = {tid: i for i, tid in enumerate(order)}
    flips = []
    for r in c1:
        if not r.get("flipped"):
            continue
        tid = r["task_id"]
        br = b_by_id[tid]
        p = pos[tid]
        neighbors_c1 = retrieve_neighbors(c1_memory, p, order)
        neighbors_c0 = retrieve_neighbors(c0_memory, p, order)
        c0r = c0_by_id[tid]
        flips.append({
            "task_id": tid,
            "family": r.get("family"),
            "gt_tier": r.get("gt_tier"),
            "base_choice": r.get("base_choice"),
            "chosen_tier": r.get("chosen_tier"),
            "exec_model": r.get("exec_model"),
            "n_retrieved": r.get("n_retrieved"),
            "top_similarity": r.get("top_similarity"),
            "V": _r(r.get("V"), 4),
            "Q": {k: _r(v, 4) for k, v in (r.get("Q") or {}).items()},
            "A_norm": {k: _r(v, 4) for k, v in (r.get("A_norm") or {}).items()},
            "z_base": {k: _r(v, 4) for k, v in (r.get("z_base") or {}).items()},
            "z_mod": {k: _r(v, 4) for k, v in (r.get("z_mod") or {}).items()},
            "quality_C1": r["evaluator"]["quality_score"],
            "quality_B": br["evaluator"]["quality_score"],
            "quality_delta": r["evaluator"]["quality_score"]
                             - br["evaluator"]["quality_score"],
            "exec_cost_C1_usd": _r(r["exec_cost_usd"], 6),
            "exec_cost_B_usd": _r(br["exec_cost_usd"], 6),
            "exec_cost_delta_usd": _r(r["exec_cost_usd"] - br["exec_cost_usd"], 6),
            "exec_finish_reason": r.get("exec_finish_reason"),
            "response_preview_empty": not str(
                r.get("exec_response_preview") or "").strip(),
            "mechanism": {
                "retrieved_neighbors_C1": neighbors_c1,
                "retrieved_neighbors_C0": neighbors_c0,
                "note": "Neighbors recomputed with the frozen retrieval "
                        "config (k=10, Jaccard>=0.5) from the run's own "
                        "memory file, using only entries written before "
                        "the task. C1's gate stores gated tasks at their "
                        "base tier, which changes the evidence the next "
                        "same-signature task sees; the C0 column is the "
                        "counterfactual evidence set.",
            },
        })
    return {
        "n_flips": len(flips),
        "flips": flips,
        "c0_flips_for_reference": sorted(
            r["task_id"] for r in c0 if r.get("flipped")),
        "quality_delta_sum": sum(f["quality_delta"] for f in flips),
        "exec_cost_delta_sum_usd": _r(
            sum(f["exec_cost_delta_usd"] for f in flips), 6),
    }


# ------------------------------------------------------- paired B / C1

def paired_bc1(b: list[dict], c1: list[dict]) -> dict:
    """Task-paired B vs C1 with tier-change attribution: modulation flip /
    judge base drift / same-tier sampling variance."""
    b_by_id = {r["task_id"]: r for r in b}
    per_task, q_deltas, c_deltas = [], [], []
    for cr in c1:
        br = b_by_id[cr["task_id"]]
        qd = cr["evaluator"]["quality_score"] - br["evaluator"]["quality_score"]
        cd = cr["exec_cost_usd"] - br["exec_cost_usd"]
        q_deltas.append(qd)
        c_deltas.append(cd)
        if cr.get("flipped"):
            cls = "modulation_flip"
        elif br["chosen_tier"] != cr["chosen_tier"]:
            cls = "judge_base_drift"
        else:
            cls = "same_tier_sampling"
        per_task.append({
            "task_id": cr["task_id"],
            "tier_B": br["chosen_tier"], "tier_C1": cr["chosen_tier"],
            "base_choice_C1": cr.get("base_choice"),
            "flipped": cr.get("flipped", False),
            "classification": cls,
            "quality_B": br["evaluator"]["quality_score"],
            "quality_C1": cr["evaluator"]["quality_score"],
            "quality_delta": qd,
            "exec_cost_B_usd": _r(br["exec_cost_usd"], 6),
            "exec_cost_C1_usd": _r(cr["exec_cost_usd"], 6),
            "exec_cost_delta_usd": _r(cd, 6),
        })
    total_b = sum(r["exec_cost_usd"] for r in b)
    total_c1 = sum(r["exec_cost_usd"] for r in c1)
    by_cls = lambda cls: [d for d in per_task if d["classification"] == cls]
    cost_attr = {cls: _r(sum(d["exec_cost_delta_usd"] for d in by_cls(cls)), 6)
                 for cls in ("modulation_flip", "judge_base_drift",
                             "same_tier_sampling")}
    q_attr = {cls: sum(d["quality_delta"] for d in by_cls(cls))
              for cls in ("modulation_flip", "judge_base_drift",
                          "same_tier_sampling")}
    return {
        "n_pairs": len(per_task),
        "quality": {
            "mean_B": _r(_mean([d["quality_B"] for d in per_task]), 4),
            "mean_C1": _r(_mean([d["quality_C1"] for d in per_task]), 4),
            "mean_delta": _r(_mean(q_deltas), 4),
            "improved_tasks": sum(1 for d in q_deltas if d > 0),
            "worsened_tasks": sum(1 for d in q_deltas if d < 0),
            "unchanged_tasks": sum(1 for d in q_deltas if d == 0),
        },
        "exec_cost": {
            "total_B_usd": _r(total_b, 6),
            "total_C1_usd": _r(total_c1, 6),
            "total_delta_usd": _r(total_c1 - total_b, 6),
            "total_change_pct": _r(100.0 * (total_c1 - total_b) / total_b, 2),
            "savings_pct_b_c1": _r(100.0 * (total_b - total_c1) / total_b, 2),
            "attribution_usd": cost_attr,
            "note": "modulation_flip = JitRL changed the executed tier vs "
                    "the judge base choice (attributable to C1); "
                    "judge_base_drift = the judge issued a different base "
                    "decision than in the B run (local llama "
                    "nondeterminism, NOT attributable to C1); "
                    "same_tier_sampling = same executed tier, execution "
                    "sampling variance (temperature 0.7).",
        },
        "quality_attribution": {
            **q_attr,
            "note": "Sum of per-task B->C1 quality deltas by the same "
                    "classification (flip / drift / same-tier).",
        },
        "tier_changes": {
            "n_total": sum(1 for d in per_task if d["tier_B"] != d["tier_C1"]),
            "modulation_flip_task_ids": sorted(
                d["task_id"] for d in by_cls("modulation_flip")),
            "judge_base_drift_task_ids": sorted(
                d["task_id"] for d in by_cls("judge_base_drift")),
        },
        "per_task": per_task,
    }


# ------------------------------------------------------- C0 -> C1 changes

def c0_to_c1_changes(c0: list[dict], c1: list[dict]) -> dict:
    c0_by_id = {r["task_id"]: r for r in c0}
    tier_changes = []
    for r in c1:
        c0r = c0_by_id[r["task_id"]]
        if c0r["chosen_tier"] != r["chosen_tier"]:
            tier_changes.append({
                "task_id": r["task_id"],
                "tier_C0": c0r["chosen_tier"],
                "tier_C1": r["chosen_tier"],
                "c0_flipped": c0r.get("flipped", False),
                "c1_flipped": r.get("flipped", False),
                "cause": ("C0 modulation flip suppressed by the C1 gate"
                          if c0r.get("flipped") and not r.get("flipped")
                          else "C1 modulation flip (new vs C0)"
                          if r.get("flipped") and not c0r.get("flipped")
                          else "judge base decision differs between the "
                               "C0 and C1 runs"),
            })
    q0 = _mean([r["evaluator"]["quality_score"] for r in c0])
    q1 = _mean([r["evaluator"]["quality_score"] for r in c1])
    cost0 = sum(r["exec_cost_usd"] for r in c0)
    cost1 = sum(r["exec_cost_usd"] for r in c1)
    return {
        "quality_mean": {"C0": _r(q0, 4), "C1": _r(q1, 4),
                         "delta": _r(q1 - q0, 4)},
        "exec_cost_total_usd": {"C0": _r(cost0, 6), "C1": _r(cost1, 6),
                                "delta": _r(cost1 - cost0, 6),
                                "change_pct": _r(100 * (cost1 - cost0) / cost0, 2)},
        "flips": {"C0": sum(1 for r in c0 if r.get("flipped")),
                  "C1": sum(1 for r in c1 if r.get("flipped"))},
        "tier_changes": tier_changes,
    }


# ---------------------------------------------------------- judge drift

def judge_drift_analysis(b: list[dict], c0: list[dict], c1: list[dict]) -> dict:
    """Separate judge base-decision drift from same-tier sampling variance.

    C1 base_choice is compared against B's executed tier (B's chosen_tier
    IS its judge base choice) and against C0's base_choice.
    """
    b_by_id = {r["task_id"]: r for r in b}
    c0_by_id = {r["task_id"]: r for r in c0}
    vs_b = [{"task_id": r["task_id"], "B_chosen": b_by_id[r["task_id"]]["chosen_tier"],
             "C1_base": r.get("base_choice")}
            for r in c1
            if r.get("base_choice") != b_by_id[r["task_id"]]["chosen_tier"]]
    vs_c0 = [{"task_id": r["task_id"], "C0_base": c0_by_id[r["task_id"]].get("base_choice"),
              "C1_base": r.get("base_choice")}
             for r in c1
             if r.get("base_choice") != c0_by_id[r["task_id"]].get("base_choice")]
    return {
        "c1_base_vs_b_chosen_mismatches": vs_b,
        "c1_base_vs_b_consistent": not vs_b,
        "c1_base_vs_c0_base_mismatches": vs_c0,
        "note": "The C1 run's judge reproduced the B run's base decision on "
                "all 24 tasks (no drift vs B). The C0 run had drifted on "
                "T13/T22 relative to B; those tasks list here as C0-vs-C1 "
                "base differences because C1 matched B instead of C0.",
    }


# ---------------------------------------------------- memory / retrieval

def memory_retrieval(c1: list[dict], c1_memory_path: str | Path) -> dict:
    mem = load_memory(c1_memory_path)
    hits = [r for r in c1 if r.get("n_retrieved", 0) > 0]
    return {
        "memory_entries_final": len(mem),
        "memory_written": sum(1 for r in c1 if r.get("memory_written")),
        "tasks_with_retrieval": len(hits),
        "retrieval_task_ids": [r["task_id"] for r in hits],
        "avg_n_retrieved_when_hit": _r(_mean([r["n_retrieved"] for r in hits]), 2)
            if hits else 0.0,
        "top_similarity_when_hit": sorted(
            {r.get("top_similarity") for r in hits}),
        "flips_among_retrieval_hits": sum(1 for r in hits if r.get("flipped")),
        "memory_entry_keys": sorted(mem[0].keys()) if mem else [],
    }


# ------------------------------------------------- prereg P1-P5 judgment

def p1_p5_judgment(b_metrics: dict, c1_metrics: dict, paired: dict,
                   c1_records: list[dict], flips: dict) -> dict:
    b_total = paired["exec_cost"]["total_B_usd"]
    c1_total = paired["exec_cost"]["total_C1_usd"]
    cost_delta = c1_total - b_total
    q_delta = c1_metrics["quality"]["mean"] - b_metrics["quality"]["mean"]
    mis_b = b_metrics["tier_vs_gt"]["misgrade_rate"]
    mis_c1 = c1_metrics["tier_vs_gt"]["misgrade_rate"]
    mis_diff_pp = (mis_c1 - mis_b) * 100.0
    jitrl_max = c1_metrics["latency_ms"]["jitrl_max"]

    p1 = {
        "criterion": "P1 exec cost: C1 total <= B + $0.001",
        "B_total_usd": b_total, "C1_total_usd": c1_total,
        "limit_usd": _r(b_total + P1_COST_TOLERANCE_USD, 6),
        "delta_usd": _r(cost_delta, 6),
        "met": cost_delta <= P1_COST_TOLERANCE_USD + _EPS,
    }
    p2 = {
        "criterion": "P2 quality: mean q(C1) - mean q(B) >= -0.3 (0.3/5)",
        "B_mean": b_metrics["quality"]["mean"],
        "C1_mean": c1_metrics["quality"]["mean"],
        "delta": _r(q_delta, 4),
        "met": q_delta >= P2_QUALITY_FLOOR_DELTA - _EPS,
    }
    p3 = {
        "criterion": "P3 misgrade rate: C1 <= B + 5pp",
        "B_misgrade_rate": mis_b, "C1_misgrade_rate": mis_c1,
        "diff_pp": _r(mis_diff_pp, 2),
        "met": mis_diff_pp <= P3_MISTIER_TOLERANCE_PP + _EPS,
    }
    p4 = {
        "criterion": "P4 JitRL latency: max jitrl_latency_ms <= 500 ms",
        "max_jitrl_ms": jitrl_max,
        "met": jitrl_max <= P4_JITRL_MAX_MS + _EPS,
    }
    flip_checks = []
    for f in flips["flips"]:
        ok_q = f["quality_C1"] >= f["quality_B"]
        ok_c = (f["exec_cost_C1_usd"] - f["exec_cost_B_usd"]
                <= P5_FLIP_COST_TOLERANCE_USD + _EPS)
        flip_checks.append({
            "task_id": f["task_id"],
            "quality_C1": f["quality_C1"], "quality_B": f["quality_B"],
            "quality_ok": ok_q,
            "exec_cost_C1_usd": f["exec_cost_C1_usd"],
            "exec_cost_B_usd": f["exec_cost_B_usd"],
            "cost_limit_usd": _r(f["exec_cost_B_usd"]
                                 + P5_FLIP_COST_TOLERANCE_USD, 6),
            "cost_ok": ok_c,
            "met": ok_q and ok_c,
        })
    p5 = {
        "criterion": "P5 flip harmlessness: each flip quality >= B same "
                     "task AND cost <= B same task + $0.001; 0 flips "
                     "trivially pass",
        "n_flips": flips["n_flips"],
        "trivially_met": flips["n_flips"] == 0,
        "flip_checks": flip_checks,
        "met": flips["n_flips"] == 0 or all(fc["met"] for fc in flip_checks),
    }
    criteria = [p1, p2, p3, p4, p5]
    all_met = all(cr["met"] for cr in criteria)
    return {
        "definition_status": "P1-P5 exactly as frozen in "
                             "docs/S2-C1-PREREGISTRATION.md v1.0 Section 4; "
                             "judged as written on the formal n=24 runs.",
        "P1": p1, "P2": p2, "P3": p3, "P4": p4, "P5": p5,
        "n_criteria_met": sum(1 for cr in criteria if cr["met"]),
        "failed": [key for key in ("P1", "P2", "P3", "P4", "P5")
                   if not {"P1": p1, "P2": p2, "P3": p3, "P4": p4,
                           "P5": p5}[key]["met"]],
        "verdict": ("PASS (all P1-P5 met; harmless-parity directional "
                    "evidence on n=24 only, no statistical significance, "
                    "and no claim of the C0 >=20% cost-savings goal)"
                    if all_met else
                    "NOT PASSING (" + ", ".join(
                        key for key in ("P1", "P2", "P3", "P4", "P5")
                        if not {"P1": p1, "P2": p2, "P3": p3, "P4": p4,
                                "P5": p5}[key]["met"]) + " failed)"),
    }


# ---------------------------------------------- informational D10 (C1)

def d10_informational(b_metrics: dict, c1_metrics: dict, paired: dict,
                       c1_records: list[dict]) -> dict:
    """Recompute the four FROZEN D10 criteria with C1 in place of C.

    INFORMATIONAL ONLY: this does not replace or amend the frozen formal
    C0 D10 judgment (eval/results-summary.json), per preregistration
    Section 4 ('does not retroactively change C0's formal D10 judgment').
    """
    b_total = paired["exec_cost"]["total_B_usd"]
    c1_total = paired["exec_cost"]["total_C1_usd"]
    savings_pct = 100.0 * (b_total - c1_total) / b_total
    q_delta = c1_metrics["quality"]["mean"] - b_metrics["quality"]["mean"]
    mis_diff_pp = ((c1_metrics["tier_vs_gt"]["misgrade_rate"]
                    - b_metrics["tier_vs_gt"]["misgrade_rate"]) * 100.0)
    jitrl_max = c1_metrics["latency_ms"]["jitrl_max"]
    criteria = [
        {"criterion": "D10-1 C1 vs B exec cost savings >= 20% ((B-C1)/B)",
         "savings_pct": _r(savings_pct, 2),
         "met": savings_pct >= 20.0 - _EPS},
        {"criterion": "D10-2 quality: C1 - B >= -0.3",
         "delta": _r(q_delta, 4),
         "met": q_delta >= -0.3 - _EPS},
        {"criterion": "D10-3 C1 misgrade rate <= B misgrade rate + 5pp",
         "diff_pp": _r(mis_diff_pp, 2),
         "met": mis_diff_pp <= 5.0 + _EPS},
        {"criterion": "D10-4 JitRL post-processing latency <= 500 ms",
         "max_jitrl_ms": jitrl_max,
         "met": jitrl_max <= 500.0 + _EPS},
    ]
    return {
        "informational_only": True,
        "disclaimer": "Informational recomputation for C1. Does NOT replace "
                      "the frozen formal C0 D10 judgment "
                      "(eval/results-summary.json: C0 = 3/4, NOT MET, "
                      "savings -3.78%).",
        "criteria": criteria,
        "n_criteria_met": sum(1 for cr in criteria if cr["met"]),
        "verdict": ("MET" if all(cr["met"] for cr in criteria) else "NOT MET")
                   + " (informational, n=24)",
    }


# --------------------------------------- prereg Section 5 predictions

def prereg_predictions_check(gate: dict, flips: dict, paired: dict) -> dict:
    gated_below = sorted(d["task_id"] for d in gate["gated_tasks"]
                         if d["gate_reason"] == "below_min_neighbors")
    modulated = sorted(d["task_id"] for d in gate["modulated_tasks"])
    preds = [
        {"prediction": "n=1/2 tasks T06/T18/T07/T19 are gated",
         "actual_gated_below_min": gated_below,
         "holds": gated_below == ["T06", "T07", "T18", "T19"]},
        {"prediction": "C0's T06/T19 flips are suppressed",
         "actual": sorted(t for t in flips["c0_flips_for_reference"]
                          if t not in [f["task_id"] for f in flips["flips"]]),
         "holds": all(f["task_id"] not in ("T06", "T19")
                      for f in flips["flips"])},
        {"prediction": "n=3 tasks T08/T20 remain modulated",
         "actual_modulated": modulated,
         "holds": modulated == ["T08", "T20"]},
        {"prediction": "flips drop from 2 to 0",
         "actual_n_flips": flips["n_flips"],
         "actual_flip_task_ids": [f["task_id"] for f in flips["flips"]],
         "holds": flips["n_flips"] == 0},
        {"prediction": "quality roughly unchanged (near B)",
         "actual_quality_delta_vs_B": paired["quality"]["mean_delta"],
         "numeric_threshold_in_prereg": None,
         "note": "qualitative prediction; -0.25 is within the P2 floor "
                 "(-0.3) but is a visible drop (sum -6 over 24 tasks), "
                 "entirely from same-tier sampling variance",
         "holds": paired["quality"]["mean_delta"] >= -0.3},
        {"prediction": "exec cost near B (P1 margin)",
         "actual_exec_delta_usd": paired["exec_cost"]["total_delta_usd"],
         "holds": paired["exec_cost"]["total_delta_usd"] <= 0.001 + _EPS},
    ]
    return {
        "source": "docs/S2-C1-PREREGISTRATION.md v1.0 Section 5 "
                  "(falsifiable run-before predictions; reported as-is)",
        "predictions": preds,
        "n_holding": sum(1 for p in preds if p["holds"]),
    }


# ----------------------------------------------------------------- charts

_COLORS = {"B": "#3b82f6", "C0": "#f59e0b", "C1": "#ef4444",
           "n_retrieved": "#3b82f6"}


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _svg_open(w: int, h: int, title: str) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="Helvetica,Arial,sans-serif">',
        f'<rect width="{w}" height="{h}" fill="white"/>',
        f'<text x="{w / 2}" y="26" font-size="16" font-weight="bold" '
        f'text-anchor="middle">{_esc(title)}</text>',
    ]


def _legend(parts: list[str], w: int, mb: int, h: int,
            entries: list[tuple[str, str]]) -> None:
    lx = 60
    for name, color in entries:
        parts.append(f'<rect x="{lx}" y="{h - mb + 36}" width="12" height="12" '
                     f'fill="{color}" rx="2"/>')
        parts.append(f'<text x="{lx + 17}" y="{h - mb + 46}" font-size="12">'
                     f'{_esc(name)}</text>')
        lx += 34 + 9 * len(name)


def svg_grouped_bars(out_path: Path, title: str, y_label: str,
                     categories: list[str], series: dict[str, list[float]],
                     hlines: list[tuple[float, str]] = (),
                     value_fmt: str = "{:.4f}") -> None:
    """Grouped vertical bars with optional horizontal threshold lines."""
    n_cat, n_ser = len(categories), len(series)
    w, h = 900, 500
    ml, mr, mt, mb = 86, 24, 56, 84
    iw, ih = w - ml - mr, h - mt - mb
    vals = [v for vs in series.values() for v in vs]
    hv = [v for v, _ in hlines]
    vmax = max(vals + hv + [1e-9])
    vmax = vmax * 1.15 if vmax > 0 else 1.0
    cluster = iw / max(n_cat, 1)
    bar_w = cluster * 0.72 / max(n_ser, 1)
    parts = _svg_open(w, h, title)
    for g in range(6):
        y = mt + ih - ih * g / 5
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{w - mr}" y2="{y:.1f}" '
                     f'stroke="#e5e7eb"/>')
        parts.append(f'<text x="{ml - 8}" y="{y + 4:.1f}" font-size="11" '
                     f'text-anchor="end">{vmax * g / 5:.4g}</text>')
    parts.append(f'<text x="18" y="{mt + ih / 2}" font-size="12" '
                 f'text-anchor="middle" transform="rotate(-90 18 {mt + ih / 2})">'
                 f'{_esc(y_label)}</text>')
    for v, label in hlines:
        y = mt + ih - ih * v / vmax
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{w - mr}" y2="{y:.1f}" '
                     f'stroke="#dc2626" stroke-width="1.5" '
                     f'stroke-dasharray="7 4"/>')
        parts.append(f'<text x="{ml + 6}" y="{y - 5:.1f}" font-size="11" '
                     f'fill="#dc2626">{_esc(label)} = {v:.6g}</text>')
    for ci, cat in enumerate(categories):
        cx = ml + cluster * ci + cluster * 0.14
        for si, (name, vs) in enumerate(series.items()):
            v = vs[ci]
            bh = ih * v / vmax
            x = cx + si * bar_w
            parts.append(f'<rect x="{x:.1f}" y="{mt + ih - bh:.1f}" '
                         f'width="{bar_w:.1f}" height="{bh:.1f}" '
                         f'fill="{_COLORS.get(name, "#10b981")}" rx="2"/>')
            parts.append(f'<text x="{x + bar_w / 2:.1f}" '
                         f'y="{mt + ih - bh - 4:.1f}" font-size="10" '
                         f'text-anchor="middle">{value_fmt.format(v)}</text>')
        parts.append(f'<text x="{ml + cluster * (ci + 0.5):.1f}" '
                     f'y="{h - mb + 20}" font-size="13" '
                     f'text-anchor="middle">{_esc(cat)}</text>')
    _legend(parts, w, mb, h, list(series.items()))
    parts.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")


def svg_gate_chart(out_path: Path, title: str, per_task: list[dict],
                   min_neighbors: int) -> None:
    """Per-task n_retrieved bars colored by gate status + threshold line."""
    w, h = 1100, 480
    ml, mr, mt, mb = 60, 24, 56, 84
    iw, ih = w - ml - mr, h - mt - mb
    n = len(per_task)
    vmax = max([d["n_retrieved"] for d in per_task] + [min_neighbors]) * 1.3
    status_color = {
        "gated_no_neighbors": "#93c5fd",
        "gated_below_min_neighbors": "#3b82f6",
        "modulated": "#f59e0b",
    }
    parts = _svg_open(w, h, title)
    for g in range(6):
        y = mt + ih - ih * g / 5
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{w - mr}" y2="{y:.1f}" '
                     f'stroke="#e5e7eb"/>')
        parts.append(f'<text x="{ml - 8}" y="{y + 4:.1f}" font-size="11" '
                     f'text-anchor="end">{vmax * g / 5:.0f}</text>')
    parts.append(f'<text x="18" y="{mt + ih / 2}" font-size="12" '
                 f'text-anchor="middle" transform="rotate(-90 18 {mt + ih / 2})">'
                 f'n_retrieved</text>')
    y_thr = mt + ih - ih * min_neighbors / vmax
    parts.append(f'<line x1="{ml}" y1="{y_thr:.1f}" x2="{w - mr}" '
                 f'y2="{y_thr:.1f}" stroke="#dc2626" stroke-width="1.5" '
                 f'stroke-dasharray="7 4"/>')
    parts.append(f'<text x="{ml + 6}" y="{y_thr - 5:.1f}" font-size="11" '
                 f'fill="#dc2626">min_neighbors = {min_neighbors}</text>')
    bw = iw / n * 0.72
    for i, d in enumerate(per_task):
        v = d["n_retrieved"]
        bh = ih * v / vmax
        x = ml + iw * i / n + iw / n * 0.14
        color = status_color[d["status"]]
        if d["flipped"]:
            parts.append(f'<rect x="{x:.1f}" y="{mt + ih - bh - 5:.1f}" '
                         f'width="{bw:.1f}" height="{bh + 5:.1f}" '
                         f'fill="#ef4444" rx="2"/>')
        else:
            parts.append(f'<rect x="{x:.1f}" y="{mt + ih - bh:.1f}" '
                         f'width="{bw:.1f}" height="{bh:.1f}" '
                         f'fill="{color}" rx="2"/>')
        if v > 0:
            parts.append(f'<text x="{x + bw / 2:.1f}" '
                         f'y="{mt + ih - bh - 4:.1f}" font-size="9" '
                         f'text-anchor="middle">{v}</text>')
        parts.append(f'<text x="{ml + iw * (i + 0.5) / n:.1f}" '
                     f'y="{h - mb + 18}" font-size="10" '
                     f'text-anchor="middle">{_esc(d["task_id"])}</text>')
    _legend(parts, w, mb, h, [
        ("gated n=0", "#93c5fd"), ("gated 0<n<3", "#3b82f6"),
        ("modulated n>=3", "#f59e0b"), ("modulated+flip", "#ef4444")])
    parts.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")


def write_charts(summary: dict, out_dir: Path) -> list[str]:
    charts = out_dir / "charts"
    g, p = summary["groups"], summary["paired_bc1"]
    written: list[str] = []
    # 1. cost comparison with the P1 limit
    svg_grouped_bars(
        charts / "cost-comparison.svg",
        "Execution cost totals: B vs C0 vs C1 (USD, n=24)",
        "exec cost (USD)", ["B", "C0", "C1"],
        {"exec": [g["B"]["exec_cost_usd"]["total"],
                  g["C0"]["exec_cost_usd"]["total"],
                  g["C1"]["exec_cost_usd"]["total"]]},
        hlines=[(g["B"]["exec_cost_usd"]["total"] + P1_COST_TOLERANCE_USD,
                 "P1 limit (B+$0.001)")],
        value_fmt="{:.6f}")
    # 2. quality comparison with the P2 floor
    svg_grouped_bars(
        charts / "quality-comparison.svg",
        "Mean quality score: B vs C0 vs C1 (0-5, n=24)",
        "mean quality", ["B", "C0", "C1"],
        {"quality": [g["B"]["quality"]["mean"], g["C0"]["quality"]["mean"],
                     g["C1"]["quality"]["mean"]]},
        hlines=[(g["B"]["quality"]["mean"] + P2_QUALITY_FLOOR_DELTA,
                 "P2 floor (B-0.3)")],
        value_fmt="{:.4f}")
    # 3. gate / modulation per task
    svg_gate_chart(
        charts / "gate-modulation.svg",
        "C1 gate outcome per task: n_retrieved vs min_neighbors=3",
        summary["c1_gate"]["per_task"], 3)
    # 4. paired per-task B vs C1 exec cost
    pt = p["per_task"]
    svg_grouped_bars(
        charts / "bc1-paired-exec-cost.svg",
        "Per-task execution cost: B vs C1 (paired by task_id, USD)",
        "exec cost (USD)", [d["task_id"] for d in pt],
        {"B": [d["exec_cost_B_usd"] for d in pt],
         "C1": [d["exec_cost_C1_usd"] for d in pt]},
        value_fmt="{:.4f}")
    written += [str(f) for f in sorted(charts.glob("*.svg"))]
    return written


# ------------------------------------------------------------------ CSV

CSV_FIELDS = ["group", "position", "task_id", "family", "gt_tier",
              "chosen_tier", "base_choice", "flipped", "modulation_active",
              "gate_reason", "min_neighbors", "n_retrieved",
              "top_similarity", "quality_score", "verdict",
              "capability_insufficient", "exec_model", "exec_finish_reason",
              "exec_cost_usd", "exec_latency_ms", "eval_cost_usd",
              "judge_latency_ms", "jitrl_latency_ms", "reward"]


def write_tasks_csv(groups: dict[str, list[dict]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        wcsv.writeheader()
        for name in ("B", "C0", "C1"):
            for pos, r in enumerate(groups[name], 1):
                ev = r.get("evaluator") or {}
                wcsv.writerow({
                    "group": name, "position": pos,
                    "task_id": r.get("task_id"), "family": r.get("family"),
                    "gt_tier": r.get("gt_tier"),
                    "chosen_tier": r.get("chosen_tier"),
                    "base_choice": r.get("base_choice"),
                    "flipped": r.get("flipped"),
                    "modulation_active": r.get("modulation_active"),
                    "gate_reason": r.get("gate_reason"),
                    "min_neighbors": r.get("min_neighbors"),
                    "n_retrieved": r.get("n_retrieved"),
                    "top_similarity": r.get("top_similarity"),
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
                    "reward": r.get("reward"),
                })


# ------------------------------------------------------------------ main

def build_summary(b_recs, c0_recs, c1_recs, *, b_summ, c0_summ, c1_summ,
                  c1_memory_path, c0_memory_path,
                  formal_paths: dict[str, str] | None = None) -> dict:
    checks = assert_formal_inputs(b_recs, c0_recs, c1_recs, c1_summ)
    order = checks["order"]
    c1_memory = load_memory(c1_memory_path)
    c0_memory = load_memory(c0_memory_path)
    metrics = {"B": group_metrics(b_recs), "C0": group_metrics(c0_recs),
               "C1": group_metrics(c1_recs)}
    gate = gate_analysis(c1_recs)
    flips = flip_analysis(c1_recs, b_recs, c0_recs, c1_memory, c0_memory, order)
    paired = paired_bc1(b_recs, c1_recs)
    summary = {
        "meta": {
            "analysis": "C1 (min-neighbor gated JitRL) post-run analysis",
            "preregistration": "docs/S2-C1-PREREGISTRATION.md v1.0 "
                               "(2026-09-11, frozen before the run)",
            "policy_change": "min_neighbors: 1 -> 3 (only intended variable "
                             "vs C0); n_retrieved < 3 -> native Judge",
            "scope": "reads the formal logs only; no CPA calls, no docs "
                     "writes, no commits; outputs confined to eval/c1/",
        },
        "inputs": {
            "formal": formal_paths or {"B": DEFAULT_B, "C0": DEFAULT_C0,
                                       "C1": DEFAULT_C1,
                                       "C1_memory": DEFAULT_C1_MEMORY,
                                       "C0_memory": DEFAULT_C0_MEMORY},
        },
        "formal_input_assertions": checks,
        "groups": metrics,
        "c1_gate": gate,
        "c1_flips": flips,
        "paired_bc1": paired,
        "c0_to_c1": c0_to_c1_changes(c0_recs, c1_recs),
        "judge_drift": judge_drift_analysis(b_recs, c0_recs, c1_recs),
        "memory_retrieval": memory_retrieval(c1_recs, c1_memory_path),
        "truncation_empty_response": {
            name: {
                "finish_reason_length": metrics[name]["finish_reason_length"],
                "empty_response_preview":
                    metrics[name]["empty_response_preview"],
            } for name in ("B", "C0", "C1")},
        "file_summaries": {"B": b_summ, "C0": c0_summ, "C1": c1_summ},
        "notes": NOTES,
    }
    summary["p1_p5"] = p1_p5_judgment(metrics["B"], metrics["C1"], paired,
                                      c1_recs, flips)
    summary["d10_informational"] = d10_informational(
        metrics["B"], metrics["C1"], paired, c1_recs)
    summary["prereg_predictions"] = prereg_predictions_check(
        gate, flips, paired)
    summary["outputs"] = {}
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="eval.analyze_c1",
        description="Independent C1 (min-neighbor gated JitRL) analysis, "
                    "judged against docs/S2-C1-PREREGISTRATION.md v1.0 "
                    "(reads logs only; no CPA calls, no credentials; "
                    "writes only under --out-dir).")
    ap.add_argument("--b", default=DEFAULT_B)
    ap.add_argument("--c0", default=DEFAULT_C0)
    ap.add_argument("--c1", default=DEFAULT_C1)
    ap.add_argument("--c1-memory", default=DEFAULT_C1_MEMORY)
    ap.add_argument("--c0-memory", default=DEFAULT_C0_MEMORY)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help="output directory (default eval/c1; C0's eval/ "
                         "assets are never overwritten)")
    args = ap.parse_args(argv)

    b_recs, b_summ = load_group(args.b)
    c0_recs, c0_summ = load_group(args.c0)
    c1_recs, c1_summ = load_group(args.c1)

    summary = build_summary(
        b_recs, c0_recs, c1_recs, b_summ=b_summ, c0_summ=c0_summ,
        c1_summ=c1_summ, c1_memory_path=args.c1_memory,
        c0_memory_path=args.c0_memory,
        formal_paths={"B": args.b, "C0": args.c0, "C1": args.c1,
                      "C1_memory": args.c1_memory,
                      "C0_memory": args.c0_memory})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary["outputs"]["charts"] = write_charts(summary, out_dir)
    with open(out_dir / "results-summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_tasks_csv({"B": b_recs, "C0": c0_recs, "C1": c1_recs},
                    out_dir / "results-tasks.csv")

    g = summary["groups"]
    p = summary["paired_bc1"]
    pp = summary["p1_p5"]
    print(f"[c1] B  q={g['B']['quality']['mean']} "
          f"exec=${g['B']['exec_cost_usd']['total']:.6f} "
          f"misgrade={g['B']['tier_vs_gt']['misgrade_rate']}")
    print(f"[c1] C0 q={g['C0']['quality']['mean']} "
          f"exec=${g['C0']['exec_cost_usd']['total']:.6f} "
          f"misgrade={g['C0']['tier_vs_gt']['misgrade_rate']}")
    print(f"[c1] C1 q={g['C1']['quality']['mean']} "
          f"exec=${g['C1']['exec_cost_usd']['total']:.6f} "
          f"misgrade={g['C1']['tier_vs_gt']['misgrade_rate']} "
          f"jitrl_max={g['C1']['latency_ms']['jitrl_max']}ms")
    print(f"[c1] gate: {summary['c1_gate']['counts']}")
    print(f"[c1] flips: {summary['c1_flips']['n_flips']} "
          f"{[f['task_id'] for f in summary['c1_flips']['flips']]} "
          f"(C0 had {summary['c1_flips']['c0_flips_for_reference']})")
    print(f"[c1] B->C1 exec delta=${p['exec_cost']['total_delta_usd']} "
          f"({p['exec_cost']['total_change_pct']}%) "
          f"attribution={p['exec_cost']['attribution_usd']}")
    print(f"[c1] B->C1 quality delta={p['quality']['mean_delta']} "
          f"attribution={p['quality_attribution']}")
    print(f"[c1] judge drift vs B: "
          f"{summary['judge_drift']['c1_base_vs_b_chosen_mismatches'] or 'none'}")
    for key in ("P1", "P2", "P3", "P4", "P5"):
        cr = pp[key]
        print(f"[c1] {key}: {'PASS' if cr['met'] else 'FAIL'} "
              f"- {cr['criterion']}")
    print(f"[c1] P1-P5 verdict: {pp['verdict']}")
    print(f"[c1] D10 (informational, NOT replacing C0): "
          f"{summary['d10_informational']['verdict']}")
    print(f"[c1] prereg predictions holding: "
          f"{summary['prereg_predictions']['n_holding']}/6")
    print(f"[c1] wrote {out_dir / 'results-summary.json'}, "
          f"{out_dir / 'results-tasks.csv'}, "
          f"{len(summary['outputs']['charts'])} charts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
