"""Session-3 Table-8 ablation analysis: T8-LM vs T8-PM (prereg v1.0, frozen).

Preregistration: S3-TAB8-PREREGISTRATION.md v1.0 (2026-09-11, frozen before
any run). This analyzer judges the FORMAL T8 inputs only, exactly as the
preregistration prescribes.

Formal inputs (prereg §6):
    LM = logs/s3_ablation_T8_LM.jsonl   (or _retry1/_retry2)
    PM = logs/s3_ablation_T8_PM.jsonl
    memory trace = eval/ablation/frozen-c0-memory-trace.jsonl
    trace hashes  = eval/ablation/frozen-c0-memory-trace.sha256.json

HARD ASSERTIONS (prereg §10.12) — the analyzer refuses (SystemExit, nonzero)
any formal input where:
  - the task order of either arm is not EXACTLY the frozen 24-task order;
  - the frozen config differs (JitRL snapshot, exec/eval token budgets,
    temperatures, read-only/prefix-replay flags, trace hash);
  - a hash mismatches (trace file vs sidecar vs per-record
    memory_source_sha256; the C0 source memory file vs sidecar);
  - 24/24 completeness is violated (record count, duplicates, any
    execution or Evaluator failure, reward missing, episodes != 1);
  - mechanism purity is violated (LM must never inject: injected_block_chars
    == 0; PM must never modulate: z_mod is null, A_norm all-zero);
  - the per-record LM math / PM choices do not reproduce from the recorded
    fields (z_mod == z_base + beta*A_norm; chosen == argmax(z_mod|z_prompt);
    base_choice == argmax(z_base));
  - the recorded prompt/block digests do not reproduce from the task
    messages + previous_tier chain + whitelist renderer (byte-level prompt
    integrity, incl. "no neighbors -> PM prompt byte-identical");
  - the two arms did not see the identical frozen memory prefix and the
    identical retrieval set per task (fairness, prereg §2.2/§10.5).

Outputs (ONLY under --out-dir, default eval/ablation; protected C0/C1
paths are refused):
    results-summary.json   full machine-readable aggregate + hypotheses
    results-tasks.csv      one row per task (LM/PM paired)
    charts/*.svg           hand-rolled SVG charts (stdlib only)

CLI:
    python eval/analyze_ablation.py \
        --lm logs/s3_ablation_T8_LM.jsonl \
        --pm logs/s3_ablation_T8_PM.jsonl \
        --out-dir eval/ablation
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.run import load_tasks  # noqa: E402
from harness.run_ablation import (  # noqa: E402
    ARM_LM,
    ARM_PM,
    FROZEN_TASK_INDEX,
    FROZEN_TASK_ORDER,
    ROUTE_MODE_LM,
    ROUTE_MODE_PM,
    render_memory_block,
    sha256_file,
    sha256_text,
)
from jitrl_core.config import TIERS, JitRLConfig  # noqa: E402
from jitrl_core.policy import choose_tier, modulate_logits  # noqa: E402
from local_judge.client import build_prompt as build_judge_prompt  # noqa: E402

DEFAULT_LM = "logs/s3_ablation_T8_LM.jsonl"
DEFAULT_PM = "logs/s3_ablation_T8_PM.jsonl"
DEFAULT_MEMORY_TRACE = "eval/ablation/frozen-c0-memory-trace.jsonl"
DEFAULT_TRACE_SHA_JSON = "eval/ablation/frozen-c0-memory-trace.sha256.json"
DEFAULT_TASKS = "eval/tasks.jsonl"
DEFAULT_OUT_DIR = "eval/ablation"

#: Frozen C0 source memory (prereg §2.1); its hash is asserted against the
#: sidecar when the file is present.
C0_SOURCE_MEMORY = "logs/s2_full_C_main24_retry2_memory.jsonl"

#: Frozen execution/evaluator conditions (prereg §5).
FROZEN_MAX_TOKENS_EXEC = 1024
FROZEN_MAX_TOKENS_EVAL = 300
FROZEN_TEMPERATURE_EXEC = 0.7
FROZEN_EVALUATOR_TEMPERATURE = 0.2

TIERS_ORDINAL = {t: i for i, t in enumerate(TIERS)}

NOTES = [
    "G is the scalar reward the Evaluator and real costs jointly produced in "
    "C0; it is information the JitRL memory already contains and its use in "
    "the T8-PM whitelist block is disclosed here (prereg §4.2).",
    "previous_tier is passed per arm from that arm's own previous executed "
    "tier (prereg §5); the two chains may diverge after the first differing "
    "choice and that divergence is part of the mechanism effect, reported in "
    "paired.previous_tier_divergence.",
    "n=24, one episode per task, single full run per arm: directional "
    "evidence only; no statistical significance is claimed (prereg header).",
    "Judge (local llama, 127.0.0.1:18080) has $0 API cost; only latency is "
    "reported. LM prompt_judge_latency_ms=0.0 is the documented sentinel "
    "for 'no prompt-injection call ran', not a measurement.",
    "Same-tier quality/usage differences are temperature=0.7 execution "
    "sampling variance and are NOT attributed to the mechanism; only "
    "tier-changed tasks are listed as mechanism-attributable (prereg §8.3).",
    "The blind Evaluator budget (eval_cost_usd) is measurement overhead, "
    "reported separately from execution cost.",
]


# ------------------------------------------------------------------ loading

def load_arm(path: str | Path) -> tuple[list[dict], dict]:
    """Load one T8 run JSONL -> (records, embedded _summary)."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"input log not found: {p}")
    records: list[dict] = []
    summary: dict | None = None
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "_summary" in obj:
            summary = obj["_summary"]
        else:
            records.append(obj)
    if summary is None:
        raise SystemExit(f"{p}: missing the trailing _summary line")
    return records, summary


def _fail(tag: str, detail: str) -> None:
    raise SystemExit(f"[analyze_ablation] FORMAL-INPUT REJECTED ({tag}): {detail}")


# ------------------------------------------------- hard input validation

def validate_trace(trace_path: Path, sha_json_path: Path) -> tuple[str, dict]:
    """Hash-assert the frozen memory trace + sidecar (prereg §2.1/§10.12)."""
    if not trace_path.is_file():
        _fail("hash", f"memory trace not found: {trace_path}")
    if not sha_json_path.is_file():
        _fail("hash", f"trace sidecar not found: {sha_json_path}")
    digest = sha256_file(trace_path)
    meta = json.loads(sha_json_path.read_text(encoding="utf-8"))
    copy_sha = meta.get("copy", {}).get("sha256")
    if copy_sha != digest:
        _fail("hash", f"trace file digest {digest} != sidecar copy sha {copy_sha}")
    if meta.get("n_entries") != len(FROZEN_TASK_ORDER):
        _fail("hash", f"sidecar n_entries {meta.get('n_entries')} != 24")
    if meta.get("frozen_task_order") != list(FROZEN_TASK_ORDER):
        _fail("order", "sidecar frozen_task_order != frozen 24-task order")
    src = meta.get("source", {})
    src_path = Path(src.get("memory_path", C0_SOURCE_MEMORY))
    if not src_path.is_file() and not src_path.is_absolute():
        src_path = REPO_ROOT / src_path
    if src_path.is_file():
        src_digest = sha256_file(src_path)
        if src.get("memory_sha256") != src_digest:
            _fail("hash",
                  f"C0 source memory {src_path} digest changed: "
                  f"{src_digest} != sidecar {src.get('memory_sha256')}")
    return digest, meta


def _recompute_block(retrieved: list[dict]) -> str | None:
    """Recompute the whitelist memory block from a record's audit copy."""
    if not retrieved:
        return None
    matches = [({"tier": m["tier"], "G": m["G"]}, m["similarity"])
               for m in retrieved]
    return render_memory_block(matches)


def validate_arm(
    records: list[dict],
    summary: dict,
    *,
    arm: str,
    log_path: Path,
    trace_sha: str,
    tasks_by_id: dict[str, dict],
) -> None:
    """All per-arm hard assertions (order / config / hash / completeness /
    mechanism purity / math / prompt integrity)."""
    is_lm = arm == ARM_LM
    route_expected = ROUTE_MODE_LM if is_lm else ROUTE_MODE_PM
    ids = [r.get("task_id") for r in records]

    # --- order (prereg §10.11/§10.12) ------------------------------------
    if ids != list(FROZEN_TASK_ORDER):
        _fail("order", f"{log_path}: task sequence != frozen 24-task order "
                       f"(n={len(ids)}, first divergence at "
                       f"{next((i for i, (a, b) in enumerate(zip(ids, FROZEN_TASK_ORDER)) if a != b), None)})")
    if len(set(ids)) != len(ids):
        _fail("order", f"{log_path}: duplicate task ids")

    # --- summary-level config (prereg §5/§10.12) --------------------------
    if summary.get("ablation_arm") != arm:
        _fail("config", f"{log_path}: _summary.ablation_arm={summary.get('ablation_arm')!r}")
    if summary.get("episodes") != 1:
        _fail("config", f"{log_path}: episodes={summary.get('episodes')} != 1")
    if summary.get("n_records") != len(FROZEN_TASK_ORDER):
        _fail("completeness", f"{log_path}: n_records={summary.get('n_records')} != 24")
    if summary.get("order_is_full_frozen_24") is not True:
        _fail("order", f"{log_path}: order_is_full_frozen_24 is not true")
    frozen_snapshot = JitRLConfig().snapshot()
    if summary.get("config_snapshot") != frozen_snapshot:
        _fail("config", f"{log_path}: config_snapshot != frozen JitRL config "
                       f"({summary.get('config_snapshot')} vs {frozen_snapshot})")
    ac = summary.get("ablation_config") or {}
    for key, expected in (
        ("max_tokens_exec", FROZEN_MAX_TOKENS_EXEC),
        ("max_tokens_eval", FROZEN_MAX_TOKENS_EVAL),
        ("temperature_exec", FROZEN_TEMPERATURE_EXEC),
        ("evaluator_temperature", FROZEN_EVALUATOR_TEMPERATURE),
    ):
        if ac.get(key) != expected:
            _fail("config", f"{log_path}: ablation_config.{key}={ac.get(key)!r} "
                           f"!= frozen {expected!r}")
    if ac.get("memory_read_only") is not True:
        _fail("config", f"{log_path}: ablation_config.memory_read_only != true")
    if ac.get("memory_trace_sha256") != trace_sha:
        _fail("hash", f"{log_path}: _summary trace hash != trace file digest")

    # --- per-record contract (prereg §7) ----------------------------------
    for r in records:
        tid = r.get("task_id")
        if tid not in tasks_by_id:
            _fail("order", f"{log_path}: task {tid!r} not present in the tasks "
                           f"file; cannot verify prompt integrity")
        where = f"{log_path}:{tid}"
        if r.get("ablation_arm") != arm:
            _fail("config", f"{where}: ablation_arm={r.get('ablation_arm')!r}")
        if r.get("route_mode") != route_expected:
            _fail("config", f"{where}: route_mode={r.get('route_mode')!r}")
        if r.get("episode_id") != 1:
            _fail("config", f"{where}: episode_id={r.get('episode_id')} != 1")
        if r.get("memory_frozen") is not True:
            _fail("config", f"{where}: memory_frozen is not true")
        if r.get("memory_written") is not False:
            _fail("completeness", f"{where}: memory_written is not false "
                                  "(frozen read-only memory was written?)")
        if r.get("memory_source_sha256") != trace_sha:
            _fail("hash", f"{where}: memory_source_sha256 != trace digest")
        if r.get("memory_prefix_size") != FROZEN_TASK_INDEX[tid]:
            _fail("config", f"{where}: memory_prefix_size="
                           f"{r.get('memory_prefix_size')} != frozen index "
                           f"{FROZEN_TASK_INDEX[tid]} (prefix replay broken)")
        retrieved = r.get("retrieved_memories")
        if not isinstance(retrieved, list):
            _fail("config", f"{where}: retrieved_memories missing")
        for m in retrieved:
            if set(m.keys()) != {"tier", "G", "similarity"}:
                _fail("config", f"{where}: retrieved_memories entry keys "
                               f"{sorted(m.keys())} != tier/G/similarity")
        if r.get("n_retrieved") != len(retrieved):
            _fail("config", f"{where}: n_retrieved={r.get('n_retrieved')} != "
                           f"len(retrieved_memories)={len(retrieved)}")

        # --- 24/24 completeness (prereg §9.4/§10.12) ----------------------
        if r.get("exec_error"):
            _fail("completeness", f"{where}: exec_error={r['exec_error']!r} "
                                  "(audit-only run, not a formal input)")
        if r.get("eval_error"):
            _fail("completeness", f"{where}: eval_error={r['eval_error']!r}")
        if r.get("evaluator") is None or r.get("reward") is None:
            _fail("completeness", f"{where}: evaluator/reward missing")

        # --- mechanism purity (prereg §3.1/§3.2, §10.8) --------------------
        if is_lm:
            if r.get("z_prompt") is not None:
                _fail("config", f"{where}: LM record carries z_prompt")
            if r.get("z_mod") is None:
                _fail("config", f"{where}: LM record missing z_mod")
            if r.get("injected_block_chars") != 0:
                _fail("config", f"{where}: LM record injected a block "
                                "(LM must never inject)")
            if r.get("injected_block_sha256") is not None:
                _fail("config", f"{where}: LM record carries a block hash")
            if r.get("judge_calls") != 1:
                _fail("config", f"{where}: LM judge_calls={r.get('judge_calls')} != 1")
            if r.get("prompt_judge_latency_ms") != 0.0:
                _fail("config", f"{where}: LM prompt_judge_latency_ms != 0.0 "
                                "sentinel")
        else:
            if r.get("z_mod") is not None:
                _fail("config", f"{where}: PM record carries z_mod "
                                "(PM must never modulate)")
            if r.get("z_prompt") is None:
                _fail("config", f"{where}: PM record missing z_prompt")
            if r.get("modulation_active") is not False:
                _fail("config", f"{where}: PM modulation_active is not false")
            if r.get("A_norm") != {t: 0.0 for t in TIERS}:
                _fail("config", f"{where}: PM A_norm is not all-zero")
            if r.get("V") != 0.0:
                _fail("config", f"{where}: PM V != 0.0")
            if r.get("judge_calls") != 2:
                _fail("config", f"{where}: PM judge_calls={r.get('judge_calls')} != 2 "
                                "(deciding + raw-prompt audit call)")

        # --- decision math reproduces from the recorded fields -------------
        z_base, z_mod, z_prompt = r.get("z_base"), r.get("z_mod"), r.get("z_prompt")
        if r.get("base_choice") != choose_tier(z_base, TIERS):
            _fail("math", f"{where}: base_choice != argmax(z_base)")
        if is_lm:
            if z_mod != modulate_logits(z_base, r.get("A_norm"), 5.0, -10.0, TIERS):
                _fail("math", f"{where}: z_mod != z_base + beta*A_norm (C0 formula)")
            if r.get("chosen_tier") != choose_tier(z_mod, TIERS):
                _fail("math", f"{where}: chosen_tier != argmax(z_mod)")
        else:
            if r.get("chosen_tier") != choose_tier(z_prompt, TIERS):
                _fail("math", f"{where}: chosen_tier != argmax(z_prompt)")
            if r.get("flipped") != (r.get("chosen_tier") != r.get("base_choice")):
                _fail("math", f"{where}: flipped inconsistent")
        if r.get("influenced") is not None and r.get("gate_reason") is not None \
                and r.get("influenced"):
            _fail("math", f"{where}: influenced=true while gate_reason="
                          f"{r.get('gate_reason')!r}")

        # --- prompt / block byte integrity (prereg §3.2/§4.3/§10.1) ---------
        message = tasks_by_id[tid]["message"]
        prev = r.get("previous_tier")
        base_prompt_sha = sha256_text(build_judge_prompt(message, prev, None))
        if r.get("base_prompt_sha256") != base_prompt_sha:
            _fail("math", f"{where}: base_prompt_sha256 does not reproduce from "
                          "task message + previous_tier (raw prompt drifted)")
        recomputed = _recompute_block(retrieved) if not is_lm else None
        if not is_lm:
            if recomputed is not None:
                if r.get("injected_block_sha256") != sha256_text(recomputed):
                    _fail("math", f"{where}: injected_block_sha256 does not "
                                  "reproduce from the whitelist renderer")
                if r.get("injected_block_chars") != len(recomputed):
                    _fail("math", f"{where}: injected_block_chars mismatch")
                decision_sha = sha256_text(
                    build_judge_prompt(message, prev, recomputed))
            else:
                if r.get("injected_block_chars") != 0:
                    _fail("math", f"{where}: block chars without neighbors")
                decision_sha = base_prompt_sha  # no-block: byte-identical prompt
            if r.get("decision_prompt_sha256") != decision_sha:
                _fail("math", f"{where}: decision_prompt_sha256 does not reproduce "
                              "(injected prompt drifted / no-block prompt differs)")
        else:
            if r.get("decision_prompt_sha256") != base_prompt_sha:
                _fail("math", f"{where}: LM decision prompt != raw prompt")


def validate_fairness(lm: list[dict], pm: list[dict]) -> None:
    """Both arms must have seen the identical prefix + retrieval per task
    (prereg §2.2/§10.5)."""
    pm_by_id = {r["task_id"]: r for r in pm}
    for lr in lm:
        tid = lr["task_id"]
        pr = pm_by_id[tid]
        if lr.get("memory_prefix_size") != pr.get("memory_prefix_size"):
            _fail("fairness", f"{tid}: memory_prefix_size differs between arms")
        if lr.get("retrieved_memories") != pr.get("retrieved_memories"):
            _fail("fairness", f"{tid}: retrieved_memories differ between arms "
                              "(the arms did not see the same frozen prefix)")


# ----------------------------------------------------------------- metrics

def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _r(x, nd):
    return None if x is None else round(x, nd)


def arm_metrics(records: list[dict], *, arm: str) -> dict:
    ok = [r for r in records if r.get("evaluator") is not None]
    q = [float(r["evaluator"]["quality_score"]) for r in ok]
    exec_costs = [float(r["exec_cost_usd"]) for r in records
                  if r.get("exec_cost_usd") is not None]
    eval_costs = [float(r["eval_cost_usd"]) for r in records
                  if r.get("eval_cost_usd") is not None]
    rewards = [float(r["reward"]) for r in records if r.get("reward") is not None]
    verdicts: dict[str, int] = {}
    for r in ok:
        v = r["evaluator"].get("verdict")
        verdicts[v] = verdicts.get(v, 0) + 1
    labeled = [r for r in records if r.get("gt_tier")]
    matches = [r for r in labeled if r["chosen_tier"] == r["gt_tier"]]
    over = [r for r in labeled
            if TIERS_ORDINAL[r["chosen_tier"]] > TIERS_ORDINAL[r["gt_tier"]]]
    under = [r for r in labeled
             if TIERS_ORDINAL[r["chosen_tier"]] < TIERS_ORDINAL[r["gt_tier"]]]

    def lat(key):
        return [float(r[key]) for r in records]

    return {
        "ablation_arm": arm,
        "n_records": len(records),
        "quality": {
            "mean": _r(_mean(q), 4),
            "sd": _r(statistics.stdev(q), 4) if len(q) > 1 else 0.0,
            "distribution": {str(s): q.count(float(s)) for s in (1, 2, 3, 4, 5)},
            "verdicts": verdicts,
        },
        "exec_cost_usd": {
            "total": _r(sum(exec_costs), 6),
            "mean": _r(_mean(exec_costs), 6),
        },
        "eval_cost_usd": {"total": _r(sum(eval_costs), 6)},
        "reward": {"mean": _r(_mean(rewards), 4)},
        "tier_vs_gt": {
            "matches": len(matches),
            "match_rate": _r(len(matches) / len(labeled), 4) if labeled else None,
            "misgrade_rate": _r(1 - len(matches) / len(labeled), 4) if labeled else None,
            "over_provisioned": len(over),
            "under_provisioned": len(under),
        },
        "retrieval_hits": sum(1 for r in records
                              if r.get("gate_reason") is None and r.get("n_retrieved", 0) > 0),
        "gated_records": sum(1 for r in records if r.get("gate_reason") is not None),
        "influenced_records": sum(1 for r in records if r.get("influenced")),
        "influenced_task_ids": [r["task_id"] for r in records if r.get("influenced")],
        "flips_vs_own_base": sum(1 for r in records if r.get("flipped")),
        "finish_reason_length": sum(1 for r in records
                                    if r.get("exec_finish_reason") == "length"),
        "empty_responses": sum(1 for r in records
                               if r.get("exec_response_preview") is not None
                               and not str(r["exec_response_preview"]).strip()),
        "latency_ms": {
            "judge_deciding_mean": _r(_mean(lat("judge_latency_ms")), 1),
            "base_judge_mean": _r(_mean(lat("base_judge_latency_ms")), 1),
            "prompt_judge_mean": _r(_mean(lat("prompt_judge_latency_ms")), 1),
            "jitrl_mean": _r(_mean(lat("jitrl_latency_ms")), 4),
            "jitrl_max": _r(max(lat("jitrl_latency_ms"), default=0.0), 4),
            "exec_mean": _r(_mean([float(r["exec_latency_ms"]) for r in records
                                   if r.get("exec_latency_ms") is not None]), 1),
            "judge_calls_total": sum(int(r.get("judge_calls", 0)) for r in records),
        },
    }


def paired_lm_pm(lm: list[dict], pm: list[dict]) -> dict:
    pm_by_id = {r["task_id"]: r for r in pm}
    per_task = []
    prev_divergence: list[str] = []
    for lr in lm:
        pr = pm_by_id[lr["task_id"]]
        if lr.get("previous_tier") != pr.get("previous_tier"):
            prev_divergence.append(lr["task_id"])
        qd = lr["evaluator"]["quality_score"] - pr["evaluator"]["quality_score"]
        cd = lr["exec_cost_usd"] - pr["exec_cost_usd"]
        per_task.append({
            "task_id": lr["task_id"],
            "family": lr.get("family"),
            "gt_tier": lr.get("gt_tier"),
            "frozen_index": FROZEN_TASK_INDEX[lr["task_id"]],
            "memory_prefix_size": lr.get("memory_prefix_size"),
            "n_retrieved": lr.get("n_retrieved"),
            "tier_LM": lr["chosen_tier"], "tier_PM": pr["chosen_tier"],
            "tier_agree": lr["chosen_tier"] == pr["chosen_tier"],
            "base_choice_LM": lr.get("base_choice"),
            "base_choice_PM": pr.get("base_choice"),
            "base_choice_agree": lr.get("base_choice") == pr.get("base_choice"),
            "flipped_LM": lr.get("flipped"), "flipped_PM": pr.get("flipped"),
            "influenced_LM": lr.get("influenced"),
            "influenced_PM": pr.get("influenced"),
            "quality_LM": lr["evaluator"]["quality_score"],
            "quality_PM": pr["evaluator"]["quality_score"],
            "quality_delta_LM_minus_PM": qd,
            "exec_cost_LM_usd": _r(lr["exec_cost_usd"], 6),
            "exec_cost_PM_usd": _r(pr["exec_cost_usd"], 6),
            "exec_cost_delta_LM_minus_PM_usd": _r(cd, 6),
            "exec_model_LM": lr.get("exec_model"),
            "exec_model_PM": pr.get("exec_model"),
            "reward_LM": lr.get("reward"), "reward_PM": pr.get("reward"),
            "verdict_LM": lr["evaluator"].get("verdict"),
            "verdict_PM": pr["evaluator"].get("verdict"),
            "finish_reason_LM": lr.get("exec_finish_reason"),
            "finish_reason_PM": pr.get("exec_finish_reason"),
            "jitrl_latency_LM_ms": _r(lr.get("jitrl_latency_ms"), 4),
            "jitrl_latency_PM_ms": _r(pr.get("jitrl_latency_ms"), 4),
            "judge_latency_LM_ms": _r(lr.get("judge_latency_ms"), 1),
            "base_judge_LM_ms": _r(lr.get("base_judge_latency_ms"), 1),
            "base_judge_PM_ms": _r(pr.get("base_judge_latency_ms"), 1),
            "prompt_judge_PM_ms": _r(pr.get("prompt_judge_latency_ms"), 1),
        })
    tier_changed = [d for d in per_task if not d["tier_agree"]]
    same_tier = [d for d in per_task if d["tier_agree"]]
    base_mismatch = [d for d in per_task if not d["base_choice_agree"]]
    # per-tier z_base drift between the two arms (H5)
    tier_drift = {}
    for t in TIERS:
        diffs = [lr["z_base"][t] - pm_by_id[lr["task_id"]]["z_base"][t]
                 for lr in lm]
        tier_drift[t] = {
            "mean_abs_diff": _r(_mean([abs(d) for d in diffs]), 4),
            "max_abs_diff": _r(max((abs(d) for d in diffs), default=0.0), 4),
        }
    # previous_tier chain divergence (prereg §5: part of the mechanism
    # effect) — computed inside the per-task loop above
    return {
        "n_pairs": len(per_task),
        "per_task": per_task,
        "tier_agreement": {
            "agree": sum(1 for d in per_task if d["tier_agree"]),
            "disagree": len(tier_changed),
            "agreement_rate": _r(sum(1 for d in per_task if d["tier_agree"])
                                 / len(per_task), 4) if per_task else None,
            "tier_changed_task_ids": [d["task_id"] for d in tier_changed],
            "same_tier_task_ids": [d["task_id"] for d in same_tier],
            "note": "only tier-changed tasks are attributable to the memory "
                    "mechanism; same-tier quality/usage differences are "
                    "temperature=0.7 execution sampling variance (prereg §8.3).",
        },
        "quality": {
            "mean_LM": _r(_mean([d["quality_LM"] for d in per_task]), 4),
            "mean_PM": _r(_mean([d["quality_PM"] for d in per_task]), 4),
            "mean_delta_LM_minus_PM": _r(
                _mean([d["quality_delta_LM_minus_PM"] for d in per_task]), 4),
            "improved_tasks": sum(1 for d in per_task
                                  if d["quality_delta_LM_minus_PM"] > 0),
            "worsened_tasks": sum(1 for d in per_task
                                  if d["quality_delta_LM_minus_PM"] < 0),
            "unchanged_tasks": sum(1 for d in per_task
                                   if d["quality_delta_LM_minus_PM"] == 0),
        },
        "exec_cost": {
            "total_LM_usd": _r(sum(d["exec_cost_LM_usd"] for d in per_task), 6),
            "total_PM_usd": _r(sum(d["exec_cost_PM_usd"] for d in per_task), 6),
            "total_delta_LM_minus_PM_usd": _r(
                sum(d["exec_cost_delta_LM_minus_PM_usd"] for d in per_task), 6),
        },
        "judge_drift_audit": {
            "base_choice_mismatch_tasks": [d["task_id"] for d in base_mismatch],
            "base_choice_mismatch_count": len(base_mismatch),
            "per_tier_z_base_diff": tier_drift,
            "note": "base-choice mismatches and z_base differences are local "
                    "Judge drift across runs (H5); they are reported, never "
                    "attributed to the memory mechanism.",
        },
        "previous_tier_divergence": {
            "task_ids_after_first_divergence": prev_divergence,
            "note": "previous_tier is per-arm (prereg §5); after the first "
                    "differing choice the two prompt chains legitimately "
                    "differ — part of the mechanism effect, listed here.",
        },
    }


# ------------------------------------------------------------- hypotheses

def evaluate_hypotheses(lm_m: dict, pm_m: dict, paired: dict) -> dict:
    q_delta = (lm_m["quality"]["mean"] - pm_m["quality"]["mean"]) \
        if lm_m["quality"]["mean"] is not None and pm_m["quality"]["mean"] is not None \
        else None
    total_lm = paired["exec_cost"]["total_LM_usd"]
    total_pm = paired["exec_cost"]["total_PM_usd"]
    cost_ratio = ((total_pm - total_lm) / total_pm) if total_pm else None
    lm_jitrl_max = lm_m["latency_ms"]["jitrl_max"]
    return {
        "H1_mechanism_influence": {
            "influenced_LM": lm_m["influenced_records"],
            "influenced_PM": pm_m["influenced_records"],
            "retrieval_hits_LM": lm_m["retrieval_hits"],
            "retrieval_hits_PM": pm_m["retrieval_hits"],
            "met": (lm_m["influenced_records"] > 0 or pm_m["influenced_records"] > 0),
            "reading": "at least one mechanism changed the raw Judge choice on "
                       "a retrieval-hit task; if both are 0 the prereg reading "
                       "is 'weak base policy / memory-format dominated, "
                       "mechanism difference not observable'.",
        },
        "H2_quality_non_inferior": {
            "mean_quality_LM_minus_PM": _r(q_delta, 4),
            "threshold": -0.3,
            "met": q_delta is not None and round(q_delta, 6) >= -0.3,
        },
        "H3_cost_comparison": {
            "total_exec_LM_usd": total_lm,
            "total_exec_PM_usd": total_pm,
            "savings_ratio_PM_to_LM": _r(cost_ratio, 4),
            "lm_favorable_claim_allowed": (
                q_delta is not None and round(q_delta, 6) >= -0.3
                and total_lm < total_pm),
            "note": "no preset 20% (D10) threshold; the ratio is reported "
                    "as-is (prereg §8.2 H3).",
        },
        "H4_latency_attribution": {
            "LM_jitrl_max_ms": lm_jitrl_max,
            "LM_jitrl_threshold_ms": 500.0,
            "met": lm_jitrl_max is not None and round(lm_jitrl_max, 6) <= 500.0,
            "PM_extra_judge_mean_ms": _r(
                (pm_m["latency_ms"]["base_judge_mean"] or 0.0), 1),
            "note": "PM's additional raw-prompt Judge call is reported in "
                    "base_judge fields and is never mixed into the "
                    "JitRL/retrieval post-processing latency.",
        },
        "H5_stability": {
            "base_choice_mismatch_count":
                paired["judge_drift_audit"]["base_choice_mismatch_count"],
            "base_choice_mismatch_tasks":
                paired["judge_drift_audit"]["base_choice_mismatch_tasks"],
            "per_tier_z_base_diff":
                paired["judge_drift_audit"]["per_tier_z_base_diff"],
        },
    }


def conclude(lm_m: dict, pm_m: dict) -> dict:
    """Prereg §8.3 conclusion rules. Conservative rule ordering: the
    'no substantive difference' rule is evaluated FIRST (a <=$0.001 cost
    edge or a <0.3 quality edge must not be overclaimed), then the LM rule
    (quality non-inferior + cheaper), then the PM rule; anything else is
    reported as not covered by the preregistered rules."""
    q_delta = lm_m["quality"]["mean"] - pm_m["quality"]["mean"]
    c_delta = lm_m["exec_cost_usd"]["total"] - pm_m["exec_cost_usd"]["total"]
    if abs(q_delta) < 0.3 and abs(c_delta) <= 0.001:
        rule, statement = (
            "no_substantive_difference",
            "未观察到实质差异 (|quality delta| < 0.3 and |cost delta| <= $0.001)")
    elif round(q_delta, 6) >= -0.3 and c_delta < 0:
        rule, statement = (
            "favors_logit_modulation",
            "本次方向性证据偏向 logit modulation (LM quality non-inferior and "
            "LM execution cost below PM)")
    elif pm_m["quality"]["mean"] > lm_m["quality"]["mean"] or c_delta > 0:
        rule, statement = (
            "favors_prompt_injection",
            "本次方向性证据偏向 prompt injection (PM better in quality or cost)")
    else:
        rule, statement = (
            "not_covered",
            "结果不在预注册结论规则覆盖范围内 (LM fails the quality "
            "non-inferiority margin while PM is not better in cost either); "
            "report the numbers as-is")
    return {
        "rule": rule,
        "statement": statement,
        "quality_delta_LM_minus_PM": _r(q_delta, 4),
        "exec_cost_delta_LM_minus_PM_usd": _r(c_delta, 6),
        "rule_ordering_note": "conservative ordering: no-substantive-"
                              "difference is checked before the LM/PM rules",
        "strength": "directional evidence only (n=24, single run per arm)",
    }


# ------------------------------------------------------------------ charts

_COLORS = {"T8-LM": "#f59e0b", "T8-PM": "#3b82f6"}


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
    n_cat, n_ser = len(categories), len(series)
    w, h = 860, 480
    ml, mr, mt, mb = 78, 24, 56, 74
    iw, ih = w - ml - mr, h - mt - mb
    vmax = _nice_max(max(v for vs in series.values() for v in vs) or 1.0)
    cluster = iw / max(n_cat, 1)
    bar_w = cluster * 0.72 / n_ser
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
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
        cx = ml + cluster * ci + cluster * 0.14
        for si, (name, vals) in enumerate(series.items()):
            v = vals[ci]
            bh = ih * v / vmax
            x = cx + si * bar_w
            y = mt + ih - bh
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                         f'height="{bh:.1f}" fill="{_COLORS.get(name, "#10b981")}" rx="2"/>')
            if v > 0:
                parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 4:.1f}" '
                             f'font-size="9" text-anchor="middle">'
                             f'{value_fmt.format(v)}</text>')
        parts.append(f'<text x="{ml + cluster * (ci + 0.5):.1f}" y="{h - mb + 18}" '
                     f'font-size="12" text-anchor="middle">{_esc(cat)}</text>')
    lx = ml
    for name in series:
        parts.append(f'<rect x="{lx}" y="{h - mb + 36}" width="12" height="12" '
                     f'fill="{_COLORS.get(name, "#10b981")}" rx="2"/>')
        parts.append(f'<text x="{lx + 17}" y="{h - mb + 46}" font-size="12">'
                     f'{_esc(name)}</text>')
        lx += 34 + 9 * len(str(name))
    parts.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")


def svg_line(title: str, y_label: str, x_labels: list[str],
             series: dict[str, list[float]], out_path: Path) -> None:
    w, h = 1020, 480
    ml, mr, mt, mb = 70, 150, 56, 64
    iw, ih = w - ml - mr, h - mt - mb
    all_v = [v for vs in series.values() for v in vs]
    lo = min(all_v)
    hi = max(all_v) + 0.001
    span = (hi - lo) or 1.0
    n = len(x_labels)
    xs = [ml + iw * i / max(n - 1, 1) for i in range(n)]

    def ys(v):
        return mt + ih - ih * (v - lo) / span

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
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
    palette = ["#f59e0b", "#3b82f6", "#ef4444"]
    for si, (name, vals) in enumerate(series.items()):
        color = _COLORS.get(name, palette[si % len(palette)])
        pts = " ".join(f"{x:.1f},{ys(v):.1f}" for x, v in zip(xs, vals))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                     f'stroke-width="2"/>')
        for x, v in zip(xs, vals):
            parts.append(f'<circle cx="{x:.1f}" cy="{ys(v):.1f}" r="3.5" '
                         f'fill="{color}"/>')
        ly = mt + 20 + si * 20
        parts.append(f'<line x1="{w - mr + 16}" y1="{ly}" x2="{w - mr + 46}" '
                     f'y2="{ly}" stroke="{color}" stroke-width="2"/>')
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


def write_charts(arms: dict, paired: dict, out_dir: Path) -> list[str]:
    charts_dir = out_dir / "charts"
    written: list[str] = []
    lm, pm = arms[ARM_LM], arms[ARM_PM]
    # 1. quality distribution
    svg_grouped_bars(
        "T8 quality score distribution (formal runs, n=24 each)",
        "tasks", ["1", "2", "3", "4", "5"],
        {ARM_LM: [lm["quality"]["distribution"][s] for s in ("1", "2", "3", "4", "5")],
         ARM_PM: [pm["quality"]["distribution"][s] for s in ("1", "2", "3", "4", "5")]},
        charts_dir / "quality-distribution.svg", value_fmt="{:.0f}")
    written.append("charts/quality-distribution.svg")
    # 2. per-task quality
    pt = paired["per_task"]
    svg_line(
        "T8 per-task quality score (frozen task order)",
        "quality (1-5)", [d["task_id"] for d in pt],
        {ARM_LM: [d["quality_LM"] for d in pt],
         ARM_PM: [d["quality_PM"] for d in pt]},
        charts_dir / "per-task-quality.svg")
    written.append("charts/per-task-quality.svg")
    # 3. per-task exec cost
    svg_line(
        "T8 per-task execution cost (USD)",
        "exec cost ($)", [d["task_id"] for d in pt],
        {ARM_LM: [d["exec_cost_LM_usd"] for d in pt],
         ARM_PM: [d["exec_cost_PM_usd"] for d in pt]},
        charts_dir / "per-task-exec-cost.svg")
    written.append("charts/per-task-exec-cost.svg")
    # 4. latency attribution (means per arm, separate columns)
    svg_grouped_bars(
        "T8 latency attribution (means; PM base_judge = additional audit call)",
        "ms", ["deciding judge", "base judge", "prompt judge", "jitrl post"],
        {ARM_LM: [lm["latency_ms"]["judge_deciding_mean"] or 0.0,
                  lm["latency_ms"]["base_judge_mean"] or 0.0, 0.0,
                  lm["latency_ms"]["jitrl_mean"] or 0.0],
         ARM_PM: [pm["latency_ms"]["judge_deciding_mean"] or 0.0,
                  pm["latency_ms"]["base_judge_mean"] or 0.0,
                  pm["latency_ms"]["prompt_judge_mean"] or 0.0,
                  pm["latency_ms"]["jitrl_mean"] or 0.0]},
        charts_dir / "latency-attribution.svg", value_fmt="{:.1f}")
    written.append("charts/latency-attribution.svg")
    return written


# -------------------------------------------------------------------- CSV

CSV_COLUMNS = [
    "task_id", "family", "gt_tier", "frozen_index", "memory_prefix_size",
    "n_retrieved", "tier_LM", "tier_PM", "tier_agree",
    "base_choice_LM", "base_choice_PM", "base_choice_agree",
    "flipped_LM", "influenced_LM", "flipped_PM", "influenced_PM",
    "quality_LM", "quality_PM", "quality_delta_LM_minus_PM",
    "exec_cost_LM_usd", "exec_cost_PM_usd", "exec_cost_delta_LM_minus_PM_usd",
    "exec_model_LM", "exec_model_PM", "reward_LM", "reward_PM",
    "verdict_LM", "verdict_PM", "finish_reason_LM", "finish_reason_PM",
    "judge_latency_LM_ms", "base_judge_LM_ms", "base_judge_PM_ms",
    "prompt_judge_PM_ms", "jitrl_latency_LM_ms", "jitrl_latency_PM_ms",
]


def write_tasks_csv(per_task: list[dict], out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in per_task:
            w.writerow(row)


# ------------------------------------------------------------------- main

def build_summary(*, lm_records, lm_summary, pm_records, pm_summary,
                  trace_digest, trace_meta, lm_path, pm_path,
                  trace_path) -> dict:
    arms = {ARM_LM: arm_metrics(lm_records, arm=ARM_LM),
            ARM_PM: arm_metrics(pm_records, arm=ARM_PM)}
    paired = paired_lm_pm(lm_records, pm_records)
    return {
        "preregistration": "S3-TAB8-PREREGISTRATION.md v1.0",
        "experiment": "T8 - Prompt Memory Injection vs Logit Modulation",
        "inputs": {
            "lm_log": str(lm_path), "pm_log": str(pm_path),
            "memory_trace": str(trace_path),
            "memory_trace_sha256": trace_digest,
            "c0_source_memory": trace_meta.get("source", {}).get("memory_path"),
            "c0_source_memory_sha256": trace_meta.get("source", {}).get("memory_sha256"),
            "validation": "hard assertions passed (order / config / hash / "
                          "24-of-24 completeness / mechanism purity / math / "
                          "prompt-byte integrity / arm fairness)",
        },
        "frozen_task_order": list(FROZEN_TASK_ORDER),
        "arms": arms,
        "paired": paired,
        "hypotheses": evaluate_hypotheses(arms[ARM_LM], arms[ARM_PM], paired),
        "conclusion": conclude(arms[ARM_LM], arms[ARM_PM]),
        "notes": NOTES,
    }


_PROTECTED_EXACT = ("eval", "eval/charts", "eval/c1", "poster-a3", "logs", ".")
_PROTECTED_INSIDE = ("eval/charts", "eval/c1", "poster-a3", "logs")


def ensure_safe_out_dir(out_dir: str | Path) -> Path:
    """Outputs only under eval/ablation (or an explicit non-protected dir
    such as a test tmp dir); never the frozen C0/C1 assets (prereg §6)."""
    od = Path(out_dir).resolve()
    repo = REPO_ROOT.resolve()
    for name in _PROTECTED_EXACT:
        p = (repo / name).resolve()
        if od == p:
            raise SystemExit(
                f"[analyze_ablation] refusing out-dir {od}: protected path "
                f"(prereg §6 — outputs only under eval/ablation)")
    for name in _PROTECTED_EXACT:
        p = (repo / name).resolve()
        if p != repo and od in p.parents:
            raise SystemExit(
                f"[analyze_ablation] refusing out-dir {od}: it would contain "
                f"the protected path {p}")
    for name in _PROTECTED_INSIDE:
        p = (repo / name).resolve()
        if p in od.parents:
            raise SystemExit(
                f"[analyze_ablation] refusing out-dir {od}: inside the "
                f"protected path {p}")
    return od


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="eval.analyze_ablation",
        description="S3 Table-8 ablation analysis (T8-LM vs T8-PM) with hard "
                    "formal-input assertions; outputs only under eval/ablation")
    ap.add_argument("--lm", default=DEFAULT_LM)
    ap.add_argument("--pm", default=DEFAULT_PM)
    ap.add_argument("--memory-trace", default=DEFAULT_MEMORY_TRACE)
    ap.add_argument("--trace-sha-json", default=DEFAULT_TRACE_SHA_JSON)
    ap.add_argument("--tasks", default=DEFAULT_TASKS)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args(argv)

    out_dir = ensure_safe_out_dir(args.out_dir)

    trace_digest, trace_meta = validate_trace(Path(args.memory_trace),
                                              Path(args.trace_sha_json))
    tasks_by_id = {t["task_id"]: t for t in load_tasks(args.tasks)}

    lm_records, lm_summary = load_arm(args.lm)
    pm_records, pm_summary = load_arm(args.pm)
    validate_arm(lm_records, lm_summary, arm=ARM_LM, log_path=Path(args.lm),
                 trace_sha=trace_digest, tasks_by_id=tasks_by_id)
    validate_arm(pm_records, pm_summary, arm=ARM_PM, log_path=Path(args.pm),
                 trace_sha=trace_digest, tasks_by_id=tasks_by_id)
    validate_fairness(lm_records, pm_records)

    summary = build_summary(
        lm_records=lm_records, lm_summary=lm_summary,
        pm_records=pm_records, pm_summary=pm_summary,
        trace_digest=trace_digest, trace_meta=trace_meta,
        lm_path=args.lm, pm_path=args.pm, trace_path=args.memory_trace)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "results-summary.json"
    csv_path = out_dir / "results-tasks.csv"
    charts = write_charts(summary["arms"], summary["paired"], out_dir)
    summary["outputs"] = {
        "results_summary": str(summary_path),
        "results_tasks_csv": str(csv_path),
        "charts": charts,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
        f.write("\n")
    write_tasks_csv(summary["paired"]["per_task"], csv_path)

    lm_m, pm_m = summary["arms"][ARM_LM], summary["arms"][ARM_PM]
    hyp = summary["hypotheses"]
    print(f"[analyze_ablation] formal inputs validated (order/config/hash/24-24)")
    print(f"[analyze_ablation] quality mean: LM={lm_m['quality']['mean']} "
          f"PM={pm_m['quality']['mean']} "
          f"delta={summary['paired']['quality']['mean_delta_LM_minus_PM']}")
    print(f"[analyze_ablation] exec cost: LM=${lm_m['exec_cost_usd']['total']} "
          f"PM=${pm_m['exec_cost_usd']['total']}")
    print(f"[analyze_ablation] tier agreement: "
          f"{summary['paired']['tier_agreement']['agree']}/24 "
          f"(changed: {summary['paired']['tier_agreement']['tier_changed_task_ids']})")
    print(f"[analyze_ablation] influenced: LM={lm_m['influenced_records']} "
          f"PM={pm_m['influenced_records']} "
          f"(retrieval hits: LM={lm_m['retrieval_hits']} PM={pm_m['retrieval_hits']})")
    print(f"[analyze_ablation] H1={hyp['H1_mechanism_influence']['met']} "
          f"H2={hyp['H2_quality_non_inferior']['met']} "
          f"H4={hyp['H4_latency_attribution']['met']}")
    print(f"[analyze_ablation] conclusion: {summary['conclusion']['statement']}")
    print(f"[analyze_ablation] wrote {summary_path}, {csv_path} and "
          f"{len(charts)} charts under {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
