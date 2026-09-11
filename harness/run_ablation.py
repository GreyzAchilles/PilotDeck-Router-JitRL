"""Session-3 Table-8 ablation harness: T8-LM vs T8-PM (prereg v1.0, frozen).

Preregistration: S3-TAB8-PREREGISTRATION.md v1.0 (2026-09-11, frozen before
any run). This module implements exactly that document; deviations must go
through a new T8.1 preregistration (prereg §9.7).

Arms (memory mechanism is the ONLY difference; execution / Evaluator /
reward / cost accounting are byte-for-byte the run_real code path):
    T8-LM  logit modulation  — raw-prompt Method B -> z_base; retrieval from
           the frozen C0 prefix memory; V/Q/A-hat via the UNMODIFIED
           JitRLEngine.decide() (exact C0 math, min_neighbors=1 gate);
           z_mod = z_base + beta * A_norm; tier = argmax z_mod.
    T8-PM  prompt injection  — the SAME retrieval from the SAME frozen
           prefix; a deterministic whitelist memory block is injected into
           the Judge prompt; Method B -> z_prompt; tier = argmax z_prompt.
           NO logit modulation (z_mod is never computed). One ADDITIONAL
           raw-prompt Method B call records z_base for influence / Judge
           drift audit (prereg §3.2); the executed tier is decided by
           z_prompt alone.

Frozen memory (prereg §2):
    - source: logs/s2_full_C_main24_retry2_memory.jsonl (the formal C0 run);
    - replay copy: eval/ablation/frozen-c0-memory-trace.jsonl, each entry
      carrying source_task_id / source_order keyed to the frozen 24-task
      order; SHA-256 in frozen-c0-memory-trace.sha256.json;
    - PREFIX REPLAY: the task at frozen position i may only read entries
      with source_order < i — never its own C0 outcome, never a future one;
    - READ-ONLY: no arm ever writes memory (engine.learn is never called;
      there is no --memory-out). Both arms therefore replay the exact C0
      memory state that existed before each task's decision.

Task order: the frozen 24-task sequence of prereg §2.2 is enforced — the
selected task list is always a subsequence of it (formal runs = all 24).
--task-ids exists ONLY for the prereg §9.2 one-task smoke runs; smoke logs
must never be used as formal results (the analyzer enforces 24/24).

Engine RNG exactness (T8-LM): one engine per run, seeded 42 like C0; only
its memory attribute is swapped between tasks (decide() itself never
mutates memory). learn() never runs, and learn() never draws from the RNG,
so the per-task decide() call sequence reproduces C0's RNG stream exactly
— validated to reproduce C0's V/Q/A_norm/z_mod bit-for-bit from the frozen
trace plus C0's z_base vectors.

Latency accounting (prereg §7/H4, separate columns):
    judge_latency_ms       deciding Judge call (LM: raw prompt; PM: injected)
    base_judge_latency_ms  raw-prompt Method B call (LM: same call as the
                           deciding one; PM: the additional audit call)
    prompt_judge_latency_ms injected-prompt call (PM; 0.0 sentinel in LM —
                           no prompt-injection call ran)
    jitrl_latency_ms       JitRL/retrieval post-processing only (LM:
                           engine.decide(); PM: retrieval + block render).
                           Never contains any Judge request time.

Failure policy (mirrors harness.run_real): execution or Evaluator failure
-> record written with exec_error/eval_error, reward null, NO memory write
(there is never a memory write anyway); Judge failure -> run aborts. Per
prereg §9.4 a run containing any execution/Evaluator failure is audit-only
(the analyzer refuses it as formal input).

Budget stop (prereg §9.8): cumulative exec+eval spend of THIS invocation is
capped at --budget-usd (default $1.00); exceeding it aborts the run.

Security: records contain no credentials. The CPA key is read from the
PilotDeck config and only ever placed in the Authorization header. Prompts
are NOT persisted (only SHA-256 digests and the block's char count).

CLI (formal, prereg §6 naming):
    python -m harness.run_ablation --arm T8-LM --judge llama \
        --tasks eval/tasks.jsonl \
        --memory-trace eval/ablation/frozen-c0-memory-trace.jsonl \
        --out logs/s3_ablation_T8_LM.jsonl
    python -m harness.run_ablation --arm T8-PM ... --out logs/s3_ablation_T8_PM.jsonl
Failed full attempts are re-run with _retry1/_retry2 output suffixes; this
module refuses to overwrite an existing output file or any frozen C0/C1
asset (logs/s2_full_*, eval/results-*, eval/charts, eval/c1, poster-a3).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from harness.cpa_client import CPAExecClient
from harness.evaluator import Evaluator
from harness.pricing import DEFAULT_MODEL, TIER_TO_MODEL, cost_from_usage
from harness.real_rewards import RealQualityReward
from harness.run import build_judge, load_tasks
from harness.run_real import (
    DEFAULT_MAX_TOKENS_EVAL,
    DEFAULT_MAX_TOKENS_EXEC,
    build_cpa_client,
)
from jitrl_core import JitRLConfig, JitRLEngine
from jitrl_core.config import TIERS
from jitrl_core.engine import GATE_BELOW_MIN_NEIGHBORS, GATE_NO_NEIGHBORS
from jitrl_core.memory import ExperienceMemory
from jitrl_core.policy import choose_tier
from jitrl_core.state import intent_class, task_signature
from local_judge.client import build_prompt as build_judge_prompt

ARM_LM = "T8-LM"
ARM_PM = "T8-PM"
ARMS = (ARM_LM, ARM_PM)
ROUTE_MODE_LM = "jitrl-frozen"
ROUTE_MODE_PM = "prompt-memory"

#: Frozen 24-task order (prereg §2.2 / §5) — also the C0 execution order.
FROZEN_TASK_ORDER: tuple[str, ...] = (
    "T01", "T05", "T09", "T13", "T17", "T21",
    "T02", "T06", "T10", "T14", "T18", "T22",
    "T03", "T07", "T11", "T15", "T19", "T23",
    "T04", "T08", "T12", "T16", "T20", "T24",
)
FROZEN_TASK_INDEX: dict[str, int] = {tid: i for i, tid in enumerate(FROZEN_TASK_ORDER)}

DEFAULT_MEMORY_TRACE = "eval/ablation/frozen-c0-memory-trace.jsonl"
DEFAULT_TASKS = "eval/tasks.jsonl"

#: Canonical C0 memory entry schema (prereg §2.1: these fields are copied
#: verbatim from the C0 memory file; only source_task_id/source_order are
#: added by the frozen-trace builder).
MEMORY_ENTRY_KEYS = ("intent_class", "signature_tokens", "tier", "G", "ts",
                     "episode_id")

SIGNATURE_PREVIEW_TOKENS = 12
RESPONSE_PREVIEW_CHARS = 600
DEFAULT_BUDGET_USD = 1.0

#: Deterministic whitelist block formatting (prereg §4.3).
G_FORMAT = "+.4f"
SIMILARITY_FORMAT = ".4f"

#: Prompt placeholders that must never appear inside a memory block
#: (build_prompt substitutes {CONTINUATION} before {USER_MESSAGE}; a
#: placeholder in the block would trigger a second replacement).
BLOCK_FORBIDDEN_PLACEHOLDERS = ("{USER_MESSAGE}", "{CONTINUATION}", "{PT}")

#: Output paths this harness must never write (prereg §6).
PROTECTED_OUT_PREFIXES = ("logs/s2_full_",)
PROTECTED_OUT_DIRS = ("eval", "poster-a3")

REPO_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ helpers

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_writable_out(out_path: str | Path) -> Path:
    """Refuse frozen-asset overwrites and re-runs over existing outputs.

    Prereg §6: retries use _retry1/_retry2 suffixes; logs/s2_full_* and the
    C0/C1 evaluation assets must never be touched."""
    p = Path(out_path)
    try:
        rel = p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        rel = None  # outside the repo (tests): only the existence rule applies
    if rel is not None:
        for prefix in PROTECTED_OUT_PREFIXES:
            if rel.startswith(prefix):
                raise SystemExit(
                    f"refusing to write {rel}: frozen C0/C1 asset (prereg §6)")
        for directory in PROTECTED_OUT_DIRS:
            if rel == directory or rel.startswith(directory + "/"):
                raise SystemExit(
                    f"refusing to write {rel}: protected directory (prereg §6)")
    if p.exists():
        raise SystemExit(
            f"output {p} already exists; use a _retry1/_retry2 suffix "
            "(prereg §6) instead of overwriting")
    return p


def load_frozen_trace(
    trace_path: str | Path,
    *,
    expected_sha256: str | None = None,
    sha_json_path: str | Path | None = None,
) -> tuple[list[dict], str]:
    """Load and validate the frozen C0 memory-trace replay copy.

    Hard checks (fail fast, never a silent fallback):
      - file digest matches the expected hash when one is known (explicit
        argument or the sidecar .sha256.json);
      - exactly 24 entries; source_order == line index (0..23);
        source_task_id == FROZEN_TASK_ORDER[source_order];
      - every entry carries the verbatim C0 schema keys.
    Returns (entries, sha256_of_file)."""
    p = Path(trace_path)
    if not p.is_file():
        raise SystemExit(f"memory trace not found: {p}")
    digest = sha256_file(p)
    if expected_sha256 is None and sha_json_path is not None:
        sj = Path(sha_json_path)
        if sj.is_file():
            meta = json.loads(sj.read_text(encoding="utf-8"))
            expected_sha256 = meta.get("copy", {}).get("sha256")
    if expected_sha256 is not None and digest != expected_sha256:
        raise SystemExit(
            f"memory trace hash mismatch for {p}: expected {expected_sha256}, "
            f"got {digest} (frozen asset modified? prereg §2.1)")

    entries: list[dict] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    if len(entries) != len(FROZEN_TASK_ORDER):
        raise SystemExit(
            f"memory trace must hold exactly {len(FROZEN_TASK_ORDER)} entries, "
            f"found {len(entries)} in {p}")
    for i, e in enumerate(entries):
        missing = [k for k in MEMORY_ENTRY_KEYS if k not in e]
        if missing:
            raise SystemExit(f"memory trace entry {i} missing keys {missing}")
        if e.get("source_order") != i:
            raise SystemExit(
                f"memory trace entry {i} has source_order={e.get('source_order')!r}")
        if e.get("source_task_id") != FROZEN_TASK_ORDER[i]:
            raise SystemExit(
                f"memory trace entry {i} has source_task_id="
                f"{e.get('source_task_id')!r}, expected {FROZEN_TASK_ORDER[i]!r}")
        if e.get("tier") not in TIERS:
            raise SystemExit(f"memory trace entry {i} has invalid tier {e.get('tier')!r}")
    return entries, digest


def entry_to_engine_schema(entry: dict) -> dict:
    """Strip the audit keys -> the exact 6-key C0 memory entry schema, so the
    prefix memory replayed into the engine is structurally identical to what
    C0's engine held at that point."""
    return {k: entry[k] for k in MEMORY_ENTRY_KEYS}


def prefix_entries(entries: list[dict], frozen_index: int) -> list[dict]:
    """Prefix replay (prereg §2.2): only entries with source_order < i."""
    return [e for e in entries if e["source_order"] < frozen_index]


def build_prefix_memory(entries: list[dict], frozen_index: int,
                        cap: int) -> ExperienceMemory:
    """Read-only ExperienceMemory holding exactly the frozen prefix for the
    task at frozen_index (insertion order preserved == C0 order, which keeps
    retrieval tie-breaking identical)."""
    mem = ExperienceMemory(cap=cap)
    for e in prefix_entries(entries, frozen_index):
        mem.add(entry_to_engine_schema(e))
    return mem


def retrieve_prefix(memory: ExperienceMemory, message: str,
                    config: JitRLConfig) -> list[tuple[dict, float]]:
    """The exact retrieval call JitRLEngine.decide() performs internally
    (same intent/token-set/k/threshold) — used for the audit copy (LM) and
    as the single retrieval for PM, guaranteeing both arms see the identical
    neighbor set for the same task."""
    return memory.retrieve(
        intent_class(message),
        frozenset(task_signature(message)),
        k=config.k,
        threshold=config.jaccard_threshold,
    )


def audit_retrieved(matches: list[tuple[dict, float]]) -> list[dict]:
    """Prereg §7: retrieved_memories audit copy — tier / G / similarity only
    (retrieval order preserved)."""
    return [{"tier": e["tier"], "G": float(e["G"]), "similarity": float(sim)}
            for e, sim in matches]


# ------------------------------------------------- whitelist memory block

BLOCK_HEADER = "## Reference - similar historical requests (anonymized outcomes)"
BLOCK_PREAMBLE = (
    "The lines below are anonymized outcome records of similar historical "
    "requests routed by this system. Treat them as weak reference only, NOT "
    "as hard rules; the final tier must follow the routing rules above and "
    "the current user message."
)


def render_memory_block(matches: list[tuple[dict, float]]) -> str:
    """Deterministic whitelist-only memory block (prereg §4).

    Allowed content ONLY: neighbor index, tier, G (fixed precision),
    Jaccard similarity, neighbor count, per-tier count and mean G. Neighbors
    are emitted in the existing retrieval order (§4.3). Everything else —
    GT tiers, checklists, Evaluator structure, raw text, task ids, family,
    timestamps, episode ids, secrets — is structurally impossible here: the
    renderer never receives anything but (tier, G, similarity)."""
    if not matches:
        raise ValueError("render_memory_block requires at least one neighbor")
    lines = [BLOCK_HEADER, BLOCK_PREAMBLE]
    lines.append(f"- neighbor_count: {len(matches)}")
    agg = []
    for t in TIERS:  # canonical tier order -> fixed field order
        gs = [float(e["G"]) for e, _ in matches if e["tier"] == t]
        mean = format(sum(gs) / len(gs), G_FORMAT) if gs else "n/a"
        agg.append(f"{t} n={len(gs)} mean_G={mean}")
    lines.append("- per_tier: " + " | ".join(agg))
    for idx, (e, sim) in enumerate(matches, start=1):
        lines.append(
            f"- neighbor_{idx}: tier={e['tier']} G={format(float(e['G']), G_FORMAT)} "
            f"similarity={format(float(sim), SIMILARITY_FORMAT)}")
    block = "\n".join(lines) + "\n"
    for placeholder in BLOCK_FORBIDDEN_PLACEHOLDERS:
        if placeholder in block:
            raise ValueError(f"memory block contains placeholder {placeholder!r}")
    return block


# ------------------------------------------------------------------ records

def clamp_z(logits: dict, z_min: float) -> dict:
    """Clamp a Judge logit vector to z_min (the engine's clamp_logits
    convention; applied to z_prompt/z_base so PM decisions use the same
    clamped space as LM and the recorded z_base is comparable across arms)."""
    return {t: max(float(logits.get(t, 0.0)), z_min) for t in TIERS}


def make_ablation_record(
    *,
    episode_id: int,
    task: dict,
    intent: str,
    signature_tokens: list[str],
    arm: str,
    route_mode: str,
    n_retrieved: int,
    top_similarity: float,
    min_neighbors: int,
    gate_reason: str | None,
    modulation_active: bool,
    V: float,
    Q: dict,
    A_norm: dict,
    z_base: dict | None,
    z_mod: dict | None,
    z_prompt: dict | None,
    chosen_tier: str,
    base_choice: str,
    flipped: bool,
    influenced: bool,
    retrieved_memories: list[dict],
    block: str | None,
    memory_prefix_size: int,
    memory_source_sha256: str,
    previous_tier: str | None,
    judge_latency_ms: float,
    base_judge_latency_ms: float,
    prompt_judge_latency_ms: float,
    jitrl_latency_ms: float,
    judge_calls: int,
    decision_prompt_sha256: str,
    base_prompt_sha256: str,
    exec_model: str,
    exec_result=None,
    exec_error: str | None,
    exec_cost_usd: float | None,
    eval_result=None,
    eval_error: str | None,
    reward: float | None,
    reward_breakdown: dict | None,
) -> dict:
    """Existing real-episode contract (run_real.make_record field set) plus
    the additive T8 audit fields of prereg §7."""
    rec = {
        # --- existing real-episode contract -----------------------------
        "episode_id": episode_id,
        "task_id": task["task_id"],
        "family": task.get("family"),
        "gt_tier": task.get("gt_tier"),
        "intent_class": intent,
        "signature_preview": signature_tokens[:SIGNATURE_PREVIEW_TOKENS],
        "n_retrieved": n_retrieved,
        "top_similarity": top_similarity,
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
        "jitrl_latency_ms": jitrl_latency_ms,
        "memory_written": False,   # frozen read-only memory: never written
        # --- additive T8 audit fields (prereg §7) ------------------------
        "ablation_arm": arm,
        "memory_frozen": True,
        "memory_prefix_size": memory_prefix_size,
        "memory_source_sha256": memory_source_sha256,
        "retrieved_memories": retrieved_memories,
        "z_prompt": z_prompt,
        "influenced": influenced,
        "injected_block_chars": len(block) if block else 0,
        "injected_block_sha256": (sha256_text(block) if block else None),
        "previous_tier": previous_tier,
        "judge_calls": judge_calls,
        "decision_prompt_sha256": decision_prompt_sha256,
        "base_prompt_sha256": base_prompt_sha256,
        "base_judge_latency_ms": base_judge_latency_ms,
        "prompt_judge_latency_ms": prompt_judge_latency_ms,
    }
    return rec


def summarize_ablation(records: list[dict], *, arm: str, judge_kind: str,
                       config: JitRLConfig, trace_path: str | Path,
                       trace_sha256: str, max_tokens_exec: int,
                       max_tokens_eval: int, temperature_exec: float,
                       budget_usd: float, spent_usd: float) -> dict:
    labeled = [r for r in records if r.get("gt_tier")]
    accuracy = (sum(1 for r in labeled if r["chosen_tier"] == r["gt_tier"])
                / len(labeled) if labeled else None)
    eval_ok = [r for r in records if r["evaluator"] is not None]
    qualities = [r["evaluator"]["quality_score"] for r in eval_ok]
    exec_costs = [r["exec_cost_usd"] for r in records if r["exec_cost_usd"] is not None]
    eval_costs = [r["eval_cost_usd"] for r in records if r["eval_cost_usd"] is not None]
    rewards = [r["reward"] for r in records if r["reward"] is not None]
    verdicts: dict[str, int] = {}
    for r in eval_ok:
        v = r["evaluator"]["verdict"]
        verdicts[v] = verdicts.get(v, 0) + 1
    task_order = [r["task_id"] for r in records]
    n = len(records) or 1

    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    return {
        "ablation_arm": arm,
        "preregistration": "S3-TAB8-PREREGISTRATION.md v1.0",
        "judge": judge_kind,
        "episodes": 1,
        "n_records": len(records),
        "task_order": task_order,
        "frozen_order_expected": list(FROZEN_TASK_ORDER),
        "order_is_full_frozen_24": task_order == list(FROZEN_TASK_ORDER),
        "config_snapshot": config.snapshot(),
        "ablation_config": {
            "max_tokens_exec": max_tokens_exec,
            "max_tokens_eval": max_tokens_eval,
            "temperature_exec": temperature_exec,
            "evaluator_temperature": 0.2,
            "memory_trace": str(trace_path),
            "memory_trace_sha256": trace_sha256,
            "memory_read_only": True,
            "memory_prefix_replay": "source_order < i",
            "prefix_gate": "no block / no modulation when n_retrieved < min_neighbors",
        },
        "memory_frozen": True,
        "memory_written": sum(1 for r in records if r["memory_written"]),
        # retrieval-hit = the memory mechanism was eligible to act
        # (n_retrieved >= min_neighbors; with the frozen min_neighbors=1
        # this is exactly n_retrieved > 0)
        "retrieval_hits": sum(1 for r in records if r["gate_reason"] is None
                              and r["n_retrieved"] > 0),
        "gated_records": sum(1 for r in records if r["gate_reason"] is not None),
        "influenced_records": sum(1 for r in records if r["influenced"]),
        "injected_records": sum(1 for r in records if r["injected_block_chars"] > 0),
        "modulated_records": sum(1 for r in records if r["modulation_active"]),
        "accuracy_vs_gt": accuracy,
        "flips": sum(1 for r in records if r["flipped"]),
        "eval_success": f"{len(eval_ok)}/{len(records)}",
        "eval_success_rate": (len(eval_ok) / len(records)) if records else None,
        "failures": [r["task_id"] for r in records
                     if r["exec_error"] or r["eval_error"] or r["evaluator"] is None],
        "avg_quality_score": _mean(qualities),
        "verdicts": verdicts,
        "avg_exec_cost_usd": _mean(exec_costs),
        "total_exec_cost_usd": sum(exec_costs) if exec_costs else 0.0,
        "total_eval_cost_usd": sum(eval_costs) if eval_costs else 0.0,
        "avg_reward_real": _mean(rewards),
        "finish_reason_length": sum(1 for r in records
                                    if r.get("exec_finish_reason") == "length"),
        "empty_responses": sum(1 for r in records
                               if r.get("exec_response_preview") is not None
                               and not str(r["exec_response_preview"]).strip()),
        "latency_ms": {
            # deciding Judge call (LM: raw prompt; PM: injected prompt)
            "judge_mean": _mean([r["judge_latency_ms"] for r in records]),
            # raw-prompt Method B call (LM: the deciding call itself;
            # PM: the additional audit-only call, prereg §3.2)
            "base_judge_mean": _mean([r["base_judge_latency_ms"] for r in records]),
            # injected-prompt call (PM; 0.0 sentinel in LM)
            "prompt_judge_mean": _mean([r["prompt_judge_latency_ms"] for r in records]),
            # JitRL/retrieval post-processing only — never Judge time
            "jitrl_mean": _mean([r["jitrl_latency_ms"] for r in records]),
            "jitrl_max": max((r["jitrl_latency_ms"] for r in records), default=0.0),
            "judge_calls_total": sum(r["judge_calls"] for r in records),
            "judge_calls_mean": sum(r["judge_calls"] for r in records) / n,
        },
        "budget_usd": budget_usd,
        "spent_usd": round(spent_usd, 6),
    }


# --------------------------------------------------------------------- run

def select_tasks(tasks: list[dict], task_ids: list[str] | None) -> list[dict]:
    """Return the selected tasks IN FROZEN ORDER (always a subsequence of
    FROZEN_TASK_ORDER; formal runs pass task_ids=None -> all 24)."""
    by_id = {t["task_id"]: t for t in tasks}
    missing = [tid for tid in FROZEN_TASK_ORDER if tid not in by_id]
    if missing:
        raise SystemExit(f"frozen task ids missing from tasks file: {missing}")
    if task_ids:
        wanted = [t.strip() for t in task_ids if t.strip()]
        unknown = [t for t in wanted if t not in FROZEN_TASK_INDEX]
        if unknown:
            raise SystemExit(
                f"task ids outside the frozen 24-task order: {unknown} "
                "(smoke runs must pick tasks from the frozen sequence)")
        keep = set(wanted)
        selected = [by_id[tid] for tid in FROZEN_TASK_ORDER if tid in keep]
    else:
        selected = [by_id[tid] for tid in FROZEN_TASK_ORDER]
    if not selected:
        raise SystemExit("no tasks selected")
    return selected


def run(
    *,
    arm: str,
    tasks_path: str | Path = DEFAULT_TASKS,
    memory_trace_path: str | Path = DEFAULT_MEMORY_TRACE,
    judge_kind: str = "llama",
    out_path: str | Path | None = None,
    task_ids: list[str] | None = None,
    config_path: str | None = None,
    max_tokens_exec: int = DEFAULT_MAX_TOKENS_EXEC,
    max_tokens_eval: int = DEFAULT_MAX_TOKENS_EVAL,
    temperature_exec: float = 0.7,
    budget_usd: float = DEFAULT_BUDGET_USD,
    trace_sha256_expected: str | None = None,
    cpa_client: CPAExecClient | None = None,
    evaluator: Evaluator | None = None,
    judge=None,
) -> dict:
    """Run one T8 arm over the frozen task order with frozen-prefix memory
    replay. cpa_client / evaluator / judge are injectable for tests."""
    if arm not in ARMS:
        raise SystemExit(f"arm must be one of {ARMS}, got {arm!r}")

    # Frozen JitRL conditions (prereg §5): the config defaults ARE the frozen
    # values (k=10, beta=5, lam=0.05, alpha=5, jaccard_threshold=0.5,
    # z_min=-10, min_neighbors=1, seed=42). No overrides are accepted.
    config = JitRLConfig()

    trace_path = Path(memory_trace_path)
    sha_json = trace_path.with_name(trace_path.stem + ".sha256.json")
    entries, trace_sha = load_frozen_trace(
        trace_path, expected_sha256=trace_sha256_expected, sha_json_path=sha_json)

    tasks = select_tasks(load_tasks(tasks_path), task_ids)

    # Validate the output path BEFORE spending any API budget: refuse frozen
    # assets and re-runs over existing outputs (prereg §6/§9.5).
    out = ensure_writable_out(out_path) if out_path is not None else None

    if judge is None:
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

    # T8-LM exactness: ONE engine per run (seed 42 like C0); only its memory
    # is swapped per task. T8-PM runs no engine at all (no modulation).
    engine: JitRLEngine | None = None
    if arm == ARM_LM:
        engine = JitRLEngine(config=config)

    records: list[dict] = []
    previous_tier: str | None = None
    spent_usd = 0.0
    episode_id = 1  # one episode per arm (prereg §5)

    for task in tasks:
        tid = task["task_id"]
        frozen_index = FROZEN_TASK_INDEX[tid]
        message = task["message"]
        intent = intent_class(message)
        signature_tokens = sorted(task_signature(message))

        # --- read-only frozen prefix + the single shared retrieval --------
        prefix_mem = build_prefix_memory(entries, frozen_index, config.memory_cap)
        matches = retrieve_prefix(prefix_mem, message, config)
        n_retrieved = len(matches)
        top_sim = matches[0][1] if matches else 0.0
        retrieved_audit = audit_retrieved(matches)
        memory_prefix_size = len(prefix_entries(entries, frozen_index))

        if arm == ARM_LM:
            # ---- T8-LM: raw-prompt Judge -> z_base -> exact C0 modulation
            jr = judge.judge(message, previous_tier=previous_tier)
            judge_lat = float(jr["latency_ms"])
            base_judge_lat = judge_lat          # the only call is the raw call
            prompt_judge_lat = 0.0              # sentinel: no injected call
            engine.memory = prefix_mem          # read-only swap; decide never mutates
            t_decide = time.perf_counter()
            decision = engine.decide(message, jr["tier_logits"])
            jitrl_lat = (time.perf_counter() - t_decide) * 1000.0
            chosen = decision.chosen_tier
            z_base, z_mod, z_prompt = decision.z_base, decision.z_mod, None
            base_choice, flipped = decision.base_choice, decision.flipped
            V, Q, A_norm = decision.V, decision.Q, decision.A_norm
            gate_reason = decision.gate_reason
            modulation_active = decision.modulation_active
            min_neighbors = decision.min_neighbors
            block = None                        # LM never injects (prereg §3.1)
            judge_calls = 1
            decision_prompt = build_judge_prompt(message, previous_tier, None)
            base_prompt = decision_prompt
        else:
            # ---- T8-PM: whitelist block -> injected Judge -> z_prompt -----
            # (plus one raw-prompt audit call; no logit modulation ever)
            t_decide = time.perf_counter()
            if n_retrieved >= config.min_neighbors and n_retrieved > 0:
                block = render_memory_block(matches)
            else:
                block = None
            jitrl_lat = (time.perf_counter() - t_decide) * 1000.0
            decision_prompt = build_judge_prompt(message, previous_tier, block)
            base_prompt = build_judge_prompt(message, previous_tier, None)

            jr_prompt = judge.judge(message, previous_tier=previous_tier,
                                    memory_block=block)
            z_prompt = clamp_z(jr_prompt["tier_logits"], config.z_min)
            prompt_judge_lat = float(jr_prompt["latency_ms"])
            chosen = choose_tier(z_prompt)      # executed tier: z_prompt only

            jr_base = judge.judge(message, previous_tier=previous_tier)
            z_base = clamp_z(jr_base["tier_logits"], config.z_min)
            base_judge_lat = float(jr_base["latency_ms"])
            base_choice = choose_tier(z_base)
            flipped = chosen != base_choice

            judge_lat = prompt_judge_lat        # deciding call
            z_mod = None                        # PM never modulates (prereg §3.2)
            V = 0.0
            Q = {t: 0.0 for t in TIERS}
            A_norm = {t: 0.0 for t in TIERS}
            if n_retrieved == 0:
                gate_reason = GATE_NO_NEIGHBORS
            elif n_retrieved < config.min_neighbors:
                gate_reason = GATE_BELOW_MIN_NEIGHBORS
            else:
                gate_reason = None
            modulation_active = False           # PM: mechanism is the prompt
            min_neighbors = config.min_neighbors
            judge_calls = 2

        influenced = gate_reason is None and chosen != base_choice

        # --- execution / Evaluator / reward: the run_real code path -------
        exec_model = TIER_TO_MODEL.get(chosen, DEFAULT_MODEL)
        exec_result = None
        exec_error: str | None = None
        eval_result = None
        eval_error: str | None = None
        reward = None
        breakdown = None
        exec_cost_usd = None

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
        # NO engine.learn(): memory is frozen read-only in both arms.

        records.append(make_ablation_record(
            episode_id=episode_id, task=task,
            intent=intent, signature_tokens=signature_tokens,
            arm=arm, route_mode=(ROUTE_MODE_LM if arm == ARM_LM else ROUTE_MODE_PM),
            n_retrieved=n_retrieved, top_similarity=top_sim,
            min_neighbors=min_neighbors, gate_reason=gate_reason,
            modulation_active=modulation_active, V=V, Q=Q, A_norm=A_norm,
            z_base=z_base, z_mod=z_mod, z_prompt=z_prompt,
            chosen_tier=chosen, base_choice=base_choice, flipped=flipped,
            influenced=influenced, retrieved_memories=retrieved_audit,
            block=block, memory_prefix_size=memory_prefix_size,
            memory_source_sha256=trace_sha, previous_tier=previous_tier,
            judge_latency_ms=judge_lat, base_judge_latency_ms=base_judge_lat,
            prompt_judge_latency_ms=prompt_judge_lat,
            jitrl_latency_ms=jitrl_lat, judge_calls=judge_calls,
            decision_prompt_sha256=sha256_text(decision_prompt),
            base_prompt_sha256=sha256_text(base_prompt),
            exec_model=exec_model, exec_result=exec_result,
            exec_error=exec_error, exec_cost_usd=exec_cost_usd,
            eval_result=eval_result, eval_error=eval_error,
            reward=reward, reward_breakdown=breakdown,
        ))
        previous_tier = chosen

        spent_usd += (exec_cost_usd or 0.0) + (
            (eval_result.cost_usd or 0.0) if eval_result is not None else 0.0)
        if spent_usd > budget_usd:
            raise RuntimeError(
                f"budget stop (prereg §9.8): spent ${spent_usd:.4f} exceeds "
                f"${budget_usd:.4f} after task {tid}; run aborted, no output "
                f"file written")

    summary = summarize_ablation(
        records, arm=arm, judge_kind=judge_kind, config=config,
        trace_path=trace_path, trace_sha256=trace_sha,
        max_tokens_exec=max_tokens_exec, max_tokens_eval=max_tokens_eval,
        temperature_exec=temperature_exec, budget_usd=budget_usd,
        spent_usd=spent_usd)

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.write(json.dumps({"_summary": summary}, ensure_ascii=False) + "\n")

    print(f"[run_ablation] arm={arm} judge={judge_kind} tasks={len(records)} "
          f"order_full_24={summary['order_is_full_frozen_24']}")
    print(f"[run_ablation] trace_sha256={trace_sha}")
    print(f"[run_ablation] retrieval_hits={summary['retrieval_hits']} "
          f"gated={summary['gated_records']} influenced={summary['influenced_records']} "
          f"injected={summary['injected_records']} modulated={summary['modulated_records']}")
    print(f"[run_ablation] eval_success={summary['eval_success']} "
          f"accuracy_vs_gt={summary['accuracy_vs_gt']} flips={summary['flips']}")
    print(f"[run_ablation] exec_cost=${summary['total_exec_cost_usd']:.6f} "
          f"eval_cost=${summary['total_eval_cost_usd']:.6f} "
          f"avg_quality={summary['avg_quality_score']}")
    print(f"[run_ablation] latency_ms judge_mean={summary['latency_ms']['judge_mean']} "
          f"base_judge_mean={summary['latency_ms']['base_judge_mean']} "
          f"prompt_judge_mean={summary['latency_ms']['prompt_judge_mean']} "
          f"jitrl_mean={summary['latency_ms']['jitrl_mean']} "
          f"jitrl_max={summary['latency_ms']['jitrl_max']}")
    print(f"[run_ablation] memory_written={summary['memory_written']} "
          f"(frozen read-only; prefix replay source_order<i)")
    return summary


# ---------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="harness.run_ablation",
        description="S3 Table-8 ablation: T8-LM (logit modulation) vs "
                    "T8-PM (prompt memory injection), frozen C0 prefix replay")
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--tasks", default=DEFAULT_TASKS)
    ap.add_argument("--task-ids", default=None,
                    help="comma-separated SUBSET of the frozen 24-task order "
                         "(smoke runs only, prereg §9.2; formal runs omit it)")
    ap.add_argument("--judge", default="llama", choices=["mock", "llama"])
    ap.add_argument("--memory-trace", default=DEFAULT_MEMORY_TRACE)
    ap.add_argument("--trace-sha256", default=None,
                    help="expected SHA-256 of the memory trace (default: "
                         "read from <trace>.sha256.json when present)")
    ap.add_argument("--out", default=None,
                    help="episode record JSONL output (default: "
                         "logs/s3_ablation_T8_{LM,PM}.jsonl, prereg §6)")
    ap.add_argument("--config", default=None,
                    help="pilotdeck.yaml path (default: PILOTDECK_CONFIG_PATH / "
                         "PILOT_HOME / ~/.pilotdeck/pilotdeck.yaml)")
    ap.add_argument("--max-tokens-exec", type=int, default=DEFAULT_MAX_TOKENS_EXEC)
    ap.add_argument("--max-tokens-eval", type=int, default=DEFAULT_MAX_TOKENS_EVAL)
    ap.add_argument("--temperature-exec", type=float, default=0.7)
    ap.add_argument("--budget-usd", type=float, default=DEFAULT_BUDGET_USD,
                    help="per-invocation exec+eval spend cap (prereg §9.8)")
    args = ap.parse_args(argv)

    out = args.out or f"logs/s3_ablation_T8_{args.arm.split('-')[1]}.jsonl"
    task_ids = ([t.strip() for t in args.task_ids.split(",")]
                if args.task_ids else None)
    run(
        arm=args.arm,
        tasks_path=args.tasks,
        memory_trace_path=args.memory_trace,
        judge_kind=args.judge,
        out_path=out,
        task_ids=task_ids,
        config_path=args.config,
        max_tokens_exec=args.max_tokens_exec,
        max_tokens_eval=args.max_tokens_eval,
        temperature_exec=args.temperature_exec,
        budget_usd=args.budget_usd,
        trace_sha256_expected=args.trace_sha256,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
