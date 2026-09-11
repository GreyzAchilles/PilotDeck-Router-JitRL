"""Continuation-probe routing evaluation — judge-only, zero CPA cost.

Session-2 deliverable for the 5 continuation probes (P01-P05) in
eval/tasks.jsonl. Unlike harness.run_real (which executes every task via
CPA and scores it with a blind evaluator), this script touches ONLY the
routing layer:

    probe message + explicit previous_tier -> local Judge (Method B)
    -> chosen tier -> hit vs gt

Why not `run_real --task-ids P01,...`: run_real chains previous_tier from
the previous row's CHOSEN tier; feeding it the probe subset alone starts
the chain at previous_tier=None, so P01's bare "继续" is judged as a fresh
task (and any upstream MISS would poison the chain). This script instead
resolves each probe's previous tier explicitly from the task set itself
and never executes/evaluates the user task (no CPA cost).

previous_tier resolution (per probe, fail-fast on any disagreement):
  1. notes reference "紧跟在 T05 后..." -> that MAIN task's gt_tier;
  2. nearest preceding main task in file order (fallback when notes are
     absent);
  3. cross-checks: the tier stated in the probe notes ("正确档位继承
     medium"), and the frozen expectation map EXPECTED_PREV_TIERS below
     (from the task-set GT). All present sources must agree.

Records are desensitized routing data only — probe id, family,
expected/previous tier, preceding task id, chosen tier, hit, tier_logits,
judge latency, attempts/error. Never the user message text, never
exec/eval/reward/cost fields.

Output: logs/s2_continuation_probes.jsonl — one JSON record per probe
plus a final {"_summary": ...} line (harness convention) — and a
standalone summary JSON next to it. Human summary on stdout.

CLI:
    python eval/run_continuation_probes.py \
        --tasks eval/tasks.jsonl --out logs/s2_continuation_probes.jsonl \
        --endpoint http://127.0.0.1:18080
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# Bootstrap repo root so this runs both as `python eval/...` and as a
# package module (eval.run_continuation_probes) under pytest.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from local_judge.config import TIERS  # noqa: E402
from harness.run import load_tasks  # noqa: E402  (same loader as run_real)

PROBE_FAMILY = "continuation"
# notes: "紧跟在 T05 后，正确档位继承 medium。"
NOTES_TASK_RE = re.compile(r"紧跟在\s*([A-Z]+\d+)\s*后")
NOTES_TIER_RE = re.compile(r"档位继承\s*([a-z]+)")

# Frozen expectation (task-set GT, session-2 directive): probe -> (its
# immediately preceding MAIN task id, the previous tier the probe must
# inherit). Used as a hard cross-check of the dynamic resolution.
EXPECTED_PREV_TIERS: dict[str, tuple[str, str]] = {
    "P01": ("T05", "medium"),
    "P02": ("T13", "reasoning"),
    "P03": ("T10", "medium"),
    "P04": ("T19", "reasoning"),
    "P05": ("T12", "reasoning"),
}

DEFAULT_TASKS = "eval/tasks.jsonl"
DEFAULT_OUT = "logs/s2_continuation_probes.jsonl"
DEFAULT_SUMMARY_OUT = "logs/s2_continuation_probes_summary.json"
DEFAULT_ENDPOINT = "http://127.0.0.1:18080"
DEFAULT_RETRIES = 3
DEFAULT_RETRY_SLEEP_S = 2.0
ERROR_MSG_MAX_CHARS = 300


# --------------------------------------------------------------- resolution
def resolve_previous_tiers(
    tasks: list[dict],
    frozen: dict[str, tuple[str, str]] | None = None,
) -> dict[str, tuple[str, str]]:
    """Map every continuation probe id -> (preceding main task id, previous_tier).

    Resolution sources per probe (all present sources must agree):
      notes task reference -> referenced main task's gt_tier;
      file order           -> nearest preceding main task's gt_tier;
      notes stated tier    -> tier spelled out in the notes;
      frozen expectation   -> EXPECTED_PREV_TIERS entry (when listed).
    Raises ValueError naming the probe and the disagreeing sources.
    """
    frozen = frozen if frozen is not None else EXPECTED_PREV_TIERS
    by_id = {t["task_id"]: t for t in tasks}
    resolved: dict[str, tuple[str, str]] = {}
    last_main: dict | None = None
    for task in tasks:
        if task.get("family") == PROBE_FAMILY:
            resolved[task["task_id"]] = _resolve_one(task, by_id, last_main, frozen)
        else:
            last_main = task
    return resolved


def _resolve_one(probe: dict, by_id: dict[str, dict], last_main: dict | None,
                 frozen: dict[str, tuple[str, str]]) -> tuple[str, str]:
    pid = probe["task_id"]
    notes = probe.get("notes") or ""
    sources: dict[str, str] = {}
    ref_id: str | None = None

    m = NOTES_TASK_RE.search(notes)
    if m:
        ref_id = m.group(1)
        ref = by_id.get(ref_id)
        if ref is None:
            raise ValueError(f"{pid}: notes reference unknown task {ref_id!r}")
        if ref.get("family") == PROBE_FAMILY:
            raise ValueError(
                f"{pid}: notes reference {ref_id} which is itself a continuation probe")
        sources[f"notes_task({ref_id}).gt_tier"] = ref.get("gt_tier")

    m = NOTES_TIER_RE.search(notes)
    if m:
        if m.group(1) not in TIERS:
            raise ValueError(f"{pid}: notes state unknown tier {m.group(1)!r}")
        sources["notes_stated_tier"] = m.group(1)

    if last_main is not None:
        sources[f"file_order({last_main['task_id']}).gt_tier"] = last_main.get("gt_tier")
        if ref_id is None:
            ref_id = last_main["task_id"]

    if pid in frozen:
        exp_id, exp_tier = frozen[pid]
        sources[f"frozen({exp_id})"] = exp_tier
        if exp_tier not in TIERS:
            raise ValueError(f"{pid}: frozen expectation tier {exp_tier!r} not in TIERS")
        if ref_id is None:
            ref_id = exp_id

    if not sources:
        raise ValueError(f"{pid}: no previous_tier source (no notes ref, no prior main task)")

    tiers = set(sources.values())
    if None in tiers or len(tiers) != 1:
        raise ValueError(f"{pid}: disagreeing previous_tier sources: {sources}")
    if ref_id is None:
        raise ValueError(f"{pid}: could not identify the preceding main task id")
    return ref_id, tiers.pop()


# ------------------------------------------------------------------ runner
def judge_with_retry(client, message: str, previous_tier: str, *,
                     retries: int, retry_sleep_s: float) -> tuple[dict | None, int, str | None]:
    """Call client.judge with previous_tier; retry network jitter.

    JudgeClient already retries transport errors per HTTP request; this
    outer loop covers fuller jitter (server restarts, bursts). Returns
    (result, attempts, error); on final failure result is None and the
    compact error string is preserved.
    """
    error: str | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            return client.judge(message, previous_tier=previous_tier), attempt, None
        except Exception as e:  # noqa: BLE001 - recorded, never fatal
            error = f"{type(e).__name__}: {e}"[:ERROR_MSG_MAX_CHARS]
            if attempt < retries:
                time.sleep(retry_sleep_s)
    return None, max(1, retries), error


def run_probes(*, tasks: list[dict], client, retries: int = DEFAULT_RETRIES,
               retry_sleep_s: float = DEFAULT_RETRY_SLEEP_S,
               frozen: dict[str, tuple[str, str]] | None = None,
               probe_ids: list[str] | None = None) -> tuple[list[dict], dict]:
    """Judge every continuation probe with its explicit previous_tier.

    Returns (records, summary). Records contain desensitized routing data
    only; probes that fail all retries are recorded with error info and
    never abort the remaining probes.
    """
    probes = [t for t in tasks if t.get("family") == PROBE_FAMILY]
    if probe_ids:
        wanted = set(probe_ids)
        probes = [t for t in probes if t["task_id"] in wanted]
        missing = wanted - {t["task_id"] for t in probes}
        if missing:
            raise SystemExit(f"probe ids not found: {sorted(missing)}")

    prev_map = resolve_previous_tiers(tasks, frozen=frozen)

    records: list[dict] = []
    for task in probes:
        pid = task["task_id"]
        prev_task_id, prev_tier = prev_map[pid]
        result, attempts, error = judge_with_retry(
            client, task["message"], prev_tier,
            retries=retries, retry_sleep_s=retry_sleep_s)
        if result is None:
            records.append({
                "probe_id": pid, "family": task.get("family"),
                "expected_tier": task.get("gt_tier"),
                "previous_tier": prev_tier, "preceding_task_id": prev_task_id,
                "chosen_tier": None, "hit": None, "tier_logits": None,
                "judge_latency_ms": None, "attempts": attempts, "error": error,
            })
            continue
        chosen = result["chosen_tier"]
        expected = task.get("gt_tier")
        records.append({
            "probe_id": pid, "family": task.get("family"),
            "expected_tier": expected,
            "previous_tier": prev_tier, "preceding_task_id": prev_task_id,
            "chosen_tier": chosen,
            "hit": (chosen == expected) if expected else None,
            "tier_logits": {t: round(float(v), 4)
                            for t, v in result["tier_logits"].items()},
            "judge_latency_ms": round(float(result["latency_ms"]), 1),
            "attempts": attempts, "error": None,
        })

    return records, summarize(records)


def summarize(records: list[dict]) -> dict:
    """Routing-only summary: accuracy over judged probes, avg judge latency."""
    judged = [r for r in records if r["error"] is None]
    hits = [r for r in judged if r["hit"]]
    lat = [r["judge_latency_ms"] for r in judged]
    return {
        "tool": "eval.run_continuation_probes",
        "judge": "llama", "z_method": "B",
        "prev_tier_resolution": "notes task ref -> preceding main task gt_tier "
                                "(cross-checked: file order, notes stated tier, frozen map)",
        "n_probes": len(records),
        "n_judged": len(judged),
        "n_errors": len(records) - len(judged),
        "hits": len(hits),
        "accuracy": (len(hits) / len(judged)) if judged else None,
        "avg_judge_latency_ms": (sum(lat) / len(lat)) if lat else None,
        "total_judge_latency_ms": sum(lat) if lat else 0.0,
        "retried_probes": [r["probe_id"] for r in records if r["attempts"] > 1],
        "per_probe": {
            r["probe_id"]: {
                "previous_tier": r["previous_tier"],
                "expected_tier": r["expected_tier"],
                "chosen_tier": r["chosen_tier"],
                "hit": r["hit"],
                "judge_latency_ms": r["judge_latency_ms"],
                "error": r["error"],
            } for r in records
        },
    }


def write_outputs(records: list[dict], summary: dict, out_path: str | Path,
                  summary_out_path: str | Path | None) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.write(json.dumps({"_summary": summary}, ensure_ascii=False) + "\n")
    if summary_out_path:
        spath = Path(summary_out_path)
        spath.parent.mkdir(parents=True, exist_ok=True)
        with open(spath, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(
        prog="eval.run_continuation_probes",
        description="Judge-only continuation-probe routing eval (no CPA execution)")
    ap.add_argument("--tasks", default=DEFAULT_TASKS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--summary-out", default=DEFAULT_SUMMARY_OUT)
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--probe-ids", default=None,
                    help="comma-separated subset of probe ids (default: all continuation probes)")
    ap.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                    help="script-level judge retries per probe on network jitter")
    ap.add_argument("--retry-sleep", type=float, default=DEFAULT_RETRY_SLEEP_S)
    args = ap.parse_args(argv)

    from local_judge.client import JudgeClient  # lazy: stdlib-only module
    tasks = load_tasks(args.tasks)
    client = JudgeClient(endpoint=args.endpoint, z_method="B")
    if client.effective_z_method() != "B":
        raise SystemExit("judge client is not on Method B — refusing to run")

    probe_ids = ([p.strip() for p in args.probe_ids.split(",")]
                 if args.probe_ids else None)
    records, summary = run_probes(
        tasks=tasks, client=client, retries=args.retries,
        retry_sleep_s=args.retry_sleep, probe_ids=probe_ids)
    summary["endpoint"] = args.endpoint
    write_outputs(records, summary, args.out, args.summary_out)

    print(f"[continuation-probes] endpoint={args.endpoint} z_method=B "
          f"probes={summary['n_probes']} judged={summary['n_judged']} "
          f"errors={summary['n_errors']} retried={summary['retried_probes']}")
    for r in records:
        mark = ("HIT " if r["hit"] else "MISS") if r["error"] is None else "ERR "
        print(f"[continuation-probes] {r['probe_id']} prev={r['previous_tier']:<9s} "
              f"(after {r['preceding_task_id']}) expected={r['expected_tier']:<9s} "
              f"chosen={r['chosen_tier']} {mark} "
              f"latency_ms={r['judge_latency_ms']} attempts={r['attempts']}"
              + (f" error={r['error']}" if r["error"] else ""))
    print(f"[continuation-probes] accuracy={summary['accuracy']} "
          f"({summary['hits']}/{summary['n_judged']}) "
          f"avg_judge_latency_ms={summary['avg_judge_latency_ms']}")
    print(f"[continuation-probes] wrote {args.out} (summary line) and {args.summary_out}")
    return 1 if summary["n_errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
