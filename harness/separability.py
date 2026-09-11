"""State separability check CLI (spec section 6).

    python -m harness.separability --tasks PATH

Pairwise Jaccard between task signatures: within-family vs cross-family
mean/min/max, a decile histogram over ALL pairs, and a verdict:
  - "bimodal"    when within_mean - cross_mean >= MARGIN (families separate)
  - "degenerate" otherwise (or when within <= cross)
Prints a JSON report to stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

from jitrl_core.state import jaccard, signature_token_set

MARGIN = 0.1
DECILES = 10


def load_tasks(path: str | Path) -> list[dict]:
    tasks: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            row = json.loads(line)
            if row.get("_summary") is not None:
                continue
            tasks.append(row)
    return tasks


def separability_report(tasks: list[dict]) -> dict:
    fams = [t.get("family", "unknown") for t in tasks]
    sets = [signature_token_set(t["message"]) for t in tasks]

    within: list[float] = []
    cross: list[float] = []
    for (i, si), (j, sj) in combinations(enumerate(sets), 2):
        sim = jaccard(si, sj)
        if fams[i] == fams[j]:
            within.append(sim)
        else:
            cross.append(sim)

    all_pairs = within + cross

    def stats(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0, "mean": None, "min": None, "max": None}
        return {
            "n": len(vals),
            "mean": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
        }

    hist = [0] * DECILES
    for s in all_pairs:
        b = min(int(s * DECILES), DECILES - 1)
        hist[b] += 1

    ws, cs = stats(within), stats(cross)
    if ws["mean"] is None or cs["mean"] is None:
        verdict = "degenerate"
    elif ws["mean"] > cs["mean"] and (ws["mean"] - cs["mean"]) >= MARGIN:
        verdict = "bimodal"
    else:
        verdict = "degenerate"

    return {
        "n_tasks": len(tasks),
        "within_family": ws,
        "cross_family": cs,
        "margin": MARGIN,
        "decile_histogram": {
            f"[{b / DECILES:.1f},{(b + 1) / DECILES:.1f})": hist[b]
            for b in range(DECILES)
        },
        "verdict": verdict,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="harness.separability")
    ap.add_argument("--tasks", required=True, help="JSONL task file")
    args = ap.parse_args(argv)

    tasks = load_tasks(args.tasks)
    report = separability_report(tasks)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
