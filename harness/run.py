"""A/B/C experiment CLI (spec section 5).

    python -m harness.run --mode {A,B,C} --tasks PATH --judge {mock,llama} \
        --out PATH --memory-out PATH --threshold FLOAT --seed INT --episodes INT

Modes:
  A  no judge; static route to the default model; chosen_tier recorded as "static"
  B  judge every task (previous_tier = previous task's CHOSEN tier);
     chosen = argmax(tier_logits); no memory, no modulation
  C  judge every task (same previous_tier passthrough); engine.decide()
     applies retrieval + modulation; chosen = argmax z';
     episode end -> stub reward (uses the row's gt_tier) -> engine.update()

--judge llama lazily imports local_judge.client.JudgeClient (clear error
"local_judge not available yet" if absent) and reads JITRL_JUDGE_ENDPOINT
(default http://127.0.0.1:18080).

--episodes N repeats the task list N times; mode C memory persists across
episodes. previous_tier chain also continues across episode boundaries
(documented choice: "previous task" = previous row in the flattened run).

Output: one DecisionRecord JSON line per (episode, task) -> --out, then a
final {"_summary": {...}} line; human-readable summary on stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from jitrl_core import JitRLConfig, JitRLEngine, MockJudgeClient
from jitrl_core.policy import choose_tier
from jitrl_core.rewards import StubRoutingReward
from jitrl_core.state import intent_class, task_signature
from harness import pricing
from harness.pricing import DEFAULT_MODEL, model_for_tier, price_ep

JUDGE_ENDPOINT_ENV = "JITRL_JUDGE_ENDPOINT"
JUDGE_ENDPOINT_DEFAULT = "http://127.0.0.1:18080"
SIGNATURE_PREVIEW_TOKENS = 12


def build_judge(kind: str, seed: int):
    if kind == "mock":
        return MockJudgeClient(seed=seed)
    if kind == "llama":
        try:
            from local_judge.client import JudgeClient as LlamaJudgeClient
        except ImportError as exc:
            raise RuntimeError("local_judge not available yet") from exc
        endpoint = os.environ.get(JUDGE_ENDPOINT_ENV, JUDGE_ENDPOINT_DEFAULT)
        return LlamaJudgeClient(endpoint=endpoint)
    raise ValueError(f"unknown judge kind: {kind}")


def load_tasks(path: str | Path) -> list[dict]:
    tasks: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            # Evaluation assets use the public schema field `id`; early harness
            # fixtures used `task_id`. Normalize both at the loader boundary.
            if "task_id" not in row and "id" in row:
                row = {**row, "task_id": row["id"]}
            if "task_id" not in row:
                continue
            tasks.append(row)
    return tasks


def make_record(
    *,
    episode_id: int,
    task: dict,
    intent: str,
    signature_tokens: list[str],
    n_retrieved: int,
    top_similarity: float,
    V: float,
    Q: dict,
    A_norm: dict,
    z_base: dict,
    z_mod: dict,
    chosen_tier: str,
    base_choice: str,
    flipped: bool,
    reward: float,
    cost: float,
    judge_latency_ms: float,
) -> dict:
    return {
        "episode_id": episode_id,
        "task_id": task["task_id"],
        "family": task.get("family"),
        "gt_tier": task.get("gt_tier"),
        "intent_class": intent,
        "signature_preview": signature_tokens[:SIGNATURE_PREVIEW_TOKENS],
        "n_retrieved": n_retrieved,
        "top_similarity": top_similarity,
        "V": V,
        "Q": Q,
        "A_norm": A_norm,
        "z_base": z_base,
        "z_mod": z_mod,
        "chosen_tier": chosen_tier,
        "base_choice": base_choice,
        "flipped": flipped,
        "reward": reward,
        "cost_per_episode": cost,
        "judge_latency_ms": judge_latency_ms,
    }


def summarize(records: list[dict], *, mode: str, judge_kind: str, episodes: int,
              memory_size: int) -> dict:
    labeled = [r for r in records if r.get("gt_tier")]
    accuracy = (
        sum(1 for r in labeled if r["chosen_tier"] == r["gt_tier"]) / len(labeled)
        if labeled else None
    )
    avg_cost = sum(r["cost_per_episode"] for r in records) / len(records) if records else 0.0
    flips = sum(1 for r in records if r["flipped"])

    fams: dict[str, list[dict]] = {}
    for r in labeled:
        fams.setdefault(r.get("family") or "unknown", []).append(r)
    family_accuracy = {
        f: sum(1 for r in rows if r["chosen_tier"] == r["gt_tier"]) / len(rows)
        for f, rows in sorted(fams.items())
    }

    cont = [r for r in labeled if r.get("family") == "continuation"]
    continuation_accuracy = (
        sum(1 for r in cont if r["chosen_tier"] == r["gt_tier"]) / len(cont)
        if cont else None
    )

    return {
        "mode": mode,
        "judge": judge_kind,
        "episodes": episodes,
        "n_records": len(records),
        "accuracy": accuracy,
        "avg_cost": avg_cost,
        "flips": flips,
        "memory_size": memory_size,
        "family_accuracy": family_accuracy,
        "continuation_accuracy": continuation_accuracy,
    }


def run(
    *,
    mode: str,
    tasks_path: str | Path,
    judge_kind: str = "mock",
    out_path: str | Path | None = None,
    memory_out_path: str | Path | None = None,
    threshold: float | None = None,
    seed: int | None = None,
    episodes: int = 1,
) -> dict:
    tasks = load_tasks(tasks_path)
    if not tasks:
        raise SystemExit(f"no tasks loaded from {tasks_path}")

    cfg_kwargs: dict = {"seed": seed if seed is not None else 42}
    if threshold is not None:
        cfg_kwargs["jaccard_threshold"] = threshold
    config = JitRLConfig(**cfg_kwargs)

    engine: JitRLEngine | None = None
    judge = None
    if mode == "C":
        engine = JitRLEngine(config=config)
    if mode in ("B", "C"):
        judge = build_judge(judge_kind, config.seed)

    reward_fn = StubRoutingReward(
        model_for_tier=model_for_tier,
        price_of_model=lambda m: price_ep(m),
        default_model=DEFAULT_MODEL,
    )

    records: list[dict] = []
    previous_tier: str | None = None

    for episode_id in range(1, episodes + 1):
        for task in tasks:
            message = task["message"]
            intent = intent_class(message)
            signature_tokens = sorted(task_signature(message))

            if mode == "A":
                chosen = "static"
                z_base: dict = {}
                z_mod: dict = {}
                n_retrieved, top_sim, V = 0, 0.0, 0.0
                Q = {t: 0.0 for t in ("simple", "medium", "complex", "reasoning")}
                A_norm = dict(Q)
                base_choice, flipped = chosen, False
                latency = 0.0
            else:
                jr = judge.judge(message, previous_tier=previous_tier)
                logits = jr["tier_logits"]
                latency = float(jr["latency_ms"])
                if mode == "B":
                    chosen = choose_tier(logits)
                    z_base = {t: max(float(logits.get(t, 0.0)), config.z_min) for t in
                              ("simple", "medium", "complex", "reasoning")}
                    z_mod = dict(z_base)
                    n_retrieved, top_sim, V = 0, 0.0, 0.0
                    Q = {t: 0.0 for t in z_base}
                    A_norm = dict(Q)
                    base_choice, flipped = chosen, False
                else:  # C
                    decision = engine.decide(message, logits)
                    chosen = decision.chosen_tier
                    z_base = decision.z_base
                    z_mod = decision.z_mod
                    n_retrieved = decision.n_retrieved
                    top_sim = decision.top_similarity
                    V = decision.V
                    Q = decision.Q
                    A_norm = decision.A_norm
                    base_choice = decision.base_choice
                    flipped = decision.flipped

            # routing-only STUB reward; chosen "static" (mode A) never matches a
            # gt tier by construction -> quality term is -0.5 (comparability).
            reward = reward_fn.reward(task, chosen)
            cost = price_ep(model_for_tier(chosen))

            if mode == "C":
                engine.learn(
                    intent=intent,
                    signature_tokens=signature_tokens,
                    tier=chosen,
                    reward=reward,
                    episode_id=episode_id,
                )

            records.append(make_record(
                episode_id=episode_id, task=task, intent=intent,
                signature_tokens=signature_tokens,
                n_retrieved=n_retrieved, top_similarity=top_sim,
                V=V, Q=Q, A_norm=A_norm, z_base=z_base, z_mod=z_mod,
                chosen_tier=chosen, base_choice=base_choice, flipped=flipped,
                reward=reward, cost=cost, judge_latency_ms=latency,
            ))
            previous_tier = chosen

    memory_size = len(engine.memory) if engine is not None else 0
    summary = summarize(records, mode=mode, judge_kind=judge_kind,
                        episodes=episodes, memory_size=memory_size)

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
        f"[harness] mode={mode} judge={judge_kind} episodes={episodes} "
        f"tasks={len(tasks)} records={len(records)}"
    )
    print(f"[harness] accuracy(vs GT)={summary['accuracy']}")
    print(f"[harness] avg_cost=${summary['avg_cost']:.6f} per episode-task")
    print(f"[harness] flips={summary['flips']} memory_size={summary['memory_size']}")
    print(f"[harness] family_accuracy={json.dumps(summary['family_accuracy'], ensure_ascii=False)}")
    print(f"[harness] continuation_accuracy={summary['continuation_accuracy']}")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="harness.run")
    ap.add_argument("--mode", required=True, choices=["A", "B", "C"])
    ap.add_argument("--tasks", default="harness/testdata/mini_tasks.jsonl")
    ap.add_argument("--judge", default="mock", choices=["mock", "llama"])
    ap.add_argument("--out", default=None, help="DecisionRecord JSONL output path")
    ap.add_argument("--memory-out", default=None, help="memory entries JSONL output path")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override jaccard_threshold (default 0.5)")
    ap.add_argument("--seed", type=int, default=None, help="override seed (default 42)")
    ap.add_argument("--episodes", type=int, default=1,
                    help="repeat the task list N times (C memory persists)")
    args = ap.parse_args(argv)

    run(
        mode=args.mode,
        tasks_path=args.tasks,
        judge_kind=args.judge,
        out_path=args.out,
        memory_out_path=args.memory_out,
        threshold=args.threshold,
        seed=args.seed,
        episodes=args.episodes,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
