"""Trajectory-level evaluation + memory update for the real harness (M1).

Unified interface (docs/optimization/MEMORY-LOOP-DECISIONS.md v1.0):
every episode — single-turn now, multi-turn in M2 — becomes a
Trajectory of TrajectorySteps. After execution:

    TrajectoryEvaluator.evaluate(traj) -> TrajectoryReport
    update_memory_from_trajectory(engine, traj, report) -> entries

M1 semantics (single-turn, length-1 trajectories):
- terminal-only evaluation: the existing blind Evaluator is called ONCE
  per episode on the final response — same call, same prompt, same
  blindness as the legacy inline path;
- reward math unchanged: RealQualityReward(quality, exec_model, usage);
- failure policy unchanged: exec failure or evaluator failure -> report
  not ok -> NO memory write (directive);
- single-step + identity credit -> G == r, and the written entry is the
  frozen 6-key schema, identical to the legacy engine.learn() call.

M2 extension point: per-step StepEvaluation fields (routing_verdict,
recommended_tier, failure_tags, feedback, confidence) already flow from
the core TrajectoryEvaluation into LifecycleMemoryUpdater metadata; the
harness fills reward/confidence today and the richer evaluator later.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from jitrl_core.credit import CreditAssigner
from jitrl_core.types import DecisionStep as CoreDecisionStep
from jitrl_core.types import RouteTrajectory as CoreRouteTrajectory
from jitrl_core.types import StepEvaluation as CoreStepEvaluation
from jitrl_core.types import TrajectoryEvaluation as CoreTrajectoryEvaluation
from jitrl_core.updater import MemoryUpdater

if TYPE_CHECKING:  # annotations only — no runtime import chain
    from harness.cpa_client import CompletionResult
    from harness.evaluator import StepVerdict, TrajectoryLevelResult
    from jitrl_core.engine import DecisionResult

EVAL_FAIL_MSG = "evaluator failed (transport/parse/validate)"


@dataclass
class TrajectoryStep:
    """Harness-level record of one routing step (richer than the engine's
    DecisionStep: carries execution, costs, and latencies for records)."""

    task: dict
    message: str
    intent: str
    signature_tokens: list[str]
    chosen_tier: str
    route_mode: str
    exec_model: str
    previous_tier: str | None
    decision: Optional["DecisionResult"] = None      # mode C only
    exec_result: Optional["CompletionResult"] = None
    exec_error: str | None = None
    exec_cost_usd: float | None = None
    judge_latency_ms: float = 0.0
    jitrl_latency_ms: float = 0.0
    ts: float = field(default_factory=time.time)


@dataclass
class Trajectory:
    traj_id: str
    episode_id: int
    steps: list[TrajectoryStep] = field(default_factory=list)


@dataclass
class StepReport:
    step_index: int
    eval_result=None
    eval_error: str | None = None
    reward: float | None = None
    breakdown: dict | None = None
    # M2 per-step review fields (terminal mode leaves them None)
    step_verdict: Optional["StepVerdict"] = None
    trajectory_result: Optional["TrajectoryLevelResult"] = None

    def eval_record(self) -> dict | None:
        """Per-turn `evaluator` payload for records. Terminal mode -> None
        (the record uses eval_result.to_record()); per-step mode -> the
        step verdict merged with the episode summary. quality_score carries
        the per-turn local quality so downstream aggregates stay on the
        same 1-5 scale."""
        if self.step_verdict is None:
            return None
        rec = self.step_verdict.to_record()
        if self.trajectory_result is not None:
            rec["trajectory_success"] = self.trajectory_result.episode_success
            rec["trajectory_quality"] = self.trajectory_result.episode_quality
            rec["trajectory_summary"] = self.trajectory_result.episode_summary
        return rec


@dataclass
class TrajectoryReport:
    ok: bool
    steps: list[StepReport] = field(default_factory=list)
    eval_result=None                     # terminal evaluation (legacy handle)
    eval_error: str | None = None
    reward: float | None = None
    breakdown: dict | None = None


class TrajectoryEvaluator:
    """Trajectory evaluation.

    mode="terminal" (M1 / arm T1): ONE blind per-response evaluation of
    the final turn — identical call, prompt, and reward math to the
    legacy inline chain; non-terminal turns get no reward (and thus no
    memory entry).

    mode="per_step" (M2 / arms T2-T3): ONE TrajectoryLevelEvaluator call
    per trajectory (episode verdict + per-turn routing verdicts);
    per-turn rewards are computed deterministically with the SAME D3
    formula (0.6*quality + 0.3*cost_saving) from the evaluator's
    local_quality and that turn's real exec model + usage. The
    evaluator's semantic output (verdict/recommended tier/failure tags/
    feedback/certainty) flows into StepReport for records and memory
    sidecars; numeric rewards stay in deterministic code."""

    def __init__(self, evaluator, reward_fn, *, mode: str = "terminal",
                 traj_evaluator=None):
        if mode not in ("terminal", "per_step"):
            raise ValueError(f"unknown trajectory eval mode: {mode!r}")
        if mode == "per_step" and traj_evaluator is None:
            raise ValueError("per_step mode requires a traj_evaluator")
        self.evaluator = evaluator
        self.reward_fn = reward_fn
        self.mode = mode
        self.traj_evaluator = traj_evaluator

    def evaluate(self, traj: Trajectory) -> TrajectoryReport:
        report = TrajectoryReport(ok=bool(traj.steps),
                                  steps=[StepReport(step_index=i)
                                         for i in range(len(traj.steps))])
        for s in traj.steps:
            if s.exec_result is None or s.exec_error is not None:
                report.ok = False
        if not report.ok:
            return report
        if self.mode == "per_step":
            return self._evaluate_per_step(traj, report)
        return self._evaluate_terminal(traj, report)

    # ------------------------------------------------------------- terminal
    def _evaluate_terminal(self, traj: Trajectory,
                           report: TrajectoryReport) -> TrajectoryReport:
        last = traj.steps[-1]
        i = len(traj.steps) - 1
        eval_result = self.evaluator.evaluate(last.task, last.exec_result.content)
        if eval_result is None:
            report.eval_error = EVAL_FAIL_MSG
            report.steps[i].eval_error = EVAL_FAIL_MSG
            report.ok = False
            return report

        reward, breakdown = self.reward_fn.reward(
            eval_result.quality_score, last.exec_model, last.exec_result.usage)
        report.eval_result = eval_result
        report.steps[i].eval_result = eval_result
        report.steps[i].reward = reward
        report.steps[i].breakdown = breakdown
        report.reward = reward
        report.breakdown = breakdown
        return report

    # ------------------------------------------------------------- per step
    def _evaluate_per_step(self, traj: Trajectory,
                           report: TrajectoryReport) -> TrajectoryReport:
        turns = [{"message": s.message, "tier": s.chosen_tier,
                  "response": (s.exec_result.content
                               if s.exec_result is not None else "")}
                 for s in traj.steps]
        res = self.traj_evaluator.evaluate_trajectory(traj.steps[0].task, turns)
        if res is None:
            report.eval_error = EVAL_FAIL_MSG
            for sr in report.steps:
                sr.eval_error = EVAL_FAIL_MSG
            report.ok = False
            return report
        for i, sv in enumerate(res.steps):
            s = traj.steps[i]
            sr = report.steps[i]
            reward, breakdown = self.reward_fn.reward(
                sv.local_quality, s.exec_model, s.exec_result.usage)
            sr.reward = reward
            sr.breakdown = breakdown
            sr.step_verdict = sv
            sr.trajectory_result = res
        report.reward = report.steps[-1].reward
        report.breakdown = report.steps[-1].breakdown
        return report


def update_memory_from_trajectory(
    engine,
    traj: Trajectory,
    report: TrajectoryReport,
    *,
    credit_assigner: CreditAssigner | None = None,
    updater: MemoryUpdater | None = None,
) -> list[dict]:
    """Write trajectory experience into engine memory via the unified M1
    path (credit assignment -> memory update). No-op unless report.ok."""
    if not report.ok:
        return []
    if not hasattr(engine, "learn_trajectory"):
        # Duck-typed engine wrapper without the M1 API (e.g. latency-test
        # SlowDecideEngine): legacy learn() per scored step — identity
        # credit, identical single-step semantics.
        entries = []
        for i, s in enumerate(traj.steps):
            sr = report.steps[i]
            if sr.reward is None:
                continue
            entries.append(engine.learn(
                intent=s.intent,
                signature_tokens=s.signature_tokens,
                tier=s.chosen_tier,
                reward=sr.reward,
                episode_id=traj.episode_id,
                ts=s.ts,
            ))
        return entries
    core_traj = CoreRouteTrajectory(trajectory_id=traj.traj_id,
                                    episode_id=traj.episode_id)
    core_eval = CoreTrajectoryEvaluation(ok=True)
    for i, s in enumerate(traj.steps):
        sr = report.steps[i]
        core_traj.add_step(CoreDecisionStep(
            step_id=i,
            user_message=s.message,
            intent_class=s.intent,
            signature_tokens=s.signature_tokens,
            chosen_tier=s.chosen_tier,
            previous_tier=s.previous_tier,
            reward=sr.reward,
            ts=s.ts,
        ))
        core_eval.steps.append(CoreStepEvaluation(
            step_id=i,
            ok=True,
            reward=sr.reward,
            confidence=(sr.eval_result.certainty
                        if sr.eval_result is not None
                        else (sr.step_verdict.certainty
                              if sr.step_verdict is not None else None)),
            routing_verdict=(sr.step_verdict.routing_verdict
                             if sr.step_verdict is not None else None),
            recommended_tier=(sr.step_verdict.recommended_tier
                              if sr.step_verdict is not None else None),
            failure_tags=(list(sr.step_verdict.failure_tags)
                          if sr.step_verdict is not None else []),
            feedback=(sr.step_verdict.feedback
                      if sr.step_verdict is not None else ""),
        ))
    core_traj.finalize()
    return engine.learn_trajectory(core_traj, core_eval,
                                   credit_assigner=credit_assigner,
                                   updater=updater)
