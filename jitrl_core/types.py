"""Trajectory-level data contracts for the JitRL memory-update loop (M1).

Design (docs/optimization/MEMORY-LOOP-DECISIONS.md v1.0):
- A learnable step is ONE real routing decision (tier choice), not a token,
  tool call, or model-internal reasoning step.
- A single-turn episode is a trajectory of length 1; under identity credit
  assignment G_1 == r_1, reproducing the legacy engine.learn() write
  exactly (same frozen 6-key memory entry).
- Lifecycle status lives in LifecycleMemory sidecars, never inside the
  entry dicts, so the frozen entry schema is untouched.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

TRAJECTORY_OPEN = "open"
TRAJECTORY_FINALIZED = "finalized"
TRAJECTORY_ABANDONED = "abandoned"


@dataclass
class DecisionStep:
    """One real routing decision, engine-facing (minimal contract).

    The harness keeps its own richer step record (exec result, latencies,
    judge logits); this is the subset the credit assigner and memory
    updater consume."""

    step_id: int
    user_message: str
    intent_class: str
    signature_tokens: list[str]          # canonical (sorted) memory form
    chosen_tier: str
    previous_tier: str | None = None
    reward: float | None = None          # filled by evaluation (terminal-only: last step)
    ts: float = field(default_factory=time.time)


@dataclass
class RouteTrajectory:
    trajectory_id: str
    steps: list[DecisionStep] = field(default_factory=list)
    episode_id: int | None = None
    status: str = TRAJECTORY_OPEN

    def add_step(self, step: DecisionStep) -> DecisionStep:
        if self.status != TRAJECTORY_OPEN:
            raise ValueError(f"cannot add step to {self.status} trajectory")
        self.steps.append(step)
        return step

    def step_by_id(self, step_id: int) -> DecisionStep | None:
        for s in self.steps:
            if s.step_id == step_id:
                return s
        return None

    def finalize(self) -> None:
        self.status = TRAJECTORY_FINALIZED

    def abandon(self) -> None:
        self.status = TRAJECTORY_ABANDONED


@dataclass
class StepEvaluation:
    """Evaluator verdict for one routing step.

    reward/confidence semantics: the evaluator makes semantic judgments;
    deterministic code computes numeric rewards. `reward is None` marks a
    step the evaluation did not score (terminal-only evaluation scores
    only the last step); such steps get no memory entry in M1."""

    step_id: int
    ok: bool
    reward: float | None = None
    confidence: float | None = None      # evaluator certainty 0.0-1.0
    routing_verdict: str | None = None   # appropriate|under_routed|over_routed (M2)
    recommended_tier: str | None = None  # (M2)
    failure_tags: list[str] = field(default_factory=list)
    feedback: str = ""
    error: str | None = None


@dataclass
class TrajectoryEvaluation:
    steps: list[StepEvaluation] = field(default_factory=list)
    ok: bool = True                      # False -> no credits, no memory write
    episode_feedback: str = ""

    def step_by_id(self, step_id: int) -> StepEvaluation | None:
        for s in self.steps:
            if s.step_id == step_id:
                return s
        return None
