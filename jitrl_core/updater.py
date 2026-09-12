"""Memory updaters: turn step credits into memory writes (M1).

- build_entry(): the ONE constructor of the frozen 6-key entry schema;
  engine.learn() and every updater share it so entries are identical.
- DirectMemoryUpdater (default): plain memory.add per credit — on a
  length-1 trajectory with identity credit this reproduces the legacy
  learn() write exactly (same entry, same ordering, no RNG).
- LifecycleMemoryUpdater: routes entries through active/provisional/
  quarantine via LifecycleMemory.add_with_status when the memory supports
  it (thresholds: confidence >= 0.80 active, >= 0.60 provisional, else
  quarantine; None confidence -> active for compatibility). Falls back
  to plain add() on a plain ExperienceMemory (duck-typed swap safety,
  e.g. run_ablation.py replaces engine.memory).
"""
from __future__ import annotations

import time
from typing import Protocol

from jitrl_core.credit import StepCredit
from jitrl_core.memory import ENTRY_ACTIVE, ENTRY_PROVISIONAL, ENTRY_QUARANTINE
from jitrl_core.types import RouteTrajectory, TrajectoryEvaluation


def build_entry(*, intent: str, signature_tokens: list[str], tier: str,
                reward: float, episode_id, ts: float | None = None) -> dict:
    """Frozen 6-key entry: intent_class, signature_tokens(sorted), tier,
    G, ts, episode_id (memory.py schema contract)."""
    return {
        "intent_class": intent,
        "signature_tokens": sorted(signature_tokens),
        "tier": tier,
        "G": float(reward),
        "ts": float(ts) if ts is not None else time.time(),
        "episode_id": episode_id,
    }


class MemoryUpdater(Protocol):
    def update(self, memory, credits: list[StepCredit],
               trajectory: RouteTrajectory,
               evaluation: TrajectoryEvaluation | None = None) -> list[dict]: ...


class DirectMemoryUpdater:
    """Append one frozen-schema entry per credit (legacy behavior)."""

    def update(self, memory, credits: list[StepCredit],
               trajectory: RouteTrajectory,
               evaluation: TrajectoryEvaluation | None = None) -> list[dict]:
        entries: list[dict] = []
        for c in credits:
            step = trajectory.step_by_id(c.step_id)
            if step is None:
                continue
            entry = build_entry(
                intent=step.intent_class,
                signature_tokens=step.signature_tokens,
                tier=step.chosen_tier,
                reward=c.G,
                episode_id=trajectory.episode_id,
                ts=step.ts,
            )
            memory.add(entry)
            entries.append(entry)
        return entries


class LifecycleMemoryUpdater(DirectMemoryUpdater):
    """Status-aware write: active/provisional/quarantine by evaluator
    confidence; metadata (verdict/recommended tier/failure tags/feedback)
    goes to the LifecycleMemory sidecar, never into the entry dict."""

    def __init__(self, active_confidence: float = 0.80,
                 provisional_confidence: float = 0.60):
        if not (0.0 <= provisional_confidence <= active_confidence <= 1.0):
            raise ValueError("require 0 <= provisional <= active <= 1")
        self.active_confidence = float(active_confidence)
        self.provisional_confidence = float(provisional_confidence)

    def _status_for(self, ev) -> str:
        if ev is None or ev.confidence is None:
            return ENTRY_ACTIVE
        if ev.confidence >= self.active_confidence:
            return ENTRY_ACTIVE
        if ev.confidence >= self.provisional_confidence:
            return ENTRY_PROVISIONAL
        return ENTRY_QUARANTINE

    def _meta_for(self, ev) -> dict:
        if ev is None:
            return {}
        return {
            "routing_verdict": ev.routing_verdict,
            "recommended_tier": ev.recommended_tier,
            "failure_tags": list(ev.failure_tags),
            "feedback": ev.feedback,
            "confidence": ev.confidence,
        }

    def update(self, memory, credits: list[StepCredit],
               trajectory: RouteTrajectory,
               evaluation: TrajectoryEvaluation | None = None) -> list[dict]:
        entries: list[dict] = []
        for c in credits:
            step = trajectory.step_by_id(c.step_id)
            if step is None:
                continue
            ev = evaluation.step_by_id(c.step_id) if evaluation else None
            entry = build_entry(
                intent=step.intent_class,
                signature_tokens=step.signature_tokens,
                tier=step.chosen_tier,
                reward=c.G,
                episode_id=trajectory.episode_id,
                ts=step.ts,
            )
            if hasattr(memory, "add_with_status"):
                memory.add_with_status(entry, self._status_for(ev),
                                       self._meta_for(ev))
            else:
                memory.add(entry)   # degraded: memory without lifecycle
            entries.append(entry)
        return entries
