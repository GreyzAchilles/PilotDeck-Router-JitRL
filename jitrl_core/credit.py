"""Credit assignment over routing trajectories (M1).

- IdentityCreditAssigner: G_t = r_t. On a length-1 trajectory this is
  bit-identical to the legacy `G = r` semantics (single-step episodes).
- DiscountedCreditAssigner: G_t = r_t + eta * sum_{k>t} gamma^(k-t) r_k.
  The delayed weight eta < 1 keeps a later successful rescue from fully
  washing out an earlier mis-route (docs/optimization/
  MEMORY-LOOP-DECISIONS.md section 3, gamma=0.8 / eta=0.5 baseline).

Pure functions: no RNG, no memory access. Steps whose evaluation carries
`reward is None` (not scored) receive no credit and contribute 0 to the
delayed sums; discount exponents use positional distance in the step
list so unevaluated steps still advance time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from jitrl_core.types import RouteTrajectory, TrajectoryEvaluation


@dataclass
class StepCredit:
    step_id: int
    G: float


class CreditAssigner(Protocol):
    def assign(self, trajectory: RouteTrajectory,
               evaluation: TrajectoryEvaluation) -> list[StepCredit]: ...


class IdentityCreditAssigner:
    """G_t = r_t (single-step legacy semantics)."""

    def assign(self, trajectory: RouteTrajectory,
               evaluation: TrajectoryEvaluation) -> list[StepCredit]:
        if not evaluation.ok:
            return []
        return [StepCredit(step_id=ev.step_id, G=float(ev.reward))
                for ev in evaluation.steps if ev.reward is not None]


class DiscountedCreditAssigner:
    """G_t = r_t + eta * sum_{k>t} gamma^(k-t) r_k over scored steps."""

    def __init__(self, gamma: float = 0.8, delayed_weight: float = 0.5):
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("gamma must be in [0, 1]")
        if not (0.0 <= delayed_weight <= 1.0):
            raise ValueError("delayed_weight must be in [0, 1]")
        self.gamma = float(gamma)
        self.delayed_weight = float(delayed_weight)

    def assign(self, trajectory: RouteTrajectory,
               evaluation: TrajectoryEvaluation) -> list[StepCredit]:
        if not evaluation.ok:
            return []
        rewards = [(ev.step_id, ev.reward) for ev in evaluation.steps]
        credits: list[StepCredit] = []
        for i, (sid, r) in enumerate(rewards):
            if r is None:
                continue
            g = float(r)
            for j in range(i + 1, len(rewards)):
                rj = rewards[j][1]
                if rj is None:
                    continue
                g += self.delayed_weight * (self.gamma ** (j - i)) * float(rj)
            credits.append(StepCredit(step_id=sid, G=g))
        return credits
