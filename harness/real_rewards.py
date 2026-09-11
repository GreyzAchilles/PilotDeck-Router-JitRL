"""Real reward from evaluator verdicts + true token costs (Session 2).

Final-plan reward (project-decisions D3):
    r = 0.6 * quality + 0.3 * cost_saving + 0.1 * revision_penalty

Session 2 realization:
    quality          = (quality_score - 3) / 2          in [-1, 1]
                       (evaluator 1-5 scale, 3 = neutral baseline)
    cost_saving      = clamp((C_default - C_actual) / C_default, -1, 1)
                       C_actual   = real cost of the executed model (usage-based)
                       C_default  = the SAME usage priced at CPA/glm-5.3
                       (denominator model = D14 static default; identical
                       token counts make the two arms comparable)
    revision_penalty = 0.0   (no revision loop in the single-turn harness;
                       hook reserved — no hidden math, documented)

Failures (evaluator unavailable / execution failed) never reach this class:
the harness skips reward computation AND the C-memory write instead.
"""
from __future__ import annotations

from harness.pricing import DEFAULT_MODEL, cost_from_usage


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class RealQualityReward:
    """r = 0.6*quality + 0.3*cost_saving (+ 0.1*revision_penalty=0)."""

    def __init__(
        self,
        prices: dict[str, dict[str, float]] | None = None,
        default_model: str = DEFAULT_MODEL,
        w_quality: float = 0.6,
        w_cost: float = 0.3,
        w_revision: float = 0.1,
    ):
        self.prices = prices
        self.default_model = default_model
        self._wq = w_quality
        self._wc = w_cost
        self._wr = w_revision

    def quality_term(self, quality_score: int) -> float:
        return (float(quality_score) - 3.0) / 2.0

    def cost_saving(self, exec_model: str, usage: dict) -> float:
        actual = cost_from_usage(exec_model, usage, self.prices)
        default = cost_from_usage(self.default_model, usage, self.prices)
        if default <= 0.0:
            return 0.0
        return clamp((default - actual) / default, -1.0, 1.0)

    def reward(self, quality_score: int, exec_model: str, usage: dict) -> tuple[float, dict]:
        """Return (r, breakdown). All inputs come from a SUCCESSFUL episode."""
        q = self.quality_term(quality_score)
        c = self.cost_saving(exec_model, usage)
        rev = 0.0
        r = self._wq * q + self._wc * c + self._wr * rev
        return r, {
            "quality_term": round(q, 4),
            "cost_saving": round(c, 4),
            "revision_penalty": rev,
            "reward": round(r, 4),
        }
