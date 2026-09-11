"""Policy: closed-form logit modulation (spec section 2, verbatim).

    z'(tier) = z(tier) + beta * A_norm(tier);  chosen = argmax z'

z_min is the documented lower clamp for logits (both base and modulated).
argmax ties resolve to the FIRST tier in canonical TIERS order.
"""
from __future__ import annotations

from jitrl_core.config import TIERS


def clamp_logits(z: dict[str, float], z_min: float, tiers: tuple[str, ...] = TIERS) -> dict[str, float]:
    return {t: max(float(z.get(t, 0.0)), z_min) for t in tiers}


def modulate_logits(
    z_base: dict[str, float],
    a_norm: dict[str, float],
    beta: float = 5.0,
    z_min: float = -10.0,
    tiers: tuple[str, ...] = TIERS,
) -> dict[str, float]:
    z_mod = {
        t: float(z_base.get(t, 0.0)) + beta * float(a_norm.get(t, 0.0))
        for t in tiers
    }
    return {t: max(v, z_min) for t, v in z_mod.items()}


def choose_tier(z: dict[str, float], tiers: tuple[str, ...] = TIERS) -> str:
    """argmax with deterministic tie-break: first max in canonical TIERS order."""
    best = tiers[0]
    best_v = float(z.get(best, 0.0))
    for t in tiers[1:]:
        v = float(z.get(t, 0.0))
        if v > best_v:
            best, best_v = t, v
    return best
