"""Value estimation (spec section 2 — implemented numerically verbatim).

Given N(s) = retrieved entries (each has "G" and "tier"):
    V = mean(G) over N(s);  empty N(s) -> V = 0 and NO modulation
    per tier t:
        |N(s,t)| > 0 -> Q(t) = mean(G over N(s,t))
        else         -> with prob lam: Q(t) = V + alpha/|N(s)| ; else Q(t) = 0
    A(t) = Q(t) - V
    A_norm(t) = A(t) / (max|A| + 1e-8);  max|A| == 0 -> A_norm(t) = 0

The `rng` is engine-owned and seeded (config.seed); inject a stub in tests.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from jitrl_core.config import TIERS


@dataclass
class ValueEstimate:
    V: float = 0.0
    Q: dict[str, float] = field(default_factory=dict)
    A: dict[str, float] = field(default_factory=dict)
    A_norm: dict[str, float] = field(default_factory=dict)
    n_retrieved: int = 0


def estimate_values(
    entries: list[dict],
    rng: random.Random | None = None,
    lam: float = 0.05,
    alpha: float = 5.0,
    tiers: tuple[str, ...] = TIERS,
) -> ValueEstimate:
    n = len(entries)
    if n == 0:
        zeros = {t: 0.0 for t in tiers}
        return ValueEstimate(V=0.0, Q=zeros, A=zeros, A_norm=zeros, n_retrieved=0)

    V = sum(float(e["G"]) for e in entries) / n

    Q: dict[str, float] = {}
    for t in tiers:
        gs = [float(e["G"]) for e in entries if e.get("tier") == t]
        if gs:
            Q[t] = sum(gs) / len(gs)
        else:
            explore = rng.random() < lam if rng is not None else False
            Q[t] = V + alpha / n if explore else 0.0

    A = {t: Q[t] - V for t in tiers}
    max_abs = max(abs(a) for a in A.values())
    if max_abs == 0.0:
        A_norm = {t: 0.0 for t in tiers}
    else:
        A_norm = {t: A[t] / (max_abs + 1e-8) for t in tiers}

    return ValueEstimate(V=V, Q=Q, A=A, A_norm=A_norm, n_retrieved=n)
