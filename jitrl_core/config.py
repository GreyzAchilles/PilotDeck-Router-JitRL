"""JitRL core: hyperparameters and canonical tier order.

Spec (Session1-T4/T5): k=10, beta=5.0, lam=0.05, alpha=5.0,
jaccard_threshold=0.5, z_min=-10.0, memory_cap=5000, seed=42.

C1 (S2-C1-PREREGISTRATION.md v1.0) adds min_neighbors (default 1 == C0
semantics; the C1 run passes 3 explicitly).
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: Canonical tier order. Also used as deterministic tie-break order for
#: argmax (first max in this sequence wins).
TIERS: tuple[str, ...] = ("simple", "medium", "complex", "reasoning")


@dataclass
class JitRLConfig:
    """Hyperparameters of the JitRL engine (spec section 2, verbatim)."""

    k: int = 10                      # top-k retrieved experiences
    beta: float = 5.0                # logit modulation strength
    lam: float = 0.05                # exploration probability for unseen tiers
    alpha: float = 5.0               # optimism bonus: Q = V + alpha/|N(s)|
    jaccard_threshold: float = 0.5   # minimum signature similarity to retrieve
    z_min: float = -10.0             # lower clamp for base/modulated logits
    memory_cap: int = 5000           # drop-oldest capacity
    seed: int = 42                   # seed for engine-side lambda draws
    # C1 min-neighbor gate (S2-C1-PREREGISTRATION.md v1.0): minimum number of
    # retrieved neighbors required before logit modulation is allowed.
    # Default 1 preserves C0 semantics exactly (>=1 neighbor => modulate);
    # the C1 run passes --min-neighbors 3 explicitly.
    min_neighbors: int = 1

    def snapshot(self) -> dict:
        return dict(self.__dict__)
