"""JitRL engine: decide (retrieve + value estimate + logit modulation) & learn.

=== JudgeClient interface contract (MUST stay verbatim-identical with the
=== parallel local_judge deliverable — do not reformat) ===

```python
from typing import TypedDict

class JudgeResult(TypedDict):
    tier_logits: dict[str, float]   # 全部 4 档
    chosen_tier: str
    raw_output: str
    latency_ms: float

class JudgeClient(Protocol):
    def judge(self, user_message: str, previous_tier: str | None = None) -> JudgeResult: ...
```
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol, TypedDict

from jitrl_core.config import JitRLConfig, TIERS
from jitrl_core.credit import CreditAssigner, IdentityCreditAssigner
from jitrl_core.memory import ExperienceMemory
from jitrl_core.policy import choose_tier, clamp_logits, modulate_logits
from jitrl_core.state import intent_class, signature_token_set, task_signature
from jitrl_core.types import RouteTrajectory, TrajectoryEvaluation
from jitrl_core.updater import (
    DirectMemoryUpdater,
    MemoryUpdater,
    build_entry,
)
from jitrl_core.value import estimate_values


class JudgeResult(TypedDict):
    tier_logits: dict[str, float]   # 全部 4 档
    chosen_tier: str
    raw_output: str
    latency_ms: float


class JudgeClient(Protocol):
    def judge(self, user_message: str, previous_tier: str | None = None) -> JudgeResult: ...


#: gate_reason semantics (C1 min-neighbor gate, S2-C1-PREREGISTRATION v1.0):
#:   "no_neighbors"       - N(s) empty: pure baseline judge (C0 behavior)
#:   "below_min_neighbors"- 1 <= n_retrieved < min_neighbors: gated, z_mod=z_base
#:   None                 - modulation allowed (n_retrieved >= min_neighbors)
GATE_NO_NEIGHBORS = "no_neighbors"
GATE_BELOW_MIN_NEIGHBORS = "below_min_neighbors"


@dataclass
class DecisionResult:
    intent_class: str
    signature_tokens: list[str]          # sorted (canonical memory form)
    n_retrieved: int
    top_similarity: float
    V: float
    Q: dict[str, float]
    A_norm: dict[str, float]
    z_base: dict[str, float]
    z_mod: dict[str, float]
    chosen_tier: str
    base_choice: str
    flipped: bool
    min_neighbors: int                   # gate threshold in force for this decide
    modulation_active: bool              # True iff the C0 modulation path ran
    gate_reason: str | None              # None | "no_neighbors" | "below_min_neighbors"
    retrieved_episode_ids: list = field(default_factory=list)


class JitRLEngine:
    """Test-time RL (no gradient): memory of (state, tier, reward);
    retrieve similar experiences, estimate per-tier advantage, modulate
    judge logits z' = z + beta * A_norm, and write episodes back.

    C1 min-neighbor gate (S2-C1-PREREGISTRATION.md v1.0): if
    0 < n_retrieved < config.min_neighbors, modulation is suppressed
    (z_mod = z_base, A_norm = 0, chosen = base_choice). min_neighbors=1
    (default) reproduces C0 exactly. estimate_values is always called so
    the RNG stream is independent of the gate."""

    def __init__(
        self,
        config: JitRLConfig | None = None,
        memory: ExperienceMemory | None = None,
        *,
        credit_assigner: CreditAssigner | None = None,
        memory_updater: MemoryUpdater | None = None,
    ):
        self.config = config or JitRLConfig()
        # NOTE: explicit None check — ExperienceMemory defines __len__, so an
        # empty (but valid) injected memory is falsy and must not be replaced.
        self.memory = (memory if memory is not None
                       else ExperienceMemory(cap=self.config.memory_cap))
        # engine-owned seeded rng for the lambda exploration branch
        self._rng = random.Random(self.config.seed)
        # M1 trajectory loop (defaults reproduce legacy single-step behavior)
        self._credit_assigner = credit_assigner
        self._memory_updater = memory_updater

    # ------------------------------------------------------------------ decide
    def decide(self, user_message: str, tier_logits: dict[str, float]) -> DecisionResult:
        cfg = self.config
        intent = intent_class(user_message)
        tokens = task_signature(user_message)
        token_set = frozenset(tokens)

        matches = self.memory.retrieve(
            intent, token_set, k=cfg.k, threshold=cfg.jaccard_threshold
        )
        top_sim = matches[0][1] if matches else 0.0
        entries = [e for e, _ in matches]

        est = estimate_values(
            entries,
            rng=self._rng,
            lam=cfg.lam,
            alpha=cfg.alpha,
            tiers=TIERS,
        )

        z_base = clamp_logits(tier_logits, cfg.z_min, TIERS)
        # estimate_values above is ALWAYS called (even when the gate below
        # fires) so the engine RNG stream stays bit-identical to C0 for the
        # same learn/decide sequence, regardless of min_neighbors.
        if est.n_retrieved == 0:
            # empty N(s): V=0 and NO modulation (pure baseline judge)
            gate_reason = GATE_NO_NEIGHBORS
            z_mod = dict(z_base)
            a_norm = {t: 0.0 for t in TIERS}
        elif est.n_retrieved < cfg.min_neighbors:
            # C1 min-neighbor gate: insufficient memory evidence -> degrade
            # to the native Judge (z_mod = z_base, A_norm = 0).
            gate_reason = GATE_BELOW_MIN_NEIGHBORS
            z_mod = dict(z_base)
            a_norm = {t: 0.0 for t in TIERS}
        else:
            # C0 modulation path: z' = z + beta * A_norm
            gate_reason = None
            a_norm = est.A_norm
            z_mod = modulate_logits(z_base, a_norm, cfg.beta, cfg.z_min, TIERS)

        base_choice = choose_tier(z_base, TIERS)
        chosen = choose_tier(z_mod, TIERS)

        return DecisionResult(
            intent_class=intent,
            signature_tokens=sorted(tokens),
            n_retrieved=est.n_retrieved,
            top_similarity=top_sim,
            V=est.V,
            Q=dict(est.Q),
            A_norm=dict(a_norm),
            z_base=z_base,
            z_mod=z_mod,
            chosen_tier=chosen,
            base_choice=base_choice,
            flipped=chosen != base_choice,
            min_neighbors=cfg.min_neighbors,
            modulation_active=gate_reason is None,
            gate_reason=gate_reason,
            retrieved_episode_ids=[e.get("episode_id") for e in entries],
        )

    # ------------------------------------------------------------------- learn
    def learn(
        self,
        *,
        intent: str,
        signature_tokens: list[str],
        tier: str,
        reward: float,
        episode_id,
        ts: float | None = None,
    ) -> dict:
        """Episode ended: append {intent_class, signature_tokens(sorted), tier,
        G, ts, episode_id}; memory enforces the cap (drop oldest)."""
        entry = build_entry(
            intent=intent,
            signature_tokens=signature_tokens,
            tier=tier,
            reward=reward,
            episode_id=episode_id,
            ts=ts,
        )
        self.memory.add(entry)
        return entry

    def update_from_message(
        self,
        user_message: str,
        chosen_tier: str,
        reward: float,
        episode_id,
        ts: float | None = None,
    ) -> dict:
        return self.learn(
            intent=intent_class(user_message),
            signature_tokens=task_signature(user_message),
            tier=chosen_tier,
            reward=reward,
            episode_id=episode_id,
            ts=ts,
        )

    # Spec cross-references call this hook "engine.update()".
    def update(
        self,
        user_message: str,
        chosen_tier: str,
        reward: float,
        episode_id,
        ts: float | None = None,
    ) -> dict:
        return self.update_from_message(
            user_message, chosen_tier, reward, episode_id, ts=ts
        )

    # ------------------------------------------------------- learn_trajectory
    def learn_trajectory(
        self,
        trajectory: RouteTrajectory,
        evaluation: TrajectoryEvaluation,
        *,
        credit_assigner: CreditAssigner | None = None,
        updater: MemoryUpdater | None = None,
    ) -> list[dict]:
        """Trajectory-level learn (M1): credit assignment -> memory update.

        Single-step trajectory + identity credit + DirectMemoryUpdater
        reproduces learn() exactly: one frozen 6-key entry with G == r,
        appended to memory in step order. Consumes no RNG (decide()'s RNG
        stream is unaffected). A failed evaluation (ok=False) writes
        nothing — same policy as the legacy single-step path."""
        ca = credit_assigner or self._credit_assigner or IdentityCreditAssigner()
        up = updater or self._memory_updater or DirectMemoryUpdater()
        credits = ca.assign(trajectory, evaluation)
        if not credits:
            return []
        return up.update(self.memory, credits, trajectory, evaluation)
