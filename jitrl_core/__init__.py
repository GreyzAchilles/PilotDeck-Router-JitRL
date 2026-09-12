"""jitrl_core: gradient-free test-time RL routing engine (Session1-T4/T5).

M1 (docs/optimization/): trajectory-level memory-update loop — types,
credit assignment, memory updaters, lifecycle memory."""
from jitrl_core.config import JitRLConfig, TIERS
from jitrl_core.credit import (
    CreditAssigner,
    DiscountedCreditAssigner,
    IdentityCreditAssigner,
    StepCredit,
)
from jitrl_core.engine import (
    DecisionResult,
    JudgeClient,
    JudgeResult,
    JitRLEngine,
)
from jitrl_core.memory import (
    ENTRY_ACTIVE,
    ENTRY_PROVISIONAL,
    ENTRY_QUARANTINE,
    ENTRY_STATUSES,
    ExperienceMemory,
    LifecycleMemory,
)
from jitrl_core.mock_judge import MockJudgeClient
from jitrl_core.policy import choose_tier, modulate_logits
from jitrl_core.rewards import RewardFn, StubRoutingReward
from jitrl_core.state import intent_class, jaccard, task_signature
from jitrl_core.types import (
    DecisionStep,
    RouteTrajectory,
    StepEvaluation,
    TrajectoryEvaluation,
)
from jitrl_core.updater import (
    DirectMemoryUpdater,
    LifecycleMemoryUpdater,
    MemoryUpdater,
    build_entry,
)
from jitrl_core.value import estimate_values

__all__ = [
    "JitRLConfig", "TIERS",
    "JudgeClient", "JudgeResult", "JitRLEngine", "DecisionResult",
    "ExperienceMemory", "MockJudgeClient",
    "choose_tier", "modulate_logits",
    "RewardFn", "StubRoutingReward",
    "intent_class", "task_signature", "jaccard", "estimate_values",
    # M1 trajectory loop
    "DecisionStep", "RouteTrajectory", "StepEvaluation", "TrajectoryEvaluation",
    "CreditAssigner", "IdentityCreditAssigner", "DiscountedCreditAssigner",
    "StepCredit",
    "MemoryUpdater", "DirectMemoryUpdater", "LifecycleMemoryUpdater",
    "build_entry",
    "LifecycleMemory", "ENTRY_ACTIVE", "ENTRY_PROVISIONAL",
    "ENTRY_QUARANTINE", "ENTRY_STATUSES",
]
