"""jitrl_core: gradient-free test-time RL routing engine (Session1-T4/T5)."""
from jitrl_core.config import JitRLConfig, TIERS
from jitrl_core.engine import (
    DecisionResult,
    JudgeClient,
    JudgeResult,
    JitRLEngine,
)
from jitrl_core.memory import ExperienceMemory
from jitrl_core.mock_judge import MockJudgeClient
from jitrl_core.policy import choose_tier, modulate_logits
from jitrl_core.rewards import RewardFn, StubRoutingReward
from jitrl_core.state import intent_class, jaccard, task_signature
from jitrl_core.value import estimate_values

__all__ = [
    "JitRLConfig", "TIERS",
    "JudgeClient", "JudgeResult", "JitRLEngine", "DecisionResult",
    "ExperienceMemory", "MockJudgeClient",
    "choose_tier", "modulate_logits",
    "RewardFn", "StubRoutingReward",
    "intent_class", "task_signature", "jaccard", "estimate_values",
]
