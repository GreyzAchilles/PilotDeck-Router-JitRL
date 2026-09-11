"""MockJudgeClient: a WEAK but credible stand-in judge (spec section 3).

Rules:
- continuation-turn messages (reuses the intent matcher from jitrl_core.state)
  -> echo previous_tier with high confidence;
- otherwise keyword/length surface heuristics + a SYSTEMATIC bias
  (always extra mass on "medium"; very long messages pushed to "reasoning"
  — classic surface heuristics) + seeded small noise;
- tier_logits argmax is always consistent with chosen_tier;
- raw_output = "<tier>{chosen}</tier>"; latency_ms fake in [5, 20].

Design note (tuning knob sanctioned by the task spec): the keyword-cue mass
is set HIGH (8.0) so that in episode 2 the lambda-exploration optimism
(alpha/|N(s)|, engine-side) cannot flip tiers the heuristic already decided.
The judge is still weak where it has no cue (data_analysis / chat / refactor
routing is systematically medium-biased), which keeps unmodulated accuracy
around ~60% on the mini fixture — clearly below 100%.
"""
from __future__ import annotations

import random

from jitrl_core.config import TIERS
from jitrl_core.engine import JudgeResult  # re-export for typing parity
from jitrl_core.state import intent_class

CODE_CUES: tuple[str, ...] = (
    "```", "def ", "函数", "单元测试", "测试用例", "实现", "重构",
    "function", "implement", "unit test",
)
DOC_CUES: tuple[str, ...] = (
    "写一封", "文档", "报告", "通知", "公告", "周报", "邮件",
    "documentation", "report", "notice", "memo", "email",
)
LONG_MESSAGE_CHARS = 300      # >= this length -> reasoning (surface heuristic)
CUE_MASS = 8.0                # confident surface-cue logit mass
MEDIUM_BIAS = 1.0             # systematic bias: prefers medium
PREV_TIER_MASS = 6.0          # continuation turn echoes previous tier
NOISE = 0.05                  # seeded uniform noise half-width


class MockJudgeClient:
    """Deterministic under a seed; interface-identical to JudgeClient."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def judge(self, user_message: str, previous_tier: str | None = None,
              memory_block: str | None = None) -> JudgeResult:
        # memory_block (S3-T8 ablation harness compatibility): accepted and
        # deliberately IGNORED — the mock has no prompt-mechanism response,
        # so T8-PM prompt effects are exercised with spy judges in tests.
        scores: dict[str, float] = {t: 0.0 for t in TIERS}

        if intent_class(user_message) == "continuation" and previous_tier in TIERS:
            scores[previous_tier] += PREV_TIER_MASS
        else:
            # systematic surface bias: always extra preference for medium
            scores["medium"] += MEDIUM_BIAS
            low = user_message.lower()
            if any(cue in low for cue in CODE_CUES):
                scores["complex"] += CUE_MASS
            elif any(cue in low for cue in DOC_CUES):
                scores["simple"] += CUE_MASS
            if len(user_message) >= LONG_MESSAGE_CHARS:
                scores["reasoning"] += CUE_MASS

        for t in TIERS:
            scores[t] += self._rng.uniform(-NOISE, NOISE)

        chosen = max(TIERS, key=lambda t: scores[t])
        latency = self._rng.uniform(5.0, 20.0)
        return {
            "tier_logits": scores,
            "chosen_tier": chosen,
            "raw_output": f"<tier>{chosen}</tier>",
            "latency_ms": latency,
        }
