"""Reward interface + routing-only STUB (spec section 2).

*** STUB — CLEARLY REPLACEABLE (session 2 swaps in real execution) ***

    r = 0.6 * quality_proxy + 0.3 * cost_saving
    quality_proxy = +1.0 if chosen_tier == gt_tier else -0.5
    cost_saving   = clamp((price_ep(default) - price_ep(chosen)) / price_ep(default), -1, 1)
    price_ep(model) = (1500 * price_in + 800 * price_out) / 1e6   [$ per episode]

Fixed assumption profile: 1500 input tokens / 800 output tokens per episode
(documented; session 2 replaces with real execution usage).

The reward function never depends on harness/ — pricing is injected.
"""
from __future__ import annotations

from typing import Any, Callable, Protocol


class RewardFn(Protocol):
    def reward(
        self,
        task: dict,
        chosen_tier: str,
        response: Any = None,
        usage: Any = None,
    ) -> float: ...


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class StubRoutingReward:
    """STUB reward: GT-tier match + price-based cost saving. See module docstring."""

    def __init__(
        self,
        model_for_tier: Callable[[str], str],
        price_of_model: Callable[[str], float],
        default_model: str,
        w_quality: float = 0.6,
        w_cost: float = 0.3,
    ):
        self._model_for_tier = model_for_tier
        self._price_of_model = price_of_model
        self._default_model = default_model
        self._w_quality = w_quality
        self._w_cost = w_cost
        self.default_price = float(price_of_model(default_model))

    def reward(
        self,
        task: dict,
        chosen_tier: str,
        response: Any = None,
        usage: Any = None,
    ) -> float:
        gt = task.get("gt_tier")
        if gt is None:
            quality_proxy = 0.0  # unlabeled task: quality term contributes 0
        else:
            quality_proxy = 1.0 if chosen_tier == gt else -0.5
        chosen_model = self._model_for_tier(chosen_tier)
        p_chosen = self._price_of_model(chosen_model)
        if self.default_price == 0.0:
            cost_saving = 0.0
        else:
            cost_saving = clamp(
                (self.default_price - p_chosen) / self.default_price, -1.0, 1.0
            )
        return self._w_quality * quality_proxy + self._w_cost * cost_saving
