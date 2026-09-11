"""Tier -> model mapping and per-episode price (spec section 4).

price_ep(model) = (1500 * price_in + 800 * price_out) / 1e6
Fixed assumption profile: 1500 input / 800 output tokens per episode
(documented; session 2 replaces with real execution usage).
"""
from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRICING_PATH = _REPO_ROOT / "eval" / "pricing.json"

#: tier -> model (spec, verbatim)
TIER_TO_MODEL: dict[str, str] = {
    "simple": "CPA/glm-5.3-flash",
    "medium": "CPA/opencode-v4-flash",
    "reasoning": "CPA/glm-5.3",
    "complex": "CPA/OpenBMB-5.3",
}

#: static/default routing (mode A) -> CPA/glm-5.3
DEFAULT_MODEL = "CPA/glm-5.3"

#: fixed assumption profile (documented deviation from real usage)
EPISODE_INPUT_TOKENS = 1500
EPISODE_OUTPUT_TOKENS = 800


def load_prices(path: str | Path | None = None) -> dict[str, dict[str, float]]:
    p = Path(path) if path else DEFAULT_PRICING_PATH
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def model_for_tier(tier: str) -> str:
    """Map a tier to its model; unknown tiers (e.g. "static") -> default."""
    return TIER_TO_MODEL.get(tier, DEFAULT_MODEL)


def price_ep(
    model: str,
    prices: dict[str, dict[str, float]] | None = None,
    in_tokens: int = EPISODE_INPUT_TOKENS,
    out_tokens: int = EPISODE_OUTPUT_TOKENS,
) -> float:
    """Per-episode price in $ under the fixed 1500-in / 800-out profile."""
    table = prices if prices is not None else _PRICES
    entry = table[model]
    return (in_tokens * entry["input"] + out_tokens * entry["output"]) / 1e6


def cost_from_usage(
    model: str,
    usage: dict,
    prices: dict[str, dict[str, float]] | None = None,
) -> float:
    """Real per-call cost in $ from normalized usage (Session 2 directive).

    Directive formula (eval/pricing.json, $/Mtok):
        cost = net_input * input + cache_read * cache_input + output * output
    `usage` is a normalized UsageResult from harness.cpa_client (net input
    already excludes cached portions). cache_write tokens (Anthropic-style;
    absent in OpenAI-format responses) are billed at the plain input rate so
    no tokens escape accounting; for OpenAI-format usage they are 0 and the
    formula reduces exactly to the directive form.
    """
    table = prices if prices is not None else _PRICES
    entry = table[model]
    net_in = int(usage.get("input_tokens", 0))
    cache_read = int(usage.get("cache_read_tokens", 0))
    cache_write = int(usage.get("cache_write_tokens", 0))
    out = int(usage.get("output_tokens", 0))
    usd = (
        net_in * entry["input"]
        + cache_write * entry["input"]
        + cache_read * entry["cache_input"]
        + out * entry["output"]
    ) / 1e6
    return usd


_PRICES: dict[str, dict[str, float]] = load_prices()
