"""Central configuration for the local judge client.

Module-level constants, each overridable via environment variable:
  ENDPOINT  <- JITRL_JUDGE_ENDPOINT   (default http://127.0.0.1:18080)
  MODEL     <- JITRL_JUDGE_MODEL      (default minicpm5-1b)
  TIMEOUT_S <- JITRL_JUDGE_TIMEOUT_S  (default 15.0)
"""
import os

ENDPOINT = os.environ.get("JITRL_JUDGE_ENDPOINT", "http://127.0.0.1:18080")
MODEL = os.environ.get("JITRL_JUDGE_MODEL", "minicpm5-1b")
TIMEOUT_S = float(os.environ.get("JITRL_JUDGE_TIMEOUT_S", "15.0"))

# The 4 cost tiers, fixed by the PilotDeck router contract.
TIERS: tuple[str, ...] = ("simple", "medium", "complex", "reasoning")

# Clamp for tiers whose logprob cannot be recovered (Method A not visible in
# top_logprobs; Method B canonical token never sampled and not in top list).
# NOTE: this is an APPROXIMATION — the true value is <= Z_MIN; -10 keeps the
# tier strictly dominated in softmax while staying finite for downstream math.
Z_MIN = -10.0

# Transport-level retries (per HTTP request).
TRANSPORT_RETRIES = 3
RETRY_SLEEP_S = 1.0
ERROR_MSG_MAX_CHARS = 300

# Canonical-tokenization retries (per tier, Method B): grammar constrains the
# emitted text but not its tokenization path; retry until the sampled token
# path matches the canonical one (see S1 report for observed failure rates).
CANONICAL_ATTEMPTS = 3

# Sampling temperature for z-extraction requests. Reported logprobs are
# pre-sampling (temperature-independent); a low temperature only reduces the
# probability of degenerate tokenization paths (probe: 7/8 canonical first
# try at 0.1 vs ~4/5 at 1.0).
Z_TEMPERATURE = 0.1

# top_logprobs requested per response. Method A reads the name position of
# the multi-tier request (task spec: 50). Method B requests 100 so the
# fallback lookup can still see canonical tokens as unlikely as ~-10 nats.
TOP_LOGPROBS_A = 50
TOP_LOGPROBS_B = 100

# Known-good canonical tokenizations measured in S1 (llama.cpp /tokenize,
# minicpm5-1b vocab). Used as fallback when the live /tokenize endpoint is
# unavailable. "reasoning" is NOT a single token in this vocab — the reason
# Method A can never recover it from a single position.
CANONICAL_TOKENS_FALLBACK: dict[str, list[str]] = {
    "simple": ["simple"],
    "medium": ["medium"],
    "complex": ["complex"],
    "reasoning": ["reason", "ing"],
}
