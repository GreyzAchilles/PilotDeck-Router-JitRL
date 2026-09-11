/**
 * Value estimation (spec section 2 — numeric port of `jitrl_core/value.py`).
 *
 * Given N(s) = retrieved entries (each has "G" and "tier"):
 *     V = mean(G) over N(s);  empty N(s) -> V = 0 and NO modulation
 *     per tier t:
 *         |N(s,t)| > 0 -> Q(t) = mean(G over N(s,t))
 *         else         -> with prob lam: Q(t) = V + alpha/|N(s)| ; else Q(t) = 0
 *     A(t) = Q(t) - V
 *     A_norm(t) = A(t) / (max|A| + 1e-8);  max|A| == 0 -> A_norm(t) = 0
 *
 * The `rng` is engine-owned and seeded (hyperparams.seed); inject a stub in tests.
 */
import { TIERS, type JitRLTier } from "./tiers.js";
import type { JitRLEpisode } from "./memory.js";

export type TierValueMap = Record<string, number>;

export type ValueEstimate = {
  V: number;
  Q: TierValueMap;
  A: TierValueMap;
  A_norm: TierValueMap;
  nRetrieved: number;
};

function zeros(): TierValueMap {
  const out: TierValueMap = {};
  for (const tier of TIERS) {
    out[tier] = 0.0;
  }
  return out;
}

export function estimateValues(
  entries: JitRLEpisode[],
  options?: {
    rng?: () => number;
    lam?: number;
    alpha?: number;
    tiers?: readonly string[];
  },
): ValueEstimate {
  const lam = options?.lam ?? 0.05;
  const alpha = options?.alpha ?? 5.0;
  const tiers = options?.tiers ?? TIERS;
  const rng = options?.rng;

  const n = entries.length;
  if (n === 0) {
    const z = zeros();
    return { V: 0.0, Q: { ...z }, A: { ...z }, A_norm: { ...z }, nRetrieved: 0 };
  }

  let gSum = 0.0;
  for (const entry of entries) {
    gSum += Number(entry.G);
  }
  const V = gSum / n;

  const Q: TierValueMap = {};
  for (const tier of tiers) {
    let gTierSum = 0.0;
    let gTierCount = 0;
    for (const entry of entries) {
      if (entry.tier === tier) {
        gTierSum += Number(entry.G);
        gTierCount += 1;
      }
    }
    if (gTierCount > 0) {
      Q[tier] = gTierSum / gTierCount;
    } else {
      const explore = rng ? rng() < lam : false;
      Q[tier] = explore ? V + alpha / n : 0.0;
    }
  }

  const A: TierValueMap = {};
  for (const tier of tiers) {
    A[tier] = (Q[tier] ?? 0.0) - V;
  }
  let maxAbs = 0.0;
  for (const tier of tiers) {
    maxAbs = Math.max(maxAbs, Math.abs(A[tier] ?? 0.0));
  }
  const A_norm: TierValueMap = {};
  if (maxAbs === 0.0) {
    for (const tier of tiers) {
      A_norm[tier] = 0.0;
    }
  } else {
    for (const tier of tiers) {
      A_norm[tier] = (A[tier] ?? 0.0) / (maxAbs + 1e-8);
    }
  }

  return { V, Q, A, A_norm, nRetrieved: n };
}

export type { JitRLTier };
