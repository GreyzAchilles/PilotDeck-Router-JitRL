/**
 * Policy: closed-form logit modulation (spec section 2, port of
 * `jitrl_core/policy.py` — verbatim).
 *
 *     z'(tier) = z(tier) + beta * A_norm(tier);  chosen = argmax z'
 *
 * zMin is the documented lower clamp for logits (both base and modulated).
 * argmax ties resolve to the FIRST tier in canonical TIERS order.
 */
import { TIERS } from "./tiers.js";

export type TierLogits = Record<string, number>;

export function clampLogits(
  z: TierLogits,
  zMin: number,
  tiers: readonly string[] = TIERS,
): TierLogits {
  const out: TierLogits = {};
  for (const tier of tiers) {
    out[tier] = Math.max(Number(z[tier] ?? 0.0), zMin);
  }
  return out;
}

export function modulateLogits(
  zBase: TierLogits,
  aNorm: TierLogits,
  options?: { beta?: number; zMin?: number; tiers?: readonly string[] },
): TierLogits {
  const beta = options?.beta ?? 5.0;
  const zMin = options?.zMin ?? -10.0;
  const tiers = options?.tiers ?? TIERS;
  const out: TierLogits = {};
  for (const tier of tiers) {
    const value = Number(zBase[tier] ?? 0.0) + beta * Number(aNorm[tier] ?? 0.0);
    out[tier] = Math.max(value, zMin);
  }
  return out;
}

/** argmax with deterministic tie-break: first max in canonical TIERS order. */
export function chooseTier(z: TierLogits, tiers: readonly string[] = TIERS): string {
  let best = tiers[0]!;
  let bestValue = Number(z[best] ?? 0.0);
  for (const tier of tiers.slice(1)) {
    const value = Number(z[tier] ?? 0.0);
    if (value > bestValue) {
      best = tier;
      bestValue = value;
    }
  }
  return best;
}
