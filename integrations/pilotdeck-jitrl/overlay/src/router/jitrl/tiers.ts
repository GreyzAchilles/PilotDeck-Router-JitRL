/**
 * JitRL canonical tier order + hyperparameters.
 *
 * TypeScript port of `jitrl_core/config.py` (Session1-T4/T5 spec, verbatim):
 * k=10, beta=5.0, lam=0.05, alpha=5.0, jaccard_threshold=0.5, z_min=-10.0,
 * memory_cap=5000, seed=42. C1 adds min_neighbors (default 1 == C0 semantics).
 */

/** Canonical tier order; also the deterministic argmax tie-break order. */
export const TIERS: readonly [string, string, string, string] = ["simple", "medium", "complex", "reasoning"] as const;

export type JitRLTier = (typeof TIERS)[number];

export function isJitRLTier(value: unknown): value is JitRLTier {
  return typeof value === "string" && (TIERS as readonly string[]).includes(value);
}

export type JitRLHyperparams = {
  /** top-k retrieved experiences */
  k: number;
  /** logit modulation strength */
  beta: number;
  /** exploration probability for unseen tiers */
  lam: number;
  /** optimism bonus: Q = V + alpha/|N(s)| */
  alpha: number;
  /** minimum signature similarity to retrieve */
  jaccardThreshold: number;
  /** lower clamp for base/modulated logits */
  zMin: number;
  /** drop-oldest memory capacity */
  memoryCap: number;
  /** seed for the engine-side lambda draws */
  seed: number;
  /**
   * C1 min-neighbor gate: minimum retrieved neighbors required before logit
   * modulation is allowed. Default 1 preserves C0 semantics exactly.
   */
  minNeighbors: number;
};

export const DEFAULT_JITRL_HYPERPARAMS: JitRLHyperparams = {
  k: 10,
  beta: 5.0,
  lam: 0.05,
  alpha: 5.0,
  jaccardThreshold: 0.5,
  zMin: -10.0,
  memoryCap: 5000,
  seed: 42,
  minNeighbors: 1,
};

export function resolveJitRLHyperparams(
  overrides?: Partial<JitRLHyperparams>,
): JitRLHyperparams {
  return { ...DEFAULT_JITRL_HYPERPARAMS, ...(overrides ?? {}) };
}

/** gate_reason semantics (C1 min-neighbor gate, S2-C1-PREREGISTRATION v1.0). */
export const GATE_NO_NEIGHBORS = "no_neighbors";
export const GATE_BELOW_MIN_NEIGHBORS = "below_min_neighbors";
export type JitRLGateReason = "no_neighbors" | "below_min_neighbors";

/**
 * Deterministic seeded PRNG (mulberry32). Used for the engine-side lambda
 * exploration draws; tests inject a stub for exact numeric control.
 *
 * Deviation from the Python reference (documented): Python uses
 * `random.Random(seed)` (MT19937). The stream values differ, but the
 * semantics (seeded, deterministic, independent of the gate path) match.
 */
export function createSeededRandom(seed: number): () => number {
  let state = (seed | 0) || 1;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
