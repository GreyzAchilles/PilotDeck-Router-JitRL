/**
 * JitRL engine: decide (retrieve + value estimate + logit modulation) & learn.
 * TypeScript port of `jitrl_core/engine.py` (test-time RL, no gradient).
 *
 * Memory of (state, tier, reward): retrieve similar experiences, estimate
 * per-tier advantage, modulate judge logits z' = z + beta * A_norm, and write
 * episodes back.
 *
 * C1 min-neighbor gate: if 0 < nRetrieved < minNeighbors, modulation is
 * suppressed (zMod = zBase, A_norm = 0, chosen = baseChoice). minNeighbors=1
 * (default) reproduces C0 exactly. estimateValues is always called so the
 * RNG stream is independent of the gate.
 */
import {
  createSeededRandom,
  GATE_BELOW_MIN_NEIGHBORS,
  GATE_NO_NEIGHBORS,
  type JitRLGateReason,
  type JitRLHyperparams,
  resolveJitRLHyperparams,
  TIERS,
} from "./tiers.js";
import { ExperienceMemory, type JitRLEpisode } from "./memory.js";
import { chooseTier, clampLogits, modulateLogits, type TierLogits } from "./policy.js";
import { estimateValues, type TierValueMap } from "./value.js";
import { intentClass, taskSignature } from "./state.js";

export type DecisionResult = {
  intentClass: string;
  /** sorted (canonical memory form) */
  signatureTokens: string[];
  nRetrieved: number;
  topSimilarity: number;
  V: number;
  Q: TierValueMap;
  A_norm: TierValueMap;
  zBase: TierLogits;
  zMod: TierLogits;
  chosenTier: string;
  baseChoice: string;
  flipped: boolean;
  /** gate threshold in force for this decide */
  minNeighbors: number;
  /** true iff the C0 modulation path ran */
  modulationActive: boolean;
  gateReason: JitRLGateReason | undefined;
  retrievedEpisodeIds: Array<string | number>;
};

export class JitRLEngine {
  readonly hyperparams: JitRLHyperparams;
  readonly memory: ExperienceMemory;
  /** engine-owned seeded rng for the lambda exploration branch */
  private readonly rng: () => number;

  constructor(
    options?: {
      hyperparams?: Partial<JitRLHyperparams>;
      memory?: ExperienceMemory;
      /** injectable for tests (replaces the seeded default) */
      rng?: () => number;
    },
  ) {
    this.hyperparams = resolveJitRLHyperparams(options?.hyperparams);
    this.memory = options?.memory ?? new ExperienceMemory(this.hyperparams.memoryCap);
    this.rng = options?.rng ?? createSeededRandom(this.hyperparams.seed);
  }

  decide(userMessage: string, tierLogits: TierLogits): DecisionResult {
    const cfg = this.hyperparams;
    const intent = intentClass(userMessage);
    const tokens = taskSignature(userMessage);
    const tokenSet = new Set(tokens);

    const matches = this.memory.retrieve(intent, tokenSet, {
      k: cfg.k,
      threshold: cfg.jaccardThreshold,
    });
    const topSim = matches.length > 0 ? matches[0]!.similarity : 0.0;
    const entries = matches.map((match) => match.entry);

    // estimateValues is ALWAYS called (even when the gate below fires) so the
    // engine RNG stream stays identical for the same learn/decide sequence,
    // regardless of minNeighbors.
    const est = estimateValues(entries, {
      rng: this.rng,
      lam: cfg.lam,
      alpha: cfg.alpha,
      tiers: TIERS,
    });

    const zBase = clampLogits(tierLogits, cfg.zMin, TIERS);
    let gateReason: JitRLGateReason | undefined;
    let zMod: TierLogits;
    let aNorm: TierValueMap;
    if (est.nRetrieved === 0) {
      // empty N(s): V=0 and NO modulation (pure baseline judge)
      gateReason = GATE_NO_NEIGHBORS;
      zMod = { ...zBase };
      aNorm = zeros();
    } else if (est.nRetrieved < cfg.minNeighbors) {
      // C1 min-neighbor gate: insufficient memory evidence -> degrade to the
      // native judge (z_mod = z_base, A_norm = 0).
      gateReason = GATE_BELOW_MIN_NEIGHBORS;
      zMod = { ...zBase };
      aNorm = zeros();
    } else {
      // C0 modulation path: z' = z + beta * A_norm
      gateReason = undefined;
      aNorm = est.A_norm;
      zMod = modulateLogits(zBase, aNorm, { beta: cfg.beta, zMin: cfg.zMin, tiers: TIERS });
    }

    const baseChoice = chooseTier(zBase, TIERS);
    const chosen = chooseTier(zMod, TIERS);

    return {
      intentClass: intent,
      signatureTokens: [...tokens].sort(),
      nRetrieved: est.nRetrieved,
      topSimilarity: topSim,
      V: est.V,
      Q: { ...est.Q },
      A_norm: { ...aNorm },
      zBase,
      zMod,
      chosenTier: chosen,
      baseChoice,
      flipped: chosen !== baseChoice,
      minNeighbors: cfg.minNeighbors,
      modulationActive: gateReason === undefined,
      gateReason,
      retrievedEpisodeIds: entries.map((entry) => entry.episode_id),
    };
  }

  /** Episode ended: append the entry; memory enforces the cap (drop oldest). */
  learn(input: {
    intent: string;
    signatureTokens: string[];
    tier: string;
    reward: number;
    episodeId: string | number;
    ts?: number;
  }): JitRLEpisode {
    const entry: JitRLEpisode = {
      intent_class: input.intent,
      signature_tokens: [...input.signatureTokens].sort(),
      tier: input.tier,
      G: Number(input.reward),
      ts: input.ts !== undefined ? Number(input.ts) : Date.now(),
      episode_id: input.episodeId,
    };
    this.memory.add(entry);
    return entry;
  }

  updateFromMessage(
    userMessage: string,
    chosenTier: string,
    reward: number,
    episodeId: string | number,
    ts?: number,
  ): JitRLEpisode {
    return this.learn({
      intent: intentClass(userMessage),
      signatureTokens: taskSignature(userMessage),
      tier: chosenTier,
      reward,
      episodeId,
      ts,
    });
  }
}

function zeros(): TierValueMap {
  const out: TierValueMap = {};
  for (const tier of TIERS) {
    out[tier] = 0.0;
  }
  return out;
}
