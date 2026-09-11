/**
 * JitRL reward: r = 0.6 * quality + 0.3 * costSaving.
 *
 * - quality: the configured evaluator model reviews the (user message,
 *   response) pair through the injected ModelRuntime.complete — never routed
 *   through the Router. Returns a [0,1] score. If the evaluation request
 *   fails, times out, or cannot be parsed, the caller MUST NOT write memory.
 * - costSaving: episode price via the router's existing pricing utilities
 *   (lookupModelPricing), normalized against the MOST EXPENSIVE candidate in
 *   the same tier candidate set and clamped to [0,1].
 *
 * The fixed assumption profile (1500 input / 800 output tokens per episode,
 * documented in jitrl_core/rewards.py) is used when no real usage is
 * available.
 */
import type { CanonicalModelRequest, ModelRuntime } from "../../model/index.js";
import type { RouterModelRef } from "../config/schema.js";
import { lookupModelPricing, type RouterModelPricingMap } from "../utils/modelPricing.js";

export const JITRL_REWARD_W_QUALITY = 0.6;
export const JITRL_REWARD_W_COST = 0.3;

/** Fixed per-episode assumption profile (documented; replaced by real usage). */
export const JITRL_EPISODE_INPUT_TOKENS = 1500;
export const JITRL_EPISODE_OUTPUT_TOKENS = 800;

export const JITRL_EVALUATOR_DEFAULT_TIMEOUT_MS = 15_000;

export type JitRLRewardInput = {
  userMessage: string;
  responseText: string;
  chosenTier: string;
  chosenModel: RouterModelRef;
  /** Same candidate set used for routing (tier → model). */
  candidates: readonly RouterModelRef[];
  usage?: { inputTokens?: number; outputTokens?: number };
};

export type JitRLRewardResult = {
  reward: number;
  quality: number;
  costSaving: number;
};

export type JitRLRewardDeps = {
  evaluatorRuntime?: ModelRuntime;
  evaluator?: RouterModelRef;
  modelPricing?: RouterModelPricingMap;
  timeoutMs?: number;
  abortSignal?: AbortSignal;
};

export function episodePrice(
  provider: string,
  model: string,
  inputTokens: number,
  outputTokens: number,
  modelPricing?: RouterModelPricingMap,
): number {
  const pricing = lookupModelPricing(provider, model, modelPricing);
  return (
    (inputTokens * (pricing.input ?? 0) + outputTokens * (pricing.output ?? 0)) / 1_000_000
  );
}

/** costSaving = clamp((mostExpensive - chosen) / mostExpensive, 0, 1). */
export function computeCostSaving(
  input: Pick<JitRLRewardInput, "chosenModel" | "candidates" | "usage">,
  modelPricing?: RouterModelPricingMap,
): number {
  const inputTokens = input.usage?.inputTokens && input.usage.inputTokens > 0
    ? input.usage.inputTokens
    : JITRL_EPISODE_INPUT_TOKENS;
  const outputTokens = input.usage?.outputTokens && input.usage.outputTokens > 0
    ? input.usage.outputTokens
    : JITRL_EPISODE_OUTPUT_TOKENS;

  const uniqueCandidates: RouterModelRef[] = [];
  for (const candidate of input.candidates) {
    if (
      !uniqueCandidates.some(
        (existing) => existing.provider === candidate.provider && existing.model === candidate.model,
      )
    ) {
      uniqueCandidates.push(candidate);
    }
  }
  if (uniqueCandidates.length === 0) {
    return 0;
  }

  let mostExpensive = 0;
  for (const candidate of uniqueCandidates) {
    mostExpensive = Math.max(
      mostExpensive,
      episodePrice(candidate.provider, candidate.model, inputTokens, outputTokens, modelPricing),
    );
  }
  const chosen = episodePrice(
    input.chosenModel.provider,
    input.chosenModel.model,
    inputTokens,
    outputTokens,
    modelPricing,
  );
  if (mostExpensive <= 0) {
    return 0;
  }
  return Math.min(1, Math.max(0, (mostExpensive - chosen) / mostExpensive));
}

export async function evaluateQuality(
  input: { userMessage: string; responseText: string },
  deps: JitRLRewardDeps,
): Promise<number | undefined> {
  const { evaluatorRuntime, evaluator } = deps;
  if (!evaluatorRuntime || !evaluator) {
    return undefined;
  }

  const request: CanonicalModelRequest = {
    provider: evaluator.provider,
    model: evaluator.model,
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: buildEvaluatorPrompt(input),
          },
        ],
      },
    ],
    maxOutputTokens: 32,
    thinking: { enabled: false },
    stream: false,
  };

  let timeout: NodeJS.Timeout | undefined;
  const controller = new AbortController();
  const forwardAbort = () => controller.abort(deps.abortSignal?.reason);
  deps.abortSignal?.addEventListener("abort", forwardAbort, { once: true });
  try {
    if (deps.abortSignal?.aborted) {
      return undefined;
    }
    const response = await Promise.race([
      evaluatorRuntime.complete(request, { signal: controller.signal }),
      new Promise<never>((_, reject) => {
        timeout = setTimeout(() => {
          controller.abort(new Error("jitrl evaluator timeout"));
          reject(new Error("jitrl evaluator timeout"));
        }, Math.max(500, deps.timeoutMs ?? JITRL_EVALUATOR_DEFAULT_TIMEOUT_MS));
      }),
    ]);
    const text = response.content
      .filter((block) => block.type === "text")
      .map((block) => (block.type === "text" ? block.text : ""))
      .join("")
      .replace(/```[a-z]*\n?/g, "")
      .replace(/```/g, "")
      .trim();
    const match = /-?\d+(?:\.\d+)?/.exec(text);
    if (!match) {
      return undefined;
    }
    const value = Number(match[0]);
    if (!Number.isFinite(value) || value < 0 || value > 10) {
      return undefined;
    }
    return value / 10;
  } catch {
    return undefined;
  } finally {
    if (timeout) {
      clearTimeout(timeout);
    }
    deps.abortSignal?.removeEventListener("abort", forwardAbort);
  }
}

export async function computeJitRLReward(
  input: JitRLRewardInput,
  deps: JitRLRewardDeps,
): Promise<JitRLRewardResult | undefined> {
  const quality = await evaluateQuality(
    { userMessage: input.userMessage, responseText: input.responseText },
    deps,
  );
  if (quality === undefined) {
    // Evaluation failed — the caller must NOT write memory.
    return undefined;
  }
  const costSaving = computeCostSaving(input, deps.modelPricing);
  return {
    reward: JITRL_REWARD_W_QUALITY * quality + JITRL_REWARD_W_COST * costSaving,
    quality,
    costSaving,
  };
}

function buildEvaluatorPrompt(input: { userMessage: string; responseText: string }): string {
  return `You are a response quality evaluator for the PilotDeck router. Rate how well the assistant response answers the user message. Judge correctness, completeness and relevance. Ignore style.

Respond with ONLY a single integer score from 0 (worthless or wrong) to 10 (excellent). No other text.

User message:
"""
${input.userMessage}
"""

Assistant response:
"""
${input.responseText.slice(0, 4000)}
"""`;
}
