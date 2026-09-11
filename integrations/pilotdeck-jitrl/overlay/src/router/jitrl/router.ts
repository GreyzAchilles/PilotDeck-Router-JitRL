/**
 * JitRL custom router (MVP): PilotDeckCustomRouter implementation wiring the
 * TypeScript JitRL engine into the PilotDeck Router through the official
 * contribution interface.
 *
 * decide(userMessage):
 *   1. judge  -> 4-tier logit vector z (ModelRuntime.complete scoring
 *      requests, or the deterministic heuristic fallback);
 *   2. engine -> z' = z + beta * A_norm over top-k Jaccard neighbors
 *      (min-neighbor gate, deterministic tie-break);
 *   3. chosen tier maps to the configured candidate model.
 *
 * onTurnOutcome(turn):
 *   reward = 0.6 * quality + 0.3 * costSaving; quality is reviewed by the
 *   configured evaluator model. Memory is written ONLY when the evaluation
 *   succeeds (directive: evaluation failure -> no learning). Execution
 *   failures produce no episode (no meaningful quality signal).
 */
import type {
  CustomRouterContext,
  CustomRouterDecideInput,
  CustomRouterTurnOutcome,
  PilotDeckCustomRouter,
} from "../customRouter/customRouter.js";
import type { RouterModelRef } from "../config/schema.js";
import { extractLastUserMessage } from "../tokenSaver/extractLastUserMessage.js";
import { JitRLJudgeClient } from "./judge.js";
import { getSharedJitRLState } from "./memoryStore.js";
import { computeJitRLReward } from "./reward.js";
import { TIERS } from "./tiers.js";

export const JITRL_ROUTER_ID = "jitrl";

export type JitRLRouterRuntimeInfo = {
  tierLogits: Record<string, number>;
  chosenTier: string;
  baseChoice: string;
  judgeSource: "model" | "fallback";
  gateReason: string | undefined;
  nRetrieved: number;
};

/** Resolve the tier -> candidate model map (customRouter.tiers, falling back to tokenSaver tiers). */
export function resolveTierCandidates(context: CustomRouterContext): Map<string, RouterModelRef> {
  const candidates = new Map<string, RouterModelRef>();
  for (const [name, ref] of Object.entries(context.config.tiers ?? {})) {
    candidates.set(name, ref);
  }
  if (candidates.size === 0 && context.routerConfig.tokenSaver?.enabled) {
    for (const [name, tier] of Object.entries(context.routerConfig.tokenSaver.tiers ?? {})) {
      candidates.set(name, tier.model);
    }
  }
  return candidates;
}

export function createJitrlCustomRouter(): PilotDeckCustomRouter {
  return {
    id: JITRL_ROUTER_ID,

    async decide(input: CustomRouterDecideInput): Promise<Partial<import("../protocol/decision.js").RouterDecision> | undefined> {
      const context = input.context;
      const userMessage = extractLastUserMessage(input.request.messages);
      if (!userMessage) {
        return undefined;
      }

      const candidates = resolveTierCandidates(context);
      if (candidates.size === 0) {
        // No tier candidates configured — let the scenario/default routing win.
        return undefined;
      }

      const shared = getSharedJitRLState({
        memoryPath: context.config.memoryPath,
        hyperparams: context.config.hyperparams,
      });

      const judge = new JitRLJudgeClient({
        judgeRuntime: context.judgeRuntime,
        judge: context.config.judge,
        timeoutMs: context.config.judgeTimeoutMs,
        abortSignal: input.abortSignal,
      });
      const judgeResult = await judge.judge({
        userMessage,
        previousTier: input.metadata?.previousTier,
      });

      const decision = shared.engine.decide(userMessage, judgeResult.tierLogits);

      const selection =
        candidates.get(decision.chosenTier) ??
        candidates.get(context.routerConfig.tokenSaver?.defaultTier ?? "") ??
        [...candidates.values()][0]!;

      console.log(
        `[jitrl] tier=${decision.chosenTier} (base=${decision.baseChoice} flipped=${decision.flipped}) ` +
        `model=${selection.provider}/${selection.model} judge=${judgeResult.source} ` +
        `n=${decision.nRetrieved} gate=${decision.gateReason ?? "modulate"}`,
      );

      return {
        provider: selection.provider,
        model: selection.model,
        tokenSaverTier: decision.chosenTier,
      };
    },

    async onTurnOutcome(outcome: CustomRouterTurnOutcome): Promise<void> {
      const context = outcome.context;
      if (outcome.error) {
        // Execution failure: no meaningful quality signal -> no episode.
        console.warn(
          `[jitrl] turn failed (tier=${outcome.decision.tokenSaverTier ?? "?"}, ` +
          `code=${outcome.error.code}) — skipping memory write`,
        );
        return;
      }

      const userMessage = outcome.userMessage;
      const tier = outcome.decision.tokenSaverTier;
      if (!userMessage || !tier) {
        return;
      }

      const candidates = [...resolveTierCandidates(context).values()];
      const evaluator = context.config.evaluator ?? context.config.judge;
      const reward = await computeJitRLReward(
        {
          userMessage,
          responseText: outcome.response ?? "",
          chosenTier: tier,
          chosenModel: {
            id: `${outcome.decision.provider}/${outcome.decision.model}`,
            provider: outcome.decision.provider,
            model: outcome.decision.model,
          },
          candidates,
          usage: outcome.usage,
        },
        {
          evaluatorRuntime: context.modelRuntime,
          evaluator,
          modelPricing: context.routerConfig.stats?.modelPricing,
          timeoutMs: context.config.evalTimeoutMs,
        },
      );

      if (!reward) {
        // Evaluator failure (or no evaluator configured): do NOT write memory.
        console.warn("[jitrl] evaluation failed — skipping memory write");
        return;
      }

      const shared = getSharedJitRLState({
        memoryPath: context.config.memoryPath,
        hyperparams: context.config.hyperparams,
      });
      shared.engine.updateFromMessage(
        userMessage,
        tier,
        reward.reward,
        `jitrl:${outcome.sessionId}:${outcome.turnId}`,
      );
      shared.store.scheduleFlush(() => shared.memory.entries());
      console.log(
        `[jitrl] learned tier=${tier} reward=${reward.reward.toFixed(3)} ` +
        `(quality=${reward.quality.toFixed(2)} costSaving=${reward.costSaving.toFixed(2)})`,
      );
    },
  };
}

/** Tier names the JitRL engine understands (re-exported for tests/UX). */
export const JITRL_TIERS = TIERS;
