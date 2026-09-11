import type { CanonicalModelError, CanonicalUsage, ModelRuntime } from "../../model/index.js";
import type { RouterConfig, RouterCustomRouterConfig } from "../config/schema.js";
import type { RouterDecision, RouterDecisionInput } from "../protocol/decision.js";

export type CustomRouterContext = {
  sessionId: string;
  isMainAgent: boolean;
  scenarios: ReadonlyArray<string>;
  /**
   * Model runtime injected by the Router. Custom routers MUST call judge /
   * evaluator models through this runtime (or `judgeRuntime`) directly —
   * never recursively through the Router itself.
   */
  modelRuntime: ModelRuntime;
  /** Judge runtime (falls back to modelRuntime when unset). */
  judgeRuntime: ModelRuntime;
  /** Parsed `router.customRouter` section (extensionId, judge, evaluator, tiers, hyperparams, memoryPath, timeouts). */
  config: RouterCustomRouterConfig;
  /** Full resolved router configuration (e.g. tokenSaver tier fallbacks). */
  routerConfig: RouterConfig;
};

export type CustomRouterDecideInput = RouterDecisionInput & {
  context: CustomRouterContext;
};

/**
 * Turn-level outcome handed to `PilotDeckCustomRouter.onTurnOutcome` after a
 * custom-routed execution settles — success or failure. Lets learning-based
 * routers (JitRL) close the loop without re-entering the Router.
 */
export type CustomRouterTurnOutcome = {
  sessionId: string;
  turnId: string;
  projectPath?: string;
  userMessage?: string;
  decision: RouterDecision;
  usage?: CanonicalUsage;
  /** Assistant text collected from the executed stream (may be partial on failure). */
  response?: string;
  error?: CanonicalModelError;
  context: CustomRouterContext;
};

export type PilotDeckCustomRouter = {
  id: string;
  decide(input: CustomRouterDecideInput): Promise<Partial<RouterDecision> | undefined>;
  /**
   * Optional learning hook. Called once per custom-routed `execute()` after
   * the stream settles: on success with usage + response, on failure with
   * the terminal error. Errors thrown by the hook never break the stream.
   */
  onTurnOutcome?(outcome: CustomRouterTurnOutcome): void | Promise<void>;
};

export type CustomRouterRegistry = {
  lookupRouter(extensionId: string): PilotDeckCustomRouter | undefined;
};

export const noopCustomRouterRegistry: CustomRouterRegistry = {
  lookupRouter: () => undefined,
};
