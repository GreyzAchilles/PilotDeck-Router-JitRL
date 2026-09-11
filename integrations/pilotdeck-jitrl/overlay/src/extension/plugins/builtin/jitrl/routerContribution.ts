import type { RouterContribution } from "../../../contributions/RouterContribution.js";
import { createJitrlCustomRouter, JITRL_ROUTER_ID } from "../../../../router/jitrl/router.js";

/**
 * Programmatic router contribution for the builtin `jitrl` plugin.
 *
 * `PluginRuntime.lookupRouter()` may construct a fresh custom-router instance
 * on every router decide; the JitRL engine, its seeded RNG stream and the
 * persisted episodes live in a process-wide shared cache keyed by the
 * configured memory path (see router/jitrl/memoryStore.ts), so instance
 * churn never resets learning state.
 */
export const jitrlRouterContribution: RouterContribution = {
  id: JITRL_ROUTER_ID,
  description:
    "JitRL test-time RL router: judge 4-tier logits modulated by experience memory (z' = z + beta*A_norm).",
  createCustomRouter: createJitrlCustomRouter,
};
