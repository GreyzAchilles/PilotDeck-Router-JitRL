/**
 * router.customRouter config parsing: extended schema (judge / evaluator /
 * tiers / hyperparams / memoryPath / timeouts) + legacy extensionId-only
 * backward compatibility.
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { parseRouterConfig } from "../../../src/router/config/parseRouterConfig.js";
import type { ModelConfig } from "../../../src/model/index.js";

function makeModelConfig(models: string[]): ModelConfig {
  const modelRecord: Record<string, { id: string; capabilities: Record<string, never>; multimodal: Record<string, never> }> = {};
  for (const model of models) {
    modelRecord[model] = { id: model, capabilities: {}, multimodal: {} };
  }
  return {
    providers: {
      test: {
        id: "test",
        protocol: "openai",
        url: "http://localhost:1",
        apiKey: "key",
        headers: {},
        models: modelRecord,
      },
    },
  } as unknown as ModelConfig;
}

const MODEL_CONFIG = makeModelConfig([
  "cheap", "mid", "big", "judge-model", "eval-model", "default-model",
]);

test("customRouter: legacy extensionId-only config stays valid", () => {
  const result = parseRouterConfig(
    { customRouter: { extensionId: "my.router" } },
    MODEL_CONFIG,
  );
  assert.equal(result.diagnostics.length, 0);
  assert.deepEqual(result.config?.customRouter, { extensionId: "my.router" });
});

test("customRouter: full extended config parses", () => {
  const result = parseRouterConfig(
    {
      customRouter: {
        extensionId: "jitrl",
        judge: "test/judge-model",
        evaluator: "test/eval-model",
        tiers: {
          simple: "test/cheap",
          medium: "test/mid",
          complex: "test/big",
          reasoning: { model: "test/big" },
        },
        hyperparams: {
          k: 7,
          beta: 3.5,
          lam: 0.02,
          alpha: 4,
          jaccardThreshold: 0.4,
          zMin: -8,
          memoryCap: 1000,
          seed: 7,
          minNeighbors: 2,
        },
        memoryPath: "D:/tmp/jitrl-memory.json",
        evalTimeoutMs: 5000,
        judgeTimeoutMs: 8000,
      },
    },
    MODEL_CONFIG,
  );
  assert.equal(result.diagnostics.length, 0);
  const custom = result.config?.customRouter;
  assert.ok(custom);
  assert.equal(custom.extensionId, "jitrl");
  assert.deepEqual(custom.judge, { id: "test/judge-model", provider: "test", model: "judge-model" });
  assert.deepEqual(custom.evaluator, { id: "test/eval-model", provider: "test", model: "eval-model" });
  assert.equal(custom.tiers?.simple?.model, "cheap");
  assert.equal(custom.tiers?.medium?.provider, "test");
  assert.equal(custom.tiers?.reasoning?.model, "big");
  assert.deepEqual(custom.hyperparams, {
    k: 7, beta: 3.5, lam: 0.02, alpha: 4, jaccardThreshold: 0.4,
    zMin: -8, memoryCap: 1000, seed: 7, minNeighbors: 2,
  });
  assert.equal(custom.memoryPath, "D:/tmp/jitrl-memory.json");
  assert.equal(custom.evalTimeoutMs, 5000);
  assert.equal(custom.judgeTimeoutMs, 8000);
});

test("customRouter: missing extensionId is fatal", () => {
  const result = parseRouterConfig({ customRouter: { judge: "test/judge-model" } }, MODEL_CONFIG);
  assert.ok(result.diagnostics.some((d) => d.code === "ROUTER_CUSTOM_ROUTER_INVALID" && d.severity === "fatal"));
  assert.equal(result.config?.customRouter, undefined);
});

test("customRouter: unknown judge ref is fatal", () => {
  const result = parseRouterConfig(
    { customRouter: { extensionId: "jitrl", judge: "test/nope" } },
    MODEL_CONFIG,
  );
  assert.ok(result.diagnostics.some((d) => d.code === "ROUTER_REF_MODEL_NOT_FOUND" && d.severity === "fatal"));
});

test("customRouter: invalid tier entry is fatal", () => {
  const result = parseRouterConfig(
    { customRouter: { extensionId: "jitrl", tiers: { simple: 42 } } },
    MODEL_CONFIG,
  );
  assert.ok(result.diagnostics.some((d) => d.code === "ROUTER_CUSTOM_ROUTER_TIER_INVALID"));
});

test("customRouter: invalid hyperparams are fatal and dropped", () => {
  const result = parseRouterConfig(
    {
      customRouter: {
        extensionId: "jitrl",
        hyperparams: { k: 0, lam: 2, minNeighbors: "three" },
      },
    },
    MODEL_CONFIG,
  );
  const hyperDiagnostics = result.diagnostics.filter((d) => d.code === "ROUTER_CUSTOM_ROUTER_HYPERPARAM_INVALID");
  assert.equal(hyperDiagnostics.length, 3);
  assert.equal(result.config?.customRouter?.hyperparams, undefined);
});

test("customRouter: invalid timeout is fatal", () => {
  const result = parseRouterConfig(
    { customRouter: { extensionId: "jitrl", evalTimeoutMs: -5 } },
    MODEL_CONFIG,
  );
  assert.ok(result.diagnostics.some((d) => d.code === "ROUTER_CUSTOM_ROUTER_TIMEOUT_INVALID"));
});

test("customRouter: partial hyperparams merge onto engine defaults at runtime", async () => {
  const { DEFAULT_JITRL_HYPERPARAMS, resolveJitRLHyperparams } = await import("../../../src/router/jitrl/tiers.js");
  const result = parseRouterConfig(
    { customRouter: { extensionId: "jitrl", hyperparams: { beta: 2.5 } } },
    MODEL_CONFIG,
  );
  const resolved = resolveJitRLHyperparams(result.config?.customRouter?.hyperparams);
  assert.equal(resolved.beta, 2.5);
  assert.equal(resolved.k, DEFAULT_JITRL_HYPERPARAMS.k);
  assert.equal(resolved.minNeighbors, DEFAULT_JITRL_HYPERPARAMS.minNeighbors);
  assert.equal(resolved.memoryCap, DEFAULT_JITRL_HYPERPARAMS.memoryCap);
});

test("customRouter: router.enabled=false drops the section entirely", () => {
  const result = parseRouterConfig(
    { enabled: false, customRouter: { extensionId: "jitrl" } },
    MODEL_CONFIG,
  );
  assert.deepEqual(result.config, { enabled: false });
});
