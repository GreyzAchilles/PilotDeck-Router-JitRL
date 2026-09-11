/**
 * JitRL reward: costSaving normalization against the most expensive
 * candidate in the same candidate set, and the evaluator quality gate.
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import type {
  CanonicalModelRequest,
  CanonicalModelResponse,
  ModelRuntime,
} from "../../../src/model/index.js";
import {
  computeCostSaving,
  computeJitRLReward,
  episodePrice,
  evaluateQuality,
} from "../../../src/router/jitrl/reward.js";

const PRICING = {
  "test/cheap": { input: 1, output: 1 },
  "test/mid": { input: 3, output: 3 },
  "test/exp": { input: 10, output: 10 },
};

function ref(providerSlashModel: string) {
  const [provider, model] = providerSlashModel.split("/");
  return { id: providerSlashModel, provider: provider!, model: model! };
}

const CANDIDATES = [ref("test/cheap"), ref("test/mid"), ref("test/exp")];

test("episodePrice: (in * priceIn + out * priceOut) / 1e6", () => {
  assert.ok(Math.abs(episodePrice("test", "exp", 100, 50, PRICING) - (100 * 10 + 50 * 10) / 1e6) < 1e-12);
});

test("costSaving: normalized against the most expensive candidate, clamped to [0,1]", () => {
  const saving = computeCostSaving(
    { chosenModel: ref("test/cheap"), candidates: CANDIDATES, usage: { inputTokens: 100, outputTokens: 50 } },
    PRICING,
  );
  // most expensive = test/exp (1500), chosen = test/cheap (150) -> 0.9
  assert.ok(Math.abs(saving - 0.9) < 1e-9);
  assert.ok(saving >= 0 && saving <= 1);

  // Choosing the most expensive candidate saves nothing.
  const zero = computeCostSaving(
    { chosenModel: ref("test/exp"), candidates: CANDIDATES, usage: { inputTokens: 100, outputTokens: 50 } },
    PRICING,
  );
  assert.equal(zero, 0);
});

test("costSaving: identical pricing -> 0; fallback token profile without usage", () => {
  const same = computeCostSaving(
    { chosenModel: ref("test/mid"), candidates: [ref("test/mid")], usage: undefined },
    PRICING,
  );
  assert.equal(same, 0);

  // No usage -> the documented 1500/800 episode profile is used.
  const withProfile = computeCostSaving(
    { chosenModel: ref("test/cheap"), candidates: CANDIDATES, usage: undefined },
    PRICING,
  );
  const expected = (1500 * 10 + 800 * 10 - (1500 * 1 + 800 * 1)) / (1500 * 10 + 800 * 10);
  assert.ok(Math.abs(withProfile - expected) < 1e-9);
});

function fakeEvaluatorRuntime(text: string | Error): ModelRuntime {
  return {
    complete: () =>
      text instanceof Error ? Promise.reject(text) : Promise.resolve({
        role: "assistant",
        content: [{ type: "text", text }],
        finishReason: "stop",
      } as CanonicalModelResponse),
    stream: function* () {} as never,
    getCapabilities: () => ({ maxOutputTokens: 4096 }) as never,
    getMultimodal: () => ({ supportedInputs: ["text"] }) as never,
    getProviderProtocol: () => "openai",
    getProviderBaseUrl: () => undefined,
  };
}

const EVALUATOR = ref("test/eval-model");

test("evaluateQuality: parses 0-10 scores into [0,1]", async () => {
  assert.equal(await evaluateQuality({ userMessage: "q", responseText: "r" }, {
    evaluatorRuntime: fakeEvaluatorRuntime("8"),
    evaluator: EVALUATOR,
  }), 0.8);
  assert.equal(await evaluateQuality({ userMessage: "q", responseText: "r" }, {
    evaluatorRuntime: fakeEvaluatorRuntime(" 7 \n"),
    evaluator: EVALUATOR,
  }), 0.7);
  // Out-of-range or unparsable -> undefined (no learning).
  assert.equal(await evaluateQuality({ userMessage: "q", responseText: "r" }, {
    evaluatorRuntime: fakeEvaluatorRuntime("11"),
    evaluator: EVALUATOR,
  }), undefined);
  assert.equal(await evaluateQuality({ userMessage: "q", responseText: "r" }, {
    evaluatorRuntime: fakeEvaluatorRuntime("great"),
    evaluator: EVALUATOR,
  }), undefined);
  // Request failure -> undefined.
  assert.equal(await evaluateQuality({ userMessage: "q", responseText: "r" }, {
    evaluatorRuntime: fakeEvaluatorRuntime(new Error("boom")),
    evaluator: EVALUATOR,
  }), undefined);
  // No evaluator configured -> undefined.
  assert.equal(await evaluateQuality({ userMessage: "q", responseText: "r" }, {}), undefined);
});

test("computeJitRLReward: 0.6*quality + 0.3*costSaving; evaluator failure -> undefined", async () => {
  const input = {
    userMessage: "分析数据",
    responseText: "答案",
    chosenTier: "simple",
    chosenModel: ref("test/cheap"),
    candidates: CANDIDATES,
    usage: { inputTokens: 100, outputTokens: 50 },
  };
  const reward = await computeJitRLReward(input, {
    evaluatorRuntime: fakeEvaluatorRuntime("8"),
    evaluator: EVALUATOR,
    modelPricing: PRICING,
  });
  assert.ok(reward);
  assert.ok(Math.abs(reward.quality - 0.8) < 1e-12);
  assert.ok(Math.abs(reward.costSaving - 0.9) < 1e-9);
  assert.ok(Math.abs(reward.reward - (0.6 * 0.8 + 0.3 * 0.9)) < 1e-9);

  const failed = await computeJitRLReward(input, {
    evaluatorRuntime: fakeEvaluatorRuntime(new Error("evaluator down")),
    evaluator: EVALUATOR,
    modelPricing: PRICING,
  });
  assert.equal(failed, undefined);
});
