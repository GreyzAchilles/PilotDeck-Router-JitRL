/**
 * End-to-end JitRL custom-router integration through the official
 * RouterRuntime surface:
 *   - decide() routes via the jitrl contribution (4-tier judge -> engine ->
 *     tier candidate model), never recursing through the Router for judge
 *     requests;
 *   - onTurnOutcome fires after custom-routed execution (success AND
 *     failure) with session/turn/user message/decision/usage/response/error;
 *   - successful turns learn (reward = 0.6*quality + 0.3*costSaving);
 *   - evaluator failures and execution failures write NO memory.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

import type {
  CanonicalModelEvent,
  CanonicalModelRequest,
  CanonicalModelResponse,
  ModelRuntime,
} from "../../../src/model/index.js";
import { createRouterRuntime } from "../../../src/router/RouterRuntime.js";
import type { RouterConfig, RouterModelRef } from "../../../src/router/config/schema.js";
import type { CustomRouterTurnOutcome } from "../../../src/router/customRouter/customRouter.js";
import { createJitrlCustomRouter } from "../../../src/router/jitrl/router.js";
import { getSharedJitRLState, resetSharedJitRLStates } from "../../../src/router/jitrl/memoryStore.js";

function ref(providerSlashModel: string): RouterModelRef {
  const [provider, model] = providerSlashModel.split("/");
  return { id: providerSlashModel, provider: provider!, model: model! };
}

function userRequest(text: string): CanonicalModelRequest {
  return {
    provider: "test",
    model: "gpt-4o",
    messages: [{ role: "user", content: [{ type: "text", text }] }],
    stream: true,
  };
}

type FakeBehavior = {
  judgeScores?: (tier: string) => number | undefined;
  evaluatorScore?: string;
  failJudge?: boolean;
  failEvaluator?: boolean;
  failStream?: boolean;
};

function makeFakeRuntime(behavior: FakeBehavior) {
  const calls = { complete: [] as Array<{ provider: string; model: string; prompt: string }>, stream: 0 };
  const runtime: ModelRuntime = {
    async complete(request: CanonicalModelRequest): Promise<CanonicalModelResponse> {
      const prompt = request.messages[0]?.content
        ?.map((block) => (block.type === "text" ? block.text : ""))
        .join("") ?? "";
      calls.complete.push({ provider: request.provider, model: request.model, prompt });
      if (request.model === "judge-model") {
        if (behavior.failJudge) {
          throw new Error("judge down");
        }
        const tierMatch = /"([a-z]+)"/.exec(prompt);
        const tier = tierMatch?.[1] ?? "medium";
        const score = behavior.judgeScores?.(tier) ?? (tier === "medium" ? 9 : 2);
        return {
          role: "assistant",
          content: [{ type: "text", text: String(score) }],
          finishReason: "stop",
        };
      }
      if (request.model === "eval-model") {
        if (behavior.failEvaluator) {
          throw new Error("evaluator down");
        }
        return {
          role: "assistant",
          content: [{ type: "text", text: behavior.evaluatorScore ?? "8" }],
          finishReason: "stop",
        };
      }
      throw new Error(`unexpected complete() model ${request.model}`);
    },
    async *stream(request: CanonicalModelRequest): AsyncGenerator<CanonicalModelEvent> {
      calls.stream += 1;
      if (behavior.failStream) {
        yield {
          type: "error",
          error: {
            provider: request.provider,
            protocol: "openai",
            code: "server_error",
            message: "model exploded",
            retryable: false,
          },
        };
        return;
      }
      yield { type: "message_start", role: "assistant" };
      yield { type: "text_delta", text: "The analysis is done." };
      yield { type: "message_end", finishReason: "stop" };
      yield { type: "usage", usage: { inputTokens: 100, outputTokens: 50, totalTokens: 150 } };
    },
    getCapabilities: () => ({ maxOutputTokens: 4096 }) as never,
    getMultimodal: () => ({ supportedInputs: ["text"] }) as never,
    getProviderProtocol: () => "openai",
    getProviderBaseUrl: () => undefined,
  };
  return { runtime, calls };
}

function buildConfig(memoryPath: string): RouterConfig {
  return {
    enabled: true,
    scenarios: { default: ref("test/gpt-4o") },
    customRouter: {
      extensionId: "jitrl",
      judge: ref("test/judge-model"),
      evaluator: ref("test/eval-model"),
      tiers: {
        simple: ref("test/gpt-4o-mini"),
        medium: ref("test/gpt-4o"),
        complex: ref("test/claude-opus"),
        reasoning: ref("test/claude-opus"),
      },
      memoryPath,
    },
  };
}

function spyRegistry(captured: CustomRouterTurnOutcome[]) {
  const jitrl = createJitrlCustomRouter();
  const learn = jitrl.onTurnOutcome?.bind(jitrl);
  return {
    lookupRouter: () => ({
      ...jitrl,
      onTurnOutcome: async (outcome: CustomRouterTurnOutcome) => {
        captured.push(outcome);
        await learn?.(outcome);
      },
    }),
  };
}

async function collectEvents(iterable: AsyncIterable<CanonicalModelEvent>): Promise<CanonicalModelEvent[]> {
  const events: CanonicalModelEvent[] = [];
  for await (const event of iterable) {
    events.push(event);
  }
  return events;
}

const USER_MESSAGE = "分析下面这组销售数据，统计环比，对比渠道表现";

async function runTurn(
  options: { behavior: FakeBehavior; memoryPath: string },
): Promise<{
  decision: Awaited<ReturnType<ReturnType<typeof createRouterRuntime>["decide"]>>;
  events: CanonicalModelEvent[];
  captured: CustomRouterTurnOutcome[];
  calls: { complete: Array<{ provider: string; model: string; prompt: string }>; stream: number };
}> {
  const { runtime, calls } = makeFakeRuntime(options.behavior);
  const captured: CustomRouterTurnOutcome[] = [];
  const routerRuntime = createRouterRuntime(buildConfig(options.memoryPath), {
    modelRuntime: runtime,
    customRouterRegistry: spyRegistry(captured),
  });
  const request = userRequest(USER_MESSAGE);
  const decision = await routerRuntime.decide({
    request,
    sessionId: "session-1",
    isMainAgent: true,
  });
  const events = await collectEvents(
    routerRuntime.execute(decision, request, { sessionId: "session-1", turnId: "turn-1" }),
  );
  await routerRuntime.shutdown();
  return { decision, events, captured, calls };
}

test("custom routing: judge scores -> engine -> tier candidate, no Router recursion", async () => {
  const memoryPath = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "jitrl-e2e-")), "memory.json");
  try {
    const { decision, calls } = await runTurn({ behavior: {}, memoryPath });
    assert.equal(decision.resolvedFrom, "custom");
    assert.equal(decision.provider, "test");
    assert.equal(decision.model, "gpt-4o");
    assert.equal(decision.tokenSaverTier, "medium");

    // Four tier-scoring judge requests, all through ModelRuntime.complete.
    const judgeCalls = calls.complete.filter((call) => call.model === "judge-model");
    assert.equal(judgeCalls.length, 4);
    for (const tier of ["simple", "medium", "complex", "reasoning"]) {
      assert.ok(judgeCalls.some((call) => call.prompt.includes(`"${tier}"`)), `missing judge prompt for ${tier}`);
    }
    // The routed completion went through stream() — exactly once.
    assert.equal(calls.stream, 1);
  } finally {
    resetSharedJitRLStates();
  }
});

test("successful turn: onTurnOutcome fires and learns reward = 0.6*quality + 0.3*costSaving", async () => {
  const memoryPath = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "jitrl-e2e-")), "memory.json");
  try {
    const { events, captured } = await runTurn({ behavior: {}, memoryPath });
    assert.ok(events.some((event) => event.type === "text_delta"));

    assert.equal(captured.length, 1);
    const outcome = captured[0]!;
    assert.equal(outcome.sessionId, "session-1");
    assert.equal(outcome.turnId, "turn-1");
    assert.equal(outcome.userMessage, USER_MESSAGE);
    assert.equal(outcome.decision.tokenSaverTier, "medium");
    assert.equal(outcome.error, undefined);
    assert.equal(outcome.response, "The analysis is done.");
    assert.equal(outcome.usage?.inputTokens, 100);
    assert.equal(outcome.usage?.outputTokens, 50);

    const shared = getSharedJitRLState({ memoryPath });
    assert.equal(shared.memory.length, 1);
    const episode = shared.memory.entries()[0]!;
    assert.equal(episode.tier, "medium");
    assert.equal(episode.intent_class, "data_analysis");
    assert.ok(episode.signature_tokens.length > 0);

    // quality = 8/10 = 0.8; candidates priced (1500*in + 800*out profile
    // replaced by observed usage 100/50): most expensive = claude-opus
    // (15, 75), chosen = gpt-4o (2.5, 10) -> costSaving = 4500/5250.
    const quality = 0.8;
    const costSaving = (100 * 15 + 50 * 75 - (100 * 2.5 + 50 * 10)) / (100 * 15 + 50 * 75);
    const expected = 0.6 * quality + 0.3 * costSaving;
    assert.ok(Math.abs(episode.G - expected) < 1e-9, `G=${episode.G} expected=${expected}`);

    // Persisted to disk.
    shared.store.flushNow(() => shared.memory.entries());
    const persisted = JSON.parse(fs.readFileSync(memoryPath, "utf8"));
    assert.equal(persisted.entries.length, 1);
    assert.equal(persisted.entries[0].tier, "medium");
  } finally {
    resetSharedJitRLStates();
  }
});

test("evaluator failure: outcome fires but memory is NOT written", async () => {
  const memoryPath = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "jitrl-e2e-")), "memory.json");
  try {
    const { captured } = await runTurn({ behavior: { failEvaluator: true }, memoryPath });
    assert.equal(captured.length, 1);
    assert.equal(captured[0]!.decision.tokenSaverTier, "medium");
    const shared = getSharedJitRLState({ memoryPath });
    assert.equal(shared.memory.length, 0);
    assert.equal(fs.existsSync(memoryPath), false);
  } finally {
    resetSharedJitRLStates();
  }
});

test("execution failure: onTurnOutcome fires with the error, no memory write", async () => {
  const memoryPath = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "jitrl-e2e-")), "memory.json");
  try {
    const { events, captured } = await runTurn({ behavior: { failStream: true }, memoryPath });
    assert.ok(events.some((event) => event.type === "error"));

    assert.equal(captured.length, 1);
    const outcome = captured[0]!;
    assert.equal(outcome.error?.code, "server_error");
    assert.equal(outcome.decision.tokenSaverTier, "medium");

    const shared = getSharedJitRLState({ memoryPath });
    assert.equal(shared.memory.length, 0);
  } finally {
    resetSharedJitRLStates();
  }
});

test("judge failure: deterministic fallback still yields a 4-tier decision", async () => {
  const memoryPath = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "jitrl-e2e-")), "memory.json");
  try {
    const { decision, calls } = await runTurn({
      behavior: {
        failJudge: true,
        // no evaluator either -> no memory write, but routing must succeed
      },
      memoryPath,
    });
    assert.equal(decision.resolvedFrom, "custom");
    // The deterministic fallback judge has no cue for this data-analysis
    // message, so its systematic medium bias wins -> gpt-4o candidate.
    assert.equal(decision.tokenSaverTier, "medium");
    assert.equal(decision.model, "gpt-4o");
    const judgeCalls = calls.complete.filter((call) => call.model === "judge-model");
    assert.equal(judgeCalls.length, 4);
  } finally {
    resetSharedJitRLStates();
  }
});
