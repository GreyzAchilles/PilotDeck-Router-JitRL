/**
 * JitRL judge client: Method-B-equivalent four scoring requests through the
 * injected ModelRuntime, score->logit conversion, and the deterministic
 * fallback (no judge configured / request failure / unparsable reply).
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import type {
  CanonicalModelRequest,
  CanonicalModelResponse,
  ModelRuntime,
} from "../../../src/model/index.js";
import { JitRLJudgeClient, scoreToLogit } from "../../../src/router/jitrl/judge.js";
import { TIERS } from "../../../src/router/jitrl/tiers.js";

function ref(providerSlashModel: string) {
  const [provider, model] = providerSlashModel.split("/");
  return { id: providerSlashModel, provider: provider!, model: model! };
}

const JUDGE = ref("test/judge-model");

function fakeCompleteRuntime(handler: (request: CanonicalModelRequest) => CanonicalModelResponse | Promise<CanonicalModelResponse>): ModelRuntime {
  return {
    complete: (request) => Promise.resolve(handler(request)),
    stream: function* () {
      // judges never stream
    } as never,
    getCapabilities: () => ({ maxOutputTokens: 4096 }) as never,
    getMultimodal: () => ({ supportedInputs: ["text"] }) as never,
    getProviderProtocol: () => "openai",
    getProviderBaseUrl: () => undefined,
  };
}

test("scoreToLogit: symmetric midpoint-bin logits", () => {
  assert.ok(Math.abs(scoreToLogit(5)) < 1e-12);
  assert.ok(Math.abs(scoreToLogit(10) - Math.log(21)) < 1e-12);
  assert.ok(Math.abs(scoreToLogit(0) + Math.log(21)) < 1e-12);
  assert.ok(scoreToLogit(10) > scoreToLogit(7));
  // out-of-range scores are clamped, not rejected
  assert.equal(scoreToLogit(99), scoreToLogit(10));
});

test("judge: four parallel tier-scoring requests produce a full logit vector", async () => {
  const prompts: string[] = [];
  const runtime = fakeCompleteRuntime((request) => {
    const prompt = request.messages[0]!.content.map((b) => (b.type === "text" ? b.text : "")).join("");
    prompts.push(prompt);
    const tier = /"([a-z]+)"/.exec(prompt)?.[1] ?? "medium";
    return {
      role: "assistant",
      content: [{ type: "text", text: tier === "reasoning" ? "10" : "2" }],
      finishReason: "stop",
    } as CanonicalModelResponse;
  });

  const client = new JitRLJudgeClient({ judgeRuntime: runtime, judge: JUDGE, timeoutMs: 1000 });
  const result = await client.judge({ userMessage: "分析这批销售数据并写总结报告" });

  assert.equal(result.source, "model");
  assert.equal(prompts.length, 4);
  for (const tier of TIERS) {
    assert.ok(prompts.some((prompt) => prompt.includes(`"${tier}"`)));
    assert.ok(Number.isFinite(result.tierLogits[tier]));
  }
  assert.equal(result.chosenTier, "reasoning");
  assert.ok(result.tierLogits.reasoning! > result.tierLogits.simple!);
  assert.equal(result.rawOutput, "<tier>reasoning</tier>");
});

test("judge: any per-tier failure degrades to the deterministic fallback", async () => {
  let calls = 0;
  const runtime = fakeCompleteRuntime(() => {
    calls += 1;
    if (calls === 3) {
      throw new Error("judge hiccup");
    }
    return { role: "assistant" as const, content: [{ type: "text" as const, text: "5" }], finishReason: "stop" as const };
  });

  const client = new JitRLJudgeClient({ judgeRuntime: runtime, judge: JUDGE, timeoutMs: 1000 });
  const result = await client.judge({ userMessage: "分析这批销售数据并写总结报告" });

  assert.equal(result.source, "fallback");
  assert.equal(calls, 4);
  for (const tier of TIERS) {
    assert.ok(Number.isFinite(result.tierLogits[tier]));
  }
  assert.ok(result.fallbackReason);
});

test("judge: unparsable reply also falls back", async () => {
  const runtime = fakeCompleteRuntime(() => ({
    role: "assistant",
    content: [{ type: "text", text: "I think medium fits well here" }],
    finishReason: "stop",
  }));
  const client = new JitRLJudgeClient({ judgeRuntime: runtime, judge: JUDGE, timeoutMs: 1000 });
  const result = await client.judge({ userMessage: "hello" });
  assert.equal(result.source, "fallback");
});

test("judge: no judge configured -> deterministic fallback, reproducible", async () => {
  const clientA = new JitRLJudgeClient({ fallbackSeed: 7 });
  const clientB = new JitRLJudgeClient({ fallbackSeed: 7 });
  const message = "给下面这个函数写一个单元测试：\ndef add(a, b):\n    return a+b";
  const a = await clientA.judge({ userMessage: message });
  const b = await clientB.judge({ userMessage: message });
  assert.equal(a.source, "fallback");
  assert.equal(a.fallbackReason, "no judge model configured");
  assert.deepEqual(a, b);

  // Code cues ("def ") mass the complex tier in the fallback heuristic.
  assert.equal(a.chosenTier, "complex");
  assert.ok(a.tierLogits.complex! > a.tierLogits.medium!);
});

test("judge: fallback continuation echo keeps the previous tier", async () => {
  const client = new JitRLJudgeClient({ fallbackSeed: 42 });
  const result = await client.judge({ userMessage: "继续", previousTier: "reasoning" });
  assert.equal(result.chosenTier, "reasoning");
  assert.ok(result.tierLogits.reasoning! > result.tierLogits.simple!);
});
