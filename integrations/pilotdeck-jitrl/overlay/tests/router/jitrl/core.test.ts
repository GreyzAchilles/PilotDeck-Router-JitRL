/**
 * Core JitRL algorithm equivalence tests (TypeScript port vs the Python
 * jitrl_core reference — hand-computed cases mirrored from
 * jitrl_core/tests/test_state.py, test_memory.py, test_value_policy.py and
 * test_c1_gate.py).
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { intentClass, jaccard, taskSignature } from "../../../src/router/jitrl/state.js";
import { ExperienceMemory, type JitRLEpisode } from "../../../src/router/jitrl/memory.js";
import { estimateValues } from "../../../src/router/jitrl/value.js";
import { chooseTier, clampLogits, modulateLogits } from "../../../src/router/jitrl/policy.js";
import { JitRLEngine } from "../../../src/router/jitrl/engine.js";
import { DEFAULT_JITRL_HYPERPARAMS, TIERS } from "../../../src/router/jitrl/tiers.js";

const ZEROS: Record<string, number> = { simple: 0, medium: 0, complex: 0, reasoning: 0 };

// --------------------------------------------------------------------- state
test("intentClass: ordered keyword rules, first match wins", () => {
  assert.equal(intentClass("你好，今天心情不错"), "chat_qa");
  assert.equal(intentClass("帮我记住：以后所有回复都用中文"), "chat_qa");
  assert.equal(intentClass("给下面这个函数写一个单元测试：\ndef add(a, b):\n    return a+b"), "code_gen");
  assert.equal(
    intentClass("分析下面这组销售数据，统计环比，对比渠道表现：\n一月 A产品 120万 B产品 300万"),
    "data_analysis",
  );
  assert.equal(intentClass("继续"), "continuation");
  assert.equal(intentClass("写一封邮件给老板"), "doc_writing");
  assert.equal(intentClass("completely unrelated words"), "other");
});

test("taskSignature: matches the Python reference token-for-token", () => {
  // Reference values produced by jitrl_core.state.task_signature (Python).
  assert.deepEqual(taskSignature("你好，今天心情不错"), ["你好", "今天", "天心", "心情", "情不", "不错"]);
  assert.deepEqual(taskSignature("继续"), ["继续"]);
  assert.deepEqual(
    taskSignature("给下面这个函数写一个单元测试：\ndef add(a, b):\n    return a+b"),
    ["给下", "下面", "面这", "这个", "个函", "函数", "数写", "写一", "一个", "个单", "单元", "元测", "测试", "def", "add", "a", "b", "<code>"],
  );
  assert.deepEqual(
    taskSignature("请访问 https://example.com/docs and read /tmp/workspace/file.md then answer 42 questions \"quoted long string here\""),
    ["请访", "访问", "<url>", "and", "read", "<path>", "then", "answer", "<num>", "questions", "<str>"],
  );
});

test("taskSignature: fenced + indented code collapse, blank-line absorption", () => {
  assert.deepEqual(
    taskSignature("review this:\n```python\nprint(1)\n```\nand this:\n    indented code line\n    more\n\nafter blank inside block?\noutside 123"),
    ["review", "this", "<code>", "and", "this", "<code>", "after", "blank", "inside", "block", "outside", "<num>"],
  );
});

test("jaccard: empty-vs-empty 1.0, one-empty 0.0, hand case", () => {
  assert.equal(jaccard(new Set(["a", "b"]), new Set(["a", "b", "c"])), 2 / 3);
  assert.equal(jaccard(new Set(), new Set()), 1.0);
  assert.equal(jaccard(new Set(), new Set(["a"])), 0.0);
  assert.equal(jaccard(new Set(["a"]), new Set()), 0.0);
});

// -------------------------------------------------------------------- memory
function entry(
  intent: string,
  tokens: string[],
  tier = "medium",
  g = 0.0,
  episodeId: string | number = 1,
): JitRLEpisode {
  return {
    intent_class: intent,
    signature_tokens: [...tokens].sort(),
    tier,
    G: g,
    ts: 0.0,
    episode_id: episodeId,
  };
}

test("memory: intent bucket filter + threshold + top-k", () => {
  const mem = new ExperienceMemory(10);
  mem.add(entry("code_gen", ["a", "b"], "complex", 1.0, 1));
  mem.add(entry("code_gen", ["a", "c"], "medium", 0.5, 2));
  mem.add(entry("code_gen", ["a", "b", "z"], "simple", 2.0, 3));
  mem.add(entry("data_analysis", ["a", "b"], "reasoning", 3.0, 4));

  const matches = mem.retrieve("code_gen", new Set(["a", "b", "x"]), { k: 10, threshold: 0.0 });
  assert.equal(matches.length, 3);
  assert.ok(matches.every((match) => match.entry.intent_class === "code_gen"));

  // similarity desc: {a,b} vs {a,b,x} = 2/3, {a,b,z} vs {a,b,x} = 0.5, {a,c} vs {a,b,x} = 0.25
  assert.ok(Math.abs(matches[0]!.similarity - 2 / 3) < 1e-12);
  assert.ok(Math.abs(matches[1]!.similarity - 0.5) < 1e-12);
  assert.ok(Math.abs(matches[2]!.similarity - 0.25) < 1e-12);

  const thresholded = mem.retrieve("code_gen", new Set(["a", "b", "x"]), { k: 10, threshold: 0.5 });
  assert.equal(thresholded.length, 2);

  const topK = mem.retrieve("code_gen", new Set(["a", "b", "x"]), { k: 1, threshold: 0.0 });
  assert.equal(topK.length, 1);
});

test("memory: ties keep insertion order (stable sort)", () => {
  const mem = new ExperienceMemory(10);
  mem.add(entry("code_gen", ["a", "b"], "complex", 1.0, "first"));
  mem.add(entry("code_gen", ["a", "b"], "medium", 0.5, "second"));
  const matches = mem.retrieve("code_gen", new Set(["a", "b"]), { k: 10, threshold: 0.5 });
  assert.deepEqual(matches.map((match) => match.entry.episode_id), ["first", "second"]);
});

test("memory: capacity evicts oldest", () => {
  const mem = new ExperienceMemory(3);
  for (let i = 0; i < 5; i += 1) {
    mem.add(entry("other", [`t${i}`], "medium", i, i));
  }
  assert.equal(mem.length, 3);
  assert.deepEqual(mem.entries().map((e) => e.episode_id), [2, 3, 4]);
});

// --------------------------------------------------------------------- value
function makeEntries(): JitRLEpisode[] {
  // N(s): G=1.0 (complex), G=0.0 (complex), G=0.5 (medium)
  return [
    entry("x", ["a"], "complex", 1.0, 1),
    entry("x", ["b"], "complex", 0.0, 2),
    entry("x", ["c"], "medium", 0.5, 3),
  ];
}

function seqRng(values: number[]): { rng: () => number; calls: number } {
  let index = 0;
  let calls = 0;
  return {
    rng: () => {
      calls += 1;
      const value = values[Math.min(index, values.length - 1)]!;
      index += 1;
      return value;
    },
    get calls() {
      return calls;
    },
  };
}

test("value: hand-computed V/Q/A/A_norm (no exploration)", () => {
  const stub = seqRng([0.99, 0.99]);
  const est = estimateValues(makeEntries(), { rng: stub.rng, lam: 0.05, alpha: 5.0 });
  assert.equal(est.V, 0.5);
  assert.equal(est.Q.complex, 0.5);
  assert.equal(est.Q.medium, 0.5);
  assert.equal(est.Q.simple, 0.0);
  assert.equal(est.Q.reasoning, 0.0);
  assert.equal(est.A.simple, -0.5);
  assert.equal(est.A.medium, 0.0);
  assert.equal(est.A.complex, 0.0);
  assert.equal(est.A.reasoning, -0.5);
  assert.ok(Math.abs(est.A_norm.simple! - -0.5 / (0.5 + 1e-8)) < 1e-6);
  assert.ok(Math.abs(est.A_norm.medium!) < 1e-9);
  assert.ok(Math.abs(est.A_norm.complex!) < 1e-9);
  assert.ok(Math.abs(est.A_norm.reasoning! - -1.0) < 1e-6);
  assert.equal(est.nRetrieved, 3);
});

test("value: lambda exploration branch draws in TIERS order", () => {
  // missing tiers (simple, reasoning) draw in TIERS order:
  // simple draws 0.01 < 0.05 -> Q = V + alpha/|N(s)| = 0.5 + 5/3
  const stub = seqRng([0.01, 0.99]);
  const est = estimateValues(makeEntries(), { rng: stub.rng, lam: 0.05, alpha: 5.0 });
  const expectedQ = 0.5 + 5.0 / 3.0;
  assert.ok(Math.abs(est.Q.simple! - expectedQ) < 1e-9);
  assert.equal(est.Q.reasoning, 0.0);
  assert.equal(stub.calls, 2);
  assert.ok(Math.abs(est.A_norm.simple! - 1.0) < 1e-6);
  assert.ok(Math.abs(est.A_norm.reasoning! - -0.5 / (expectedQ - 0.5 + 1e-8)) < 1e-6);
});

test("value: empty memory -> zeros, no modulation", () => {
  const est = estimateValues([], { lam: 0.05, alpha: 5.0 });
  assert.equal(est.V, 0.0);
  assert.equal(est.nRetrieved, 0);
  for (const tier of TIERS) {
    assert.equal(est.A_norm[tier], 0.0);
  }
});

test("value: max|A| == 0 -> A_norm all zero", () => {
  const entries = TIERS.map((tier, index) => entry("x", [`t${index}`], tier, 0.5, index));
  const est = estimateValues(entries, { lam: 0.05, alpha: 5.0 });
  assert.equal(est.V, 0.5);
  for (const tier of TIERS) {
    assert.equal(est.A_norm[tier], 0.0);
  }
});

// -------------------------------------------------------------------- policy
const Z = { simple: 1.0, medium: 2.0, complex: 0.0, reasoning: 1.0 };
const A_NORM = { simple: -1.0, medium: 0.0, complex: 0.0, reasoning: -1.0 };

test("policy: modulation hand case, no flip", () => {
  const zMod = modulateLogits(Z, A_NORM, { beta: 5.0, zMin: -10.0 });
  assert.equal(zMod.simple, 1.0 - 5.0);
  assert.equal(zMod.medium, 2.0);
  assert.equal(zMod.complex, 0.0);
  assert.equal(zMod.reasoning, 1.0 - 5.0);
  assert.equal(chooseTier(zMod), "medium");
});

test("policy: optimistic advantage flips the argmax", () => {
  const aNorm = { simple: 1.0, medium: 0.0, complex: 0.0, reasoning: 0.0 };
  const zMod = modulateLogits(Z, aNorm, { beta: 5.0, zMin: -10.0 });
  assert.equal(zMod.simple, 6.0);
  assert.equal(chooseTier(Z), "medium");
  assert.equal(chooseTier(zMod), "simple");
});

test("policy: z_min clamp on base and modulated logits", () => {
  const clamped = clampLogits({ simple: -11.0, medium: 0.5 }, -10.0);
  assert.equal(clamped.simple, -10.0);
  assert.equal(clamped.medium, 0.5);
  const zMod = modulateLogits(
    { simple: 0.0, medium: 0.0 },
    { simple: -2.5, medium: 0.0 },
    { beta: 5.0, zMin: -10.0 },
  );
  assert.equal(zMod.simple, -10.0);
});

test("policy: deterministic tie-break is first max in TIERS order", () => {
  assert.equal(chooseTier({ simple: 1.0, medium: 1.0, complex: 1.0, reasoning: 1.0 }), "simple");
  assert.equal(chooseTier({ reasoning: 2.0, complex: 2.0 }), "complex");
});

// -------------------------------------------------------------------- engine
const MSG = "分析这批销售数据，统计环比，对比渠道表现";
const LOGITS = { simple: 0.0, medium: 1.0, complex: 0.0, reasoning: 0.0 };

function learnedEngine(
  options?: { minNeighbors?: number; n?: number; reward?: number; rng?: () => number },
): JitRLEngine {
  const engine = new JitRLEngine({
    hyperparams: options?.minNeighbors !== undefined ? { minNeighbors: options.minNeighbors } : undefined,
    ...(options?.rng ? { rng: options.rng } : {}),
  });
  for (let i = 0; i < (options?.n ?? 0); i += 1) {
    engine.learn({
      intent: "data_analysis",
      signatureTokens: taskSignature(MSG),
      tier: "medium",
      reward: options?.reward ?? -0.3,
      episodeId: i + 1,
      ts: 0,
    });
  }
  return engine;
}

test("engine: explicit minNeighbors=1 equals the default bit-for-bit", () => {
  for (const n of [0, 1, 2, 3]) {
    const dDefault = learnedEngine({ n }).decide(MSG, LOGITS);
    const dOne = learnedEngine({ n, minNeighbors: 1 }).decide(MSG, LOGITS);
    assert.deepEqual(dOne, dDefault);
  }
});

test("engine: default modulates whenever n >= 1 (C0 formula)", () => {
  for (const n of [1, 2, 3, 4]) {
    const d = learnedEngine({ n }).decide(MSG, LOGITS);
    assert.equal(d.nRetrieved, n);
    assert.equal(d.minNeighbors, 1);
    assert.equal(d.modulationActive, true);
    assert.equal(d.gateReason, undefined);
    // exactly the C0 formula, unchanged
    const expected = modulateLogits(d.zBase, d.A_norm, {
      beta: DEFAULT_JITRL_HYPERPARAMS.beta,
      zMin: DEFAULT_JITRL_HYPERPARAMS.zMin,
    });
    assert.deepEqual(d.zMod, expected);
    assert.notDeepEqual(d.zMod, d.zBase);
  }
});

test("engine: empty memory is pure baseline (no_neighbors gate)", () => {
  const d = learnedEngine({ n: 0 }).decide(MSG, LOGITS);
  assert.equal(d.nRetrieved, 0);
  assert.equal(d.modulationActive, false);
  assert.equal(d.gateReason, "no_neighbors");
  assert.deepEqual(d.zMod, d.zBase);
  assert.deepEqual(d.A_norm, ZEROS);
  assert.equal(d.chosenTier, d.baseChoice);
  assert.equal(d.flipped, false);
});

test("engine: C1 gate suppresses modulation below min_neighbors", () => {
  const d = learnedEngine({ n: 2, minNeighbors: 3 }).decide(MSG, LOGITS);
  assert.equal(d.nRetrieved, 2);
  assert.equal(d.minNeighbors, 3);
  assert.equal(d.modulationActive, false);
  assert.equal(d.gateReason, "below_min_neighbors");
  assert.deepEqual(d.zMod, d.zBase);
  assert.deepEqual(d.A_norm, ZEROS);
  assert.equal(d.chosenTier, d.baseChoice);
  assert.equal(d.flipped, false);
});

test("engine: gate does not perturb the RNG stream (estimateValues always runs)", () => {
  let draws = 0;
  const makeCountingRng = () => () => {
    draws += 1;
    return 0.99;
  };
  draws = 0;
  learnedEngine({ n: 2, rng: makeCountingRng() }).decide(MSG, LOGITS);
  const drawsC0 = draws;
  draws = 0;
  learnedEngine({ n: 2, minNeighbors: 3, rng: makeCountingRng() }).decide(MSG, LOGITS);
  const drawsC1 = draws;
  assert.equal(drawsC0, drawsC1);
  assert.ok(drawsC0 > 0);
});

test("engine: modulated choice can flip from the base choice", () => {
  // 3 medium episodes with negative reward: A_norm(medium)=0 but the unseen
  // tiers gain positive advantage -> the argmax flips away from medium.
  const d = learnedEngine({ n: 3, reward: -0.3, rng: () => 0.99 }).decide(MSG, LOGITS);
  assert.equal(d.baseChoice, "medium");
  assert.equal(d.chosenTier, "simple");
  assert.equal(d.flipped, true);
});

test("engine: learn stores canonical sorted entries; updateFromMessage matches", () => {
  const engine = new JitRLEngine();
  const stored = engine.learn({
    intent: "data_analysis",
    signatureTokens: ["b", "a", "b"],
    tier: "medium",
    reward: 0.42,
    episodeId: "e1",
    ts: 123.0,
  });
  assert.deepEqual(stored.signature_tokens, ["a", "b", "b"]);
  assert.equal(stored.G, 0.42);
  assert.equal(stored.tier, "medium");
  assert.equal(stored.ts, 123.0);

  const viaMessage = engine.updateFromMessage("继续", "simple", 0.1, "e2", 5.0);
  assert.equal(viaMessage.intent_class, "continuation");
  assert.deepEqual(viaMessage.signature_tokens, taskSignature("继续"));
  assert.equal(engine.memory.length, 2);
});
