/**
 * JitRL memory persistence: atomic writes, throttled flush, cap eviction on
 * disk, and the process-wide shared state cache that survives
 * `lookupRouter()` instance churn.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

import { JitRLMemoryStore, getSharedJitRLState, resetSharedJitRLStates } from "../../../src/router/jitrl/memoryStore.js";
import { taskSignature } from "../../../src/router/jitrl/state.js";
import type { JitRLEpisode } from "../../../src/router/jitrl/memory.js";

function tmpDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "jitrl-memory-"));
}

function episode(tier: string, g: number, id: string | number): JitRLEpisode {
  return {
    intent_class: "data_analysis",
    signature_tokens: ["a", "b"],
    tier,
    G: g,
    ts: 1.0,
    episode_id: id,
  };
}

test("store: saveSync + load round trip (atomic, no leftover tmp files)", () => {
  const dir = tmpDir();
  const filePath = path.join(dir, "router", "jitrl-memory.json");
  const store = new JitRLMemoryStore(filePath);
  store.saveSync([episode("medium", 0.42, "e1"), episode("simple", 0.1, 2)]);

  const loaded = store.load();
  assert.equal(loaded.length, 2);
  assert.deepEqual(loaded[0], { intent_class: "data_analysis", signature_tokens: ["a", "b"], tier: "medium", G: 0.42, ts: 1.0, episode_id: "e1" });

  const leftovers = fs.readdirSync(path.dirname(filePath)).filter((name) => name.includes(".tmp-"));
  assert.deepEqual(leftovers, []);

  const payload = JSON.parse(fs.readFileSync(filePath, "utf8"));
  assert.equal(payload.version, 1);
  assert.equal(payload.entries.length, 2);
});

test("store: load tolerates missing/corrupt files", () => {
  const dir = tmpDir();
  const store = new JitRLMemoryStore(path.join(dir, "missing.json"));
  assert.deepEqual(store.load(), []);

  const corrupt = path.join(dir, "corrupt.json");
  fs.writeFileSync(corrupt, "{not json", "utf8");
  assert.deepEqual(new JitRLMemoryStore(corrupt).load(), []);

  const badShape = path.join(dir, "bad.json");
  fs.writeFileSync(badShape, JSON.stringify({ version: 1, entries: [{ nope: true }] }), "utf8");
  assert.deepEqual(new JitRLMemoryStore(badShape).load(), []);
});

test("store: throttled flush coalesces bursty saves", async () => {
  const dir = tmpDir();
  const filePath = path.join(dir, "jitrl-memory.json");
  const store = new JitRLMemoryStore(filePath);

  let writes = 0;
  const originalSave = store.saveSync.bind(store);
  (store as unknown as { saveSync: (entries: JitRLEpisode[]) => void }).saveSync = (entries) => {
    writes += 1;
    originalSave(entries);
  };

  const entriesA = [episode("medium", 0.1, 1)];
  const entriesB = [episode("medium", 0.1, 1), episode("simple", 0.2, 2)];
  store.scheduleFlush(() => entriesA, 50);
  store.scheduleFlush(() => entriesB, 50);
  await new Promise((resolve) => setTimeout(resolve, 150));

  assert.equal(writes, 1);
  assert.equal(store.load().length, 2);
});

test("store: flushNow writes pending entries immediately", () => {
  const dir = tmpDir();
  const store = new JitRLMemoryStore(path.join(dir, "jitrl-memory.json"));
  store.flushNow(() => [episode("reasoning", 0.9, "r1")]);
  assert.equal(store.load().length, 1);
  assert.equal(store.load()[0]!.tier, "reasoning");
});

test("shared state: identical instances across lookupRouter-style churn", () => {
  const dir = tmpDir();
  const memoryPath = path.join(dir, "jitrl-memory.json");
  try {
    const first = getSharedJitRLState({ memoryPath });
    const second = getSharedJitRLState({ memoryPath });
    assert.equal(second, first);

    // Simulate PluginRuntime.lookupRouter() building a fresh custom router
    // per decide: the engine instance must be the shared one.
    first.engine.updateFromMessage("分析这批销售数据，统计环比", "medium", 0.5, "e1", 1);
    assert.equal(second.memory.length, 1);
    assert.equal(second.engine, first.engine);
  } finally {
    resetSharedJitRLStates();
  }
});

test("shared state: persists across resets and enforces cap on disk", () => {
  const dir = tmpDir();
  const memoryPath = path.join(dir, "jitrl-memory.json");
  try {
    const state = getSharedJitRLState({ memoryPath, hyperparams: { memoryCap: 5 } });
    for (let i = 0; i < 8; i += 1) {
      state.engine.updateFromMessage("分析这批销售数据，统计环比", "medium", 0.1 * i, `e${i}`, i);
    }
    assert.equal(state.memory.length, 5);
    state.store.flushNow(() => state.memory.entries());
    assert.equal(new JitRLMemoryStore(memoryPath).load().length, 5);

    // A fresh shared state (post-reset) reloads the persisted 5 entries.
    resetSharedJitRLStates();
    const reloaded = getSharedJitRLState({ memoryPath, hyperparams: { memoryCap: 5 } });
    assert.equal(reloaded.memory.length, 5);
  } finally {
    resetSharedJitRLStates();
  }
});

test("shared state: hyperparameter change rebuilds the engine but keeps episodes", () => {
  const dir = tmpDir();
  const memoryPath = path.join(dir, "jitrl-memory.json");
  try {
    const state = getSharedJitRLState({ memoryPath, hyperparams: { minNeighbors: 1 } });
    state.engine.updateFromMessage("分析这批销售数据，统计环比", "medium", 0.5, "e1", 1);
    const tuned = getSharedJitRLState({ memoryPath, hyperparams: { minNeighbors: 3, beta: 2 } });
    assert.notEqual(tuned.engine, state.engine);
    assert.equal(tuned.memory.length, 1);
    assert.equal(tuned.hyperparams.minNeighbors, 3);
    assert.equal(tuned.hyperparams.beta, 2);
  } finally {
    resetSharedJitRLStates();
  }
});

test("learned episodes use canonical signature tokens", () => {
  const dir = tmpDir();
  const memoryPath = path.join(dir, "jitrl-memory.json");
  try {
    const state = getSharedJitRLState({ memoryPath });
    const message = "给下面这个函数写一个单元测试：\ndef add(a, b):\n    return a+b";
    state.engine.updateFromMessage(message, "medium", 0.7, "e-canonical", 1);
    const stored = state.memory.entries()[0]!;
    assert.equal(stored.intent_class, "code_gen");
    assert.deepEqual(stored.signature_tokens, [...taskSignature(message)].sort());
  } finally {
    resetSharedJitRLStates();
  }
});
