/**
 * Persistent JitRL memory: `<pilotHome>/router/jitrl-memory.json`.
 *
 * - Atomic write (tmp file + rename) with a trailing-edge throttle, so burst
 *   learns coalesce into one flush.
 * - Shared per-path state cache: `PluginRuntime.lookupRouter()` may create a
 *   fresh custom-router instance on every router decide, so the engine, its
 *   seeded RNG stream and the in-memory episodes MUST be cached per resolved
 *   memory path (test injectable) rather than per plugin instance.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { resolvePilotHome } from "../../pilot/paths.js";
import { ExperienceMemory, type JitRLEpisode } from "./memory.js";
import { JitRLEngine } from "./engine.js";
import type { JitRLHyperparams } from "./tiers.js";

const DEFAULT_MEMORY_FILE = "jitrl-memory.json";
const DEFAULT_FLUSH_DELAY_MS = 500;

export type JitRLMemoryFileShape = {
  version: 1;
  entries: JitRLEpisode[];
};

export class JitRLMemoryStore {
  private readonly filePath: string;
  private flushTimer: NodeJS.Timeout | undefined;
  private pendingEntries: JitRLEpisode[] | undefined;

  constructor(filePath: string) {
    this.filePath = filePath;
  }

  get path(): string {
    return this.filePath;
  }

  load(): JitRLEpisode[] {
    try {
      const raw = fs.readFileSync(this.filePath, "utf8");
      const parsed = JSON.parse(raw) as Partial<JitRLMemoryFileShape>;
      if (!parsed || !Array.isArray(parsed.entries)) {
        return [];
      }
      return parsed.entries.filter(isEpisode);
    } catch {
      // Missing or corrupt file — start from empty (documented recovery).
      return [];
    }
  }

  /** Synchronous immediate save (atomic: tmp + rename). */
  saveSync(entries: JitRLEpisode[]): void {
    const payload: JitRLMemoryFileShape = { version: 1, entries };
    try {
      fs.mkdirSync(path.dirname(this.filePath), { recursive: true });
    } catch {
      // ignore — writeFileSync below surfaces the real error
    }
    const tmpPath = `${this.filePath}.tmp-${process.pid}-${Date.now()}`;
    try {
      fs.writeFileSync(tmpPath, JSON.stringify(payload), "utf8");
      fs.renameSync(tmpPath, this.filePath);
    } catch {
      try {
        fs.rmSync(tmpPath, { force: true });
      } catch {
        // best-effort cleanup
      }
      throw new Error(`JitRL memory persistence failed at ${this.filePath}`);
    }
  }

  /** Schedule a throttled save; bursty learns coalesce into one flush. */
  scheduleFlush(readEntries: () => JitRLEpisode[], delayMs: number = DEFAULT_FLUSH_DELAY_MS): void {
    this.pendingEntries = readEntries();
    if (this.flushTimer) {
      return;
    }
    this.flushTimer = setTimeout(() => {
      this.flushTimer = undefined;
      const pending = this.pendingEntries;
      this.pendingEntries = undefined;
      if (pending) {
        try {
          this.saveSync(pending);
        } catch (error) {
          console.warn(
            "[jitrl] memory flush failed:",
            error instanceof Error ? error.message : String(error),
          );
        }
      }
    }, delayMs);
    // Do not keep the event loop alive for a background flush.
    this.flushTimer.unref?.();
  }

  /** Flush any pending throttled write immediately (used by tests/shutdown). */
  flushNow(readEntries?: () => JitRLEpisode[]): void {
    const pending = readEntries ? readEntries() : this.pendingEntries;
    this.pendingEntries = undefined;
    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
      this.flushTimer = undefined;
    }
    if (pending) {
      this.saveSync(pending);
    }
  }
}

function isEpisode(value: unknown): value is JitRLEpisode {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const entry = value as Partial<JitRLEpisode>;
  return (
    typeof entry.intent_class === "string" &&
    Array.isArray(entry.signature_tokens) &&
    entry.signature_tokens.every((token) => typeof token === "string") &&
    typeof entry.tier === "string" &&
    typeof entry.G === "number" &&
    typeof entry.ts === "number" &&
    (typeof entry.episode_id === "string" || typeof entry.episode_id === "number")
  );
}

export type SharedJitRLState = {
  memory: ExperienceMemory;
  engine: JitRLEngine;
  store: JitRLMemoryStore;
  hyperparams: JitRLHyperparams;
  memoryPath: string;
};

const sharedStates = new Map<string, SharedJitRLState>();

export function defaultJitRLMemoryPath(): string {
  return path.join(resolvePilotHome(), "router", DEFAULT_MEMORY_FILE);
}

function sameHyperparams(a: JitRLHyperparams, b: JitRLHyperparams): boolean {
  return (
    a.k === b.k &&
    a.beta === b.beta &&
    a.lam === b.lam &&
    a.alpha === b.alpha &&
    a.jaccardThreshold === b.jaccardThreshold &&
    a.zMin === b.zMin &&
    a.memoryCap === b.memoryCap &&
    a.seed === b.seed &&
    a.minNeighbors === b.minNeighbors
  );
}

/**
 * Get (or create) the process-wide shared state for a memory path. The
 * in-memory episodes survive plugin-instance churn; on a hyperparameter
 * change the engine is rebuilt around the SAME episodes (memory contents are
 * preserved; only the seeded RNG stream restarts — documented deviation).
 */
export function getSharedJitRLState(input: {
  memoryPath?: string;
  hyperparams?: Partial<JitRLHyperparams>;
}): SharedJitRLState {
  const memoryPath = input.memoryPath ?? defaultJitRLMemoryPath();
  const existing = sharedStates.get(memoryPath);
  if (existing && sameHyperparams(existing.hyperparams, resolveHyperparamsFor(input))) {
    return existing;
  }
  const hyperparams = resolveHyperparamsFor(input);
  const store = existing?.store ?? new JitRLMemoryStore(memoryPath);
  const seedEntries = existing ? existing.memory.entries() : store.load();
  const memory = new ExperienceMemory(hyperparams.memoryCap);
  for (const entry of seedEntries) {
    memory.add(entry);
  }
  const engine = new JitRLEngine({ hyperparams, memory });
  const state: SharedJitRLState = { memory, engine, store, hyperparams, memoryPath };
  sharedStates.set(memoryPath, state);
  return state;
}

function resolveHyperparamsFor(input: { hyperparams?: Partial<JitRLHyperparams> }): JitRLHyperparams {
  const resolved: JitRLHyperparams = {
    k: input.hyperparams?.k ?? 10,
    beta: input.hyperparams?.beta ?? 5.0,
    lam: input.hyperparams?.lam ?? 0.05,
    alpha: input.hyperparams?.alpha ?? 5.0,
    jaccardThreshold: input.hyperparams?.jaccardThreshold ?? 0.5,
    zMin: input.hyperparams?.zMin ?? -10.0,
    memoryCap: input.hyperparams?.memoryCap ?? 5000,
    seed: input.hyperparams?.seed ?? 42,
    minNeighbors: input.hyperparams?.minNeighbors ?? 1,
  };
  return resolved;
}

/** Test hook: drop all cached shared states. */
export function resetSharedJitRLStates(): void {
  for (const state of sharedStates.values()) {
    try {
      state.store.flushNow(() => state.memory.entries());
    } catch {
      // best-effort flush during reset
    }
  }
  sharedStates.clear();
}
