/**
 * Experience memory: (state, tier, reward) episodes with top-k Jaccard
 * retrieval. TypeScript port of `jitrl_core/memory.py` (verbatim semantics).
 *
 * Entry schema (spec, verbatim keys):
 *   { intent_class, signature_tokens (sorted), tier, G, ts, episode_id }
 *
 * Retrieval: stage 1 filter by same intent_class; stage 2 token-set Jaccard
 * >= threshold, sorted by similarity desc, top-k. Ties keep insertion order
 * (stable sort). Capacity: memoryCap, drop oldest on overflow.
 */
import { jaccard } from "./state.js";

export type JitRLEpisode = {
  intent_class: string;
  signature_tokens: string[];
  tier: string;
  G: number;
  ts: number;
  episode_id: string | number;
};

export type RetrievedEpisode = { entry: JitRLEpisode; similarity: number };

export class ExperienceMemory {
  private readonly cap: number;
  private readonly entriesInternal: JitRLEpisode[] = [];

  constructor(cap: number = 5000) {
    if (cap <= 0) {
      throw new Error("cap must be positive");
    }
    this.cap = Math.floor(cap);
  }

  get length(): number {
    return this.entriesInternal.length;
  }

  /** Append an entry; evict oldest beyond cap. */
  add(entry: JitRLEpisode): void {
    this.entriesInternal.push(entry);
    if (this.entriesInternal.length > this.cap) {
      const drop = this.entriesInternal.length - this.cap;
      this.entriesInternal.splice(0, drop);
    }
  }

  /** Copy of all entries (oldest first). */
  entries(): JitRLEpisode[] {
    return [...this.entriesInternal];
  }

  /**
   * Stage-1 intent filter + stage-2 Jaccard threshold, top-k by similarity.
   * Stable sort: ties resolve to insertion order (oldest first).
   */
  retrieve(
    intent: string,
    tokenSet: ReadonlySet<string>,
    options?: { k?: number; threshold?: number },
  ): RetrievedEpisode[] {
    const k = options?.k ?? 10;
    const threshold = options?.threshold ?? 0.5;
    const scored: RetrievedEpisode[] = [];
    for (const entry of this.entriesInternal) {
      if (entry.intent_class !== intent) {
        continue;
      }
      const sim = jaccard(new Set(entry.signature_tokens ?? []), tokenSet);
      if (sim >= threshold) {
        scored.push({ entry, similarity: sim });
      }
    }
    scored.sort((a, b) => b.similarity - a.similarity);
    return scored.slice(0, k);
  }
}
