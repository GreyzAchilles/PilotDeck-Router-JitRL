"""Experience memory: (state, tier, reward) episodes with top-k Jaccard retrieval.

Entry schema (spec, verbatim keys):
    {"intent_class", "signature_tokens" (sorted list), "tier", "G", "ts", "episode_id"}

Retrieval (spec): stage 1 filter by same intent_class; stage 2 token-set
Jaccard >= threshold, sorted by similarity desc, top-k. Ties keep insertion
order (stable sort). Capacity: memory_cap, drop oldest on overflow.
"""
from __future__ import annotations

from jitrl_core.state import jaccard


class ExperienceMemory:
    def __init__(self, cap: int = 5000):
        if cap <= 0:
            raise ValueError("cap must be positive")
        self.cap = int(cap)
        self._entries: list[dict] = []

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, entry: dict) -> None:
        """Append an entry; evict oldest beyond cap."""
        self._entries.append(entry)
        if len(self._entries) > self.cap:
            drop = len(self._entries) - self.cap
            del self._entries[:drop]

    def entries(self) -> list[dict]:
        """Copy of all entries (oldest first)."""
        return list(self._entries)

    def retrieve(
        self,
        intent: str,
        token_set: frozenset[str] | set[str],
        k: int = 10,
        threshold: float = 0.5,
    ) -> list[tuple[dict, float]]:
        """Stage-1 intent filter + stage-2 Jaccard threshold, top-k by similarity."""
        scored: list[tuple[dict, float]] = []
        for e in self._entries:
            if e.get("intent_class") != intent:
                continue
            sim = jaccard(frozenset(e.get("signature_tokens", ())), token_set)
            if sim >= threshold:
                scored.append((e, sim))
        # stable sort: ties resolve to insertion order (oldest first)
        scored.sort(key=lambda pair: -pair[1])
        return scored[:k]

    def retrieve_similarities(
        self,
        intent: str,
        token_set: frozenset[str] | set[str],
        k: int = 10,
        threshold: float = 0.5,
    ) -> tuple[list[dict], float]:
        """Convenience: (entries, top_similarity) with top_similarity=0.0 if empty."""
        matches = self.retrieve(intent, token_set, k=k, threshold=threshold)
        top = matches[0][1] if matches else 0.0
        return [e for e, _ in matches], top
