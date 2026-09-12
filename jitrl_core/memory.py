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


# ---------------------------------------------------------------------------
# M1 lifecycle memory (docs/optimization/MEMORY-LOOP-DECISIONS.md section 4)
# ---------------------------------------------------------------------------
#: Entry lifecycle statuses. Only ACTIVE entries are retrieved; provisional
#: and quarantined entries stay in memory (audit/memory-out) but never
#: modulate routing until promoted.
ENTRY_ACTIVE = "active"
ENTRY_PROVISIONAL = "provisional"
ENTRY_QUARANTINE = "quarantine"
ENTRY_STATUSES = (ENTRY_ACTIVE, ENTRY_PROVISIONAL, ENTRY_QUARANTINE)


class LifecycleMemory(ExperienceMemory):
    """ExperienceMemory + active/provisional/quarantine sidecar (M1).

    Entry dicts keep the frozen 6-key schema; status and metadata live in
    sidecar maps keyed by id(entry). Plain add() defaults to ACTIVE, so a
    LifecycleMemory used through the legacy path behaves exactly like
    ExperienceMemory. Statuses are pruned on add, so evicted entries never
    leave stale keys behind (an evicted id cannot be reused by the entry
    that triggered its own eviction: the new dict is allocated first)."""

    def __init__(self, cap: int = 5000):
        super().__init__(cap)
        self._status: dict[int, str] = {}
        self._meta: dict[int, dict] = {}

    # -- writes ------------------------------------------------------------
    def add(self, entry: dict, status: str = ENTRY_ACTIVE,
            meta: dict | None = None) -> None:
        if status not in ENTRY_STATUSES:
            raise ValueError(f"unknown entry status: {status!r}")
        super().add(entry)
        self._status[id(entry)] = status
        if meta is not None:
            self._meta[id(entry)] = dict(meta)
        self._prune()

    def add_with_status(self, entry: dict, status: str,
                        meta: dict | None = None) -> None:
        self.add(entry, status=status, meta=meta)

    # -- status access -----------------------------------------------------
    def status_of(self, entry: dict) -> str:
        return self._status.get(id(entry), ENTRY_ACTIVE)

    def meta_of(self, entry: dict) -> dict:
        return self._meta.get(id(entry), {})

    def set_status(self, entry: dict, status: str) -> None:
        if status not in ENTRY_STATUSES:
            raise ValueError(f"unknown entry status: {status!r}")
        self._status[id(entry)] = status

    def promote(self, entry: dict) -> None:
        self.set_status(entry, ENTRY_ACTIVE)

    def quarantine(self, entry: dict) -> None:
        self.set_status(entry, ENTRY_QUARANTINE)

    def status_counts(self) -> dict[str, int]:
        counts = {s: 0 for s in ENTRY_STATUSES}
        for e in self._entries:
            counts[self.status_of(e)] += 1
        return counts

    # -- retrieval: ACTIVE only ---------------------------------------------
    def retrieve(
        self,
        intent: str,
        token_set: frozenset[str] | set[str],
        k: int = 10,
        threshold: float = 0.5,
    ) -> list[tuple[dict, float]]:
        """Same scoring/ordering as the base class, but provisional and
        quarantined entries are filtered out BEFORE the top-k cut."""
        wide = super().retrieve(intent, token_set,
                                k=max(k, len(self._entries)),
                                threshold=threshold)
        active = [(e, sim) for e, sim in wide if self.status_of(e) == ENTRY_ACTIVE]
        return active[:k]

    def _prune(self) -> None:
        live = {id(e) for e in self._entries}
        self._status = {i: s for i, s in self._status.items() if i in live}
        self._meta = {i: m for i, m in self._meta.items() if i in live}
