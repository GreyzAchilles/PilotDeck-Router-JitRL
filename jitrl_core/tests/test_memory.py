"""memory: intent bucket filter + threshold + top-k; capacity eviction."""
from __future__ import annotations

import unittest

from jitrl_core.memory import ExperienceMemory


def entry(ic, toks, tier="medium", g=0.0, eid=1):
    return {
        "intent_class": ic,
        "signature_tokens": sorted(toks),
        "tier": tier,
        "G": g,
        "ts": 0.0,
        "episode_id": eid,
    }


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.mem = ExperienceMemory(cap=10)
        self.mem.add(entry("code_gen", ["a", "b"], "complex", 1.0))       # sim 0.5
        self.mem.add(entry("code_gen", ["a", "c"], "medium", 0.5))        # sim 0.25
        self.mem.add(entry("code_gen", ["a", "b", "z"], "simple", 2.0))   # sim 0.5
        self.mem.add(entry("data_analysis", ["a", "b"], "reasoning", 3.0))  # filtered

    def test_intent_bucket_filter(self):
        q = frozenset(["a", "b", "x"])
        matches = self.mem.retrieve("code_gen", q, k=10, threshold=0.0)
        self.assertTrue(all(m["intent_class"] == "code_gen" for m, _ in matches))
        self.assertEqual(len(matches), 3)

    def test_threshold_filters(self):
        q = frozenset(["a", "b", "x"])
        matches = self.mem.retrieve("code_gen", q, k=10, threshold=0.5)
        # sim(e1 {a,b})   = 2/3  >= 0.5  survive
        # sim(e2 {a,c})   = 1/4  <  0.5  filtered
        # sim(e3 {a,b,z}) = 2/4  >= 0.5  survive
        self.assertEqual([m["signature_tokens"] for m, _ in matches],
                         [["a", "b"], ["a", "b", "z"]])

    def test_top_k_and_ordering(self):
        mem = ExperienceMemory(cap=10)
        mem.add(entry("i", list("abcdef"), g=1.0))  # sim 6/9
        mem.add(entry("i", list("abcghij"), g=2.0))  # sim 3/9... vs q below
        mem.add(entry("i", list("abcd"), g=3.0))    # sim 4/9
        q = frozenset("abcdefghi")
        matches = mem.retrieve("i", q, k=2, threshold=0.0)
        sims = [s for _, s in matches]
        self.assertEqual(len(matches), 2)
        self.assertGreaterEqual(sims[0], sims[1])
        self.assertAlmostEqual(sims[0], 6 / 9)

    def test_tie_keeps_insertion_order(self):
        mem = ExperienceMemory(cap=10)
        # both sim = 2/3 against q={a,b} -> exact tie
        mem.add(entry("i", ["a", "b", "c"], eid=1))
        mem.add(entry("i", ["a", "b", "d"], eid=2))
        matches = mem.retrieve("i", frozenset(["a", "b"]), k=10, threshold=0.0)
        self.assertEqual(matches[0][1], matches[1][1])
        self.assertEqual(matches[0][0]["episode_id"], 1)  # stable sort


class TestCapacity(unittest.TestCase):
    def test_eviction_drops_oldest(self):
        mem = ExperienceMemory(cap=3)
        for i in range(4):
            mem.add(entry("i", ["t"], eid=i))
        self.assertEqual(len(mem), 3)
        self.assertEqual([e["episode_id"] for e in mem.entries()], [1, 2, 3])

    def test_retrieve_similarities_helper(self):
        mem = ExperienceMemory(cap=10)
        mem.add(entry("i", ["a", "b"]))
        entries, top = mem.retrieve_similarities("i", frozenset(["a", "b"]), k=5, threshold=0.5)
        self.assertEqual(len(entries), 1)
        self.assertEqual(top, 1.0)
        entries2, top2 = mem.retrieve_similarities("i", frozenset(["zzz"]), k=5, threshold=0.5)
        self.assertEqual(entries2, [])
        self.assertEqual(top2, 0.0)


if __name__ == "__main__":
    unittest.main()
