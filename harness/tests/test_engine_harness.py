"""engine+harness integration: B vs C on the mini fixture (mock judge).

Assertions (spec section 7):
- mock unmodulated accuracy on fixture in [0.5, 0.7] (clearly < 100%)
- mode C memory grows; JSONL outputs are valid; record schema exact
- --episodes 2: C episode-2 accuracy on variant families >= B episode-2
  (if this ever fails, tune the MOCK's bias — never the engine math)
- separability: within-family mean Jaccard > cross-family mean
"""
from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from harness import run as harness_run
from harness.separability import load_tasks, separability_report

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "harness" / "testdata" / "mini_tasks.jsonl"

RECORD_KEYS = {
    "episode_id", "task_id", "family", "gt_tier", "intent_class",
    "signature_preview", "n_retrieved", "top_similarity", "V", "Q",
    "A_norm", "z_base", "z_mod", "chosen_tier", "base_choice", "flipped",
    "reward", "cost_per_episode", "judge_latency_ms",
}


def run_mode(mode: str, episodes: int, tmp: str):
    out = Path(tmp) / f"out_{mode}_{episodes}.jsonl"
    mem = Path(tmp) / f"mem_{mode}_{episodes}.jsonl"
    summary = harness_run.run(
        mode=mode, tasks_path=FIXTURE, judge_kind="mock",
        out_path=out, memory_out_path=mem, episodes=episodes,
    )
    records, summary_line = [], None
    with open(out, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "_summary" in obj:
                summary_line = obj["_summary"]
            else:
                records.append(obj)
    mem_entries = []
    with open(mem, encoding="utf-8") as f:
        mem_entries = [json.loads(line) for line in f if line.strip()]
    return summary, summary_line, records, mem_entries


def variant_family_task_ids() -> set[str]:
    """Families with >= 2 tasks ('variant families')."""
    tasks = load_tasks(FIXTURE)
    fam_counts = Counter(t["family"] for t in tasks)
    return {t["task_id"] for t in tasks if fam_counts[t["family"]] >= 2}


def episode_accuracy(records: list[dict], episode_id: int, task_ids: set[str]) -> float:
    rows = [r for r in records
            if r["episode_id"] == episode_id and r["task_id"] in task_ids
            and r.get("gt_tier")]
    hits = sum(1 for r in rows if r["chosen_tier"] == r["gt_tier"])
    return hits / len(rows)


class TestHarnessBAndC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = cls._tmp.name
        cls.b1_sum, _, cls.b1_records, _ = run_mode("B", 1, tmp)
        cls.b2_sum, _, cls.b2_records, cls.b2_mem = run_mode("B", 2, tmp)
        cls.c1_sum, _, cls.c1_records, cls.c1_mem = run_mode("C", 1, tmp)
        cls.c2_sum, cls.c2_line, cls.c2_records, cls.c2_mem = run_mode("C", 2, tmp)
        cls.variant_ids = variant_family_task_ids()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    # ------------------------------------------------------- mock quality
    def test_mock_unmodulated_accuracy_in_band(self):
        acc = self.b1_sum["accuracy"]
        self.assertGreaterEqual(acc, 0.5)
        self.assertLessEqual(acc, 0.7)   # clearly < 1.0

    # ------------------------------------------------------- record schema
    def test_jsonl_records_valid_and_exact_schema(self):
        self.assertEqual(len(self.c2_records), 20)
        for r in self.c2_records:
            self.assertEqual(set(r.keys()), RECORD_KEYS)
            self.assertIsInstance(r["flipped"], bool)
            self.assertLessEqual(len(r["signature_preview"]), 12)
            self.assertIsInstance(r["reward"], float)
            self.assertIsInstance(r["cost_per_episode"], float)
        self.assertIsNotNone(self.c2_line)
        self.assertEqual(self.c2_line["mode"], "C")

    def test_loader_accepts_public_id_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "public.jsonl"
            path.write_text(json.dumps({
                "id": "T01", "family": "chat_qa", "message": "你好",
                "gt_tier": "simple",
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            rows = harness_run.load_tasks(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["task_id"], "T01")
        self.assertEqual(rows[0]["id"], "T01")

    # ------------------------------------------------------- mode A basics
    def test_mode_A_static_routing(self):
        with tempfile.TemporaryDirectory() as tmp:
            a_sum, _, a_records, a_mem = run_mode("A", 1, tmp)
        self.assertEqual(a_sum["memory_size"], 0)
        self.assertEqual(a_mem, [])
        for r in a_records:
            self.assertEqual(r["chosen_tier"], "static")
            self.assertEqual(r["judge_latency_ms"], 0.0)
            self.assertEqual(r["n_retrieved"], 0)

    # ------------------------------------------------------- memory growth
    def test_C_memory_grows_and_schema(self):
        n_tasks = len(self.c1_records)
        self.assertEqual(len(self.c1_mem), n_tasks)          # ep1: one per task
        self.assertEqual(len(self.c2_mem), 2 * n_tasks)      # persists across eps
        for e in self.c2_mem:
            self.assertEqual(set(e.keys()),
                             {"intent_class", "signature_tokens", "tier",
                              "G", "ts", "episode_id"})
        # B never writes memory
        self.assertEqual(self.b2_mem, [])

    # ------------------------------------------------- C ep2 >= B ep2 variants
    def test_C_episode2_variant_accuracy_ge_B(self):
        b_acc = episode_accuracy(self.b2_records, 2, self.variant_ids)
        c_acc = episode_accuracy(self.c2_records, 2, self.variant_ids)
        self.assertGreaterEqual(c_acc, b_acc,
                                f"C ep2 variant acc {c_acc} < B {b_acc}")

    def test_C_episode2_not_worse_than_B_overall(self):
        def acc(records, ep):
            rows = [r for r in records if r["episode_id"] == ep and r.get("gt_tier")]
            return sum(1 for r in rows if r["chosen_tier"] == r["gt_tier"]) / len(rows)
        self.assertGreaterEqual(acc(self.c2_records, 2), acc(self.b2_records, 2))

    # ------------------------------------------------- modulation did something
    def test_C_episode2_has_flips(self):
        flips = [r for r in self.c2_records
                 if r["episode_id"] == 2 and r["flipped"]]
        self.assertGreaterEqual(len(flips), 1)

    # ------------------------------------------------- continuation probe
    def test_continuation_inherits_previous_chosen_tier(self):
        recs = {r["task_id"]: r for r in self.b1_records}
        self.assertEqual(recs["t04"]["chosen_tier"], recs["t03"]["chosen_tier"])
        self.assertEqual(self.b1_sum["continuation_accuracy"], 1.0)

    # ------------------------------------------------- summary line shape
    def test_summary_metrics_present(self):
        s = self.c2_line
        for key in ("accuracy", "avg_cost", "flips", "memory_size",
                    "family_accuracy", "continuation_accuracy"):
            self.assertIn(key, s)
        self.assertEqual(s["memory_size"], 20)
        self.assertAlmostEqual(s["avg_cost"], self.c2_sum["avg_cost"])


class TestSeparability(unittest.TestCase):
    def test_within_family_mean_greater_than_cross(self):
        tasks = load_tasks(FIXTURE)
        report = separability_report(tasks)
        w = report["within_family"]["mean"]
        c = report["cross_family"]["mean"]
        self.assertIsNotNone(w)
        self.assertIsNotNone(c)
        self.assertGreater(w, c,
                           f"within-family mean {w} !> cross-family mean {c}")
        self.assertIn(report["verdict"], {"bimodal", "degenerate"})
        total = sum(report["decile_histogram"].values())
        n = report["n_tasks"]
        self.assertEqual(total, n * (n - 1) // 2)


if __name__ == "__main__":
    unittest.main()
