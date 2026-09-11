"""Tests for eval/run_continuation_probes.py (no network, no CPA).

Covers the two directive-critical behaviors:
  1. previous_tier mapping — each P-probe resolves to its immediately
     preceding MAIN task's gt_tier (P01/T05 medium, P02/T13 reasoning,
     P03/T10 medium, P04/T19 reasoning, P05/T12 reasoning), never None;
  2. previous_tier passing — the runner calls judge(message,
     previous_tier=<resolved>) for every probe and records only
     desensitized routing fields, with retry/error retention.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.run_continuation_probes import (  # noqa: E402
    EXPECTED_PREV_TIERS,
    judge_with_retry,
    resolve_previous_tiers,
    run_probes,
    summarize,
)
from harness.run import load_tasks  # noqa: E402

TASKS_PATH = _REPO_ROOT / "eval" / "tasks.jsonl"


def make_task(tid, family, gt, notes=""):
    return {"task_id": tid, "id": tid, "family": family, "gt_tier": gt,
            "message": f"msg-{tid}", "notes": notes}


class FakeJudge:
    """Records (message, previous_tier) calls; echoes previous_tier as chosen."""

    def __init__(self, chosen=None, fail_first=0):
        self.calls = []
        self.chosen = chosen or {}
        self.fail_first = fail_first

    def judge(self, user_message, previous_tier=None):
        self.calls.append((user_message, previous_tier))
        if self.fail_first > 0:
            self.fail_first -= 1
            raise RuntimeError("simulated network jitter")
        tier = self.chosen.get(previous_tier, previous_tier)
        return {"tier_logits": {"simple": -10.0, "medium": -1.0,
                                "complex": -9.0, "reasoning": -2.0},
                "chosen_tier": tier, "raw_output": f"<tier>{tier}</tier>",
                "latency_ms": 100.0}


class TestResolvePreviousTiersRealTasks(unittest.TestCase):
    """Mapping on the real eval/tasks.jsonl — the directive's core requirement."""

    @classmethod
    def setUpClass(cls):
        cls.tasks = load_tasks(TASKS_PATH)
        cls.resolved = resolve_previous_tiers(cls.tasks)

    def test_exactly_five_probes(self):
        probes = [t for t in self.tasks if t.get("family") == "continuation"]
        self.assertEqual([t["task_id"] for t in probes],
                         ["P01", "P02", "P03", "P04", "P05"])
        self.assertEqual(set(self.resolved), {"P01", "P02", "P03", "P04", "P05"})

    def test_mapping_matches_frozen_expectation(self):
        for pid, (task_id, tier) in EXPECTED_PREV_TIERS.items():
            self.assertEqual(self.resolved[pid], (task_id, tier),
                             f"{pid} must inherit {tier} from {task_id}")

    def test_specific_mapping(self):
        self.assertEqual(self.resolved["P01"], ("T05", "medium"))
        self.assertEqual(self.resolved["P02"], ("T13", "reasoning"))
        self.assertEqual(self.resolved["P03"], ("T10", "medium"))
        self.assertEqual(self.resolved["P04"], ("T19", "reasoning"))
        self.assertEqual(self.resolved["P05"], ("T12", "reasoning"))

    def test_previous_tier_is_never_none(self):
        for pid, (_, tier) in self.resolved.items():
            self.assertIn(tier, ("simple", "medium", "complex", "reasoning"))


class TestResolvePreviousTiersSynthetic(unittest.TestCase):
    def test_notes_missing_falls_back_to_file_order(self):
        tasks = [make_task("T05", "code_gen", "medium"),
                 make_task("P01", "continuation", "medium", notes="no ref here")]
        self.assertEqual(resolve_previous_tiers(tasks), {"P01": ("T05", "medium")})

    def test_frozen_map_used_when_no_sources(self):
        tasks = [make_task("P01", "continuation", "medium", notes="orphan probe")]
        resolved = resolve_previous_tiers(tasks)
        self.assertEqual(resolved["P01"], ("T05", "medium"))

    def test_disagreeing_sources_raise(self):
        tasks = [make_task("T05", "code_gen", "medium"),
                 make_task("T09", "doc_writing", "reasoning"),
                 make_task("P01", "continuation", "medium",
                           notes="紧跟在 T05 后，正确档位继承 reasoning。")]
        with self.assertRaisesRegex(ValueError, "P01.*disagreeing"):
            resolve_previous_tiers(tasks)

    def test_frozen_map_mismatch_raises(self):
        tasks = [make_task("T05", "code_gen", "medium"),
                 make_task("P01", "continuation", "medium",
                           notes="紧跟在 T05 后，正确档位继承 medium。")]
        frozen = {"P01": ("T05", "reasoning")}
        with self.assertRaisesRegex(ValueError, "P01.*disagreeing"):
            resolve_previous_tiers(tasks, frozen=frozen)

    def test_probe_referencing_probe_raises(self):
        tasks = [make_task("T05", "code_gen", "medium"),
                 make_task("P01", "continuation", "medium",
                           notes="紧跟在 T05 后，正确档位继承 medium。"),
                 make_task("P02", "continuation", "reasoning",
                           notes="紧跟在 P01 后，正确档位继承 medium。")]
        with self.assertRaisesRegex(ValueError, "P02.*itself a continuation"):
            resolve_previous_tiers(tasks)

    def test_unknown_notes_reference_raises(self):
        tasks = [make_task("T05", "code_gen", "medium"),
                 make_task("P01", "continuation", "medium",
                           notes="紧跟在 T99 后，正确档位继承 medium。")]
        with self.assertRaisesRegex(ValueError, "P01.*unknown task"):
            resolve_previous_tiers(tasks)


class TestRunProbesPreviousTierPassing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = load_tasks(TASKS_PATH)

    def test_judge_receives_explicit_previous_tier_per_probe(self):
        fake = FakeJudge()
        records, summary = run_probes(tasks=self.tasks, client=fake, retries=1)
        self.assertEqual(len(fake.calls), 5)
        by_pid = {t["task_id"]: t for t in self.tasks
                  if t.get("family") == "continuation"}
        for (message, prev), (pid, task) in zip(fake.calls, by_pid.items()):
            self.assertEqual(message, task["message"])       # probe message judged
            self.assertIsNotNone(prev)                        # never the None-chain bug
            self.assertEqual(prev, EXPECTED_PREV_TIERS[pid][1])
        self.assertEqual(len(records), 5)
        self.assertEqual(summary["n_errors"], 0)

    def test_echo_judge_hits_all_and_records_are_desensitized(self):
        fake = FakeJudge()  # echoes previous_tier as chosen -> all HIT
        records, summary = run_probes(tasks=self.tasks, client=fake, retries=1)
        self.assertEqual(summary["hits"], 5)
        self.assertEqual(summary["accuracy"], 1.0)
        self.assertAlmostEqual(summary["avg_judge_latency_ms"], 100.0)
        allowed = {"probe_id", "family", "expected_tier", "previous_tier",
                   "preceding_task_id", "chosen_tier", "hit", "tier_logits",
                   "judge_latency_ms", "attempts", "error"}
        for r in records:
            self.assertLessEqual(set(r), allowed)             # no message/exec/eval leak
            self.assertNotIn("message", r)
            self.assertEqual(r["hit"], True)
            self.assertEqual(r["chosen_tier"], r["previous_tier"])
            self.assertEqual(r["expected_tier"], r["previous_tier"])

    def test_miss_recorded_when_judge_chooses_wrong_tier(self):
        fake = FakeJudge(chosen={"medium": "simple", "reasoning": "simple"})
        records, summary = run_probes(tasks=self.tasks, client=fake, retries=1)
        self.assertEqual(summary["hits"], 0)
        self.assertEqual(summary["accuracy"], 0.0)
        for r in records:
            self.assertEqual(r["chosen_tier"], "simple")
            self.assertEqual(r["hit"], False)

    def test_network_jitter_retried_and_failure_info_kept(self):
        fake = FakeJudge(fail_first=1)  # first probe fails once, then succeeds
        records, summary = run_probes(tasks=self.tasks, client=fake, retries=3,
                                      retry_sleep_s=0.0)
        self.assertEqual(summary["n_errors"], 0)
        self.assertEqual(records[0]["attempts"], 2)
        self.assertIsNone(records[0]["error"])
        self.assertEqual(summary["retried_probes"], ["P01"])

    def test_persistent_failure_recorded_without_aborting_rest(self):
        class AlwaysFail:
            def judge(self, message, previous_tier=None):
                raise RuntimeError("judge down")

        records, summary = run_probes(tasks=self.tasks, client=AlwaysFail(),
                                      retries=2, retry_sleep_s=0.0)
        self.assertEqual(summary["n_errors"], 5)
        self.assertEqual(summary["n_judged"], 0)
        self.assertIsNone(summary["accuracy"])
        for r in records:
            self.assertIsNone(r["chosen_tier"])
            self.assertIsNone(r["hit"])
            self.assertIn("judge down", r["error"])
            self.assertEqual(r["attempts"], 2)
            # failure records still carry the resolved context
            self.assertIn(r["previous_tier"], ("medium", "reasoning"))


class TestJudgeWithRetry(unittest.TestCase):
    def test_success_first_try(self):
        result, attempts, error = judge_with_retry(
            FakeJudge(), "继续", "medium", retries=3, retry_sleep_s=0.0)
        self.assertIsNotNone(result)
        self.assertEqual(attempts, 1)
        self.assertIsNone(error)

    def test_exhausted_retries_return_error(self):
        class AlwaysFail:
            def judge(self, message, previous_tier=None):
                raise RuntimeError("boom")

        result, attempts, error = judge_with_retry(
            AlwaysFail(), "继续", "reasoning", retries=3, retry_sleep_s=0.0)
        self.assertIsNone(result)
        self.assertEqual(attempts, 3)
        self.assertIn("RuntimeError: boom", error)


class TestSummarize(unittest.TestCase):
    def test_summary_fields(self):
        records = [
            {"probe_id": "P01", "previous_tier": "medium", "expected_tier": "medium",
             "chosen_tier": "medium", "hit": True, "judge_latency_ms": 100.0,
             "attempts": 1, "error": None},
            {"probe_id": "P02", "previous_tier": "reasoning",
             "expected_tier": "reasoning", "chosen_tier": "medium", "hit": False,
             "judge_latency_ms": 300.0, "attempts": 2, "error": None},
        ]
        s = summarize(records)
        self.assertEqual(s["n_probes"], 2)
        self.assertEqual(s["n_judged"], 2)
        self.assertEqual(s["hits"], 1)
        self.assertEqual(s["accuracy"], 0.5)
        self.assertAlmostEqual(s["avg_judge_latency_ms"], 200.0)
        self.assertEqual(s["retried_probes"], ["P02"])
        self.assertEqual(set(s["per_probe"]), {"P01", "P02"})


if __name__ == "__main__":
    unittest.main()
