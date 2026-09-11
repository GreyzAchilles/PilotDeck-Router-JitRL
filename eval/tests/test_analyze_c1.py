"""Tests for eval/analyze_c1.py (pure functions, no network, no CPA).

Covers the directive-critical behaviors:
  1. formal-input assertions: exactly 24 records, eval 24/24, identical
     B/C0/C1 order equal to the frozen preregistration sequence, and a
     config snapshot check that enforces min_neighbors=3 (plus every other
     frozen prereg parameter);
  2. gate analysis: no_neighbors / below_min_neighbors / modulated
     classification and counts;
  3. neighbor recomputation from a memory file (Jaccard >= 0.5, top-k,
     only entries written before the task) with formula parity against
     jitrl_core.state.jaccard;
  4. P1-P5 preregistered judgment logic, including boundary cases and the
     P5 zero-flip trivial pass;
  5. paired B/C1 attribution: modulation flip vs judge base drift vs
     same-tier sampling;
  6. informational D10 recomputation stays informational and does not
     claim to replace the frozen C0 judgment;
  7. end-to-end main() writes summary JSON, tasks CSV and SVG charts
     strictly under the out dir, and never touches C0's eval/ assets.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.analyze_c1 import (  # noqa: E402
    DEFAULT_OUT_DIR,
    PREREG_CONFIG,
    PREREG_ORDER,
    assert_formal_inputs,
    build_summary,
    d10_informational,
    flip_analysis,
    gate_analysis,
    group_metrics,
    jaccard,
    judge_drift_analysis,
    load_group,
    main,
    p1_p5_judgment,
    paired_bc1,
    retrieve_neighbors,
)

TIERS = ("simple", "medium", "complex", "reasoning")


# ------------------------------------------------------------- fixtures

def make_record(task_id: str, *, tier: str = "simple", base: str = "simple",
                flipped: bool = False, quality: int = 5,
                cost: float = 0.0005, n_retrieved: int = 0,
                modulation_active: bool = False,
                gate_reason: str | None = "no_neighbors",
                jitrl_ms: float = 0.2, gt_tier: str = "simple",
                family: str = "fam", memory_written: bool = True,
                finish_reason: str = "stop", response: str = "text",
                min_neighbors: int = 3) -> dict:
    return {
        "episode_id": 1, "task_id": task_id, "family": family,
        "gt_tier": gt_tier, "n_retrieved": n_retrieved,
        "top_similarity": 1.0 if n_retrieved else 0.0,
        "min_neighbors": min_neighbors,
        "modulation_active": modulation_active, "gate_reason": gate_reason,
        "chosen_tier": tier, "route_mode": "jitrl", "base_choice": base,
        "flipped": flipped, "exec_model": "CPA/test",
        "exec_response_preview": response, "exec_finish_reason": finish_reason,
        "exec_cost_usd": cost, "exec_latency_ms": 1000.0,
        "exec_error": None,
        "evaluator": {"quality_score": quality, "verdict": "useful",
                      "capability_insufficient": False,
                      "evaluator_latency_ms": 100.0},
        "eval_cost_usd": 0.005, "eval_error": None,
        "judge_latency_ms": 1000.0, "jitrl_latency_ms": jitrl_ms,
        "reward": 0.5, "memory_written": memory_written,
    }


def make_run(**overrides) -> list[dict]:
    """A full 24-record run in the frozen prereg order."""
    out = []
    for i, tid in enumerate(PREREG_ORDER):
        r = make_record(tid)
        for k, v in overrides.items():
            r[k] = v(tid, r) if callable(v) else v
        out.append(r)
    return out


def make_c1_summary(min_neighbors: int = 3) -> dict:
    return {"_summary": {
        "mode": "C", "judge": "llama", "episodes": 1, "n_records": 24,
        "config_snapshot": {**PREREG_CONFIG, "min_neighbors": min_neighbors},
    }}


def write_run(path: Path, records: list[dict], summary: dict | None) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if summary is not None:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")


def write_memory(path: Path, entries: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


# ------------------------------------------------- formal input assertions

class TestFormalInputAssertions(unittest.TestCase):
    def test_ok(self):
        checks = assert_formal_inputs(make_run(), make_run(), make_run(),
                                      make_c1_summary()["_summary"])
        self.assertTrue(checks["order_identical_b_c0_c1"])
        self.assertTrue(checks["order_equals_prereg"])
        self.assertEqual(checks["order"], PREREG_ORDER)
        self.assertTrue(checks["c1_config_matches_prereg"])
        self.assertEqual(checks["c1_config_snapshot"]["min_neighbors"], 3)
        self.assertEqual(checks["eval_success"],
                         {"B": "24/24", "C0": "24/24", "C1": "24/24"})

    def test_wrong_record_count_refused(self):
        run = make_run()[:-1]
        with self.assertRaises(SystemExit):
            assert_formal_inputs(run, make_run(), make_run(),
                                 make_c1_summary()["_summary"])

    def test_eval_failure_refused(self):
        run = make_run()
        run[3]["evaluator"] = None
        with self.assertRaises(SystemExit):
            assert_formal_inputs(make_run(), make_run(), run,
                                 make_c1_summary()["_summary"])

    def test_exec_error_refused(self):
        run = make_run()
        run[3]["exec_error"] = "boom"
        with self.assertRaises(SystemExit):
            assert_formal_inputs(make_run(), make_run(), run,
                                 make_c1_summary()["_summary"])

    def test_order_mismatch_refused(self):
        run = make_run()
        run[0], run[1] = run[1], run[0]
        with self.assertRaises(SystemExit):
            assert_formal_inputs(make_run(), make_run(), run,
                                 make_c1_summary()["_summary"])

    def test_non_prereg_order_refused(self):
        run = make_run()
        run[0], run[1] = run[1], run[0]
        with self.assertRaises(SystemExit):
            assert_formal_inputs(run, run, run,
                                 make_c1_summary()["_summary"])

    def test_min_neighbors_not_3_refused(self):
        with self.assertRaises(SystemExit):
            assert_formal_inputs(make_run(), make_run(), make_run(),
                                 make_c1_summary(min_neighbors=1)["_summary"])

    def test_missing_config_key_refused(self):
        summ = make_c1_summary()["_summary"]
        del summ["config_snapshot"]["seed"]
        with self.assertRaises(SystemExit):
            assert_formal_inputs(make_run(), make_run(), make_run(), summ)

    def test_wrong_frozen_param_refused(self):
        summ = make_c1_summary()["_summary"]
        summ["config_snapshot"]["beta"] = 4.0
        with self.assertRaises(SystemExit):
            assert_formal_inputs(make_run(), make_run(), make_run(), summ)


# ------------------------------------------------------------ gate analysis

class TestGateAnalysis(unittest.TestCase):
    def test_classification_and_counts(self):
        c1 = make_run(
            n_retrieved=lambda tid, r: {"T06": 1, "T18": 1, "T07": 2,
                                        "T19": 2, "T08": 3,
                                        "T20": 3}.get(tid, 0),
            modulation_active=lambda tid, r: tid in ("T08", "T20"),
            gate_reason=lambda tid, r: (None if tid in ("T08", "T20")
                                        else "below_min_neighbors"
                                        if tid in ("T06", "T07", "T18", "T19")
                                        else "no_neighbors"),
        )
        c1_by = {r["task_id"]: r for r in c1}
        c1_by["T20"].update(flipped=True, chosen_tier="complex",
                            base_choice="simple")
        gate = gate_analysis(c1)
        self.assertEqual(gate["counts"]["total"], 24)
        self.assertEqual(gate["counts"]["gated"], 22)
        self.assertEqual(gate["counts"]["gated_no_neighbors"], 18)
        self.assertEqual(gate["counts"]["gated_below_min_neighbors"], 4)
        self.assertEqual(gate["counts"]["modulated"], 2)
        self.assertEqual(gate["counts"]["modulated_and_flipped"], 1)
        self.assertEqual([d["task_id"] for d in gate["modulated_tasks"]],
                         ["T08", "T20"])
        self.assertEqual(gate["modulated_tasks"][1]["chosen_tier"], "complex")
        below = sorted(d["task_id"] for d in gate["gated_tasks"]
                       if d["gate_reason"] == "below_min_neighbors")
        self.assertEqual(below, ["T06", "T07", "T18", "T19"])


# ------------------------------------------------------- neighbor recompute

class TestRetrieveNeighbors(unittest.TestCase):
    def test_jaccard_parity_with_jitrl_core(self):
        try:
            from jitrl_core.state import jaccard as core_jaccard
        except ImportError:
            self.skipTest("jitrl_core not importable")
        a, b = {"a", "b", "c"}, {"b", "c", "d"}
        self.assertAlmostEqual(jaccard(a, b), core_jaccard(
            frozenset(a), frozenset(b)))
        self.assertEqual(jaccard(set(), set()), 0.0)

    def test_retrieval_threshold_k_and_visibility(self):
        order = ["T1", "T2", "T3", "T4"]
        # T4={a,b,c} matches T3 exactly (sim 1.0) and T2 partially
        # (sim 2/3), T1 not at all (sim 0); k=2 cap
        mem = [
            {"signature_tokens": ["x", "y", "z", "w"], "tier": "simple",
             "G": 0.1},
            {"signature_tokens": ["a", "b"], "tier": "complex", "G": -0.6},
            {"signature_tokens": ["a", "b", "c"], "tier": "simple",
             "G": -0.3},
            {"signature_tokens": ["a", "b", "c"], "tier": "simple",
             "G": -0.2},
        ]
        got = retrieve_neighbors(mem, 3, order, k=2)
        # highest similarity first
        self.assertEqual([n["task_id"] for n in got], ["T3", "T2"])
        self.assertEqual(got[0]["similarity"], 1.0)
        self.assertEqual(got[0]["stored_tier"], "simple")
        self.assertEqual(got[1]["stored_tier"], "complex")
        self.assertTrue(all(n["similarity"] >= 0.5 for n in got))
        # the task's own entry (index 3) and later entries are invisible
        self.assertNotIn("T4", [n["task_id"] for n in got])
        # below-threshold neighbors are excluded (T1: jaccard 0 < 0.5)
        self.assertNotIn("T1", [n["task_id"] for n in got])


# ---------------------------------------------------------------- flips

class TestFlipAnalysis(unittest.TestCase):
    def test_flip_evidence_and_mechanism(self):
        b = make_run()
        c1 = make_run()
        by = {r["task_id"]: r for r in c1}
        t20 = by["T20"]
        t20.update(flipped=True, chosen_tier="complex", base_choice="simple",
                   cost=lambda *_: None)
        t20["exec_cost_usd"] = 0.0047254
        by["T20"]["evaluator"]["quality_score"] = 1
        b_by = {r["task_id"]: r for r in b}
        b_by["T20"]["evaluator"]["quality_score"] = 1
        b_by["T20"]["exec_cost_usd"] = 0.00053555
        c0 = make_run()
        pos = {tid: i for i, tid in enumerate(PREREG_ORDER)}
        mem_c1 = [{"signature_tokens": [f"t{i}"], "tier": "simple",
                   "G": -0.33} for i in range(24)]
        mem_c0 = [{"signature_tokens": [f"t{i}"], "tier": "simple",
                   "G": -0.33} for i in range(24)]
        mem_c0[pos["T19"]]["tier"] = "complex"
        mem_c0[pos["T19"]]["G"] = -0.6
        # give T17/T18/T19/T20 identical signatures so T20 retrieves them
        for tid in ("T17", "T18", "T19", "T20"):
            mem_c1[pos[tid]]["signature_tokens"] = ["refactor"]
            mem_c0[pos[tid]]["signature_tokens"] = ["refactor"]
        flips = flip_analysis(c1, b, c0, mem_c1, mem_c0, PREREG_ORDER)
        self.assertEqual(flips["n_flips"], 1)
        f = flips["flips"][0]
        self.assertEqual(f["task_id"], "T20")
        self.assertEqual(f["quality_C1"], 1)
        self.assertEqual(f["quality_B"], 1)
        self.assertAlmostEqual(f["exec_cost_delta_usd"],
                               0.0047254 - 0.00053555, places=6)
        self.assertEqual(
            [n["task_id"] for n in f["mechanism"]["retrieved_neighbors_C1"]],
            ["T17", "T18", "T19"])
        c0_n = {n["task_id"]: n
                for n in f["mechanism"]["retrieved_neighbors_C0"]}
        self.assertEqual(c0_n["T19"]["stored_tier"], "complex")
        self.assertEqual(c0_n["T19"]["stored_G"], -0.6)
        self.assertEqual(flips["quality_delta_sum"], 0)
        self.assertAlmostEqual(flips["exec_cost_delta_sum_usd"],
                               0.00418985, places=6)


# ------------------------------------------------------- paired attribution

class TestPairedBC1(unittest.TestCase):
    def test_flip_drift_and_same_tier_separation(self):
        b = make_run()
        c1 = make_run()
        b_by = {r["task_id"]: r for r in b}
        c1_by = {r["task_id"]: r for r in c1}
        # T20: modulation flip (tier changes, flipped=True)
        c1_by["T20"].update(flipped=True, chosen_tier="complex",
                            base_choice="simple")
        c1_by["T20"]["exec_cost_usd"] = 0.0047
        # T13: judge base drift (tier changes, NOT flipped: base changed)
        c1_by["T13"].update(chosen_tier="complex", base_choice="complex")
        b_by["T13"]["chosen_tier"] = "simple"
        b_by["T13"]["exec_cost_usd"] = 0.0005
        c1_by["T13"]["exec_cost_usd"] = 0.0045
        # T09: same-tier sampling variance (cost/quality differ, same tier)
        c1_by["T09"]["evaluator"]["quality_score"] = 2
        b_by["T09"]["evaluator"]["quality_score"] = 5
        c1_by["T09"]["exec_cost_usd"] = 0.0006
        b_by["T09"]["exec_cost_usd"] = 0.0003
        p = paired_bc1(b, c1)
        self.assertEqual(p["tier_changes"]["n_total"], 2)
        self.assertEqual(p["tier_changes"]["modulation_flip_task_ids"],
                         ["T20"])
        self.assertEqual(p["tier_changes"]["judge_base_drift_task_ids"],
                         ["T13"])
        cost_attr = p["exec_cost"]["attribution_usd"]
        self.assertAlmostEqual(cost_attr["modulation_flip"], 0.0042,
                               places=6)
        self.assertAlmostEqual(cost_attr["judge_base_drift"], 0.0040,
                               places=6)
        self.assertAlmostEqual(cost_attr["same_tier_sampling"], 0.0003,
                               places=6)
        q_attr = p["quality_attribution"]
        self.assertEqual(q_attr["same_tier_sampling"], -3)
        self.assertEqual(q_attr["modulation_flip"], 0)
        self.assertEqual(q_attr["judge_base_drift"], 0)


class TestJudgeDrift(unittest.TestCase):
    def test_c1_matches_b_and_differs_from_c0(self):
        b = make_run()
        c1 = make_run()
        c0 = make_run()
        b_by = {r["task_id"]: r for r in b}
        c0_by = {r["task_id"]: r for r in c0}
        # C0's judge drifted on T13 (simple instead of B's complex)
        c0_by["T13"].update(chosen_tier="simple", base_choice="simple")
        b_by["T13"]["chosen_tier"] = "complex"
        c1_by = {r["task_id"]: r for r in c1}
        c1_by["T13"].update(chosen_tier="complex", base_choice="complex")
        d = judge_drift_analysis(b, c0, c1)
        self.assertTrue(d["c1_base_vs_b_consistent"])
        self.assertEqual(d["c1_base_vs_b_chosen_mismatches"], [])
        self.assertEqual([m["task_id"]
                          for m in d["c1_base_vs_c0_base_mismatches"]],
                         ["T13"])


# ------------------------------------------------------------ P1-P5 logic

class TestP1toP5(unittest.TestCase):
    @staticmethod
    def _judgment(b_cost_total, c1_cost_total, q_b, q_c1, mis_b, mis_c1,
                  jitrl_max, flips, b_for_flips=None):
        b_metrics = group_metrics(make_run())
        c1_metrics = group_metrics(make_run())
        b_metrics["exec_cost_usd"]["total"] = b_cost_total
        c1_metrics["exec_cost_usd"]["total"] = c1_cost_total
        b_metrics["quality"]["mean"] = q_b
        c1_metrics["quality"]["mean"] = q_c1
        b_metrics["tier_vs_gt"]["misgrade_rate"] = mis_b
        c1_metrics["tier_vs_gt"]["misgrade_rate"] = mis_c1
        c1_metrics["latency_ms"]["jitrl_max"] = jitrl_max
        paired = {"exec_cost": {"total_B_usd": b_cost_total,
                                "total_C1_usd": c1_cost_total}}
        flip_objs = []
        for tid, (q1, qb, c1c, bc) in flips.items():
            flip_objs.append({
                "task_id": tid, "quality_C1": q1, "quality_B": qb,
                "exec_cost_C1_usd": c1c, "exec_cost_B_usd": bc})
        flips_obj = {"n_flips": len(flip_objs), "flips": flip_objs}
        return p1_p5_judgment(b_metrics, c1_metrics, paired, [], flips_obj)

    def test_all_pass(self):
        j = self._judgment(0.028, 0.0285, 2.7, 2.7, 0.7, 0.7, 1.0, {})
        self.assertTrue(all(j[k]["met"] for k in ("P1", "P2", "P3", "P4")))
        self.assertTrue(j["P5"]["met"] and j["P5"]["trivially_met"])
        self.assertEqual(j["n_criteria_met"], 5)
        self.assertTrue(j["verdict"].startswith("PASS"))

    def test_p1_boundary_and_violation(self):
        self.assertTrue(self._judgment(
            0.028, 0.029, 2.7, 2.7, 0.7, 0.7, 1.0, {})["P1"]["met"])
        self.assertFalse(self._judgment(
            0.028, 0.029001, 2.7, 2.7, 0.7, 0.7, 1.0, {})["P1"]["met"])

    def test_p2_boundary_and_violation(self):
        self.assertTrue(self._judgment(
            0.028, 0.028, 2.7, 2.4, 0.7, 0.7, 1.0, {})["P2"]["met"])
        self.assertFalse(self._judgment(
            0.028, 0.028, 2.7, 2.3999, 0.7, 0.7, 1.0, {})["P2"]["met"])

    def test_p3_boundary_and_violation(self):
        self.assertTrue(self._judgment(
            0.028, 0.028, 2.7, 2.7, 0.70, 0.75, 1.0, {})["P3"]["met"])
        self.assertFalse(self._judgment(
            0.028, 0.028, 2.7, 2.7, 0.70, 0.751, 1.0, {})["P3"]["met"])

    def test_p4_boundary_and_violation(self):
        self.assertTrue(self._judgment(
            0.028, 0.028, 2.7, 2.7, 0.7, 0.7, 500.0, {})["P4"]["met"])
        self.assertFalse(self._judgment(
            0.028, 0.028, 2.7, 2.7, 0.7, 0.7, 500.1, {})["P4"]["met"])

    def test_p5_zero_flips_trivially_met(self):
        j = self._judgment(0.028, 0.028, 2.7, 2.7, 0.7, 0.7, 1.0, {})
        self.assertTrue(j["P5"]["met"])
        self.assertTrue(j["P5"]["trivially_met"])
        self.assertEqual(j["P5"]["flip_checks"], [])

    def test_p5_flip_passes_at_boundaries(self):
        j = self._judgment(0.028, 0.028, 2.7, 2.7, 0.7, 0.7, 1.0,
                           {"T20": (3, 3, 0.0015, 0.0005)})
        self.assertTrue(j["P5"]["met"])
        self.assertFalse(j["P5"]["trivially_met"])

    def test_p5_flip_fails_on_quality(self):
        j = self._judgment(0.028, 0.028, 2.7, 2.7, 0.7, 0.7, 1.0,
                           {"T20": (2, 3, 0.0015, 0.0005)})
        self.assertFalse(j["P5"]["met"])
        self.assertFalse(j["P5"]["flip_checks"][0]["quality_ok"])

    def test_p5_flip_fails_on_cost(self):
        j = self._judgment(0.028, 0.028, 2.7, 2.7, 0.7, 0.7, 1.0,
                           {"T20": (3, 3, 0.0047254, 0.00053555)})
        self.assertFalse(j["P5"]["met"])
        self.assertTrue(j["P5"]["flip_checks"][0]["quality_ok"])
        self.assertFalse(j["P5"]["flip_checks"][0]["cost_ok"])

    def test_verdict_names_failed_criteria(self):
        j = self._judgment(0.028, 0.033, 2.7, 2.45, 0.7, 0.7, 1.0,
                           {"T20": (1, 1, 0.0047254, 0.00053555)})
        self.assertEqual(j["n_criteria_met"], 3)
        self.assertEqual(j["failed"], ["P1", "P5"])
        self.assertIn("NOT PASSING", j["verdict"])


# ------------------------------------------------------ informational D10

class TestD10Informational(unittest.TestCase):
    def test_informational_and_disclaimer(self):
        b_metrics = group_metrics(make_run())
        c1_metrics = group_metrics(make_run())
        c1_metrics["exec_cost_usd"]["total"] = 0.033
        b_metrics["exec_cost_usd"]["total"] = 0.028
        paired = {"exec_cost": {"total_B_usd": 0.028,
                                "total_C1_usd": 0.033}}
        d = d10_informational(b_metrics, c1_metrics, paired, [])
        self.assertTrue(d["informational_only"])
        self.assertIn("Does NOT replace", d["disclaimer"])
        # savings = (0.028-0.033)/0.028 < 0 -> criterion 1 fails
        self.assertFalse(d["criteria"][0]["met"])
        self.assertEqual(d["n_criteria_met"], 3)
        self.assertTrue(d["verdict"].startswith("NOT MET"))

    def test_all_four_met_when_cheaper(self):
        b_metrics = group_metrics(make_run())
        c1_metrics = group_metrics(make_run())
        b_metrics["exec_cost_usd"]["total"] = 0.028
        c1_metrics["exec_cost_usd"]["total"] = 0.020
        paired = {"exec_cost": {"total_B_usd": 0.028,
                                "total_C1_usd": 0.020}}
        d = d10_informational(b_metrics, c1_metrics, paired, [])
        self.assertEqual(d["n_criteria_met"], 4)
        self.assertTrue(d["verdict"].startswith("MET"))


# ------------------------------------------------------------- end-to-end

class TestMainEndToEnd(unittest.TestCase):
    def _write_formal_logs(self, tmp: Path):
        b = make_run(cost=lambda tid, r: 0.0005)
        c0 = make_run(cost=lambda tid, r: 0.0005)
        c1 = make_run(cost=lambda tid, r: 0.0005)
        b_by = {r["task_id"]: r for r in b}
        c1_by = {r["task_id"]: r for r in c1}
        pos = {tid: i for i, tid in enumerate(PREREG_ORDER)}
        # a realistic C1: gate everywhere except T08/T20, one flip on T20
        for tid in ("T06", "T07", "T18", "T19"):
            c1_by[tid]["n_retrieved"] = {"T06": 1, "T07": 2,
                                         "T18": 1, "T19": 2}[tid]
            c1_by[tid]["gate_reason"] = "below_min_neighbors"
        for tid in ("T08", "T20"):
            c1_by[tid].update(n_retrieved=3, modulation_active=True,
                              gate_reason=None)
        c1_by["T20"].update(flipped=True, chosen_tier="complex")
        c1_by["T20"]["exec_cost_usd"] = 0.0047254
        b_by["T20"]["exec_cost_usd"] = 0.00053555
        mem = [{"signature_tokens": [f"s{i}"], "tier": "simple", "G": -0.3,
                "episode_id": 1} for i in range(24)]
        for tid in ("T17", "T18", "T19", "T20"):
            mem[pos[tid]]["signature_tokens"] = ["refactor"]
        write_run(tmp / "b.jsonl", b, None)
        write_run(tmp / "c0.jsonl", c0, None)
        write_run(tmp / "c1.jsonl", c1, make_c1_summary())
        write_memory(tmp / "c1_mem.jsonl", mem)
        write_memory(tmp / "c0_mem.jsonl", mem)
        return b, c0, c1

    def test_main_writes_assets_only_under_out_dir(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            self._write_formal_logs(tmp)
            out = tmp / "eval" / "c1"
            rc = main(["--b", str(tmp / "b.jsonl"),
                       "--c0", str(tmp / "c0.jsonl"),
                       "--c1", str(tmp / "c1.jsonl"),
                       "--c1-memory", str(tmp / "c1_mem.jsonl"),
                       "--c0-memory", str(tmp / "c0_mem.jsonl"),
                       "--out-dir", str(out)])
            self.assertEqual(rc, 0)
            summary = json.loads(
                (out / "results-summary.json").read_text(encoding="utf-8"))
            # every written file lives strictly under the out dir
            written = [p for p in out.rglob("*") if p.is_file()]
            self.assertTrue(written)
            self.assertTrue(all(out in p.parents for p in written))
            self.assertEqual(
                sorted(p.name for p in written),
                ["bc1-paired-exec-cost.svg", "cost-comparison.svg",
                 "gate-modulation.svg", "quality-comparison.svg",
                 "results-summary.json", "results-tasks.csv"])
            # CSV: 24 rows x 3 groups + header
            lines = (out / "results-tasks.csv").read_text(
                encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 73)
            # judgment reflects the synthetic flip
            self.assertEqual(summary["c1_flips"]["n_flips"], 1)
            self.assertEqual(summary["c1_flips"]["flips"][0]["task_id"],
                             "T20")
            self.assertFalse(summary["p1_p5"]["P1"]["met"])
            self.assertFalse(summary["p1_p5"]["P5"]["met"])
            self.assertTrue(summary["p1_p5"]["P2"]["met"])
            self.assertTrue(summary["p1_p5"]["P3"]["met"])
            self.assertTrue(summary["p1_p5"]["P4"]["met"])
            self.assertIn("NOT PASSING", summary["p1_p5"]["verdict"])
            self.assertTrue(summary["d10_informational"]["informational_only"])
            self.assertEqual(
                summary["c1_gate"]["counts"]["modulated_and_flipped"], 1)
            # default out dir constant points at eval/c1, never eval/
            self.assertEqual(Path(DEFAULT_OUT_DIR).as_posix(), "eval/c1")

    def test_main_refuses_order_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            self._write_formal_logs(tmp)
            bad = make_run()
            bad[0], bad[1] = bad[1], bad[0]
            write_run(tmp / "bad.jsonl", bad, make_c1_summary())
            with self.assertRaises(SystemExit):
                main(["--b", str(tmp / "b.jsonl"),
                      "--c0", str(tmp / "c0.jsonl"),
                      "--c1", str(tmp / "bad.jsonl"),
                      "--c1-memory", str(tmp / "c1_mem.jsonl"),
                      "--c0-memory", str(tmp / "c0_mem.jsonl"),
                      "--out-dir", str(tmp / "out")])

    def test_c0_assets_not_overwritten(self):
        """The analyzer must never write outside its --out-dir; C0's
        eval/results-summary.json, eval/results-tasks.csv and eval/charts
        stay untouched (hash-compare before/after a run)."""
        import hashlib

        def digest(p: Path) -> str | None:
            return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() \
                else None

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            self._write_formal_logs(tmp)
            # stand-ins for C0's frozen assets at eval/ root
            c0_dir = tmp / "eval"
            c0_dir.mkdir()
            for name in ("results-summary.json", "results-tasks.csv"):
                (c0_dir / name).write_text("C0 ASSET", encoding="utf-8")
            (c0_dir / "charts").mkdir()
            (c0_dir / "charts" / "cost-summary.svg").write_text(
                "C0 CHART", encoding="utf-8")
            before = {p: digest(p) for p in c0_dir.rglob("*") if p.is_file()}
            rc = main(["--b", str(tmp / "b.jsonl"),
                       "--c0", str(tmp / "c0.jsonl"),
                       "--c1", str(tmp / "c1.jsonl"),
                       "--c1-memory", str(tmp / "c1_mem.jsonl"),
                       "--c0-memory", str(tmp / "c0_mem.jsonl"),
                       "--out-dir", str(c0_dir / "c1")])
            self.assertEqual(rc, 0)
            after = {p: digest(p) for p in c0_dir.rglob("*") if p.is_file()}
            for p, h in before.items():
                self.assertEqual(after[p], h,
                                 f"C0 asset {p.name} was modified")
            self.assertTrue((c0_dir / "c1" / "results-summary.json").exists())


# ------------------------------------------------------- build_summary glue

class TestBuildSummary(unittest.TestCase):
    def test_sections_present(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            b = make_run()
            c0 = make_run()
            c1 = make_run()
            write_run(tmp / "c1.jsonl", c1, make_c1_summary())
            write_memory(tmp / "c1_mem.jsonl", [
                {"signature_tokens": [f"s{i}"], "tier": "simple", "G": -0.3}
                for i in range(24)])
            s = build_summary(b, c0, c1, b_summ=None, c0_summ=None,
                              c1_summ=make_c1_summary()["_summary"],
                              c1_memory_path=tmp / "c1_mem.jsonl",
                              c0_memory_path=tmp / "c0_missing.jsonl")
            for key in ("formal_input_assertions", "groups", "c1_gate",
                        "c1_flips", "paired_bc1", "c0_to_c1", "judge_drift",
                        "memory_retrieval", "truncation_empty_response",
                        "p1_p5", "d10_informational",
                        "prereg_predictions", "notes"):
                self.assertIn(key, s)
            self.assertEqual(s["groups"]["B"]["n_records"], 24)
            self.assertEqual(
                s["memory_retrieval"]["memory_entries_final"], 24)
            self.assertEqual(
                s["prereg_predictions"]["predictions"][3]
                ["actual_n_flips"], 0)
            self.assertTrue(
                s["prereg_predictions"]["predictions"][3]["holds"])


if __name__ == "__main__":
    unittest.main()
