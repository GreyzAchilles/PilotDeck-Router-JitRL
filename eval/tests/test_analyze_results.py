"""Tests for eval/analyze_results.py (pure functions, no network, no CPA).

Covers the directive-critical behaviors:
  1. group metrics: quality mean/distribution, exec vs evaluator cost,
     tier-vs-GT match/over/under, finish_reason=length, latencies;
  2. B/C pairing joins by task_id and refuses mismatched id sets;
     tier changes split into modulation flips vs judge base drift;
  3. flip analysis: quality/cost/GT impact of modulation flips;
  4. learning-curve segment means (tertiles + halves);
  5. failure audit of excluded runs (exec/eval failures only);
  6. D10 gate logic on the FROZEN project-decisions criteria
     (cost savings >=20%, quality C-B >= -0.3, misgrade <= B+5pp,
     JitRL latency increment <=500ms; no pseudo-thresholds);
  7. end-to-end main() writes summary JSON, tasks CSV and SVG charts,
     reframes A's tier-vs-GT rate as a non-routing coincidence, and
     HARD-FAILS unless A/B/C share the exact same task order.
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

from eval.analyze_results import (  # noqa: E402
    audit_failures,
    build_summary,
    consistency_checks,
    d10_judgment,
    flip_analysis,
    group_metrics,
    learning_curve,
    load_group,
    main,
    paired_bc,
)


def rec(task_id, *, gt, chosen, quality=4, cost=0.001, eval_cost=0.005,
        finish="stop", flipped=False, base=None, judge_ms=100.0,
        jitrl_ms=0.0, retrieved=0, top_sim=0.0, reward=0.1,
        evaluator=True, exec_error=None, eval_error=None, family="f",
        mode="judge", empty_response=False, memory=False, V=0.0):
    return {
        "episode_id": 1, "task_id": task_id, "family": family,
        "gt_tier": gt, "chosen_tier": chosen,
        "base_choice": base if base is not None else chosen,
        "route_mode": mode, "flipped": flipped,
        "exec_model": f"CPA/{chosen}", "exec_finish_reason": finish,
        "exec_response_preview": "" if empty_response else "x",
        "exec_cost_usd": cost, "exec_latency_ms": 500.0,
        "exec_error": exec_error,
        "evaluator": None if not evaluator else {
            "quality_score": quality, "verdict": "useful",
            "capability_insufficient": False,
            "evaluator_latency_ms": 1000.0,
        },
        "eval_cost_usd": eval_cost if evaluator else None,
        "eval_error": eval_error,
        "reward": reward if evaluator else None,
        "judge_latency_ms": judge_ms, "jitrl_latency_ms": jitrl_ms,
        "n_retrieved": retrieved, "top_similarity": top_sim, "V": V,
        "memory_written": memory,
    }


def write_jsonl(path: Path, records, summary=None):
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    if summary is not None:
        lines.append(json.dumps({"_summary": summary}, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestGroupMetrics(unittest.TestCase):
    def test_basic_metrics(self):
        rs = [
            rec("T1", gt="simple", chosen="simple", quality=5, cost=0.0002),
            rec("T2", gt="medium", chosen="simple", quality=1, cost=0.0005,
                finish="length", empty_response=True),
            rec("T3", gt="reasoning", chosen="reasoning", quality=4,
                cost=0.004, evaluator=False),
        ]
        m = group_metrics(rs)
        self.assertEqual(m["n_records"], 3)
        self.assertEqual(m["eval_success"], "2/3")
        self.assertAlmostEqual(m["quality"]["mean"], 3.0)
        self.assertEqual(m["quality"]["distribution"]["5"], 1)
        self.assertEqual(m["quality"]["distribution"]["1"], 1)
        self.assertAlmostEqual(m["exec_cost_usd"]["total"], 0.0047, places=6)
        # evaluator budget is separate and only counts evaluated records
        self.assertAlmostEqual(m["eval_cost_usd"]["total"], 0.01, places=6)
        tv = m["tier_vs_gt"]
        self.assertEqual(tv["matches"], 2)  # T3 still counts (labeled, exec ok)
        self.assertEqual(tv["under_provisioned"], 1)
        self.assertEqual(tv["over_provisioned"], 0)
        self.assertAlmostEqual(tv["misgrade_rate"], round(1 / 3, 4))
        self.assertEqual(m["finish_reason_length"], 1)
        self.assertEqual(m["empty_response"], 1)
        self.assertAlmostEqual(m["latency_ms"]["judge_mean"], 100.0)
        self.assertAlmostEqual(m["latency_ms"]["evaluator_mean"], 1000.0)

    def test_over_provision_uses_tier_ordinal(self):
        rs = [rec("T1", gt="medium", chosen="complex")]
        self.assertEqual(group_metrics(rs)["tier_vs_gt"]["over_provisioned"], 1)


class TestPairedAndFlips(unittest.TestCase):
    def setUp(self):
        # 3 tasks; C flips T2 simple->complex (quality same, cost up),
        # improves T1 (same tier, re-sampled); T3 identical.
        self.b = [
            rec("T1", gt="simple", chosen="simple", quality=3, cost=0.001),
            rec("T2", gt="medium", chosen="simple", quality=1, cost=0.001),
            rec("T3", gt="reasoning", chosen="reasoning", quality=5,
                cost=0.004),
        ]
        self.c = [
            rec("T1", gt="simple", chosen="simple", quality=4, cost=0.001),
            rec("T2", gt="medium", chosen="complex", quality=1, cost=0.004,
                flipped=True, base="simple", retrieved=2, top_sim=1.0,
                empty_response=True, finish="length"),
            rec("T3", gt="reasoning", chosen="reasoning", quality=5,
                cost=0.004),
        ]

    def test_paired_quality_and_cost(self):
        p = paired_bc(self.b, self.c)
        self.assertEqual(p["n_pairs"], 3)
        # deltas: T1 +1 (same tier, re-sampled), T2 0 (flip), T3 0
        self.assertAlmostEqual(p["quality"]["mean_delta"], round(1 / 3, 4))
        self.assertEqual(p["quality"]["improved_tasks"], 1)
        self.assertEqual(p["quality"]["worsened_tasks"], 0)
        self.assertEqual(p["quality"]["unchanged_tasks"], 2)
        # total cost 0.006 -> 0.009 = +50% change, i.e. -50% savings
        self.assertAlmostEqual(p["exec_cost"]["total_change_pct"], 50.0)
        self.assertAlmostEqual(p["exec_cost"]["savings_pct_bc"], -50.0)
        self.assertAlmostEqual(p["exec_cost"]["flip_tasks_delta_usd"], 0.003)
        self.assertAlmostEqual(p["exec_cost"]["nonflip_tasks_delta_usd"], 0.0)
        # tier changes: only T2, and it is a modulation flip (no drift)
        self.assertEqual(p["tier_changes"]["n_total"], 1)
        self.assertEqual(p["tier_changes"]["task_ids"], ["T2"])
        self.assertEqual(p["tier_changes"]["modulation_flip_task_ids"],
                         ["T2"])
        self.assertEqual(p["tier_changes"]["judge_base_drift_task_ids"], [])
        # whole quality delta comes from same-tier (non-flip) tasks
        self.assertEqual(p["quality_attribution"]["flip_tasks_quality_delta"], 0)
        self.assertEqual(
            p["quality_attribution"]["judge_drift_tasks_quality_delta"], 0)
        self.assertEqual(
            p["quality_attribution"]["same_tier_tasks_quality_delta"], 1)

    def test_paired_reports_judge_base_drift(self):
        # B judged T2 complex but the C run's judge base was simple:
        # a tier change that is NOT a modulation flip.
        b = [rec("T1", gt="simple", chosen="simple", quality=3, cost=0.001),
             rec("T2", gt="medium", chosen="complex", quality=2, cost=0.004)]
        c = [rec("T1", gt="simple", chosen="simple", quality=3, cost=0.001),
             rec("T2", gt="medium", chosen="simple", quality=4, cost=0.001,
                 base="simple")]
        p = paired_bc(b, c)
        self.assertEqual(p["tier_changes"]["n_total"], 1)
        self.assertEqual(p["tier_changes"]["modulation_flip_task_ids"], [])
        self.assertEqual(p["tier_changes"]["judge_base_drift_task_ids"],
                         ["T2"])
        self.assertAlmostEqual(
            p["quality_attribution"]["judge_drift_tasks_quality_delta"], 2)
        self.assertAlmostEqual(
            p["exec_cost"]["judge_drift_tasks_delta_usd"], -0.003)

    def test_paired_refuses_id_mismatch(self):
        with self.assertRaises(SystemExit):
            paired_bc(self.b, self.c[:2])

    def test_flip_analysis(self):
        f = flip_analysis(self.c, self.b)
        self.assertEqual(f["n_flips"], 1)
        fl = f["flips"][0]
        self.assertEqual(fl["task_id"], "T2")
        self.assertEqual(fl["base_choice"], "simple")
        self.assertEqual(fl["chosen_tier"], "complex")
        self.assertFalse(fl["gt_correct_before"])
        self.assertFalse(fl["gt_correct_after"])
        self.assertEqual(fl["quality_delta"], 0)
        self.assertAlmostEqual(fl["exec_cost_delta_usd"], 0.003)
        self.assertEqual(f["improved"], 0)
        self.assertEqual(f["worsened"], 0)
        self.assertEqual(f["gt_correctness_changes"], 0)


class TestConsistencyAndCurve(unittest.TestCase):
    def test_consistency_detects_order_but_same_set(self):
        a = [rec("T1", gt="simple", chosen="reasoning", mode="static"),
             rec("T2", gt="medium", chosen="reasoning", mode="static")]
        b = [rec("T1", gt="simple", chosen="simple"),
             rec("T2", gt="medium", chosen="simple")]
        c = [rec("T2", gt="medium", chosen="simple"),
             rec("T1", gt="simple", chosen="simple", base="simple")]
        cc = consistency_checks(a, b, c)
        self.assertTrue(cc["abc_same_id_set"])
        self.assertFalse(cc["abc_same_order"])
        self.assertTrue(cc["bc_same_id_set"])
        self.assertFalse(cc["bc_same_order"])
        self.assertEqual(cc["b_order_style"], "numeric_ascending")
        self.assertEqual(cc["c_order_style"], "other")
        self.assertTrue(cc["bc_judge_base_choice_consistent"])
        self.assertEqual(cc["bc_judge_base_choice_mismatches"], [])
        self.assertEqual(cc["route_modes"], {"A": ["static"], "B": ["judge"],
                                             "C": ["judge"]})

    def test_consistency_reports_judge_base_mismatch(self):
        a = [rec("T2", gt="simple", chosen="reasoning", mode="static"),
             rec("T1", gt="simple", chosen="reasoning", mode="static")]
        b = [rec("T2", gt="simple", chosen="complex"),
             rec("T1", gt="simple", chosen="simple")]
        c = [rec("T2", gt="simple", chosen="simple", base="simple"),
             rec("T1", gt="simple", chosen="simple", base="simple")]
        cc = consistency_checks(a, b, c)
        self.assertTrue(cc["abc_same_order"])
        self.assertEqual(cc["b_order_style"], "family_blocked")
        self.assertFalse(cc["bc_judge_base_choice_consistent"])
        self.assertEqual(cc["bc_judge_base_choice_mismatches"],
                         [{"task_id": "T2", "b_chosen_tier": "complex",
                           "c_base_choice": "simple"}])

    def test_learning_curve_segments(self):
        c = [rec(f"T{i:02d}", gt="simple", chosen="simple",
                 quality=q, reward=q / 10.0)
             for i, q in enumerate([5, 5, 5, 1, 1, 1], 1)]
        lc = learning_curve(c)
        self.assertAlmostEqual(lc["segments"]["first_half"]["quality_mean"], 5.0)
        self.assertAlmostEqual(lc["segments"]["second_half"]["quality_mean"], 1.0)
        self.assertAlmostEqual(lc["segments"]["pre_tertile"]["quality_mean"], 5.0)
        self.assertAlmostEqual(lc["segments"]["post_tertile"]["quality_mean"], 1.0)
        self.assertIn("confound", lc)


class TestAuditAndD10(unittest.TestCase):
    def test_audit_failures_lists_both_kinds(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "run.jsonl"
            write_jsonl(p, [
                rec("T1", gt="simple", chosen="simple"),
                rec("T2", gt="simple", chosen="simple", evaluator=False,
                    exec_error="CPATransportError: HTTP 500"),
                rec("T3", gt="simple", chosen="simple", evaluator=False,
                    eval_error="evaluator failed (transport/parse/validate)",
                    finish="length"),
            ])
            audit = audit_failures([p])
            self.assertEqual(len(audit), 1)
            self.assertEqual(audit[0]["eval_success"], "1/3")
            self.assertEqual({f["task_id"] for f in audit[0]["failures"]},
                             {"T2", "T3"})

    def test_d10_frozen_criteria(self):
        def mk(total_b=0.03, total_c=0.02, q_b=2.5, q_c=2.6,
               mis_b=0.70, mis_c=0.72, jitrl_c_mean=0.24, jitrl_c_max=0.67):
            return {
                "groups": {
                    "B": {"quality": {"mean": q_b},
                          "tier_vs_gt": {"misgrade_rate": mis_b},
                          "latency_ms": {"jitrl_mean": 0.0,
                                         "jitrl_max": 0.0}},
                    "C": {"quality": {"mean": q_c},
                          "tier_vs_gt": {"misgrade_rate": mis_c},
                          "latency_ms": {"jitrl_mean": jitrl_c_mean,
                                         "jitrl_max": jitrl_c_max}},
                },
                "paired_bc": {"exec_cost": {"total_B_usd": total_b,
                                            "total_C_usd": total_c}},
                # flip worsening is NOT a D10 criterion anymore
                "flips": {"worsened": 1},
            }
        # all four frozen criteria pass: savings 33.33% >= 20%, quality
        # +0.1 >= -0.3, misgrade +2pp <= +5pp, latency 0.67ms <= 500ms
        out = d10_judgment(mk())
        self.assertIn("MET", out["verdict"])
        self.assertEqual(out["n_criteria_met"], 4)
        self.assertTrue(all(cr["met"] for cr in out["criteria"]))
        # quality boundary: delta of exactly -0.3 passes, below it fails
        # (3.0 -> 2.7 is a float-exact -0.2999... >= -0.3)
        self.assertTrue(
            d10_judgment(mk(q_b=3.0, q_c=2.7))["criteria"][1]["met"])
        self.assertFalse(
            d10_judgment(mk(q_b=3.0, q_c=2.6))["criteria"][1]["met"])
        # cost: C more expensive than B -> negative savings -> fail
        out2 = d10_judgment(mk(total_b=0.028333, total_c=0.029406))
        self.assertEqual(out2["verdict"], "NOT MET")
        self.assertFalse(out2["criteria"][0]["met"])
        self.assertTrue(out2["criteria"][1]["met"])
        # misgrade: exactly +5pp passes, +6pp fails
        self.assertTrue(d10_judgment(mk(mis_c=0.75))["criteria"][2]["met"])
        self.assertFalse(d10_judgment(mk(mis_c=0.76))["criteria"][2]["met"])
        # JitRL latency increment: 500ms passes, above fails (judged on max)
        self.assertTrue(
            d10_judgment(mk(jitrl_c_max=500.0))["criteria"][3]["met"])
        self.assertFalse(
            d10_judgment(mk(jitrl_c_max=500.1))["criteria"][3]["met"])


class TestEndToEnd(unittest.TestCase):
    def test_main_writes_all_assets(self):
        with tempfile.TemporaryDirectory() as td:
            tp = Path(td)
            a = [rec(f"T{i:02d}", gt="reasoning", chosen="reasoning",
                     quality=2, mode="static", base="reasoning")
                 for i in (1, 2, 3)]
            b = [rec(f"T{i:02d}", gt="simple" if i == 1 else "medium",
                     chosen="simple", quality=3)
                 for i in (1, 2, 3)]
            # C in the SAME family-blocked order as A/B, with one flip
            c = [rec("T01", gt="simple", chosen="simple", quality=5,
                     mode="jitrl", jitrl_ms=0.5, memory=True),
                 rec("T02", gt="medium", chosen="complex", quality=3,
                     flipped=True, base="simple", mode="jitrl",
                     jitrl_ms=0.5, memory=True),
                 rec("T03", gt="medium", chosen="simple", quality=3,
                     mode="jitrl", jitrl_ms=0.5, memory=True)]
            failed = [rec("T01", gt="simple", chosen="simple", evaluator=False,
                          exec_error="CPATransportError: HTTP 500")]
            pa, pb, pc = tp / "a.jsonl", tp / "b.jsonl", tp / "c.jsonl"
            pf = tp / "failed.jsonl"
            pm = tp / "mem.jsonl"
            write_jsonl(pa, a, {"mode": "A"})
            write_jsonl(pb, b, {"mode": "B"})
            write_jsonl(pc, c, {"mode": "C"})
            write_jsonl(pf, failed)
            pm.write_text('{"sig": 1}\n{"sig": 2}\n', encoding="utf-8")
            out = tp / "out"
            rc = main(["--a", str(pa), "--b", str(pb), "--c", str(pc),
                       "--audit-c", str(pf), "--c-memory", str(pm),
                       "--out-dir", str(out)])
            self.assertEqual(rc, 0)
            summary = json.loads(
                (out / "results-summary.json").read_text(encoding="utf-8"))
            # A reframed: coincidence, never a routing accuracy
            self.assertIn("static_tier_gt_coincidence", summary["groups"]["A"])
            self.assertNotIn("tier_vs_gt", summary["groups"]["A"])
            self.assertEqual(summary["groups"]["C"]["flips"], 1)
            self.assertEqual(summary["failure_audit"][0]["eval_success"], "0/1")
            self.assertEqual(summary["retrieval_memory"]["memory_entries_final"], 2)
            # hard order assertion holds for same-order inputs
            self.assertTrue(summary["consistency"]["abc_same_id_set"])
            self.assertTrue(summary["consistency"]["abc_same_order"])
            self.assertTrue(summary["consistency"]["bc_same_id_set"])
            self.assertTrue(summary["consistency"]["bc_same_order"])
            csv_text = (out / "results-tasks.csv").read_text(encoding="utf-8")
            self.assertEqual(csv_text.count("\n"), 10)  # header + 9 rows
            charts = sorted((out / "charts").glob("*.svg"))
            self.assertEqual(len(charts), 5)
            for ch in charts:
                text = ch.read_text(encoding="utf-8")
                self.assertIn("<svg", text)
                self.assertIn("</svg>", text)

    def test_main_refuses_abc_order_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            tp = Path(td)
            a = [rec(f"T{i:02d}", gt="reasoning", chosen="reasoning",
                     mode="static", base="reasoning")
                 for i in (1, 2, 3)]
            b = [rec(f"T{i:02d}", gt="simple" if i == 1 else "medium",
                     chosen="simple")
                 for i in (1, 2, 3)]
            # C transposed (T02 first) -> order differs from A/B
            c = [rec("T02", gt="medium", chosen="simple", mode="jitrl"),
                 rec("T01", gt="simple", chosen="simple", mode="jitrl"),
                 rec("T03", gt="medium", chosen="simple", mode="jitrl")]
            pa, pb, pc = tp / "a.jsonl", tp / "b.jsonl", tp / "c.jsonl"
            write_jsonl(pa, a)
            write_jsonl(pb, b)
            write_jsonl(pc, c)
            out = tp / "out"
            with self.assertRaises(SystemExit) as ctx:
                main(["--a", str(pa), "--b", str(pb), "--c", str(pc),
                      "--audit-c", "--c-memory", str(tp / "none.jsonl"),
                      "--out-dir", str(out)])
            self.assertIn("consistency assertion failed",
                          str(ctx.exception))
            self.assertFalse(out.exists())  # nothing written on refusal


if __name__ == "__main__":
    unittest.main()
