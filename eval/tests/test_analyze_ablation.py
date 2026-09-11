"""Tests for eval/analyze_ablation.py (S3-TAB8-PREREGISTRATION §10.12).

Covers the analyzer's formal-input hard assertions and output boundaries:

  - a VALID mock-produced T8-LM/T8-PM pair (real frozen trace, real tasks,
    mock judge + scripted CPA) passes every hard assertion end to end and
    writes results-summary.json / results-tasks.csv / charts only under the
    out dir;
  - order violations (swapped records, missing record) are rejected;
  - config violations (tampered JitRL snapshot, tampered ablation_config,
    wrong arm label, wrong episode count) are rejected;
  - hash violations (record memory_source_sha256, trace file vs sidecar,
    sidecar task order, C0 source memory digest) are rejected;
  - 24/24 completeness violations (exec failure, eval failure) are rejected;
  - mechanism purity violations (LM injecting, PM modulating) are rejected;
  - prefix-replay and cross-arm fairness violations are rejected;
  - prompt/block byte-integrity violations are rejected (hash tamper,
    z_mod formula tamper);
  - protected out dirs (eval root, eval/c1, eval/charts, logs, poster-a3,
    repo root) are refused while eval/ablation and external tmp dirs pass;
  - conclusion rule ordering and hypothesis boundaries (H1/H2/H4).

No network, no CPA, no writes outside TemporaryDirectory (plus read-only
use of the repo's frozen assets).
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import eval.analyze_ablation as aa  # noqa: E402
from harness.cpa_client import CompletionResult  # noqa: E402
import harness.run_ablation as ra  # noqa: E402

TRACE_PATH = REPO_ROOT / "eval" / "ablation" / "frozen-c0-memory-trace.jsonl"
SHA_JSON_PATH = REPO_ROOT / "eval" / "ablation" / "frozen-c0-memory-trace.sha256.json"
TASKS_PATH = REPO_ROOT / "eval" / "tasks.jsonl"

EVAL_JSON = json.dumps({
    "quality_score": 4, "verdict": "useful", "certainty": 0.8,
    "capability_insufficient": False,
    "checklist": {"hits": ["x"], "misses": [], "notes": "ok"},
}, ensure_ascii=False)


def usage(n_in=100, n_out=50):
    return {"input_tokens": n_in, "cache_read_tokens": 0, "cache_write_tokens": 0,
            "output_tokens": n_out, "total_tokens": n_in + n_out, "native_cost": None}


class FakeCPAClient:
    def completion(self, model, messages, *, max_tokens=512, temperature=0.7,
                   extra_body=None):
        api = model.split("/")[-1]
        content = EVAL_JSON if api == "gpt-5.6-sol" else "exec reply text"
        return CompletionResult(content=content, usage=usage(), model=api,
                                finish_reason="stop", latency_ms=5.0)


def _run_mock_pair(tmp: Path) -> tuple[Path, Path]:
    lm, pm = tmp / "lm.jsonl", tmp / "pm.jsonl"
    ra.run(arm="T8-LM", tasks_path=TASKS_PATH, memory_trace_path=TRACE_PATH,
           judge_kind="mock", out_path=lm, cpa_client=FakeCPAClient())
    ra.run(arm="T8-PM", tasks_path=TASKS_PATH, memory_trace_path=TRACE_PATH,
           judge_kind="mock", out_path=pm, cpa_client=FakeCPAClient())
    return lm, pm


def _read_log(path: Path) -> tuple[list[dict], dict]:
    records, summary = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if "_summary" in obj:
            summary = obj["_summary"]
        else:
            records.append(obj)
    return records, summary


def _write_log(path: Path, records: list[dict], summary: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.write(json.dumps({"_summary": summary}, ensure_ascii=False) + "\n")


def _analyze(lm: Path, pm: Path, out_dir: Path, *,
             trace=TRACE_PATH, sha_json=SHA_JSON_PATH):
    return aa.main(["--lm", str(lm), "--pm", str(pm), "--out-dir", str(out_dir),
                    "--memory-trace", str(trace), "--trace-sha-json",
                    str(sha_json), "--tasks", str(TASKS_PATH)])


class AnalyzerTestBase(unittest.TestCase):
    """Builds one valid mock pair per class; mutation tests copy + tweak."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.lm_path, cls.pm_path = _run_mock_pair(tmp)
        cls.lm_records, cls.lm_summary = _read_log(cls.lm_path)
        cls.pm_records, cls.pm_summary = _read_log(cls.pm_path)
        assert len(cls.lm_records) == 24 and len(cls.pm_records) == 24

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def mutated(self, *, arm="T8-LM", rec=None, summary=None) -> Path:
        """Write a mutated copy of one arm log into a fresh tmp file."""
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: _rmtree(tmp))
        if arm == "T8-LM":
            records, summ = self.lm_records, self.lm_summary
        else:
            records, summ = self.pm_records, self.pm_summary
        records = [dict(r) for r in records]
        for r in records:
            for k in ("z_base", "z_mod", "z_prompt", "A_norm", "Q"):
                if r.get(k) is not None:
                    r[k] = dict(r[k])
        if rec is not None:
            rec(records)
        if summary is not None:
            summ = summary(summ)
        out = tmp / f"mut_{arm.replace('-', '_')}.jsonl"
        _write_log(out, records, summ)
        return out

    def expect_reject(self, lm, pm, tag: str) -> None:
        out_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: _rmtree(out_dir))
        with self.assertRaises(SystemExit) as cm:
            _analyze(lm, pm, out_dir)
        self.assertIn("FORMAL-INPUT REJECTED", str(cm.exception))
        self.assertIn(f"({tag})", str(cm.exception),
                      f"expected tag ({tag}) in: {cm.exception}")


def _rmtree(path: Path):
    import shutil
    shutil.rmtree(path, ignore_errors=True)


# ------------------------------------------------- §10.12 hard assertions

class TestFormalInputValidation(AnalyzerTestBase):

    def test_valid_pair_passes_and_writes_outputs(self):
        out_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: _rmtree(out_dir))
        rc = _analyze(self.lm_path, self.pm_path, out_dir)
        self.assertEqual(rc, 0)
        summary = json.loads((out_dir / "results-summary.json")
                             .read_text(encoding="utf-8"))
        self.assertEqual(summary["preregistration"],
                         "S3-TAB8-PREREGISTRATION.md v1.0")
        self.assertIn("T8-LM", summary["arms"])
        self.assertIn("T8-PM", summary["arms"])
        self.assertEqual(summary["paired"]["n_pairs"], 24)
        self.assertIn("H1_mechanism_influence", summary["hypotheses"])
        self.assertIn("rule", summary["conclusion"])
        csv_text = (out_dir / "results-tasks.csv").read_text(encoding="utf-8")
        self.assertEqual(len(csv_text.splitlines()), 25)  # header + 24
        charts = sorted(p.name for p in (out_dir / "charts").glob("*.svg"))
        self.assertEqual(len(charts), 4)
        # nothing written outside the out dir
        self.assertEqual(sorted(p.name for p in out_dir.iterdir()),
                         ["charts", "results-summary.json", "results-tasks.csv"])

    def test_rejects_swapped_order(self):
        def rec(records):
            records[3], records[4] = records[4], records[3]
        self.expect_reject(self.mutated(rec=rec), self.pm_path, "order")

    def test_rejects_missing_record(self):
        lm = self.mutated(rec=lambda records: records.pop(9))
        self.expect_reject(lm, self.pm_path, "order")

    def test_rejects_exec_failure(self):
        def rec(records):
            records[5]["exec_error"] = "CPATransportError: boom"
        self.expect_reject(self.mutated(rec=rec), self.pm_path, "completeness")

    def test_rejects_eval_failure(self):
        def rec(records):
            records[5]["eval_error"] = "evaluator failed"
            records[5]["evaluator"] = None
        self.expect_reject(self.mutated(rec=rec), self.pm_path, "completeness")

    def test_rejects_wrong_arm_label(self):
        def rec(records):
            records[0]["ablation_arm"] = "T8-PM"
        self.expect_reject(self.mutated(rec=rec), self.pm_path, "config")

    def test_rejects_tampered_jitrl_snapshot(self):
        def summary(s):
            s = dict(s)
            s["config_snapshot"] = dict(s["config_snapshot"])
            s["config_snapshot"]["beta"] = 4.0
            return s
        self.expect_reject(self.mutated(summary=summary), self.pm_path, "config")

    def test_rejects_tampered_ablation_config(self):
        def summary(s):
            s = dict(s)
            s["ablation_config"] = dict(s["ablation_config"])
            s["ablation_config"]["temperature_exec"] = 0.0
            return s
        self.expect_reject(self.mutated(summary=summary), self.pm_path, "config")

    def test_rejects_wrong_episode_count(self):
        def summary(s):
            s = dict(s, episodes=2)
            return s
        self.expect_reject(self.mutated(summary=summary), self.pm_path, "config")

    def test_rejects_record_hash_tamper(self):
        def rec(records):
            records[0]["memory_source_sha256"] = "0" * 64
        self.expect_reject(self.mutated(rec=rec), self.pm_path, "hash")

    def test_rejects_trace_file_tamper(self):
        # trace content changed but the (real) sidecar still holds the
        # frozen digest -> hash rejection before anything else
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: _rmtree(tmp))
        tampered = tmp / "tampered-trace.jsonl"
        lines = TRACE_PATH.read_text(encoding="utf-8").splitlines()
        first = json.loads(lines[0])
        first["G"] = first["G"] + 0.5
        lines[0] = json.dumps(first, ensure_ascii=False)
        tampered.write_text("\n".join(lines) + "\n", encoding="utf-8")
        out_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: _rmtree(out_dir))
        with self.assertRaises(SystemExit) as cm:
            _analyze(self.lm_path, self.pm_path, out_dir, trace=tampered)
        self.assertIn("(hash)", str(cm.exception))

    def test_rejects_sidecar_order_tamper(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: _rmtree(tmp))
        bad_sidecar = tmp / "bad.sha256.json"
        meta = json.loads(SHA_JSON_PATH.read_text(encoding="utf-8"))
        meta["frozen_task_order"] = list(reversed(meta["frozen_task_order"]))
        bad_sidecar.write_text(json.dumps(meta), encoding="utf-8")
        out_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: _rmtree(out_dir))
        with self.assertRaises(SystemExit) as cm:
            _analyze(self.lm_path, self.pm_path, out_dir, sha_json=bad_sidecar)
        self.assertIn("(order)", str(cm.exception))

    def test_rejects_lm_injection(self):
        def rec(records):
            records[7]["injected_block_chars"] = 42  # T06: retrieval hit
        self.expect_reject(self.mutated(rec=rec), self.pm_path, "config")

    def test_rejects_pm_modulation(self):
        def rec(records):
            records[7]["z_mod"] = dict(records[7]["z_prompt"])
        self.expect_reject(self.lm_path,
                           self.mutated(arm="T8-PM", rec=rec), "config")

    def test_rejects_prefix_size_tamper(self):
        def rec(records):
            records[7]["memory_prefix_size"] = 0
        self.expect_reject(self.lm_path,
                           self.mutated(arm="T8-PM", rec=rec), "config")

    def test_rejects_prompt_hash_tamper(self):
        def rec(records):
            records[0]["base_prompt_sha256"] = "0" * 64
        self.expect_reject(self.mutated(rec=rec), self.pm_path, "math")

    def test_rejects_block_hash_tamper(self):
        def rec(records):
            hit = next(r for r in records if r["injected_block_chars"] > 0)
            hit["injected_block_sha256"] = "0" * 64
        self.expect_reject(self.lm_path,
                           self.mutated(arm="T8-PM", rec=rec), "math")

    def test_rejects_zmod_formula_tamper(self):
        def rec(records):
            records[7]["z_mod"]["simple"] += 0.5
        self.expect_reject(self.mutated(rec=rec), self.pm_path, "math")

    def test_fairness_violation_rejected(self):
        # unit level: differing retrieval sets between arms must be refused
        lm = [dict(r) for r in self.lm_records]
        pm = [dict(r) for r in self.pm_records]
        pm[7]["retrieved_memories"] = []
        with self.assertRaises(SystemExit) as cm:
            aa.validate_fairness(lm, pm)
        self.assertIn("(fairness)", str(cm.exception))

    def test_validate_trace_rejects_source_digest_change(self):
        # synthetic sidecar whose source hash no longer matches the file
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: _rmtree(tmp))
        bad = tmp / "bad.sha256.json"
        meta = json.loads(SHA_JSON_PATH.read_text(encoding="utf-8"))
        meta["source"]["memory_sha256"] = "0" * 64
        bad.write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaises(SystemExit) as cm:
            aa.validate_trace(TRACE_PATH, bad)
        self.assertIn("(hash)", str(cm.exception))


# ------------------------------------------------------ output boundaries

class TestOutDirProtection(unittest.TestCase):
    def test_protected_dirs_refused(self):
        for rel in ("eval", "eval/c1", "eval/charts", "logs", "poster-a3", "."):
            with self.assertRaises(SystemExit, msg=rel):
                aa.ensure_safe_out_dir(REPO_ROOT / rel)

    def test_ablation_dir_and_tmp_allowed(self):
        self.assertEqual(aa.ensure_safe_out_dir(REPO_ROOT / "eval" / "ablation"),
                         (REPO_ROOT / "eval" / "ablation").resolve())
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(aa.ensure_safe_out_dir(tmp), Path(tmp).resolve())
        # nested subdirectory of eval/ablation is fine too
        nested = REPO_ROOT / "eval" / "ablation" / "sub"
        self.assertEqual(aa.ensure_safe_out_dir(nested), nested.resolve())


# ------------------------------------------------- conclusion + hypotheses

def _arm_metrics(quality_mean, exec_total, *, jitrl_max=1.0, influenced=0):
    return {
        "quality": {"mean": quality_mean},
        "exec_cost_usd": {"total": exec_total},
        "latency_ms": {"jitrl_max": jitrl_max, "base_judge_mean": 10.0},
        "influenced_records": influenced,
        "retrieval_hits": 6,
    }


class TestConclusionRules(unittest.TestCase):
    def test_no_substantive_difference_takes_precedence(self):
        # tiny edges on BOTH axes must not be overclaimed as LM-favorable
        lm = _arm_metrics(4.1, 0.0300)
        pm = _arm_metrics(4.0, 0.0304)
        c = aa.conclude(lm, pm)
        self.assertEqual(c["rule"], "no_substantive_difference")

    def test_favors_logit_modulation(self):
        lm = _arm_metrics(4.0, 0.020)
        pm = _arm_metrics(3.9, 0.030)
        c = aa.conclude(lm, pm)
        self.assertEqual(c["rule"], "favors_logit_modulation")

    def test_quality_non_inferior_boundary_is_inclusive(self):
        lm = _arm_metrics(3.7, 0.020)   # delta exactly -0.3
        pm = _arm_metrics(4.0, 0.030)
        c = aa.conclude(lm, pm)
        self.assertEqual(c["rule"], "favors_logit_modulation")

    def test_favors_prompt_injection_on_cost(self):
        lm = _arm_metrics(4.0, 0.040)
        pm = _arm_metrics(4.0, 0.030)
        c = aa.conclude(lm, pm)
        self.assertEqual(c["rule"], "favors_prompt_injection")

    def test_favors_prompt_injection_on_quality_beyond_margin(self):
        lm = _arm_metrics(3.0, 0.020)
        pm = _arm_metrics(4.0, 0.030)
        c = aa.conclude(lm, pm)
        self.assertEqual(c["rule"], "favors_prompt_injection")

    def test_not_covered_when_lm_quality_edge_without_cost_edge(self):
        # LM clearly better in quality but NOT cheaper: H3 forbids the
        # LM-favorable claim and PM is not better either -> not covered
        lm = _arm_metrics(4.5, 0.030)
        pm = _arm_metrics(4.0, 0.030)
        c = aa.conclude(lm, pm)
        self.assertEqual(c["rule"], "not_covered")

    def test_pm_better_quality_with_cost_tie_favors_pm(self):
        lm = _arm_metrics(3.0, 0.030)
        pm = _arm_metrics(4.0, 0.030)
        c = aa.conclude(lm, pm)
        self.assertEqual(c["rule"], "favors_prompt_injection")


class TestHypothesisBoundaries(unittest.TestCase):
    def test_h1_requires_at_least_one_influence(self):
        h = aa.evaluate_hypotheses(
            _arm_metrics(4, 0.03, influenced=0),
            _arm_metrics(4, 0.03, influenced=0),
            {"exec_cost": {"total_LM_usd": 0.03, "total_PM_usd": 0.03},
             "judge_drift_audit": {"base_choice_mismatch_count": 0,
                                   "base_choice_mismatch_tasks": [],
                                   "per_tier_z_base_diff": {}}})
        self.assertFalse(h["H1_mechanism_influence"]["met"])
        self.assertEqual(h["H1_mechanism_influence"]["reading"],
                         "at least one mechanism changed the raw Judge choice on "
                         "a retrieval-hit task; if both are 0 the prereg reading "
                         "is 'weak base policy / memory-format dominated, "
                         "mechanism difference not observable'.")

    def test_h2_boundary_inclusive(self):
        lm = _arm_metrics(3.7, 0.03)
        pm = _arm_metrics(4.0, 0.03)
        h = aa.evaluate_hypotheses(
            lm, pm, {"exec_cost": {"total_LM_usd": 0.03, "total_PM_usd": 0.03},
                     "judge_drift_audit": {"base_choice_mismatch_count": 0,
                                           "base_choice_mismatch_tasks": [],
                                           "per_tier_z_base_diff": {}}})
        self.assertTrue(h["H2_quality_non_inferior"]["met"])

    def test_h4_boundary_inclusive(self):
        lm = _arm_metrics(4, 0.03, jitrl_max=500.0)
        pm = _arm_metrics(4, 0.03)
        h = aa.evaluate_hypotheses(
            lm, pm, {"exec_cost": {"total_LM_usd": 0.03, "total_PM_usd": 0.03},
                     "judge_drift_audit": {"base_choice_mismatch_count": 0,
                                           "base_choice_mismatch_tasks": [],
                                           "per_tier_z_base_diff": {}}})
        self.assertTrue(h["H4_latency_attribution"]["met"])
        lm_over = _arm_metrics(4, 0.03, jitrl_max=500.001)
        h2 = aa.evaluate_hypotheses(
            lm_over, pm, {"exec_cost": {"total_LM_usd": 0.03,
                                        "total_PM_usd": 0.03},
                          "judge_drift_audit": {"base_choice_mismatch_count": 0,
                                                "base_choice_mismatch_tasks": [],
                                                "per_tier_z_base_diff": {}}})
        self.assertFalse(h2["H4_latency_attribution"]["met"])

    def test_h3_ratio_and_lm_claim_gate(self):
        lm = _arm_metrics(4.0, 0.020)
        pm = _arm_metrics(4.0, 0.030)
        h = aa.evaluate_hypotheses(
            lm, pm, {"exec_cost": {"total_LM_usd": 0.020,
                                   "total_PM_usd": 0.030},
                     "judge_drift_audit": {"base_choice_mismatch_count": 0,
                                           "base_choice_mismatch_tasks": [],
                                           "per_tier_z_base_diff": {}}})
        self.assertAlmostEqual(h["H3_cost_comparison"]["savings_ratio_PM_to_LM"],
                               1 / 3, places=3)
        self.assertTrue(h["H3_cost_comparison"]["lm_favorable_claim_allowed"])


if __name__ == "__main__":
    unittest.main()
