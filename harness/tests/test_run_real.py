"""Unit tests for the Evaluator + real-episode harness (mocked network)."""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from harness.cpa_client import CPATransportError, CompletionResult
from harness.evaluator import Evaluator, extract_json, validate_evaluation
from harness.real_rewards import RealQualityReward
from harness import run_real
from jitrl_core.config import TIERS
from jitrl_core.policy import modulate_logits

TASKS = [
    {
        "task_id": "R01", "family": "chat_qa", "gt_tier": "simple",
        "message": "你好呀，简单打个招呼",
        "quality_checklist": ["回复包含友好问候"],
    },
    {
        "task_id": "R02", "family": "data_analysis", "gt_tier": "reasoning",
        "message": "分析这组数据：月:1,2,3 销量:10,50,11，指出峰值并解释",
        "quality_checklist": ["指出第 2 月销量最高", "提醒异常需结合业务解释"],
    },
]

EVAL_JSON = json.dumps({
    "quality_score": 4, "verdict": "useful", "certainty": 0.8,
    "capability_insufficient": False,
    "checklist": {"hits": ["指出第 2 月销量最高"], "misses": ["提醒异常需结合业务解释"],
                  "notes": "基本完成"},
}, ensure_ascii=False)


def usage(n_in=100, n_out=50, cache_read=0):
    return {"input_tokens": n_in, "cache_read_tokens": cache_read,
            "cache_write_tokens": 0, "output_tokens": n_out,
            "total_tokens": n_in + cache_read + n_out, "native_cost": None}


class FakeCPAClient:
    """Scripted stand-in: distinguishes evaluator calls by model name."""

    def __init__(self, exec_content="这是执行模型的回复。",
                 eval_content=EVAL_JSON, eval_fail=False,
                 exec_usage=None, eval_usage=None):
        self.exec_content = exec_content
        self.eval_content = eval_content
        self.eval_fail = eval_fail
        self.exec_usage = exec_usage or usage(1000, 200)
        self.eval_usage = eval_usage or usage(1500, 200)
        self.exec_calls: list[tuple] = []
        self.eval_calls: list[tuple] = []

    def completion(self, model, messages, *, max_tokens=512,
                   temperature=0.7, extra_body=None):
        api = model.split("/")[-1]
        if api == "gpt-5.6-sol":
            self.eval_calls.append((model, messages, max_tokens))
            if self.eval_fail:
                raise CPATransportError("CPA request failed (HTTP 503)")
            return CompletionResult(content=self.eval_content, usage=self.eval_usage,
                                    model=api, finish_reason="stop", latency_ms=5.0)
        self.exec_calls.append((model, messages, max_tokens))
        return CompletionResult(content=self.exec_content, usage=self.exec_usage,
                                model=api, finish_reason="stop", latency_ms=5.0)


def write_tasks(tmp: str) -> Path:
    p = Path(tmp) / "tasks.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for t in TASKS:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    return p


def run_mode(mode, tmp, cpa=None, episodes=1, judge="mock", **run_kwargs):
    out = Path(tmp) / f"out_{mode}.jsonl"
    mem = Path(tmp) / f"mem_{mode}.jsonl"
    cpa = cpa or FakeCPAClient()
    summary = run_real.run(
        mode=mode, tasks_path=write_tasks(tmp), judge_kind=judge,
        out_path=out, memory_out_path=mem, episodes=episodes,
        cpa_client=cpa, **run_kwargs,
    )
    records, summary_line = [], None
    with open(out, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "_summary" in obj:
                summary_line = obj["_summary"]
            else:
                records.append(obj)
    with open(mem, encoding="utf-8") as f:
        mem_entries = [json.loads(l) for l in f if l.strip()]
    return summary, summary_line, records, mem_entries, cpa, out


# ------------------------------------------------------------- JSON parsing
class TestExtractJson(unittest.TestCase):
    def test_bare_json(self):
        obj = extract_json('{"quality_score": 5}')
        self.assertEqual(obj["quality_score"], 5)

    def test_fenced_json(self):
        obj = extract_json('评分如下：\n```json\n{"quality_score": 3}\n```\n完毕')
        self.assertEqual(obj["quality_score"], 3)

    def test_prose_wrapped(self):
        obj = extract_json('好的，我的评审结果是 {"quality_score": 2, '
                           '"verdict": "harmful"} 请查收')
        self.assertEqual(obj["verdict"], "harmful")

    def test_nested_braces_in_strings(self):
        obj = extract_json('{"quality_score": 4, "checklist": '
                           '{"hits": ["a{b}c"], "misses": []}}')
        self.assertEqual(obj["checklist"]["hits"], ["a{b}c"])

    def test_invalid_returns_none(self):
        for bad in ("", "no json at all", "{broken", '{"quality_score": }',
                    "```\nnot json\n```"):
            self.assertIsNone(extract_json(bad), msg=bad)


class TestValidateEvaluation(unittest.TestCase):
    def test_valid_full(self):
        res = validate_evaluation(json.loads(EVAL_JSON))
        self.assertEqual(res.quality_score, 4)
        self.assertEqual(res.verdict, "useful")
        self.assertEqual(res.certainty, 0.8)
        self.assertFalse(res.capability_insufficient)
        self.assertEqual(len(res.checklist_hits), 1)
        self.assertEqual(len(res.checklist_misses), 1)

    def test_verdict_case_normalized(self):
        res = validate_evaluation({"quality_score": 5, "verdict": "USEFUL",
                                   "certainty": 0.9,
                                   "capability_insufficient": "yes"})
        self.assertEqual(res.verdict, "useful")
        self.assertTrue(res.capability_insufficient)

    def test_score_out_of_range_rejected(self):
        for bad in (0, 6, "3.5", None):
            self.assertIsNone(validate_evaluation(
                {"quality_score": bad, "verdict": "useful", "certainty": 0.5}))

    def test_bad_verdict_rejected(self):
        self.assertIsNone(validate_evaluation(
            {"quality_score": 3, "verdict": "excellent", "certainty": 0.5}))

    def test_certainty_clamped_not_fatal(self):
        res = validate_evaluation({"quality_score": 3, "verdict": "neutral",
                                   "certainty": 7})
        self.assertEqual(res.certainty, 1.0)


# --------------------------------------------------------------- evaluator
class TestEvaluator(unittest.TestCase):
    def test_success_path(self):
        cpa = FakeCPAClient()
        ev = Evaluator(client=cpa)
        res = ev.evaluate(TASKS[1], "分析结果：第2月峰值50")
        self.assertIsNotNone(res)
        self.assertEqual(res.quality_score, 4)
        self.assertEqual(len(cpa.eval_calls), 1)
        model, messages, max_tokens = cpa.eval_calls[0]
        self.assertEqual(model, "CPA/gpt-5.6-sol")
        self.assertLessEqual(max_tokens, 300)
        # blind: prompt must not mention which model produced the response
        self.assertNotIn("glm-5.3", messages[0]["content"].split("## 待评审回复")[0])

    def test_transport_failure_returns_none(self):
        ev = Evaluator(client=FakeCPAClient(eval_fail=True))
        self.assertIsNone(ev.evaluate(TASKS[0], "hi"))

    def test_unparseable_returns_none(self):
        ev = Evaluator(client=FakeCPAClient(eval_content="我觉得挺好的，但是无法给出 JSON"))
        self.assertIsNone(ev.evaluate(TASKS[0], "hi"))

    def test_invalid_schema_returns_none(self):
        ev = Evaluator(client=FakeCPAClient(
            eval_content='{"quality_score": 9, "verdict": "great"}'))
        self.assertIsNone(ev.evaluate(TASKS[0], "hi"))

    def test_prompt_includes_checklist(self):
        ev = Evaluator(client=FakeCPAClient())
        prompt = ev.build_prompt(TASKS[1], "resp")
        self.assertIn("指出第 2 月销量最高", prompt)
        prompt2 = ev.build_prompt({"message": "hi"}, "resp")
        self.assertIn("未提供评分要点", prompt2)


# ------------------------------------------------------------ reward math
class TestRealReward(unittest.TestCase):
    def test_reward_math_known_usage(self):
        rw = RealQualityReward()
        # glm-5.3-flash execution, 1000 net in / 200 out
        u = usage(1000, 200)
        r, b = rw.reward(5, "CPA/glm-5.3-flash", u)
        actual = (1000 * 0.15 + 200 * 0.50) / 1e6
        default = (1000 * 1.40 + 200 * 4.40) / 1e6
        expected = 0.6 * 1.0 + 0.3 * ((default - actual) / default)
        self.assertAlmostEqual(r, expected, places=6)
        self.assertEqual(b["quality_term"], 1.0)

    def test_quality_term_scale(self):
        rw = RealQualityReward()
        self.assertEqual(rw.quality_term(5), 1.0)
        self.assertEqual(rw.quality_term(3), 0.0)
        self.assertEqual(rw.quality_term(1), -1.0)

    def test_premium_model_negative_cost_saving(self):
        rw = RealQualityReward()
        # OpenBMB-5.3 shares glm-5.3's price entry -> cost_saving ~ 0
        c = rw.cost_saving("CPA/OpenBMB-5.3", usage(1000, 200))
        self.assertAlmostEqual(c, 0.0, places=9)
        # gpt-5.6-sol is far pricier than the default -> negative saving
        c2 = rw.cost_saving("CPA/gpt-5.6-sol", usage(1000, 200))
        self.assertLess(c2, 0.0)


# ------------------------------------------------------------- run_real ABC
class TestRunRealFlow(unittest.TestCase):
    def test_mode_A_semantics_fixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary, _, records, mem, cpa, _ = run_mode("A", tmp)
        self.assertEqual(mem, [])                    # A never writes memory
        for r in records:
            self.assertEqual(r["chosen_tier"], "reasoning")   # actual executed tier
            self.assertEqual(r["route_mode"], "static")
            self.assertEqual(r["exec_model"], "CPA/glm-5.3")  # D14 static default
            self.assertFalse(r["memory_written"])
        self.assertEqual(summary["accuracy_vs_gt"], 0.5)      # R01 simple vs reasoning

    def test_mode_B_judge_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary, _, records, mem, cpa, _ = run_mode("B", tmp)
        self.assertEqual(mem, [])
        for r in records:
            self.assertEqual(r["route_mode"], "judge")
            self.assertIn(r["chosen_tier"],
                          ("simple", "medium", "complex", "reasoning"))
            self.assertIsNotNone(r["reward"])
            self.assertTrue(r["memory_written"] is False)
        self.assertEqual(len(cpa.exec_calls), len(records))
        self.assertEqual(len(cpa.eval_calls), len(records))

    def test_mode_C_writes_memory_on_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary, _, records, mem, cpa, _ = run_mode("C", tmp, episodes=2)
        self.assertEqual(len(mem), 4)                # 2 tasks x 2 episodes
        self.assertEqual(summary["memory_written"], 4)
        for e in mem:
            self.assertIn(e["tier"], ("simple", "medium", "complex", "reasoning"))
            self.assertIsInstance(e["G"], float)

    def test_evaluator_failure_skips_memory(self):
        cpa = FakeCPAClient(eval_fail=True)
        with tempfile.TemporaryDirectory() as tmp:
            summary, _, records, mem, _, _ = run_mode("C", tmp, cpa=cpa)
        self.assertEqual(mem, [])                    # directive: failure -> no write
        for r in records:
            self.assertIsNone(r["evaluator"])
            self.assertIsNotNone(r["eval_error"])
            self.assertIsNone(r["reward"])
            self.assertFalse(r["memory_written"])
        self.assertEqual(summary["memory_size"], 0)

    def test_exec_failure_skips_eval_and_memory(self):
        cpa = FakeCPAClient()
        cpa.completion = lambda *a, **k: (_ for _ in ()).throw(
            CPATransportError("CPA request failed (HTTP 503)"))
        with tempfile.TemporaryDirectory() as tmp:
            summary, _, records, mem, _, _ = run_mode("C", tmp, cpa=cpa)
        self.assertEqual(mem, [])
        for r in records:
            self.assertIsNotNone(r["exec_error"])
            self.assertIsNone(r["evaluator"])
            self.assertIsNone(r["reward"])

    def test_record_schema_and_no_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, records, _, _, out = run_mode("C", tmp)
            text = out.read_text(encoding="utf-8")
        required = {"episode_id", "task_id", "family", "gt_tier", "chosen_tier",
                    "route_mode", "exec_model", "exec_usage", "exec_cost_usd",
                    "exec_finish_reason", "evaluator", "reward",
                    "reward_breakdown", "memory_written",
                    "judge_latency_ms", "jitrl_latency_ms"}
        for r in records:
            self.assertTrue(required.issubset(set(r.keys())))
            self.assertEqual(r["exec_finish_reason"], "stop")
            self.assertIn(r["exec_model"], ("CPA/glm-5.3-flash",
                                            "CPA/opencode-v4-flash",
                                            "CPA/glm-5.3", "CPA/OpenBMB-5.3"))
        for marker in ("apiKey", "api_key", "Bearer ", "Authorization"):
            self.assertNotIn(marker, text)

    def test_exec_default_budget_accounts_for_reasoning_preamble(self):
        # Calibration s2: reasoning models burn 400+ tokens on reasoning_content
        # before visible content; 512 truncated (finish=length, content="").
        self.assertGreaterEqual(run_real.DEFAULT_MAX_TOKENS_EXEC, 1024)

    def test_task_ids_filter_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks_path = write_tasks(tmp)
            out = Path(tmp) / "f.jsonl"
            summary = run_real.run(
                mode="A", tasks_path=tasks_path, task_ids=["R01"],
                judge_kind="mock", out_path=out,
                memory_out_path=Path(tmp) / "m.jsonl",
                cpa_client=FakeCPAClient())
            records = [json.loads(l) for l in
                       out.read_text(encoding="utf-8").splitlines()
                       if "_summary" not in l]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["task_id"], "R01")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                run_real.run(mode="A", tasks_path=write_tasks(tmp),
                             task_ids=["NOPE"], cpa_client=FakeCPAClient(),
                             out_path=Path(tmp) / "x.jsonl")


# --------------------------------------------- JitRL latency instrumentation
class FakeJudge:
    """Deterministic judge reporting a large latency_ms, to prove the
    4-request judge latency never leaks into jitrl_latency_ms."""

    def judge(self, user_message, previous_tier=None):
        return {"tier_logits": {"simple": -2.0, "medium": -1.0,
                                "complex": -3.0, "reasoning": -2.5},
                "chosen_tier": "medium", "raw_output": "<tier>medium</tier>",
                "latency_ms": 123.0}


class SlowDecideEngine:
    """Delegates to a real engine but makes decide() take ~sleep_s wall time,
    so the instrumentation can be checked against a known duration."""

    def __init__(self, engine, sleep_s):
        self._engine = engine
        self._sleep_s = sleep_s
        self.memory = engine.memory

    def decide(self, message, logits):
        time.sleep(self._sleep_s)
        return self._engine.decide(message, logits)

    def learn(self, **kwargs):
        return self._engine.learn(**kwargs)


class TestJitRLLatencyInstrumentation(unittest.TestCase):
    def test_mode_C_times_decide_only(self):
        real_engine_cls = run_real.JitRLEngine

        def slow_engine_factory(config=None):
            return SlowDecideEngine(real_engine_cls(config=config), 0.02)

        with mock.patch.object(run_real, "build_judge",
                               return_value=FakeJudge()), \
             mock.patch.object(run_real, "JitRLEngine",
                               side_effect=slow_engine_factory):
            with tempfile.TemporaryDirectory() as tmp:
                summary, summary_line, records, _, _, _ = run_mode("C", tmp)
        self.assertEqual(len(records), 2)
        vals = []
        for r in records:
            self.assertIsInstance(r["jitrl_latency_ms"], float)
            # decide includes ~20ms of deliberate sleep: strictly positive and
            # far below the 123ms judge latency -> the two clocks don't mix
            self.assertGreater(r["jitrl_latency_ms"], 0.0)
            self.assertLess(r["jitrl_latency_ms"], 100.0)
            # judge latency is reported separately and uncontaminated
            self.assertEqual(r["judge_latency_ms"], 123.0)
            vals.append(r["jitrl_latency_ms"])
        # summary average matches the per-record values (both in-memory
        # summary and the _summary line serialized to JSONL)
        expected_avg = sum(vals) / len(vals)
        self.assertAlmostEqual(summary["avg_jitrl_latency_ms"], expected_avg,
                               places=6)
        self.assertAlmostEqual(summary_line["avg_jitrl_latency_ms"],
                               expected_avg, places=6)

    def test_mode_A_and_B_jitrl_latency_is_documented_zero(self):
        for mode in ("A", "B"):
            with tempfile.TemporaryDirectory() as tmp:
                summary, summary_line, records, _, _, _ = run_mode(mode, tmp)
            for r in records:
                # documented convention: no JitRL post-processing in A/B
                self.assertEqual(r["jitrl_latency_ms"], 0.0)
            self.assertEqual(summary["avg_jitrl_latency_ms"], 0.0)
            self.assertEqual(summary_line["avg_jitrl_latency_ms"], 0.0)


# --------------------------------------------- C1 min-neighbor gate
# (docs/S2-C1-PREREGISTRATION.md v1.0: default min_neighbors=1 == C0;
#  the C1 run passes --min-neighbors 3 explicitly)
class TestC1MinNeighborGate(unittest.TestCase):
    def test_default_min_neighbors_reproduces_c0(self):
        # no flag: config default 1. Episode 2 retrieves exactly 1 neighbor
        # per task and MUST modulate (old C0 semantics).
        with tempfile.TemporaryDirectory() as tmp:
            summary, summary_line, records, _, _, _ = run_mode(
                "C", tmp, episodes=2)
        self.assertEqual(len(records), 4)
        for r in records:
            self.assertEqual(r["min_neighbors"], 1)
            self.assertIn(r["gate_reason"],
                          ("no_neighbors", "below_min_neighbors", None))
        for r in (x for x in records if x["episode_id"] == 1):
            self.assertEqual(r["n_retrieved"], 0)
            self.assertFalse(r["modulation_active"])
            self.assertEqual(r["gate_reason"], "no_neighbors")
        for r in (x for x in records if x["episode_id"] == 2):
            self.assertEqual(r["n_retrieved"], 1)
            self.assertTrue(r["modulation_active"])   # C0: n>=1 modulates
            self.assertIsNone(r["gate_reason"])
            # unchanged C0 formula at the harness level
            self.assertEqual(r["z_mod"], modulate_logits(
                r["z_base"], r["A_norm"], 5.0, -10.0, TIERS))
        self.assertEqual(summary["modulated_records"], 2)
        self.assertEqual(summary["gated_records"], 2)
        self.assertEqual(summary["config_snapshot"]["min_neighbors"], 1)
        # JSONL _summary line carries the same C1 fields
        self.assertEqual(summary_line["modulated_records"], 2)
        self.assertEqual(summary_line["gated_records"], 2)
        self.assertEqual(summary_line["config_snapshot"]["min_neighbors"], 1)

    def test_min_neighbors_3_gates_n1(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary, summary_line, records, _, _, _ = run_mode(
                "C", tmp, episodes=2, min_neighbors=3)
        for r in records:
            self.assertEqual(r["min_neighbors"], 3)
            self.assertFalse(r["modulation_active"])
            self.assertFalse(r["flipped"])
            self.assertEqual(r["chosen_tier"], r["base_choice"])
            self.assertEqual(r["z_mod"], r["z_base"])
            self.assertEqual(r["A_norm"],
                             {t: 0.0 for t in r["A_norm"]})
        for r in (x for x in records if x["episode_id"] == 1):
            self.assertEqual(r["gate_reason"], "no_neighbors")
        for r in (x for x in records if x["episode_id"] == 2):
            self.assertEqual(r["n_retrieved"], 1)
            self.assertEqual(r["gate_reason"], "below_min_neighbors")
        self.assertEqual(summary["modulated_records"], 0)
        self.assertEqual(summary["gated_records"], 4)
        self.assertEqual(summary["config_snapshot"]["min_neighbors"], 3)
        self.assertEqual(summary_line["modulated_records"], 0)
        self.assertEqual(summary_line["gated_records"], 4)

    def test_ab_modes_have_stable_gate_fields(self):
        for mode in ("A", "B"):
            with tempfile.TemporaryDirectory() as tmp:
                summary, summary_line, records, _, _, _ = run_mode(mode, tmp)
            for r in records:
                # schema-stable sentinels: no engine ran, gate never evaluated
                self.assertEqual(r["min_neighbors"], 1)
                self.assertFalse(r["modulation_active"])
                self.assertIsNone(r["gate_reason"])
            self.assertEqual(summary["modulated_records"], 0)
            self.assertEqual(summary["gated_records"], 0)
            self.assertEqual(summary_line["modulated_records"], 0)
            self.assertEqual(summary_line["gated_records"], 0)
            snap = summary["config_snapshot"]
            self.assertIsNotNone(snap)
            self.assertEqual(snap["min_neighbors"], 1)
            self.assertIn("seed", snap)
            self.assertIn("k", snap)

    def _cli_run(self, tmp, extra_args):
        out = Path(tmp) / "cli.jsonl"
        argv = ["--mode", "C", "--tasks", str(write_tasks(tmp)),
                "--judge", "mock", "--episodes", "2",
                "--out", str(out),
                "--memory-out", str(Path(tmp) / "cli_mem.jsonl")]
        argv += extra_args
        with mock.patch.object(run_real, "build_cpa_client",
                               return_value=FakeCPAClient()):
            rc = run_real.main(argv)
        self.assertEqual(rc, 0)
        records, summary_line = [], None
        for line in out.read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            if "_summary" in obj:
                summary_line = obj["_summary"]
            else:
                records.append(obj)
        return records, summary_line

    def test_cli_min_neighbors_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            records, summary = self._cli_run(tmp, ["--min-neighbors", "3"])
        self.assertEqual(len(records), 4)
        for r in records:
            self.assertEqual(r["min_neighbors"], 3)
            self.assertFalse(r["modulation_active"])
        self.assertEqual(summary["config_snapshot"]["min_neighbors"], 3)
        self.assertEqual(summary["modulated_records"], 0)
        self.assertEqual(summary["gated_records"], 4)

    def test_cli_default_is_min_neighbors_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            records, summary = self._cli_run(tmp, [])
        for r in records:
            self.assertEqual(r["min_neighbors"], 1)
        self.assertEqual(summary["config_snapshot"]["min_neighbors"], 1)
        self.assertEqual(summary["modulated_records"], 2)
        self.assertEqual(summary["gated_records"], 2)


if __name__ == "__main__":
    unittest.main()
