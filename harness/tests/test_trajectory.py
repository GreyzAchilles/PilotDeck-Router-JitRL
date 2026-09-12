"""M1 unified trajectory interface in the real harness (mocked network)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harness.cpa_client import CPATransportError, CompletionResult
from harness.evaluator import Evaluator
from harness.real_rewards import RealQualityReward
from harness.trajectory import (
    Trajectory,
    TrajectoryEvaluator,
    TrajectoryStep,
    update_memory_from_trajectory,
)
from harness import run_real
from jitrl_core import JitRLEngine

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
    "checklist": {"hits": ["指出第 2 月销量最高"],
                  "misses": ["提醒异常需结合业务解释"], "notes": "基本完成"},
}, ensure_ascii=False)


def usage(n_in=100, n_out=50):
    return {"input_tokens": n_in, "cache_read_tokens": 0,
            "cache_write_tokens": 0, "output_tokens": n_out,
            "total_tokens": n_in + n_out, "native_cost": None}


class FakeCPAClient:
    def __init__(self, exec_content="这是执行模型的回复。",
                 eval_content=EVAL_JSON, eval_fail=False):
        self.exec_content = exec_content
        self.eval_content = eval_content
        self.eval_fail = eval_fail
        self.exec_calls: list[tuple] = []
        self.eval_calls: list[tuple] = []

    def completion(self, model, messages, *, max_tokens=512,
                   temperature=0.7, extra_body=None):
        api = model.split("/")[-1]
        if api == "gpt-5.6-sol":
            self.eval_calls.append((model, messages, max_tokens))
            if self.eval_fail:
                raise CPATransportError("CPA request failed (HTTP 503)")
            return CompletionResult(content=self.eval_content,
                                    usage=usage(1500, 200), model=api,
                                    finish_reason="stop", latency_ms=5.0)
        self.exec_calls.append((model, messages, max_tokens))
        return CompletionResult(content=self.exec_content,
                                usage=usage(1000, 200), model=api,
                                finish_reason="stop", latency_ms=5.0)


def make_step(task=TASKS[1], exec_result=None, exec_error=None) -> TrajectoryStep:
    from jitrl_core.state import intent_class, task_signature
    msg = task["message"]
    if exec_result is None and exec_error is None:
        exec_result = CompletionResult(
            content="分析结果：第2月峰值50", usage=usage(1000, 200),
            model="glm-5.3-flash", finish_reason="stop", latency_ms=5.0)
    return TrajectoryStep(
        task=task, message=msg, intent=intent_class(msg),
        signature_tokens=sorted(task_signature(msg)), chosen_tier="complex",
        route_mode="jitrl", exec_model="CPA/glm-5.3-flash",
        previous_tier=None, exec_result=exec_result, exec_error=exec_error,
        exec_cost_usd=0.001, judge_latency_ms=10.0, jitrl_latency_ms=0.1,
        ts=2000.0)


def make_traj(step) -> Trajectory:
    return Trajectory(traj_id="1:R02", episode_id=1, steps=[step])


# ------------------------------------------------------------- evaluator
class TestTrajectoryEvaluator(unittest.TestCase):
    def setUp(self):
        self.cpa = FakeCPAClient()
        self.evaluator = Evaluator(client=self.cpa)
        self.traj_eval = TrajectoryEvaluator(
            evaluator=self.evaluator, reward_fn=RealQualityReward())

    def test_single_step_reward_matches_legacy_reward_math(self):
        step = make_step()
        report = self.traj_eval.evaluate(make_traj(step))
        self.assertTrue(report.ok)
        self.assertIsNone(report.eval_error)
        # identical math to the legacy inline call
        legacy_r, legacy_b = RealQualityReward().reward(
            4, "CPA/glm-5.3-flash", usage(1000, 200))
        self.assertAlmostEqual(report.reward, legacy_r, places=12)
        self.assertEqual(report.breakdown["quality_term"],
                         legacy_b["quality_term"])
        self.assertEqual(report.reward, report.steps[0].reward)
        # exactly ONE blind evaluator call per episode
        self.assertEqual(len(self.cpa.eval_calls), 1)

    def test_blindness_preserved(self):
        self.traj_eval.evaluate(make_traj(make_step()))
        model, messages, _ = self.cpa.eval_calls[0]
        self.assertEqual(model, "CPA/gpt-5.6-sol")
        self.assertNotIn("glm-5.3",
                         messages[0]["content"].split("## 待评审回复")[0])

    def test_exec_failure_is_not_ok_no_eval_call(self):
        step = make_step(exec_result=None, exec_error="CPATransportError: x")
        report = self.traj_eval.evaluate(make_traj(step))
        self.assertFalse(report.ok)
        self.assertIsNone(report.reward)
        self.assertEqual(self.cpa.eval_calls, [])

    def test_eval_failure_is_not_ok(self):
        cpa = FakeCPAClient(eval_fail=True)
        traj_eval = TrajectoryEvaluator(
            evaluator=Evaluator(client=cpa), reward_fn=RealQualityReward())
        report = traj_eval.evaluate(make_traj(make_step()))
        self.assertFalse(report.ok)
        self.assertIsNotNone(report.eval_error)
        self.assertIsNone(report.reward)


# ----------------------------------------------------------------- update
class TestUpdateMemory(unittest.TestCase):
    def test_single_step_writes_one_frozen_entry_with_g_equal_reward(self):
        cpa = FakeCPAClient()
        traj_eval = TrajectoryEvaluator(
            evaluator=Evaluator(client=cpa), reward_fn=RealQualityReward())
        traj = make_traj(make_step())
        report = traj_eval.evaluate(traj)
        engine = JitRLEngine()
        entries = update_memory_from_trajectory(engine, traj, report)
        self.assertEqual(len(entries), 1)
        self.assertEqual(len(engine.memory), 1)
        entry = entries[0]
        self.assertEqual(
            set(entry.keys()),
            {"intent_class", "signature_tokens", "tier", "G", "ts",
             "episode_id"})
        self.assertEqual(entry["episode_id"], 1)
        self.assertAlmostEqual(entry["G"], report.reward, places=12)
        self.assertEqual(entry["tier"], "complex")

    def test_failed_report_writes_nothing(self):
        engine = JitRLEngine()
        traj = make_traj(make_step(exec_result=None, exec_error="boom"))
        report = TrajectoryEvaluator(
            evaluator=Evaluator(client=FakeCPAClient()),
            reward_fn=RealQualityReward()).evaluate(traj)
        self.assertEqual(
            update_memory_from_trajectory(engine, traj, report), [])
        self.assertEqual(len(engine.memory), 0)


# ------------------------------------------------------- run_real integration
def run_mode(mode, tmp, cpa=None, episodes=1, **run_kwargs):
    out = Path(tmp) / f"traj_out_{mode}.jsonl"
    mem = Path(tmp) / f"traj_mem_{mode}.jsonl"
    cpa = cpa or FakeCPAClient()
    summary = run_real.run(
        mode=mode, tasks_path=write_tasks(tmp), judge_kind="mock",
        out_path=out, memory_out_path=mem, episodes=episodes,
        cpa_client=cpa, **run_kwargs,
    )
    records = []
    with open(out, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "_summary" not in obj:
                records.append(obj)
    with open(mem, encoding="utf-8") as f:
        mem_entries = [json.loads(l) for l in f if l.strip()]
    return summary, records, mem_entries


def write_tasks(tmp: str) -> Path:
    p = Path(tmp) / "traj_tasks.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for t in TASKS:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    return p


class TestRunRealTrajectoryPath(unittest.TestCase):
    def test_mode_c_memory_entries_frozen_schema_and_g_equals_reward(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary, records, mem = run_mode("C", tmp, episodes=2)
        self.assertEqual(len(mem), 4)                      # 2 tasks x 2 eps
        self.assertEqual(summary["memory_written"], 4)
        for e in mem:
            self.assertEqual(
                set(e.keys()),
                {"intent_class", "signature_tokens", "tier", "G", "ts",
                 "episode_id"})
        # trajectory path writes entries in record order: G == record reward
        for e, r in zip(mem, records):
            self.assertAlmostEqual(e["G"], r["reward"], places=12)
            self.assertEqual(e["tier"], r["chosen_tier"])

    def test_records_carry_additive_trajectory_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, records, _ = run_mode("C", tmp)
        for r in records:
            self.assertEqual(r["n_steps"], 1)
            self.assertTrue(isinstance(r["traj_id"], str))
            self.assertEqual(r["traj_id"], f"{r['episode_id']}:{r['task_id']}")

    def test_mode_b_writes_no_memory_but_has_traj_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, records, mem = run_mode("B", tmp)
        self.assertEqual(mem, [])
        for r in records:
            self.assertEqual(r["n_steps"], 1)
            self.assertFalse(r["memory_written"])

    def test_eval_failure_still_skips_memory_on_trajectory_path(self):
        cpa = FakeCPAClient(eval_fail=True)
        with tempfile.TemporaryDirectory() as tmp:
            summary, records, mem = run_mode("C", tmp, cpa=cpa)
        self.assertEqual(mem, [])
        self.assertEqual(summary["memory_size"], 0)
        for r in records:
            self.assertIsNone(r["reward"])
            self.assertFalse(r["memory_written"])


if __name__ == "__main__":
    unittest.main()
