"""M2: multi-turn grouping, conversation building, trajectory-level
evaluation, per-step rewards, and run_real arms (mocked network)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harness.cpa_client import CPATransportError, CompletionResult
from harness.evaluator import (
    Evaluator,
    TrajectoryLevelEvaluator,
    validate_trajectory_evaluation,
)
from harness.multiturn import build_conversation, group_trajectories
from harness.real_rewards import RealQualityReward
from harness.trajectory import (
    Trajectory,
    TrajectoryEvaluator,
    TrajectoryStep,
)
from harness import run_real
from jitrl_core import (
    ENTRY_ACTIVE,
    ENTRY_PROVISIONAL,
    ENTRY_QUARANTINE,
    LifecycleMemory,
)

MT_TASKS = [
    {"id": "MT1", "traj_id": "MTR1", "turn_index": 0, "family": "code_gen",
     "gt_tier": "complex", "message": "写一个两数相加的函数",
     "quality_checklist": ["正确实现"]},
    {"id": "MT2", "traj_id": "MTR1", "turn_index": 1, "family": "code_gen",
     "gt_tier": "reasoning", "message": "审查它的边界场景并修复问题",
     "quality_checklist": ["识别边界"]},
    {"id": "MT3", "family": "chat_qa", "gt_tier": "simple",
     "message": "你好，简单打个招呼", "quality_checklist": ["友好问候"]},
]

TRAJ_JSON = json.dumps({
    "episode": {"success": True, "quality_score": 4, "summary": "任务完成"},
    "steps": [
        {"turn_index": 0, "routing_verdict": "appropriate", "local_quality": 5,
         "recommended_tier": "complex", "failure_tags": [],
         "feedback": "首路由档位合理", "certainty": 0.9},
        {"turn_index": 1, "routing_verdict": "under_routed", "local_quality": 3,
         "recommended_tier": "reasoning",
         "failure_tags": ["capability_mismatch"],
         "feedback": "边界审查需要更强模型", "certainty": 0.65},
    ],
}, ensure_ascii=False)

EVAL_JSON = json.dumps({
    "quality_score": 4, "verdict": "useful", "certainty": 0.8,
    "capability_insufficient": False,
    "checklist": {"hits": ["识别边界"], "misses": [], "notes": "ok"},
}, ensure_ascii=False)

#: 1-turn trajectory review (for singleton rows under per-step arms)
TRAJ_JSON_1 = json.dumps({
    "episode": {"success": True, "quality_score": 4, "summary": "打招呼完成"},
    "steps": [
        {"turn_index": 0, "routing_verdict": "appropriate", "local_quality": 4,
         "recommended_tier": "simple", "failure_tags": [],
         "feedback": "闲聊用 simple 恰当", "certainty": 0.85},
    ],
}, ensure_ascii=False)


def usage(n_in=1000, n_out=200):
    return {"input_tokens": n_in, "cache_read_tokens": 0,
            "cache_write_tokens": 0, "output_tokens": n_out,
            "total_tokens": n_in + n_out, "native_cost": None}


class FakeCPA:
    """Returns scripted content for evaluator calls (gpt-5.6-sol); exec
    calls succeed unless fail_on_exec_call is set (1-based counter)."""

    def __init__(self, eval_content=TRAJ_JSON, fail_on_exec_call=None,
                 exec_fail=False, eval_fail=False):
        self.eval_content = eval_content
        self.fail_on_exec_call = fail_on_exec_call
        self.exec_fail = exec_fail
        self.eval_fail = eval_fail
        self.exec_calls: list[tuple] = []
        self.eval_calls: list[tuple] = []
        self._exec_count = 0

    def completion(self, model, messages, *, max_tokens=512,
                   temperature=0.7, extra_body=None):
        api = model.split("/")[-1]
        if api == "gpt-5.6-sol":
            self.eval_calls.append((model, messages, max_tokens))
            if self.eval_fail:
                raise CPATransportError("CPA request failed (HTTP 503)")
            return CompletionResult(content=self.eval_content,
                                    usage=usage(1500, 300), model=api,
                                    finish_reason="stop", latency_ms=5.0)
        self._exec_count += 1
        self.exec_calls.append((model, messages, max_tokens))
        if self.exec_fail or self.fail_on_exec_call == self._exec_count:
            raise CPATransportError("CPA request failed (HTTP 503)")
        return CompletionResult(content=f"turn-{self._exec_count} 的回复",
                                usage=usage(), model=api,
                                finish_reason="stop", latency_ms=5.0)


def write_tasks(tmp, tasks=MT_TASKS) -> Path:
    p = Path(tmp) / "mt_tasks.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    return p


def run_arm(mode, tmp, arm="T1", cpa=None, episodes=1, tasks=MT_TASKS,
            judge="mock"):
    out = Path(tmp) / f"mt_out_{mode}_{arm}.jsonl"
    mem = Path(tmp) / f"mt_mem_{mode}_{arm}.jsonl"
    cpa = cpa if cpa is not None else FakeCPA()
    summary = run_real.run(
        mode=mode, tasks_path=write_tasks(tmp, tasks), judge_kind=judge,
        out_path=out, memory_out_path=mem, episodes=episodes,
        cpa_client=cpa, arm=arm,
    )
    records = []
    with open(out, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "_summary" not in obj:
                records.append(obj)
    with open(mem, encoding="utf-8") as f:
        mem_entries = [json.loads(l) for l in f if l.strip()]
    return summary, records, mem_entries, cpa


# ------------------------------------------------------------ grouping
class TestGroupTrajectories(unittest.TestCase):
    def test_groups_by_traj_id_with_singletons(self):
        groups = group_trajectories(MT_TASKS)
        self.assertEqual(len(groups), 2)
        self.assertEqual([t["id"] for t in groups[0]], ["MT1", "MT2"])
        self.assertEqual([t["id"] for t in groups[1]], ["MT3"])

    def test_turn_index_sorts_within_group(self):
        shuffled = [MT_TASKS[1], MT_TASKS[2], MT_TASKS[0]]
        groups = group_trajectories(shuffled)
        self.assertEqual([t["id"] for t in groups[0]], ["MT1", "MT2"])

    def test_all_singletons_without_traj_id(self):
        rows = [{"task_id": f"S{i}", "message": "m"} for i in range(3)]
        groups = group_trajectories(rows)
        self.assertEqual([len(g) for g in groups], [1, 1, 1])


class TestBuildConversation(unittest.TestCase):
    def _step(self, message, content=None):
        exec_result = None
        if content is not None:
            exec_result = CompletionResult(content=content, usage=usage(),
                                           model="glm-5.3-flash",
                                           finish_reason="stop", latency_ms=1.0)
        return TrajectoryStep(
            task=MT_TASKS[0], message=message, intent="code_gen",
            signature_tokens=["a"], chosen_tier="complex", route_mode="jitrl",
            exec_model="CPA/glm-5.3-flash", previous_tier=None,
            exec_result=exec_result)

    def test_first_turn_is_legacy_single_message(self):
        msgs = build_conversation([], "你好")
        self.assertEqual(msgs, [{"role": "user", "content": "你好"}])

    def test_multi_turn_interleaves_history(self):
        steps = [self._step("第一问", "第一答"), self._step("第二问", "第二答")]
        msgs = build_conversation(steps, "第三问")
        self.assertEqual([m["role"] for m in msgs],
                         ["user", "assistant", "user", "assistant", "user"])
        self.assertEqual(msgs[1]["content"], "第一答")
        self.assertEqual(msgs[-1]["content"], "第三问")

    def test_failed_turn_keeps_user_message_only(self):
        steps = [self._step("第一问", None)]
        msgs = build_conversation(steps, "第二问")
        self.assertEqual([m["role"] for m in msgs], ["user", "user"])

    def test_history_truncated_to_cap(self):
        from harness.multiturn import HISTORY_PREVIEW_CHARS
        long = "x" * (HISTORY_PREVIEW_CHARS + 100)
        msgs = build_conversation([self._step("q", long)], "next")
        self.assertEqual(len(msgs[1]["content"]), HISTORY_PREVIEW_CHARS)
        self.assertEqual(len(msgs[2]["content"]), 4)  # current never truncated


# ------------------------------------------------- trajectory-level evaluator
class TestTrajectoryLevelEvaluator(unittest.TestCase):
    def _turns(self):
        return [{"message": "写函数", "tier": "complex", "response": "def add..."},
                {"message": "审查边界", "tier": "medium", "response": "存在边界问题"}]

    def test_prompt_shows_tiers_but_not_models(self):
        ev = TrajectoryLevelEvaluator(client=FakeCPA())
        prompt = ev.build_prompt(MT_TASKS[0], self._turns())
        self.assertIn("路由档位：complex", prompt)
        self.assertIn("路由档位：medium", prompt)
        self.assertIn("审查边界", prompt)
        self.assertNotIn("glm-5.3", prompt)
        self.assertNotIn("opencode", prompt)

    def test_success_path_one_call(self):
        cpa = FakeCPA()
        ev = TrajectoryLevelEvaluator(client=cpa)
        res = ev.evaluate_trajectory(MT_TASKS[0], self._turns())
        self.assertIsNotNone(res)
        self.assertTrue(res.episode_success)
        self.assertEqual(res.episode_quality, 4)
        self.assertEqual(len(res.steps), 2)
        self.assertEqual(res.steps[1].routing_verdict, "under_routed")
        self.assertEqual(res.steps[0].certainty, 0.9)
        self.assertEqual(len(cpa.eval_calls), 1)

    def test_failure_returns_none(self):
        self.assertIsNone(TrajectoryLevelEvaluator(
            client=FakeCPA(eval_fail=True)).evaluate_trajectory(
                MT_TASKS[0], self._turns()))
        self.assertIsNone(TrajectoryLevelEvaluator(
            client=FakeCPA(eval_content="无法输出 JSON")).evaluate_trajectory(
                MT_TASKS[0], self._turns()))

    def test_validate_rejects_bad_payloads(self):
        good = json.loads(TRAJ_JSON)
        self.assertIsNotNone(validate_trajectory_evaluation(good, 2))
        # wrong step count
        self.assertIsNone(validate_trajectory_evaluation(good, 3))
        # turn_index mismatch
        bad = json.loads(TRAJ_JSON)
        bad["steps"][0]["turn_index"] = 1
        self.assertIsNone(validate_trajectory_evaluation(bad, 2))
        # bad verdict
        bad2 = json.loads(TRAJ_JSON)
        bad2["steps"][0]["routing_verdict"] = "great"
        self.assertIsNone(validate_trajectory_evaluation(bad2, 2))
        # local_quality out of range
        bad3 = json.loads(TRAJ_JSON)
        bad3["steps"][0]["local_quality"] = 9
        self.assertIsNone(validate_trajectory_evaluation(bad3, 2))
        # certainty clamped, not fatal
        bad4 = json.loads(TRAJ_JSON)
        bad4["steps"][0]["certainty"] = 2.0
        res = validate_trajectory_evaluation(bad4, 2)
        self.assertEqual(res.steps[0].certainty, 1.0)


# ------------------------------------------------ per-step trajectory evaluation
class TestPerStepEvaluation(unittest.TestCase):
    def _traj(self):
        cpa = FakeCPA()
        steps = []
        prev = None
        for i, t in enumerate(MT_TASKS[:2]):
            steps.append(TrajectoryStep(
                task=t, message=t["message"], intent="code_gen",
                signature_tokens=[f"tok{i}"], chosen_tier="complex",
                route_mode="jitrl", exec_model="CPA/glm-5.3-flash",
                previous_tier=prev,
                exec_result=CompletionResult(
                    content=f"回复{i}", usage=usage(), model="glm-5.3-flash",
                    finish_reason="stop", latency_ms=1.0)))
            prev = "complex"
        return Trajectory(traj_id="1:MTR1", episode_id=1, steps=steps), cpa

    def test_per_step_rewards_use_d3_formula_on_local_quality(self):
        traj, cpa = self._traj()
        te = TrajectoryEvaluator(
            evaluator=Evaluator(client=cpa), reward_fn=RealQualityReward(),
            mode="per_step",
            traj_evaluator=TrajectoryLevelEvaluator(client=cpa))
        report = te.evaluate(traj)
        self.assertTrue(report.ok)
        rw = RealQualityReward()
        for i, local_q in enumerate((5, 3)):     # from TRAJ_JSON
            exp_r, exp_b = rw.reward(local_q, "CPA/glm-5.3-flash", usage())
            self.assertAlmostEqual(report.steps[i].reward, exp_r, places=12)
            self.assertEqual(report.steps[i].breakdown["quality_term"],
                             exp_b["quality_term"])
            self.assertEqual(report.steps[i].step_verdict.local_quality, local_q)
        # semantic fields flow for memory sidecars
        self.assertEqual(report.steps[1].step_verdict.routing_verdict,
                         "under_routed")
        self.assertEqual(report.steps[1].step_verdict.recommended_tier,
                         "reasoning")
        # exactly ONE trajectory-level evaluator call
        self.assertEqual(len(cpa.eval_calls), 1)

    def test_traj_eval_failure_marks_all_turns(self):
        traj, cpa = self._traj()
        cpa.eval_fail = True
        te = TrajectoryEvaluator(
            evaluator=Evaluator(client=cpa), reward_fn=RealQualityReward(),
            mode="per_step",
            traj_evaluator=TrajectoryLevelEvaluator(client=cpa))
        report = te.evaluate(traj)
        self.assertFalse(report.ok)
        self.assertIsNotNone(report.eval_error)
        for sr in report.steps:
            self.assertIsNotNone(sr.eval_error)
            self.assertIsNone(sr.reward)

    def test_mid_traj_exec_failure_not_ok(self):
        traj, _ = self._traj()
        traj.steps[1].exec_result = None
        traj.steps[1].exec_error = "CPATransportError: boom"
        cpa = FakeCPA()
        te = TrajectoryEvaluator(
            evaluator=Evaluator(client=cpa), reward_fn=RealQualityReward(),
            mode="per_step",
            traj_evaluator=TrajectoryLevelEvaluator(client=cpa))
        report = te.evaluate(traj)
        self.assertFalse(report.ok)
        self.assertEqual(cpa.eval_calls, [])   # no evaluator call at all

    def test_per_step_requires_traj_evaluator(self):
        with self.assertRaises(ValueError):
            TrajectoryEvaluator(evaluator=None, reward_fn=None,
                                mode="per_step")


# ------------------------------------------------------------- run_real arms
class TestRunRealArms(unittest.TestCase):
    def test_T1_multiturn_writes_only_terminal_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary, records, mem, cpa = run_arm("C", tmp, arm="T1",
                                                 cpa=FakeCPA(eval_content=EVAL_JSON))
        self.assertEqual(len(records), 3)          # 2 turns + 1 singleton
        self.assertEqual(len(mem), 2)              # terminal of MTR1 + singleton
        mt = [r for r in records if r["traj_id"] == "1:MTR1"]
        self.assertEqual([r["memory_written"] for r in mt], [False, True])
        self.assertEqual([r["turn_index"] for r in mt], [0, 1])
        self.assertEqual(mt[0]["n_turns"], 2)
        self.assertEqual(mt[1]["evaluator"]["quality_score"], 4)
        self.assertIsNone(mt[0]["evaluator"])
        # legacy per-response evaluator used (single JSON verdict schema)
        self.assertIn("verdict", mt[1]["evaluator"])

    def test_T2_discounted_entries_per_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary, records, mem, cpa = run_arm("C", tmp, arm="T2",
                                                 tasks=MT_TASKS[:2])
        mt_mem = mem
        self.assertEqual(len(mt_mem), 2)           # one entry PER TURN
        mt = [r for r in records if r["traj_id"] == "1:MTR1"]
        self.assertEqual(len(mt), 2)
        rw = RealQualityReward()
        # per-turn rewards use each turn's ACTUAL exec model + usage
        r0, _ = rw.reward(5, mt[0]["exec_model"], usage())
        r1, _ = rw.reward(3, mt[1]["exec_model"], usage())
        # discounted credit: G0 = r0 + eta*gamma*r1 ; G1 = r1
        self.assertAlmostEqual(mt_mem[0]["G"], r0 + 0.5 * 0.8 * r1, places=12)
        self.assertAlmostEqual(mt_mem[1]["G"], r1, places=12)
        self.assertEqual(mt_mem[0]["tier"], mt[0]["chosen_tier"])
        # per-turn step verdicts recorded (per-step schema, no verdict key)
        self.assertEqual(mt[0]["evaluator"]["routing_verdict"], "appropriate")
        self.assertEqual(mt[1]["evaluator"]["routing_verdict"], "under_routed")
        self.assertEqual(mt[1]["evaluator"]["recommended_tier"], "reasoning")
        self.assertNotIn("verdict", mt[0]["evaluator"])
        # eval cost booked once (terminal turn only)
        self.assertIsNone(mt[0]["eval_cost_usd"])
        self.assertIsNotNone(mt[1]["eval_cost_usd"])

    def test_T3_lifecycle_gates_by_confidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary, records, mem, cpa = run_arm("C", tmp, arm="T3",
                                                 tasks=MT_TASKS[:2])
        # certainties 0.9 / 0.65 -> active / provisional
        self.assertEqual(summary["memory_status_counts"],
                         {ENTRY_ACTIVE: 1, ENTRY_PROVISIONAL: 1,
                          ENTRY_QUARANTINE: 0})
        self.assertEqual(len(mem), 2)   # provisional kept for audit
        self.assertEqual(summary["n_trajectories"], 1)

    def test_mode_B_multiturn_no_memory_and_previous_tier_chains(self):
        calls = []

        class ChainJudge:
            def judge(self, user_message, previous_tier=None):
                calls.append((user_message, previous_tier))
                return {"tier_logits": {"simple": -1.0, "medium": 2.0,
                                        "complex": -1.0, "reasoning": -2.0},
                        "chosen_tier": "medium", "raw_output": "<tier>medium</tier>",
                        "latency_ms": 5.0}

        import harness.run as run_mod
        orig = run_real.build_judge
        run_real.build_judge = lambda kind, seed: ChainJudge()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                summary, records, mem, cpa = run_arm("B", tmp, arm="T2")
        finally:
            run_real.build_judge = orig
        self.assertEqual(mem, [])
        # previous_tier chains within the trajectory: 2nd call saw 1st choice
        self.assertEqual(calls[1][1], "medium")
        self.assertIsNone(calls[0][1])
        # multi-turn exec saw the conversation history
        mt_exec = [c for c in cpa.exec_calls if len(c[1]) == 3]
        self.assertEqual(len(mt_exec), 1)

    def test_mid_traj_exec_failure_blocks_memory(self):
        cpa = FakeCPA(fail_on_exec_call=2,   # 2nd turn of MTR1 fails
                      eval_content=TRAJ_JSON_1)  # singleton MT3: 1-step review
        with tempfile.TemporaryDirectory() as tmp:
            summary, records, mem, _ = run_arm("C", tmp, arm="T2", cpa=cpa)
        mt = [r for r in records if r["traj_id"] == "1:MTR1"]
        self.assertIsNotNone(mt[1]["exec_error"])
        self.assertFalse(all(r["memory_written"] for r in mt))
        # MTR1 wrote nothing; only the singleton MT3 wrote (1 entry)
        self.assertEqual(len(mem), 1)
        self.assertEqual(summary["memory_written"], 1)
        st = [r for r in records if r["task_id"] == "MT3"][0]
        self.assertEqual(mem[0]["tier"], st["chosen_tier"])

    def test_single_turn_regression_arm_field(self):
        single = [MT_TASKS[2]]
        with tempfile.TemporaryDirectory() as tmp:
            summary, records, mem, _ = run_arm(
                "C", tmp, arm="T1", cpa=FakeCPA(eval_content=EVAL_JSON),
                tasks=single)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["arm"], "T1")
        self.assertEqual(records[0]["n_turns"], 1)
        self.assertEqual(records[0]["traj_id"], "1:MT3")
        self.assertEqual(len(mem), 1)

    def test_unknown_arm_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_arm("C", tmp, arm="T9")


if __name__ == "__main__":
    unittest.main()
