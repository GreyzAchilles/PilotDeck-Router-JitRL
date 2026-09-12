"""M1 trajectory loop: types, credit assignment, updaters, lifecycle memory,
and engine.learn_trajectory single-step equivalence."""
from __future__ import annotations

import unittest

from jitrl_core.config import JitRLConfig
from jitrl_core.credit import (
    DiscountedCreditAssigner,
    IdentityCreditAssigner,
    StepCredit,
)
from jitrl_core.engine import JitRLEngine
from jitrl_core.memory import (
    ENTRY_ACTIVE,
    ENTRY_PROVISIONAL,
    ENTRY_QUARANTINE,
    ExperienceMemory,
    LifecycleMemory,
)
from jitrl_core.state import task_signature
from jitrl_core.types import (
    DecisionStep,
    RouteTrajectory,
    StepEvaluation,
    TrajectoryEvaluation,
)
from jitrl_core.updater import (
    DirectMemoryUpdater,
    LifecycleMemoryUpdater,
    build_entry,
)

MSG = "分析这批销售数据，统计环比，对比渠道表现"
LOGITS = {"simple": 0.0, "medium": 1.0, "complex": 0.0, "reasoning": 0.0}
ENTRY_KEYS = {"intent_class", "signature_tokens", "tier", "G", "ts", "episode_id"}


def single_step_trajectory(reward: float, ts: float = 1000.0,
                           episode_id: int = 7) -> RouteTrajectory:
    traj = RouteTrajectory(trajectory_id="t1", episode_id=episode_id)
    traj.add_step(DecisionStep(
        step_id=0, user_message=MSG, intent_class="data_analysis",
        signature_tokens=task_signature(MSG), chosen_tier="medium",
        previous_tier=None, reward=reward, ts=ts,
    ))
    traj.finalize()
    return traj


def evaluation_for(traj: RouteTrajectory, rewards: list, *,
                   ok: bool = True, confidences: list | None = None) -> TrajectoryEvaluation:
    confidences = confidences or [None] * len(rewards)
    return TrajectoryEvaluation(
        ok=ok,
        steps=[StepEvaluation(step_id=s.step_id, ok=ok, reward=r,
                              confidence=c)
               for s, r, c in zip(traj.steps, rewards, confidences)],
    )


# ------------------------------------------------------------------- types
class TestTrajectoryTypes(unittest.TestCase):
    def test_add_step_only_when_open(self):
        traj = RouteTrajectory(trajectory_id="t")
        step = traj.add_step(DecisionStep(
            step_id=0, user_message="m", intent_class="chat_qa",
            signature_tokens=["a"], chosen_tier="simple"))
        self.assertEqual(traj.step_by_id(0), step)
        traj.finalize()
        with self.assertRaises(ValueError):
            traj.add_step(DecisionStep(
                step_id=1, user_message="m2", intent_class="chat_qa",
                signature_tokens=["b"], chosen_tier="simple"))

    def test_step_by_id_missing_returns_none(self):
        traj = RouteTrajectory(trajectory_id="t")
        self.assertIsNone(traj.step_by_id(99))
        ev = TrajectoryEvaluation(steps=[StepEvaluation(step_id=3, ok=True)])
        self.assertIsNone(ev.step_by_id(1))
        self.assertIsNotNone(ev.step_by_id(3))


# ------------------------------------------------------------------ credit
class TestIdentityCredit(unittest.TestCase):
    def test_single_step_g_equals_r(self):
        traj = single_step_trajectory(reward=-0.38)
        credits = IdentityCreditAssigner().assign(
            traj, evaluation_for(traj, [-0.38]))
        self.assertEqual(credits, [StepCredit(step_id=0, G=-0.38)])

    def test_failed_evaluation_yields_no_credits(self):
        traj = single_step_trajectory(reward=0.5)
        ev = evaluation_for(traj, [0.5], ok=False)
        self.assertEqual(IdentityCreditAssigner().assign(traj, ev), [])
        self.assertEqual(DiscountedCreditAssigner().assign(traj, ev), [])

    def test_unscored_step_gets_no_credit(self):
        traj = single_step_trajectory(reward=None)
        credits = IdentityCreditAssigner().assign(
            traj, evaluation_for(traj, [None]))
        self.assertEqual(credits, [])


class TestDiscountedCredit(unittest.TestCase):
    def _three_step(self):
        traj = RouteTrajectory(trajectory_id="t3", episode_id=1)
        for i, r in enumerate((-0.7, 0.2, 0.8)):
            traj.add_step(DecisionStep(
                step_id=i, user_message=f"m{i}", intent_class="data_analysis",
                signature_tokens=[f"tok{i}"], chosen_tier="medium",
                reward=r, ts=1000.0 + i))
        traj.finalize()
        return traj

    def test_hand_computed_gamma_eta(self):
        # gamma=0.8, eta=0.5, rewards (-0.7, 0.2, 0.8):
        # G0 = -0.7 + 0.5*(0.8*0.2 + 0.64*0.8) = -0.7 + 0.5*0.672 = -0.364
        # G1 =  0.2 + 0.5*(0.8*0.8)            =  0.2 + 0.32     =  0.52
        # G2 =  0.8
        traj = self._three_step()
        credits = DiscountedCreditAssigner(gamma=0.8, delayed_weight=0.5).assign(
            traj, evaluation_for(traj, [-0.7, 0.2, 0.8]))
        self.assertAlmostEqual(credits[0].G, -0.364, places=12)
        self.assertAlmostEqual(credits[1].G, 0.52, places=12)
        self.assertAlmostEqual(credits[2].G, 0.8, places=12)

    def test_rescue_does_not_wash_out_misroute(self):
        # The design example: r=(-0.7, +0.8) must leave G0 < 0
        traj = RouteTrajectory(trajectory_id="t2", episode_id=1)
        for i, r in enumerate((-0.7, 0.8)):
            traj.add_step(DecisionStep(
                step_id=i, user_message=f"m{i}", intent_class="data_analysis",
                signature_tokens=[f"tok{i}"], chosen_tier="medium",
                reward=r, ts=1000.0 + i))
        credits = DiscountedCreditAssigner().assign(
            traj, evaluation_for(traj, [-0.7, 0.8]))
        self.assertLess(credits[0].G, 0.0)   # -0.38
        self.assertAlmostEqual(credits[0].G, -0.38, places=12)
        self.assertAlmostEqual(credits[1].G, 0.8, places=12)

    def test_unscored_middle_step_contributes_zero_but_advances_time(self):
        # rewards (0.2, None, 0.8): G0 = 0.2 + 0.5*(0.8^2 * 0.8) = 0.2 + 0.256
        traj = RouteTrajectory(trajectory_id="t3b", episode_id=1)
        for i, r in enumerate((0.2, None, 0.8)):
            traj.add_step(DecisionStep(
                step_id=i, user_message=f"m{i}", intent_class="data_analysis",
                signature_tokens=[f"tok{i}"], chosen_tier="medium",
                reward=r, ts=1000.0 + i))
        credits = DiscountedCreditAssigner().assign(
            traj, evaluation_for(traj, [0.2, None, 0.8]))
        self.assertEqual([c.step_id for c in credits], [0, 2])
        self.assertAlmostEqual(credits[0].G, 0.456, places=12)

    def test_invalid_params_rejected(self):
        with self.assertRaises(ValueError):
            DiscountedCreditAssigner(gamma=1.5)
        with self.assertRaises(ValueError):
            DiscountedCreditAssigner(delayed_weight=-0.1)


# ----------------------------------------------------------------- updater
class TestBuildEntry(unittest.TestCase):
    def test_frozen_schema(self):
        e = build_entry(intent="data_analysis",
                        signature_tokens=["b", "a"], tier="medium",
                        reward=-0.5, episode_id=3, ts=42.0)
        self.assertEqual(set(e.keys()), ENTRY_KEYS)
        self.assertEqual(e["signature_tokens"], ["a", "b"])
        self.assertEqual(e["G"], -0.5)
        self.assertIsInstance(e["G"], float)
        self.assertEqual(e["ts"], 42.0)
        self.assertEqual(e["episode_id"], 3)


class TestDirectMemoryUpdater(unittest.TestCase):
    def test_writes_frozen_entries_in_step_order(self):
        traj = single_step_trajectory(reward=0.25)
        mem = ExperienceMemory()
        entries = DirectMemoryUpdater().update(
            mem, [StepCredit(step_id=0, G=0.25)], traj,
            evaluation_for(traj, [0.25]))
        self.assertEqual(len(mem), 1)
        self.assertEqual(entries[0], mem.entries()[0])
        self.assertEqual(set(entries[0].keys()), ENTRY_KEYS)
        self.assertEqual(entries[0]["episode_id"], 7)   # from trajectory


# --------------------------------------------------------- lifecycle memory
class TestLifecycleMemory(unittest.TestCase):
    def setUp(self):
        self.mem = LifecycleMemory(cap=10)
        self.e_active = {"intent_class": "i", "signature_tokens": ["a", "b"],
                         "tier": "simple", "G": 0.1, "ts": 1.0, "episode_id": 1}
        self.e_prov = {"intent_class": "i", "signature_tokens": ["a", "b"],
                       "tier": "medium", "G": 0.2, "ts": 2.0, "episode_id": 2}
        self.e_quar = {"intent_class": "i", "signature_tokens": ["a", "b"],
                       "tier": "complex", "G": 0.3, "ts": 3.0, "episode_id": 3}

    def test_plain_add_defaults_active_and_retrievable(self):
        self.mem.add(self.e_active)
        self.assertEqual(self.mem.status_of(self.e_active), ENTRY_ACTIVE)
        got = self.mem.retrieve("i", frozenset(["a", "b"]), threshold=0.0)
        self.assertEqual(len(got), 1)

    def test_provisional_not_retrieved_until_promoted(self):
        self.mem.add(self.e_prov, status=ENTRY_PROVISIONAL)
        self.assertEqual(self.mem.status_counts()[ENTRY_PROVISIONAL], 1)
        self.assertEqual(
            self.mem.retrieve("i", frozenset(["a", "b"]), threshold=0.0), [])
        self.mem.promote(self.e_prov)
        self.assertEqual(
            len(self.mem.retrieve("i", frozenset(["a", "b"]), threshold=0.0)), 1)

    def test_quarantine_excluded_but_kept_for_audit(self):
        self.mem.add(self.e_quar, status=ENTRY_QUARANTINE)
        self.assertIn(self.e_quar, self.mem.entries())        # audit/JSONL
        self.assertEqual(
            self.mem.retrieve("i", frozenset(["a", "b"]), threshold=0.0), [])
        self.mem.quarantine(self.e_quar)  # idempotent
        self.assertEqual(self.mem.status_counts()[ENTRY_QUARANTINE], 1)

    def test_active_filter_applies_before_top_k_cut(self):
        # 1 active + 3 provisional: k=2 must still return the active entry
        self.mem.add(self.e_active)
        for j in range(3):
            e = {"intent_class": "i", "signature_tokens": ["a", "b"],
                 "tier": "simple", "G": 0.5, "ts": float(j), "episode_id": j}
            self.mem.add(e, status=ENTRY_PROVISIONAL)
        got = self.mem.retrieve("i", frozenset(["a", "b"]), k=2, threshold=0.0)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0][0]["episode_id"], 1)

    def test_meta_sidecar_and_eviction_prune(self):
        mem = LifecycleMemory(cap=2)
        e1 = {"intent_class": "i", "signature_tokens": ["a"], "tier": "simple",
              "G": 0.1, "ts": 1.0, "episode_id": 1}
        e2 = {"intent_class": "i", "signature_tokens": ["a"], "tier": "simple",
              "G": 0.2, "ts": 2.0, "episode_id": 2}
        e3 = {"intent_class": "i", "signature_tokens": ["a"], "tier": "simple",
              "G": 0.3, "ts": 3.0, "episode_id": 3}
        mem.add_with_status(e1, ENTRY_PROVISIONAL, {"confidence": 0.7})
        mem.add_with_status(e2, ENTRY_ACTIVE)
        mem.add(e3)  # evicts e1 (oldest)
        self.assertEqual([e["episode_id"] for e in mem.entries()], [2, 3])
        self.assertEqual(mem.status_counts(),
                         {ENTRY_ACTIVE: 2, ENTRY_PROVISIONAL: 0,
                          ENTRY_QUARANTINE: 0})
        self.assertEqual(mem.meta_of(e2), {})

    def test_unknown_status_rejected(self):
        with self.assertRaises(ValueError):
            self.mem.add(self.e_active, status="bogus")


class TestLifecycleMemoryUpdater(unittest.TestCase):
    def _traj_and_eval(self, confidence: float):
        traj = single_step_trajectory(reward=0.4)
        return traj, evaluation_for(traj, [0.4], confidences=[confidence])

    def test_confidence_thresholds_map_to_statuses(self):
        cases = [
            (0.9, ENTRY_ACTIVE),
            (0.8, ENTRY_ACTIVE),      # boundary
            (0.79, ENTRY_PROVISIONAL),
            (0.6, ENTRY_PROVISIONAL),  # boundary
            (0.59, ENTRY_QUARANTINE),
            (None, ENTRY_ACTIVE),      # legacy: no confidence signal
        ]
        for confidence, expected in cases:
            with self.subTest(confidence=confidence):
                traj, ev = self._traj_and_eval(confidence)
                mem = LifecycleMemory()
                entries = LifecycleMemoryUpdater().update(
                    mem, [StepCredit(step_id=0, G=0.4)], traj, ev)
                self.assertEqual(mem.status_of(entries[0]), expected)

    def test_meta_sidecar_populated(self):
        traj = single_step_trajectory(reward=0.4)
        ev = evaluation_for(traj, [0.4], confidences=[0.9])
        ev.steps[0].routing_verdict = "under_routed"
        ev.steps[0].recommended_tier = "reasoning"
        ev.steps[0].failure_tags = ["capability_mismatch"]
        ev.steps[0].feedback = "初始任务已包含跨文件分析需求"
        mem = LifecycleMemory()
        entries = LifecycleMemoryUpdater().update(
            mem, [StepCredit(step_id=0, G=0.4)], traj, ev)
        meta = mem.meta_of(entries[0])
        self.assertEqual(meta["routing_verdict"], "under_routed")
        self.assertEqual(meta["recommended_tier"], "reasoning")
        self.assertEqual(meta["failure_tags"], ["capability_mismatch"])
        self.assertEqual(meta["confidence"], 0.9)
        # entry itself stays frozen 6-key
        self.assertEqual(set(entries[0].keys()), ENTRY_KEYS)

    def test_plain_memory_fallback(self):
        # duck-typed safety: run_ablation swaps in ExperienceMemory
        traj, ev = self._traj_and_eval(0.2)
        mem = ExperienceMemory()
        entries = LifecycleMemoryUpdater().update(
            mem, [StepCredit(step_id=0, G=0.4)], traj, ev)
        self.assertEqual(len(mem), 1)
        self.assertEqual(set(entries[0].keys()), ENTRY_KEYS)


# ---------------------------------------------------- engine.learn_trajectory
class TestEngineLearnTrajectory(unittest.TestCase):
    def test_single_step_equivalent_to_legacy_learn(self):
        # Same seed + same decide/learn sequence: the trajectory path must
        # produce bit-identical memory entries and leave the RNG untouched.
        ts = 12345.0
        eng_legacy = JitRLEngine(config=JitRLConfig(seed=42))
        eng_traj = JitRLEngine(config=JitRLConfig(seed=42))

        for eng in (eng_legacy, eng_traj):
            d = eng.decide(MSG, LOGITS)
            self.assertEqual(d.chosen_tier, "medium")

        legacy_entry = eng_legacy.learn(
            intent="data_analysis",
            signature_tokens=task_signature(MSG),
            tier="medium", reward=0.35, episode_id=7, ts=ts)

        traj = single_step_trajectory(reward=0.35, ts=ts, episode_id=7)
        traj_entries = eng_traj.learn_trajectory(
            traj, evaluation_for(traj, [0.35]))

        self.assertEqual(len(eng_traj.memory), 1)
        self.assertEqual(len(traj_entries), 1)
        self.assertEqual(traj_entries[0], legacy_entry)
        self.assertEqual(eng_traj.memory.entries()[0],
                         eng_legacy.memory.entries()[0])
        # RNG stream untouched by the trajectory path
        self.assertEqual(eng_traj._rng.getstate(), eng_legacy._rng.getstate())

    def test_failed_evaluation_writes_nothing(self):
        eng = JitRLEngine()
        traj = single_step_trajectory(reward=0.5)
        ev = evaluation_for(traj, [0.5], ok=False)
        self.assertEqual(eng.learn_trajectory(traj, ev), [])
        self.assertEqual(len(eng.memory), 0)

    def test_injected_updater_and_assigner(self):
        eng = JitRLEngine(
            memory_updater=LifecycleMemoryUpdater(),
            credit_assigner=DiscountedCreditAssigner(gamma=0.9,
                                                     delayed_weight=0.5))
        traj = RouteTrajectory(trajectory_id="t", episode_id=1)
        for i, r in enumerate((0.2, 0.8)):
            traj.add_step(DecisionStep(
                step_id=i, user_message=f"m{i}", intent_class="data_analysis",
                signature_tokens=[f"tok{i}"], chosen_tier="medium",
                reward=r, ts=1000.0 + i))
        ev = evaluation_for(traj, [0.2, 0.8], confidences=[None, 0.95])
        entries = eng.learn_trajectory(traj, ev)
        self.assertEqual(len(entries), 2)
        # G0 = 0.2 + 0.5*0.9*0.8 = 0.56 ; G1 = 0.8
        self.assertAlmostEqual(entries[0]["G"], 0.56, places=12)
        self.assertAlmostEqual(entries[1]["G"], 0.8, places=12)
        # engine-owned LifecycleMemory? No: default engine memory is plain
        # ExperienceMemory -> fallback path wrote both entries
        self.assertEqual(len(eng.memory), 2)

    def test_engine_with_lifecycle_memory(self):
        mem = LifecycleMemory()
        eng = JitRLEngine(memory=mem, memory_updater=LifecycleMemoryUpdater())
        traj = single_step_trajectory(reward=0.1)
        ev = evaluation_for(traj, [0.1], confidences=[0.65])
        eng.learn_trajectory(traj, ev)
        # 0.65 -> provisional: kept but not retrieved
        self.assertEqual(mem.status_counts()[ENTRY_PROVISIONAL], 1)
        self.assertEqual(mem.retrieve("data_analysis",
                                      frozenset(task_signature(MSG)),
                                      threshold=0.0), [])


if __name__ == "__main__":
    unittest.main()
