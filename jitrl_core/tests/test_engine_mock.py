"""MockJudgeClient contract + JitRLEngine decide/learn behavior."""
from __future__ import annotations

import unittest

from jitrl_core.config import JitRLConfig, TIERS
from jitrl_core.engine import JitRLEngine
from jitrl_core.mock_judge import MockJudgeClient
from jitrl_core.state import task_signature


class TestMockJudgeContract(unittest.TestCase):
    def setUp(self):
        self.judge = MockJudgeClient(seed=42)

    def test_result_schema_and_consistency(self):
        res = self.judge.judge("写一个函数，实现快排，写单元测试")
        self.assertEqual(set(res.keys()), {"tier_logits", "chosen_tier",
                                           "raw_output", "latency_ms"})
        self.assertEqual(set(res["tier_logits"].keys()), set(TIERS))
        best = max(TIERS, key=lambda t: res["tier_logits"][t])
        self.assertEqual(res["chosen_tier"], best)          # argmax self-consistent
        self.assertEqual(res["raw_output"], f"<tier>{res['chosen_tier']}</tier>")
        self.assertGreaterEqual(res["latency_ms"], 5.0)
        self.assertLessEqual(res["latency_ms"], 20.0)

    def test_continuation_echoes_previous_tier(self):
        res = self.judge.judge("继续，好的，接着", previous_tier="complex")
        self.assertEqual(res["chosen_tier"], "complex")
        self.assertGreater(res["tier_logits"]["complex"], 5.0)

    def test_seeded_determinism(self):
        a = MockJudgeClient(seed=7).judge("分析这份数据")
        b = MockJudgeClient(seed=7).judge("分析这份数据")
        self.assertEqual(a, b)

    def test_medium_bias_and_cue_heuristics(self):
        # no cue -> medium (systematic bias)
        r1 = self.judge.judge("随便聊聊今天的天气情况吧")
        self.assertEqual(r1["chosen_tier"], "medium")
        # code cue -> complex despite bias
        r2 = self.judge.judge("帮我实现一个函数，输入校验")
        self.assertEqual(r2["chosen_tier"], "complex")


class TestEngine(unittest.TestCase):
    def test_empty_memory_is_pure_baseline(self):
        eng = JitRLEngine()
        d = eng.decide("分析这份数据", {"simple": 0.0, "medium": 1.0,
                                        "complex": 0.0, "reasoning": 0.0})
        self.assertEqual(d.n_retrieved, 0)
        self.assertEqual(d.top_similarity, 0.0)
        self.assertEqual(d.V, 0.0)
        self.assertEqual(d.z_mod, d.z_base)      # no modulation
        self.assertEqual(d.chosen_tier, "medium")
        self.assertFalse(d.flipped)

    def test_learn_then_decide_modulates(self):
        eng = JitRLEngine()
        msg = "分析这批销售数据，统计环比，对比渠道表现"
        toks = task_signature(msg)
        for g in (-0.3, -0.3):
            eng.learn(intent="data_analysis",
                      signature_tokens=toks,
                      tier="medium", reward=g, episode_id=1)
        d = eng.decide(msg, {"simple": 0.0, "medium": 1.0,
                             "complex": 0.0, "reasoning": 0.0})
        self.assertEqual(d.n_retrieved, 2)
        self.assertGreater(d.top_similarity, 0.0)
        self.assertAlmostEqual(d.V, -0.3)
        self.assertEqual(d.Q["medium"], -0.3)
        self.assertTrue(d.flipped)               # medium de-modulated, argmax moved
        self.assertNotEqual(d.chosen_tier, "medium")
        # entry lands in memory with sorted tokens
        self.assertEqual(len(eng.memory), 2)
        entry = eng.memory.entries()[0]
        self.assertEqual(set(entry.keys()),
                         {"intent_class", "signature_tokens", "tier", "G",
                          "ts", "episode_id"})
        self.assertEqual(entry["signature_tokens"],
                         sorted(entry["signature_tokens"]))

    def test_threshold_zero_retrieves_everything_same_intent(self):
        eng = JitRLEngine(config=JitRLConfig(jaccard_threshold=0.0))
        eng.learn(intent="other", signature_tokens=["qqq"], tier="simple",
                  reward=1.0, episode_id=1)
        d = eng.decide("zzz wwww", {"simple": 1.0, "medium": 0.0,
                                    "complex": 0.0, "reasoning": 0.0})
        self.assertEqual(d.n_retrieved, 1)
        self.assertEqual(d.top_similarity, 0.0)


if __name__ == "__main__":
    unittest.main()
