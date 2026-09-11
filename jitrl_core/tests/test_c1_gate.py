"""C1 min-neighbor gate — pre-registered in docs/S2-C1-PREREGISTRATION.md v1.0.

The ONLY policy change: when 0 < n_retrieved < min_neighbors, modulation is
suppressed (z_mod = z_base, A_norm = 0, chosen = base_choice, flipped=False).
Backward compatibility: default min_neighbors=1 must reproduce C0 exactly
(>=1 retrieved neighbor => modulation allowed). estimate_values is always
called, so the engine RNG stream must stay bit-identical regardless of the
gate. No value/policy/reward math is changed.
"""
from __future__ import annotations

import unittest

from jitrl_core.config import JitRLConfig, TIERS
from jitrl_core.engine import JitRLEngine
from jitrl_core.policy import modulate_logits
from jitrl_core.state import task_signature

MSG = "分析这批销售数据，统计环比，对比渠道表现"
LOGITS = {"simple": 0.0, "medium": 1.0, "complex": 0.0, "reasoning": 0.0}
ZEROS = {t: 0.0 for t in TIERS}


def c0_expected_z_mod(d) -> dict:
    """The unchanged C0 formula: z' = clamp(z + beta * A_norm, z_min)."""
    return modulate_logits(d.z_base, d.A_norm, 5.0, -10.0, TIERS)


def learned_engine(min_neighbors: int | None = None, n: int = 0,
                   reward: float = -0.3) -> JitRLEngine:
    cfg = (JitRLConfig(min_neighbors=min_neighbors)
           if min_neighbors is not None else JitRLConfig())
    eng = JitRLEngine(config=cfg)
    for i in range(n):
        eng.learn(intent="data_analysis",
                  signature_tokens=task_signature(MSG),
                  tier="medium", reward=reward, episode_id=i + 1)
    return eng


class TestConfigMinNeighbors(unittest.TestCase):
    def test_default_is_one_c0_semantics(self):
        self.assertEqual(JitRLConfig().min_neighbors, 1)

    def test_snapshot_carries_min_neighbors(self):
        self.assertEqual(JitRLConfig().snapshot()["min_neighbors"], 1)
        self.assertEqual(
            JitRLConfig(min_neighbors=3).snapshot()["min_neighbors"], 3)


class TestDefaultReproducesC0(unittest.TestCase):
    def test_explicit_one_equals_default_bit_for_bit(self):
        seq = [0, 1, 2, 3]  # growing neighborhood
        for n in seq:
            d_default = learned_engine(n=n).decide(MSG, LOGITS)
            d_one = learned_engine(min_neighbors=1, n=n).decide(MSG, LOGITS)
            self.assertEqual(d_default, d_one)

    def test_default_modulates_whenever_n_ge_1(self):
        for n in (1, 2, 3, 4):
            d = learned_engine(n=n).decide(MSG, LOGITS)
            self.assertEqual(d.n_retrieved, n)
            self.assertEqual(d.min_neighbors, 1)
            self.assertTrue(d.modulation_active)
            self.assertIsNone(d.gate_reason)
            # exactly the C0 formula, unchanged
            self.assertEqual(d.z_mod, c0_expected_z_mod(d))
            if n >= 1:
                self.assertNotEqual(d.z_mod, d.z_base)

    def test_empty_memory_still_pure_baseline(self):
        d = learned_engine(n=0).decide(MSG, LOGITS)
        self.assertEqual(d.n_retrieved, 0)
        self.assertFalse(d.modulation_active)
        self.assertEqual(d.gate_reason, "no_neighbors")
        self.assertEqual(d.z_mod, d.z_base)
        self.assertEqual(d.A_norm, ZEROS)
        self.assertEqual(d.chosen_tier, d.base_choice)
        self.assertFalse(d.flipped)


class TestMinThreeGate(unittest.TestCase):
    def test_n1_and_n2_are_gated(self):
        for n in (1, 2):
            d = learned_engine(min_neighbors=3, n=n).decide(MSG, LOGITS)
            self.assertEqual(d.n_retrieved, n)
            self.assertEqual(d.min_neighbors, 3)
            self.assertEqual(d.gate_reason, "below_min_neighbors")
            self.assertFalse(d.modulation_active)
            # pre-registered gated outcome
            self.assertEqual(d.z_mod, d.z_base)
            self.assertEqual(d.A_norm, ZEROS)
            self.assertEqual(d.chosen_tier, d.base_choice)
            self.assertFalse(d.flipped)
            self.assertEqual(d.chosen_tier, "medium")  # pure judge argmax
            # diagnostics of the neighborhood are still recorded (not zeroed)
            self.assertAlmostEqual(d.V, -0.3)
            self.assertAlmostEqual(d.Q["medium"], -0.3)

    def test_n3_boundary_allows_modulation(self):
        d = learned_engine(min_neighbors=3, n=3).decide(MSG, LOGITS)
        self.assertEqual(d.n_retrieved, 3)
        self.assertIsNone(d.gate_reason)
        self.assertTrue(d.modulation_active)
        # C0 modulation path re-activates exactly at the boundary
        self.assertEqual(d.z_mod, c0_expected_z_mod(d))
        self.assertNotEqual(d.z_mod, d.z_base)
        self.assertTrue(any(v != 0.0 for v in d.A_norm.values()))

    def test_no_neighbors_reason_independent_of_min(self):
        for mn in (1, 3):
            d = learned_engine(min_neighbors=mn, n=0).decide(MSG, LOGITS)
            self.assertEqual(d.gate_reason, "no_neighbors")
            self.assertFalse(d.modulation_active)
            self.assertEqual(d.z_mod, d.z_base)


class TestRNGDeterminism(unittest.TestCase):
    SEQUENCE = [
        # (tier learned after the decide, reward)
        ("medium", -0.3), ("medium", -0.3), ("complex", -0.1),
        ("medium", -0.5), ("complex", 0.2), ("medium", -0.3),
    ]

    def _run_sequence(self, min_neighbors: int | None):
        eng = learned_engine(min_neighbors=min_neighbors)
        decisions = []
        for i, (tier, g) in enumerate(self.SEQUENCE):
            decisions.append(eng.decide(MSG, LOGITS))
            eng.learn(intent="data_analysis",
                      signature_tokens=task_signature(MSG),
                      tier=tier, reward=g, episode_id=i + 1)
        decisions.append(eng.decide(MSG, LOGITS))
        return eng, decisions

    def test_gate_does_not_shift_rng_phase(self):
        # The gate changes z_mod but must NOT change how many RNG draws
        # estimate_values consumes: same seed + same sequence => identical
        # RNG state, whatever min_neighbors is.
        eng1, dec1 = self._run_sequence(1)
        eng3, dec3 = self._run_sequence(3)
        # sanity: the gate actually fired differently along the way
        gated = [d for d in dec3 if d.gate_reason == "below_min_neighbors"]
        self.assertTrue(gated)
        self.assertTrue(any(d.modulation_active for d in dec1))
        self.assertEqual(eng1._rng.getstate(), eng3._rng.getstate())

    def test_same_seed_same_gate_is_deterministic(self):
        _, dec_a = self._run_sequence(3)
        _, dec_b = self._run_sequence(3)
        self.assertEqual(dec_a, dec_b)


if __name__ == "__main__":
    unittest.main()
