"""value/policy: hand-computed numeric cases, argmax flip, lambda branch, z_min."""
from __future__ import annotations

import unittest

from jitrl_core.config import TIERS
from jitrl_core.policy import choose_tier, clamp_logits, modulate_logits
from jitrl_core.value import estimate_values


class SeqRng:
    """Deterministic rng stub: returns queued values for .random()."""

    def __init__(self, values):
        self._values = list(values)

    def random(self) -> float:
        return self._values.pop(0)


def make_entries():
    # N(s): G=1.0 (complex), G=0.0 (complex), G=0.5 (medium)
    return [
        {"tier": "complex", "G": 1.0},
        {"tier": "complex", "G": 0.0},
        {"tier": "medium", "G": 0.5},
    ]


class TestValueHandComputed(unittest.TestCase):
    def test_no_exploration(self):
        est = estimate_values(make_entries(), rng=SeqRng([0.99, 0.99]),
                              lam=0.05, alpha=5.0)
        self.assertAlmostEqual(est.V, 0.5)
        self.assertAlmostEqual(est.Q["complex"], 0.5)
        self.assertAlmostEqual(est.Q["medium"], 0.5)
        self.assertAlmostEqual(est.Q["simple"], 0.0)
        self.assertAlmostEqual(est.Q["reasoning"], 0.0)
        # A = Q - V
        self.assertAlmostEqual(est.A["simple"], -0.5)
        self.assertAlmostEqual(est.A["medium"], 0.0)
        self.assertAlmostEqual(est.A["complex"], 0.0)
        self.assertAlmostEqual(est.A["reasoning"], -0.5)
        # A_norm = A / (max|A| + 1e-8), max|A| = 0.5
        self.assertAlmostEqual(est.A_norm["simple"], -0.5 / (0.5 + 1e-8), places=6)
        self.assertAlmostEqual(est.A_norm["medium"], 0.0, places=9)
        self.assertAlmostEqual(est.A_norm["complex"], 0.0, places=9)
        self.assertAlmostEqual(est.A_norm["reasoning"], -1.0, places=6)
        self.assertEqual(est.n_retrieved, 3)

    def test_exploration_branch_lambda(self):
        # missing tiers (simple, reasoning) draw in TIERS order:
        # simple draws 0.01 < 0.05 -> Q = V + alpha/|N(s)| = 0.5 + 5/3
        est = estimate_values(make_entries(), rng=SeqRng([0.01, 0.99]),
                              lam=0.05, alpha=5.0)
        expected_q = 0.5 + 5.0 / 3.0
        self.assertAlmostEqual(est.Q["simple"], expected_q)
        self.assertAlmostEqual(est.Q["reasoning"], 0.0)
        self.assertAlmostEqual(est.A["simple"], expected_q - 0.5)
        # max|A| is the explored tier's advantage; unexplored reasoning has A=-0.5
        self.assertAlmostEqual(est.A_norm["simple"], 1.0, places=6)
        self.assertAlmostEqual(est.A_norm["reasoning"],
                               -0.5 / (expected_q - 0.5 + 1e-8), places=6)

    def test_empty_memory_no_modulation(self):
        est = estimate_values([], rng=None, lam=0.05, alpha=5.0)
        self.assertEqual(est.V, 0.0)
        self.assertEqual(est.n_retrieved, 0)
        for t in TIERS:
            self.assertEqual(est.A_norm[t], 0.0)

    def test_max_abs_zero_gives_zero_advantage(self):
        # every tier observed with G == V -> all A == 0 (no rng draws needed)
        entries = [{"tier": t, "G": 0.5} for t in TIERS]
        est = estimate_values(entries, rng=None, lam=0.05, alpha=5.0)
        self.assertAlmostEqual(est.V, 0.5)
        for t in TIERS:
            self.assertEqual(est.A_norm[t], 0.0)


class TestPolicy(unittest.TestCase):
    Z = {"simple": 1.0, "medium": 2.0, "complex": 0.0, "reasoning": 1.0}
    A_NORM = {"simple": -1.0, "medium": 0.0, "complex": 0.0, "reasoning": -1.0}

    def test_modulation_hand_case(self):
        z_mod = modulate_logits(self.Z, self.A_NORM, beta=5.0, z_min=-10.0)
        self.assertAlmostEqual(z_mod["simple"], 1.0 - 5.0)
        self.assertAlmostEqual(z_mod["medium"], 2.0)
        self.assertAlmostEqual(z_mod["complex"], 0.0)
        self.assertAlmostEqual(z_mod["reasoning"], 1.0 - 5.0)
        # no argmax flip here
        self.assertEqual(choose_tier(z_mod), "medium")

    def test_flip_case(self):
        # optimistic advantage on a non-baseline tier flips the argmax
        a_norm = {"simple": 1.0, "medium": 0.0, "complex": 0.0, "reasoning": 0.0}
        z_mod = modulate_logits(self.Z, a_norm, beta=5.0, z_min=-10.0)
        self.assertAlmostEqual(z_mod["simple"], 6.0)
        self.assertEqual(choose_tier(self.Z), "medium")      # base choice
        self.assertEqual(choose_tier(z_mod), "simple")       # flipped

    def test_z_min_clamp(self):
        clamped = clamp_logits({"simple": -11.0, "medium": 0.5}, z_min=-10.0)
        self.assertEqual(clamped["simple"], -10.0)
        self.assertEqual(clamped["medium"], 0.5)
        z_mod = modulate_logits({"simple": 0.0, "medium": 0.0},
                                {"simple": -2.5, "medium": 0.0},
                                beta=5.0, z_min=-10.0)
        self.assertEqual(z_mod["simple"], -10.0)  # 0 - 12.5 clamped up to z_min

    def test_tie_break_first_in_tiers(self):
        self.assertEqual(choose_tier({"simple": 1.0, "medium": 1.0,
                                      "complex": 1.0, "reasoning": 1.0}), "simple")
        self.assertEqual(choose_tier({"reasoning": 2.0, "complex": 2.0}), "complex")


if __name__ == "__main__":
    unittest.main()
