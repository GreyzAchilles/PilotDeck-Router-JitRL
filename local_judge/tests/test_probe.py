"""Unit tests for probe.py b10903 response compatibility (no network).

Same failure mode covered by test_client.py: llama-server b10903 (thinking
chat template) routes grammar-forced output into `message.reasoning_content`
and leaves `message.content` empty. The probe must read the real emission
there so its z tables are correct instead of crashing (method B content
assertion) or reporting missing / Z_MIN-clamped z values.
"""
from __future__ import annotations

import unittest

from local_judge import probe


def lp_entry(token: str, logprob: float, top=None) -> dict:
    return {"token": token, "logprob": logprob,
            "top_logprobs": top if top is not None else []}


def b10903_response(tier: str, name_logprob: float = -8.009,
                    top_logprobs=None) -> dict:
    """Response shape from llama-server b10903 (thinking chat template):
    emitted text in reasoning_content, empty content, trailing '' token
    artifact after </tier>. Mirrors the fixture in test_client.py."""
    entries = [
        lp_entry("<", -8.061), lp_entry("t", -6.625), lp_entry("ier", -0.001),
        lp_entry(">", -0.184),
        lp_entry(tier, name_logprob, top=top_logprobs),
        lp_entry("</", -0.024), lp_entry("t", -0.0), lp_entry("ier", -0.0),
        lp_entry(">", -0.0), lp_entry("", -0.0),
    ]
    return {"choices": [{
        "finish_reason": "stop",
        "message": {"role": "assistant", "content": "",
                    "reasoning_content": f"<tier>{tier}</tier>"},
        "logprobs": {"content": entries},
    }]}


class TestProbeEmittedText(unittest.TestCase):
    def test_prefers_nonempty_content(self):
        resp = {"choices": [{"message": {
            "content": "<tier>simple</tier>",
            "reasoning_content": "<tier>medium</tier>"}}]}
        self.assertEqual(probe._emitted_text(resp), "<tier>simple</tier>")

    def test_falls_back_to_reasoning_content(self):
        resp = b10903_response("medium")
        self.assertEqual(probe._emitted_text(resp), "<tier>medium</tier>")

    def test_empty_everywhere(self):
        resp = {"choices": [{"message": {"content": "",
                                         "reasoning_content": ""}}]}
        self.assertEqual(probe._emitted_text(resp), "")


class TestNameTokensInfoB10903(unittest.TestCase):
    def test_reads_z_from_reasoning_content(self):
        # Without the fallback this raised AssertionError (empty content !=
        # '<tier>medium</tier>') and method B produced no z at all.
        resp = b10903_response("medium")
        count, total, toks = probe.name_tokens_info(resp, "medium")
        self.assertEqual((count, toks), (1, ["medium"]))
        self.assertAlmostEqual(total, -8.009)
        self.assertNotEqual(total, probe.Z_MIN)

    def test_classic_content_still_works(self):
        resp = b10903_response("simple")
        resp["choices"][0]["message"] = {"role": "assistant",
                                         "content": "<tier>simple</tier>"}
        count, total, toks = probe.name_tokens_info(resp, "simple")
        self.assertEqual((count, toks), (1, ["simple"]))
        self.assertAlmostEqual(total, -8.009)


class TestMethodAB10903(unittest.TestCase):
    def test_z_table_complete_not_zmin(self):
        # All four tier names visible in top_logprobs at the name position;
        # the probe must report them, not all-None / Z_MIN entries.
        top = [lp_entry("medium", -8.009), lp_entry("simple", -8.5),
               lp_entry("complex", -9.1), lp_entry("reasoning", -9.7)]
        resp = b10903_response("medium", top_logprobs=top)
        orig_chat = probe.chat
        probe.chat = lambda *a, **k: (resp, 12.0)
        try:
            out = probe.method_a("prompt")
        finally:
            probe.chat = orig_chat
        self.assertEqual(out["chosen"], "medium")   # not '??'
        self.assertAlmostEqual(out["z"]["medium"], -8.009)
        self.assertAlmostEqual(out["z"]["simple"], -8.5)
        self.assertAlmostEqual(out["z"]["complex"], -9.1)
        self.assertAlmostEqual(out["z"]["reasoning"], -9.7)
        self.assertTrue(all(v is not None for v in out["z"].values()))
        self.assertTrue(all(v != probe.Z_MIN for v in out["z"].values()))


if __name__ == "__main__":
    unittest.main()
