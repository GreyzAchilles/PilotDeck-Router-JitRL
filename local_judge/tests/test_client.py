"""Unit tests for JudgeClient response parsing (no network).

Covers the b10903 llama-server compatibility fix: grammar-forced output
arriving in `message.reasoning_content` with `message.content` empty.
Replicates the exact response shape observed live on 2026-09-13.
"""
from __future__ import annotations

import unittest

from local_judge.client import JudgeClient


def lp_entry(token: str, logprob: float, top=None) -> dict:
    return {"token": token, "logprob": logprob,
            "top_logprobs": top if top is not None else []}


def b10903_response(tier: str, name_logprob: float = -8.009) -> dict:
    """Response shape from llama-server b10903 (thinking chat template):
    emitted text in reasoning_content, empty content, trailing '' token
    artifact after </tier>."""
    entries = [
        lp_entry("<", -8.061), lp_entry("t", -6.625), lp_entry("ier", -0.001),
        lp_entry(">", -0.184),
        lp_entry(tier, name_logprob,
                 top=[lp_entry(tier, name_logprob), lp_entry("reason", -3.2)]),
        lp_entry("</", -0.024), lp_entry("t", -0.0), lp_entry("ier", -0.0),
        lp_entry(">", -0.0), lp_entry("", -0.0),
    ]
    return {"choices": [{
        "finish_reason": "stop",
        "message": {"role": "assistant", "content": "",
                    "reasoning_content": f"<tier>{tier}</tier>"},
        "logprobs": {"content": entries},
    }]}


def classic_response(tier: str) -> dict:
    """Pre-b10903 shape: text in message.content, no reasoning field."""
    resp = b10903_response(tier)
    msg = resp["choices"][0]["message"]
    msg["content"] = f"<tier>{tier}</tier>"
    del msg["reasoning_content"]
    return resp


class TestEmittedText(unittest.TestCase):
    def test_prefers_nonempty_content(self):
        resp = {"choices": [{"message": {
            "content": "<tier>simple</tier>",
            "reasoning_content": "<tier>medium</tier>"}}]}
        self.assertEqual(JudgeClient._emitted_text(resp), "<tier>simple</tier>")

    def test_falls_back_to_reasoning_content(self):
        resp = b10903_response("medium")
        self.assertEqual(JudgeClient._emitted_text(resp), "<tier>medium</tier>")

    def test_empty_everywhere(self):
        resp = {"choices": [{"message": {"content": "",
                                         "reasoning_content": ""}}]}
        self.assertEqual(JudgeClient._emitted_text(resp), "")


class TestParseForced(unittest.TestCase):
    def setUp(self):
        self.client = JudgeClient()

    def test_accepts_reasoning_content_response(self):
        parsed = self.client._parse_forced(b10903_response("medium"), "medium")
        self.assertIsNotNone(parsed)
        toks, total, top = parsed
        self.assertEqual(toks, ["medium"])
        self.assertAlmostEqual(total, -8.009)
        self.assertEqual(top.get("medium"), -8.009)

    def test_accepts_classic_response(self):
        parsed = self.client._parse_forced(classic_response("medium"), "medium")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed[0], ["medium"])

    def test_rejects_wrong_text(self):
        resp = b10903_response("medium")
        parsed = self.client._parse_forced(resp, "simple")
        self.assertIsNone(parsed)


class TestMethodAParse(unittest.TestCase):
    def test_method_a_reads_reasoning_content(self):
        client = JudgeClient()
        entries = [
            lp_entry("<", -5.0), lp_entry("tier", -0.1), lp_entry(">", -0.1),
            lp_entry("reasoning", -2.0,
                     top=[lp_entry("reasoning", -2.0), lp_entry("medium", -2.7)]),
            lp_entry("</", -0.1), lp_entry("tier", -0.1), lp_entry(">", -0.1),
        ]
        fake = {"choices": [{
            "message": {"role": "assistant", "content": "",
                        "reasoning_content": "<tier>reasoning</tier>"},
            "logprobs": {"content": entries},
        }]}
        client._chat = lambda *a, **k: fake
        result = client._judge_method_a("prompt")
        self.assertEqual(result["chosen_tier"], "reasoning")
        self.assertEqual(result["raw_output"], "<tier>reasoning</tier>")
        self.assertAlmostEqual(result["tier_logits"]["reasoning"], -2.0)
        self.assertAlmostEqual(result["tier_logits"]["medium"], -2.7)


if __name__ == "__main__":
    unittest.main()
