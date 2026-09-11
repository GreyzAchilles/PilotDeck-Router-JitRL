"""Tests for harness/run_ablation.py and the local_judge memory_block hook.

Implementation gate of S3-TAB8-PREREGISTRATION.md v1.0 §10, item by item:

 1. memory_block=None -> Judge prompt byte-identical to the current version;
 2. Method B's four requests use exactly the same injected prompt;
 3. prefix replay loads only source_order < i for the i-th frozen task;
 4. the current task's own C0 outcome and future outcomes are NOT
    retrievable (with a leaky full-trace control that WOULD retrieve them);
 5. both arms see the same memory prefix and retrieval set per task;
 6. the whitelist block renders no GT / checklist / Evaluator structure /
    raw text / task ids / timestamps / secrets;
 7. no neighbors -> PM prompt unchanged AND LM not modulated;
 8. T8-PM never applies logit modulation; T8-LM never injects a block;
 9. the frozen memory source file and the replay copy are hash-identical
    before and after a run;
10. memory_written=false everywhere and the memory entry count is unchanged;
11. the executed task order strictly equals the frozen 24-task order;
12. (analyzer side — see eval/tests/test_analyze_ablation.py);
13. all pre-existing tests keep passing (suite-level).

Plus the exact-C0-LM-math gate: replaying the frozen trace through the
unmodified engine reproduces C0's V/Q/A_norm/z_mod/tier choices
bit-for-bit from the C0 z_base vectors.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock  # noqa: F401  (unittest.mock patching in TestCli)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.cpa_client import CompletionResult  # noqa: E402
import harness.run_ablation as ra  # noqa: E402
from harness.run import load_tasks  # noqa: E402
from jitrl_core import JitRLConfig, JitRLEngine  # noqa: E402
from jitrl_core.config import TIERS  # noqa: E402
from jitrl_core.memory import ExperienceMemory  # noqa: E402
from jitrl_core.policy import modulate_logits  # noqa: E402
from jitrl_core.state import intent_class, task_signature  # noqa: E402
from local_judge import config as judge_config  # noqa: E402
from local_judge.client import (  # noqa: E402
    CONT_TMPL,
    PROMPT_TMPL,
    JudgeClient,
    build_prompt,
)

TRACE_PATH = REPO_ROOT / "eval" / "ablation" / "frozen-c0-memory-trace.jsonl"
SHA_JSON_PATH = REPO_ROOT / "eval" / "ablation" / "frozen-c0-memory-trace.sha256.json"
C0_MEMORY_PATH = REPO_ROOT / "logs" / "s2_full_C_main24_retry2_memory.jsonl"
C0_RUN_PATH = REPO_ROOT / "logs" / "s2_full_C_main24_retry2.jsonl"
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
    """Scripted stand-in (mirrors harness/tests/test_run_real.py)."""

    def completion(self, model, messages, *, max_tokens=512, temperature=0.7,
                   extra_body=None):
        api = model.split("/")[-1]
        content = EVAL_JSON if api == "gpt-5.6-sol" else "exec reply text"
        return CompletionResult(content=content, usage=usage(), model=api,
                                finish_reason="stop", latency_ms=5.0)


class SpyJudge:
    """Records every judge call (message, previous_tier, memory_block) and
    returns a fixed logit vector with a distinctive latency."""

    def __init__(self, logits=None, latency_ms=123.0):
        self._logits = logits or {"simple": -2.0, "medium": -1.0,
                                  "complex": -3.0, "reasoning": -2.5}
        self.latency_ms = latency_ms
        self.calls: list[tuple[str, str | None, str | None]] = []

    def judge(self, user_message, previous_tier=None, memory_block=None):
        self.calls.append((user_message, previous_tier, memory_block))
        return {"tier_logits": dict(self._logits), "chosen_tier": "medium",
                "raw_output": "<tier>medium</tier>",
                "latency_ms": self.latency_ms}


def _lp_entry(token, logprob, top=None):
    return {"token": token, "logprob": logprob,
            "top_logprobs": top if top is not None else []}


class MethodBCaptureClient(JudgeClient):
    """JudgeClient with the transport replaced by a recorder: captures the
    exact prompt of every Method B (forced-emission) request, no network."""

    def __init__(self):
        super().__init__()
        # pre-seed the canonical-token cache -> no /tokenize HTTP call
        self._canonical = {t: list(p) for t, p in
                           judge_config.CANONICAL_TOKENS_FALLBACK.items()}
        self.requests: list[tuple[str, str]] = []

    def _chat(self, prompt, grammar, top_logprobs, temperature):
        tier = re.search(r'\("([^"]+)"\)', grammar).group(1)
        self.requests.append((prompt, tier))
        pieces = self._canonical[tier]
        entries = [_lp_entry("<", -8.0), _lp_entry("t", -6.0),
                   _lp_entry("ier", -0.1), _lp_entry(">", -0.1)]
        for k, piece in enumerate(pieces):
            entries.append(_lp_entry(piece, -2.0 - 0.1 * k,
                                     top=[_lp_entry(piece, -2.0 - 0.1 * k)]))
        entries += [_lp_entry("</", -0.1), _lp_entry("tier", -0.1),
                    _lp_entry(">", -0.1)]
        return {"choices": [{
            "finish_reason": "stop",
            "message": {"role": "assistant",
                        "content": f"<tier>{tier}</tier>"},
            "logprobs": {"content": entries},
        }]}


def load_trace_entries():
    return [json.loads(l) for l in
            TRACE_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_c0_records():
    return [json.loads(l) for l in
            C0_RUN_PATH.read_text(encoding="utf-8").splitlines()
            if l.strip() and "_summary" not in l]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def run_arm(arm, tmp, *, task_ids=None, judge=None, cpa=None, **kw):
    """Run one arm over the real tasks + frozen trace into a tmp file."""
    out = Path(tmp) / f"out_{arm.replace('-', '_')}.jsonl"
    summary = ra.run(
        arm=arm, tasks_path=TASKS_PATH, memory_trace_path=TRACE_PATH,
        judge_kind="mock", out_path=out, task_ids=task_ids,
        cpa_client=cpa or FakeCPAClient(), judge=judge, **kw)
    records = []
    for line in out.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        if "_summary" not in obj:
            records.append(obj)
    return summary, records, out


# ------------------------------------------- §10.1/§10.2 prompt byte identity

class TestPromptByteIdentity(unittest.TestCase):
    MSG = "给我分析一下这组数据并写一段总结"
    PREV = "medium"
    BLOCK = ("## Reference - similar historical requests (anonymized outcomes)\n"
             "- neighbor_count: 1\n"
             "- neighbor_1: tier=simple G=+0.1234 similarity=0.5678\n")

    def test_none_block_byte_identical_with_prev(self):
        expected = (PROMPT_TMPL
                    .replace("{CONTINUATION}", CONT_TMPL.replace("{PT}", self.PREV))
                    .replace("{USER_MESSAGE}", self.MSG))
        self.assertEqual(build_prompt(self.MSG, self.PREV), expected)
        self.assertEqual(build_prompt(self.MSG, self.PREV, None), expected)
        self.assertEqual(build_prompt(self.MSG, self.PREV, memory_block=None),
                         build_prompt(self.MSG, self.PREV))

    def test_none_block_byte_identical_without_prev(self):
        expected = (PROMPT_TMPL.replace("{CONTINUATION}", "")
                    .replace("{USER_MESSAGE}", self.MSG))
        self.assertEqual(build_prompt(self.MSG), expected)
        self.assertEqual(build_prompt(self.MSG, None, None), expected)

    def test_judge_none_block_sends_identical_prompts(self):
        # §10.1 at the client level: judge(..., memory_block=None) must issue
        # the same 4 Method B requests as judge(...) without the argument.
        base = MethodBCaptureClient()
        base.judge(self.MSG, previous_tier=self.PREV)
        with_none = MethodBCaptureClient()
        with_none.judge(self.MSG, previous_tier=self.PREV, memory_block=None)
        self.assertEqual(len(base.requests), 4)
        self.assertEqual([p for p, _ in base.requests],
                         [p for p, _ in with_none.requests])
        self.assertTrue(all(p == build_prompt(self.MSG, self.PREV)
                            for p, _ in base.requests))
        # tiers are requested once each
        self.assertEqual(sorted(t for _, t in base.requests), sorted(TIERS))

    def test_method_b_four_requests_share_injected_prompt(self):
        # §10.2: all four forced-emission requests use the SAME injected
        # prompt, byte-equal to build_prompt(message, prev, block).
        client = MethodBCaptureClient()
        client.judge(self.MSG, previous_tier=self.PREV, memory_block=self.BLOCK)
        prompts = [p for p, _ in client.requests]
        self.assertEqual(len(prompts), 4)
        self.assertEqual(len(set(prompts)), 1)
        self.assertEqual(prompts[0], build_prompt(self.MSG, self.PREV, self.BLOCK))
        # the block sits before "User message:" and the user text stays single
        self.assertIn(self.BLOCK.rstrip(), prompts[0])
        self.assertEqual(prompts[0].count(self.MSG), 1)

    def test_placeholder_block_refused(self):
        for evil in ("x {USER_MESSAGE} y", "{CONTINUATION}", "{PT}",
                     "tier {USER_MESSAGE}"):
            with self.assertRaises(ValueError, msg=evil):
                build_prompt(self.MSG, None, evil)

    def test_empty_block_refused(self):
        with self.assertRaises(ValueError):
            build_prompt(self.MSG, None, "   \n ")


# --------------------------------------- §10.3/§10.4 frozen trace + replay

class TestFrozenTraceAndPrefixReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entries = load_trace_entries()
        cls.tasks = {t["task_id"]: t for t in load_tasks(TASKS_PATH)}

    def test_trace_schema_order_and_preserved_fields(self):
        self.assertEqual(len(self.entries), 24)
        for i, e in enumerate(self.entries):
            self.assertEqual(e["source_order"], i)
            self.assertEqual(e["source_task_id"], ra.FROZEN_TASK_ORDER[i])
            # original C0 fields preserved verbatim
            self.assertIn("intent_class", e)
            self.assertIsInstance(e["signature_tokens"], list)
            self.assertIn(e["tier"], TIERS)
            self.assertIsInstance(e["G"], float)
            self.assertIn("ts", e)
            self.assertIn("episode_id", e)

    def test_sidecar_hashes_match_files(self):
        meta = json.loads(SHA_JSON_PATH.read_text(encoding="utf-8"))
        self.assertEqual(meta["copy"]["sha256"], sha256_of(TRACE_PATH))
        self.assertEqual(meta["source"]["memory_sha256"],
                         sha256_of(C0_MEMORY_PATH))
        self.assertEqual(meta["frozen_task_order"], list(ra.FROZEN_TASK_ORDER))

    def test_prefix_replay_loads_only_source_order_below_i(self):
        # §10.3: for every frozen position i the prefix holds exactly the
        # entries with source_order < i, in ascending source_order.
        for i in range(24):
            prefix = ra.prefix_entries(self.entries, i)
            self.assertEqual(len(prefix), i, f"index {i}")
            self.assertEqual([e["source_order"] for e in prefix], list(range(i)))
            mem = ra.build_prefix_memory(self.entries, i, 5000)
            self.assertEqual(len(mem), i)

    def test_current_and_future_outcomes_not_retrievable(self):
        # §10.4: no task may retrieve its own C0 outcome or any later one.
        # Control: a LEAKY full-trace memory WOULD retrieve them (the
        # code_gen family shares signature tokens, similarity 1.0), proving
        # the prefix filter is what blocks the leak.
        cfg = JitRLConfig()
        leaky = ExperienceMemory(cap=5000)
        for e in self.entries:
            leaky.add(ra.entry_to_engine_schema(e))

        def source_order_of(match_entry):
            for x in self.entries:
                if x["ts"] == match_entry["ts"] and x["G"] == match_entry["G"]:
                    return x["source_order"]
            return None

        any_hit = False
        for i, tid in enumerate(ra.FROZEN_TASK_ORDER):
            message = self.tasks[tid]["message"]
            prefix_mem = ra.build_prefix_memory(self.entries, i, cfg.memory_cap)
            got = ra.retrieve_prefix(prefix_mem, message, cfg)
            for e, _sim in got:  # every neighbor strictly predates task i
                so = source_order_of(e)
                self.assertIsNotNone(so, tid)
                self.assertLess(so, i, f"{tid} retrieved source_order={so} >= {i}")
            if not got:
                continue
            any_hit = True
            # leaky control retrieves strictly more, including self/future
            leaky_got = leaky.retrieve(intent_class(message),
                                       frozenset(task_signature(message)),
                                       k=cfg.k, threshold=cfg.jaccard_threshold)
            self.assertGreater(len(leaky_got), len(got), tid)
            leaky_orders = [source_order_of(e) for e, _ in leaky_got]
            self.assertTrue(any(so is not None and so >= i for so in leaky_orders),
                            f"{tid}: leaky control did not retrieve self/future")
        self.assertTrue(any_hit, "expected at least one retrieval-hit task")

    def test_prefix_sizes_follow_frozen_index_in_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, records, _ = run_arm("T8-LM", tmp)
        for r in records:
            self.assertEqual(r["memory_prefix_size"],
                             ra.FROZEN_TASK_INDEX[r["task_id"]])


# --------------------------------------------- exact C0 LM math (prereg §3.1)

class TestExactC0Replay(unittest.TestCase):
    def test_engine_replay_reproduces_c0_records_bit_for_bit(self):
        """One engine per run + per-task prefix swap must reproduce C0's
        V/Q/A_norm/z_mod/chosen/base_choice/flipped from the frozen trace
        and C0's own z_base — including the RNG-dependent exploration Q."""
        entries = load_trace_entries()
        tasks = {t["task_id"]: t for t in load_tasks(TASKS_PATH)}
        c0 = load_c0_records()
        cfg = JitRLConfig()
        engine = JitRLEngine(config=cfg)
        for i, tid in enumerate(ra.FROZEN_TASK_ORDER):
            rec = c0[i]
            self.assertEqual(rec["task_id"], tid)
            engine.memory = ra.build_prefix_memory(entries, i, cfg.memory_cap)
            d = engine.decide(tasks[tid]["message"], rec["z_base"])
            self.assertEqual(d.V, rec["V"], tid)
            self.assertEqual(d.Q, rec["Q"], tid)
            self.assertEqual(d.A_norm, rec["A_norm"], tid)
            self.assertEqual(d.z_base, rec["z_base"], tid)
            self.assertEqual(d.z_mod, rec["z_mod"], tid)
            self.assertEqual(d.chosen_tier, rec["chosen_tier"], tid)
            self.assertEqual(d.base_choice, rec["base_choice"], tid)
            self.assertEqual(d.flipped, rec["flipped"], tid)
            self.assertEqual(d.n_retrieved, rec["n_retrieved"], tid)
            self.assertEqual(d.top_similarity, rec["top_similarity"], tid)
            # C0 formula parity at the record level
            self.assertEqual(rec["z_mod"],
                             modulate_logits(rec["z_base"], rec["A_norm"],
                                             5.0, -10.0, TIERS))


# ---------------------------------------------------- §10.6 whitelist block

class TestWhitelistBlock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entries = load_trace_entries()
        cls.tasks = {t["task_id"]: t for t in load_tasks(TASKS_PATH)}
        cfg = JitRLConfig()
        i = ra.FROZEN_TASK_INDEX["T20"]  # 3 code_gen neighbors
        mem = ra.build_prefix_memory(cls.entries, i, cfg.memory_cap)
        message = cls.tasks["T20"]["message"]
        cls.matches = ra.retrieve_prefix(mem, message, cfg)
        assert len(cls.matches) > 0, "T20 must have C0-prefix neighbors"
        cls.task = cls.tasks["T20"]

    def test_block_is_deterministic_with_fixed_precision(self):
        b1 = ra.render_memory_block(self.matches)
        b2 = ra.render_memory_block(list(self.matches))
        self.assertEqual(b1, b2)
        self.assertRegex(b1, r"G=[+-]\d\.\d{4}")   # fixed signed precision
        self.assertRegex(b1, r"similarity=\d\.\d{4}")

    def test_block_whitelist_only_no_forbidden_content(self):
        block = ra.render_memory_block(self.matches)
        forbidden = [
            # placeholders (must not trigger second substitution)
            "{USER_MESSAGE}", "{CONTINUATION}", "{PT}",
            # GT / checklist / evaluator structure
            "gt_tier", "quality", "verdict", "certainty", "checklist",
            "capability", "quality_score",
            # task identity / provenance / order
            "T01", "T05", "T20", "T06", "family", "episode", "source_order",
            "source_task", "ts", "order", "task",
            # secrets / endpoints / config
            "api", "key", "http", "Bearer", "pilotdeck",
        ]
        for marker in forbidden:
            # word-boundary match: 'ts' must not fire inside 'requests' etc.
            pattern = (r"\b" + re.escape(marker) + r"\b" if marker.isascii()
                       else re.escape(marker))
            self.assertIsNone(re.search(pattern, block),
                              f"block leaks {marker!r}")
        # raw user text fragments of the current task family (CJK bigrams)
        for cjk in ("重构", "代码", "下面", "说明"):
            self.assertNotIn(cjk, block)
        # whitelist content present
        self.assertIn("neighbor_count", block)
        self.assertIn("per_tier", block)
        self.assertIn("mean_G", block)
        self.assertIn("neighbor_1: tier=", block)
        self.assertIn("similarity=", block)
        # anonymization framing (prereg §4.1)
        self.assertIn("anonymized", block.lower())
        self.assertIn("NOT as hard rules", block)
        # the block is pure ASCII (no CJK raw-text fragments can leak in)
        self.assertTrue(block.isascii())
        # and every word comes from the fixed whitelist vocabulary
        allowed = {
            "reference", "similar", "historical", "requests", "anonymized",
            "outcomes", "outcome", "the", "lines", "below", "are", "records",
            "of", "routed", "by", "this", "system", "treat", "them", "as",
            "weak", "only", "not", "hard", "rules", "final", "tier", "must",
            "follow", "routing", "above", "and", "current", "user", "message",
            "neighbor_count", "neighbor_", "per_tier", "mean_g", "n", "na", "a",
            "g", "similarity", "simple", "medium", "complex", "reasoning",
        }
        words = {w.lower() for w in re.findall(r"[A-Za-z_]+", block)}
        self.assertTrue(words <= allowed, f"unexpected words: {words - allowed}")

    def test_audit_copy_is_tier_g_similarity_only(self):
        audit = ra.audit_retrieved(self.matches)
        for m in audit:
            self.assertEqual(set(m.keys()), {"tier", "G", "similarity"})
        self.assertEqual([m["tier"] for m in audit],
                         [e["tier"] for e, _ in self.matches])


# ------------------------------------------------------- §10.5/7/8/9/10/11

class TestArmRuns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_sha_before = sha256_of(C0_MEMORY_PATH)
        cls.copy_sha_before = sha256_of(TRACE_PATH)
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.lm_summary, cls.lm_records, cls.lm_out = run_arm("T8-LM", tmp)
        cls.pm_summary, cls.pm_records, cls.pm_out = run_arm("T8-PM", tmp)
        cls.source_sha_after = sha256_of(C0_MEMORY_PATH)
        cls.copy_sha_after = sha256_of(TRACE_PATH)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_order_strictly_equals_frozen_order(self):
        # §10.11 (both arms, full formal selection)
        for records in (self.lm_records, self.pm_records):
            self.assertEqual([r["task_id"] for r in records],
                             list(ra.FROZEN_TASK_ORDER))
        self.assertTrue(self.lm_summary["order_is_full_frozen_24"])
        self.assertTrue(self.pm_summary["order_is_full_frozen_24"])

    def test_both_arms_same_prefix_and_retrieval(self):
        # §10.5: per task, identical memory prefix and identical retrieval
        pm_by_id = {r["task_id"]: r for r in self.pm_records}
        for lr in self.lm_records:
            pr = pm_by_id[lr["task_id"]]
            self.assertEqual(lr["memory_prefix_size"], pr["memory_prefix_size"])
            self.assertEqual(lr["retrieved_memories"], pr["retrieved_memories"])
            self.assertEqual(lr["n_retrieved"], pr["n_retrieved"])
            self.assertEqual(lr["top_similarity"], pr["top_similarity"])

    def test_no_neighbors_pm_prompt_unchanged_lm_unmodulated(self):
        # §10.7: T01 is first in the frozen order -> empty prefix -> no
        # neighbors. PM must send a byte-identical raw prompt for BOTH its
        # calls; LM must not modulate (z_mod == z_base).
        with tempfile.TemporaryDirectory() as tmp:
            spy_lm, spy_pm = SpyJudge(), SpyJudge()
            _, lm_records, _ = run_arm("T8-LM", tmp, task_ids=["T01"], judge=spy_lm)
            _, pm_records, _ = run_arm("T8-PM", tmp, task_ids=["T01"], judge=spy_pm)
        tasks = {t["task_id"]: t for t in load_tasks(TASKS_PATH)}
        message = tasks["T01"]["message"]
        expected_prompt = build_prompt(message, None, None)
        # LM: exactly one raw-prompt call
        self.assertEqual(spy_lm.calls, [(message, None, None)])
        # PM: two calls, both on the identical raw prompt, no block anywhere
        self.assertEqual(len(spy_pm.calls), 2)
        for msg, prev, block in spy_pm.calls:
            self.assertIsNone(block)
            self.assertEqual(build_prompt(msg, prev, block), expected_prompt)
        for r in lm_records + pm_records:
            self.assertEqual(r["n_retrieved"], 0)
            self.assertEqual(r["gate_reason"], "no_neighbors")
            self.assertEqual(r["injected_block_chars"], 0)
            self.assertIsNone(r["injected_block_sha256"])
            self.assertFalse(r["influenced"])
        for r in lm_records:
            self.assertEqual(r["z_mod"], r["z_base"])
            self.assertFalse(r["modulation_active"])
            self.assertEqual(r["chosen_tier"], r["base_choice"])
            self.assertFalse(r["flipped"])
        for r in pm_records:
            self.assertIsNone(r["injected_block_sha256"])
            self.assertEqual(r["decision_prompt_sha256"], r["base_prompt_sha256"])

    def test_pm_never_modulates_lm_never_injects(self):
        # §10.8: on retrieval-hit tasks (T06/T20 have C0-prefix neighbors)
        # PM still must not modulate and LM still must not inject.
        hits = [r for r in self.pm_records if r["n_retrieved"] > 0]
        self.assertTrue(hits)
        for r in hits:
            self.assertIsNone(r["z_mod"])
            self.assertEqual(r["A_norm"], {t: 0.0 for t in TIERS})
            self.assertEqual(r["V"], 0.0)
            self.assertFalse(r["modulation_active"])
            self.assertIsNone(r["gate_reason"])
            self.assertGreater(r["injected_block_chars"], 0)
            self.assertIsNotNone(r["injected_block_sha256"])
            self.assertNotEqual(r["decision_prompt_sha256"],
                                r["base_prompt_sha256"])
        lm_hits = [r for r in self.lm_records if r["n_retrieved"] > 0]
        self.assertEqual(len(lm_hits), len(hits))
        for r in lm_hits:
            self.assertIsNone(r["z_prompt"])
            self.assertEqual(r["injected_block_chars"], 0)
            self.assertIsNone(r["injected_block_sha256"])
            self.assertEqual(r["decision_prompt_sha256"], r["base_prompt_sha256"])
            self.assertIsNotNone(r["z_mod"])
            self.assertTrue(r["modulation_active"])
            # C0 formula parity from the recorded fields
            self.assertEqual(r["z_mod"],
                             modulate_logits(r["z_base"], r["A_norm"],
                                             5.0, -10.0, TIERS))

    def test_memory_files_unchanged_and_never_written(self):
        # §10.9 + §10.10
        self.assertEqual(self.source_sha_before, self.source_sha_after)
        self.assertEqual(self.copy_sha_before, self.copy_sha_after)
        self.assertEqual(len(load_trace_entries()), 24)
        for r in self.lm_records + self.pm_records:
            self.assertFalse(r["memory_written"])
            self.assertTrue(r["memory_frozen"])
            self.assertIsNotNone(r["memory_source_sha256"])
            self.assertEqual(r["memory_source_sha256"], self.copy_sha_before)
        self.assertEqual(self.lm_summary["memory_written"], 0)
        self.assertEqual(self.pm_summary["memory_written"], 0)

    def test_record_schema_contract_and_no_secrets(self):
        existing_contract = {
            "episode_id", "task_id", "family", "gt_tier", "intent_class",
            "signature_preview", "n_retrieved", "top_similarity",
            "min_neighbors", "modulation_active", "gate_reason", "V", "Q",
            "A_norm", "z_base", "z_mod", "chosen_tier", "route_mode",
            "base_choice", "flipped", "exec_model", "exec_response_preview",
            "exec_finish_reason", "exec_usage", "exec_cost_usd",
            "exec_latency_ms", "exec_error", "evaluator", "eval_cost_usd",
            "eval_error", "reward", "reward_breakdown", "judge_latency_ms",
            "jitrl_latency_ms", "memory_written",
        }
        additive = {
            "ablation_arm", "memory_frozen", "memory_prefix_size",
            "memory_source_sha256", "retrieved_memories", "z_prompt",
            "influenced", "injected_block_chars", "injected_block_sha256",
            "previous_tier", "judge_calls", "decision_prompt_sha256",
            "base_prompt_sha256", "base_judge_latency_ms",
            "prompt_judge_latency_ms",
        }
        for r in self.lm_records + self.pm_records:
            self.assertTrue(existing_contract.issubset(r.keys()))
            self.assertTrue(additive.issubset(r.keys()))
            self.assertEqual(r["route_mode"],
                             "jitrl-frozen" if r["ablation_arm"] == "T8-LM"
                             else "prompt-memory")
        for out in (self.lm_out, self.pm_out):
            text = out.read_text(encoding="utf-8")
            for marker in ("apiKey", "api_key", "Bearer ", "Authorization"):
                self.assertNotIn(marker, text)

    def test_latency_fields_separated(self):
        # §10 (H4): PM's two judge calls are reported in separate columns and
        # NEVER leak into jitrl_latency_ms; LM has one call and the 0.0
        # sentinel for the prompt column.
        with tempfile.TemporaryDirectory() as tmp:
            spy = SpyJudge(latency_ms=250.0)
            _, lm_records, _ = run_arm("T8-LM", tmp, task_ids=["T06"], judge=spy)
            _, pm_records, _ = run_arm("T8-PM", tmp, task_ids=["T06"], judge=spy)
        for r in lm_records:
            self.assertEqual(r["judge_calls"], 1)
            self.assertEqual(r["judge_latency_ms"], 250.0)
            self.assertEqual(r["base_judge_latency_ms"], 250.0)
            self.assertEqual(r["prompt_judge_latency_ms"], 0.0)  # sentinel
            self.assertLess(r["jitrl_latency_ms"], 100.0)
        for r in pm_records:
            self.assertEqual(r["judge_calls"], 2)
            self.assertEqual(r["judge_latency_ms"], 250.0)  # deciding call
            self.assertEqual(r["prompt_judge_latency_ms"], 250.0)
            self.assertEqual(r["base_judge_latency_ms"], 250.0)  # audit call
            self.assertLess(r["jitrl_latency_ms"], 100.0)
        self.assertEqual(len(spy.calls), 3)  # 1 LM + 2 PM

    def test_previous_tier_chain_follows_own_choices(self):
        for records in (self.lm_records, self.pm_records):
            prev = None
            for r in records:
                self.assertEqual(r["previous_tier"], prev)
                prev = r["chosen_tier"]

    def test_retrieval_gate_symmetry_between_arms(self):
        # the injection gate (PM) and the modulation gate (LM) agree task by
        # task under the frozen min_neighbors=1
        pm_by_id = {r["task_id"]: r for r in self.pm_records}
        for lr in self.lm_records:
            pr = pm_by_id[lr["task_id"]]
            self.assertEqual(lr["gate_reason"], pr["gate_reason"])
            # LM modulated  <=>  PM injected (same gate, frozen min_neighbors=1)
            self.assertEqual(lr["modulation_active"],
                             pr["injected_block_chars"] > 0)
            self.assertEqual(lr["n_retrieved"] > 0, pr["injected_block_chars"] > 0)

    def test_task_subset_preserves_frozen_order_and_unknown_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, records, _ = run_arm("T8-PM", tmp,
                                    task_ids=["T20", "T06", "T01"])
            self.assertEqual([r["task_id"] for r in records],
                             ["T01", "T06", "T20"])
            with self.assertRaises(SystemExit):
                run_arm("T8-PM", tmp, task_ids=["P01"])  # continuation probe
            with self.assertRaises(SystemExit):
                run_arm("T8-PM", tmp, task_ids=["NOPE"])

    def test_budget_guard_aborts_without_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "budget.jsonl"
            with self.assertRaises(RuntimeError):
                ra.run(arm="T8-LM", tasks_path=TASKS_PATH,
                       memory_trace_path=TRACE_PATH, judge_kind="mock",
                       out_path=out, task_ids=["T01", "T06"],
                       cpa_client=FakeCPAClient(), budget_usd=1e-9)
            self.assertFalse(out.exists())

    def test_out_path_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            # existing output: refuse (retry-suffix rule, prereg §6)
            run_arm("T8-PM", tmp, task_ids=["T01"])
            existing = Path(tmp) / "out_T8_PM.jsonl"
            self.assertTrue(existing.exists())
            with self.assertRaises(SystemExit):
                ra.run(arm="T8-PM", tasks_path=TASKS_PATH,
                       memory_trace_path=TRACE_PATH, judge_kind="mock",
                       out_path=existing, task_ids=["T01"],
                       cpa_client=FakeCPAClient())
        # frozen C0 log namespace: refused even though the file is new
        with self.assertRaises(SystemExit):
            ra.ensure_writable_out("logs/s2_full_C_never_write_this.jsonl")
        with self.assertRaises(SystemExit):
            ra.ensure_writable_out("eval/definitely-not-allowed.jsonl")
        with self.assertRaises(SystemExit):
            ra.ensure_writable_out("poster-a3/nope.jsonl")

    def test_summary_fields(self):
        for summary, arm in ((self.lm_summary, "T8-LM"),
                             (self.pm_summary, "T8-PM")):
            self.assertEqual(summary["ablation_arm"], arm)
            self.assertEqual(summary["episodes"], 1)
            self.assertEqual(summary["n_records"], 24)
            self.assertEqual(summary["memory_written"], 0)
            self.assertEqual(summary["config_snapshot"],
                             JitRLConfig().snapshot())
            self.assertEqual(summary["ablation_config"]["max_tokens_exec"], 1024)
            self.assertEqual(summary["ablation_config"]["temperature_exec"], 0.7)
            self.assertEqual(summary["ablation_config"]["max_tokens_eval"], 300)
            self.assertTrue(summary["ablation_config"]["memory_read_only"])
            self.assertEqual(summary["latency_ms"]["judge_calls_total"], 24
                             if arm == "T8-LM" else 48)
        self.assertEqual(self.lm_summary["injected_records"], 0)
        self.assertEqual(self.pm_summary["modulated_records"], 0)
        self.assertEqual(self.lm_summary["retrieval_hits"],
                         self.pm_summary["retrieval_hits"])


class TestCli(unittest.TestCase):
    """CLI paths that never touch the real CPA config/endpoint."""

    def _run_cli(self, tmp, extra):
        out = Path(tmp) / "cli.jsonl"
        argv = ["--arm", "T8-PM", "--tasks", str(TASKS_PATH),
                "--memory-trace", str(TRACE_PATH), "--judge", "mock",
                "--out", str(out)] + extra
        with mock.patch.object(ra, "build_cpa_client",
                               return_value=FakeCPAClient()):
            rc = ra.main(argv)
        return rc, out

    def test_cli_smoke_subset_and_default_sidecar_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, out = self._run_cli(tmp, ["--task-ids", "T01,T06"])
            self.assertEqual(rc, 0)
            records = [json.loads(l) for l in
                       out.read_text(encoding="utf-8").splitlines()
                       if "_summary" not in l]
            self.assertEqual([r["task_id"] for r in records], ["T01", "T06"])
            # sidecar (.sha256.json) auto-discovered next to the trace
            for r in records:
                self.assertEqual(r["memory_source_sha256"],
                                 sha256_of(TRACE_PATH))

    def test_cli_refuses_unknown_task_before_any_cpa_construction(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                self._run_cli(tmp, ["--task-ids", "NOPE"])

    def test_cli_refuses_protected_out_before_any_cpa_construction(self):
        argv = ["--arm", "T8-LM", "--tasks", str(TASKS_PATH),
                "--memory-trace", str(TRACE_PATH), "--judge", "mock",
                "--out", "logs/s2_full_never_write_this.jsonl"]
        with self.assertRaises(SystemExit):
            ra.main(argv)

    def test_cli_bad_trace_hash_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_trace = Path(tmp) / "bad-trace.jsonl"
            bad_trace.write_text("{}\n", encoding="utf-8")
            out = Path(tmp) / "cli.jsonl"
            argv = ["--arm", "T8-LM", "--tasks", str(TASKS_PATH),
                    "--memory-trace", str(bad_trace), "--judge", "mock",
                    "--out", str(out)]
            with self.assertRaises(SystemExit):
                ra.main(argv)


if __name__ == "__main__":
    unittest.main()
