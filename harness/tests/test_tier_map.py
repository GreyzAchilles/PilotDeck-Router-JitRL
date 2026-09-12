"""S4-M2 amendment v1.1: switched cross-provider tier map, pricing entries,
and ExecClientRouter routing."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harness import run_real
from harness.pricing import (
    DEFAULT_MODEL,
    SWITCHED_TIER_TO_MODEL,
    TIER_TO_MODEL,
    cost_from_usage,
    get_tier_to_model,
)
from harness.cpa_client import CompletionResult


def usage(n_in=1000, n_out=200):
    return {"input_tokens": n_in, "cache_read_tokens": 0,
            "cache_write_tokens": 0, "output_tokens": n_out,
            "total_tokens": n_in + n_out, "native_cost": None}


class TestTierMaps(unittest.TestCase):
    def test_spec_map_is_the_frozen_default(self):
        self.assertIs(get_tier_to_model(), TIER_TO_MODEL)
        self.assertIs(get_tier_to_model("spec"), TIER_TO_MODEL)

    def test_switched_map_is_cross_provider_and_complete(self):
        self.assertEqual(set(SWITCHED_TIER_TO_MODEL), set(TIER_TO_MODEL))
        providers = {m.split("/")[0] for m in SWITCHED_TIER_TO_MODEL.values()}
        self.assertEqual(providers, {"CPA", "provider1"})
        self.assertEqual(SWITCHED_TIER_TO_MODEL["complex"],
                         TIER_TO_MODEL["complex"])   # OpenBMB stays on CPA
        self.assertNotEqual(SWITCHED_TIER_TO_MODEL["simple"],
                            TIER_TO_MODEL["simple"])
        self.assertNotEqual(SWITCHED_TIER_TO_MODEL["reasoning"],
                            TIER_TO_MODEL["reasoning"])

    def test_unknown_map_rejected(self):
        with self.assertRaises(ValueError):
            get_tier_to_model("bogus")

    def test_all_models_in_both_maps_have_pricing(self):
        for m in set(TIER_TO_MODEL.values()) | set(SWITCHED_TIER_TO_MODEL.values()) \
                | {DEFAULT_MODEL}:
            cost = cost_from_usage(m, usage())   # raises KeyError if missing
            self.assertIsInstance(cost, float)
            self.assertGreaterEqual(cost, 0.0)

    def test_switched_reasoning_priced_as_default_gives_zero_saving(self):
        # provider1/glm-5.3 carries the glm-5.3 price class: cost_saving of
        # the reasoning tier against the glm-5.3 default is exactly 0,
        # same structural property as the spec map.
        actual = cost_from_usage("provider1/glm-5.3", usage())
        default = cost_from_usage(DEFAULT_MODEL, usage())
        self.assertAlmostEqual(actual, default, places=9)


class TestExecClientRouter(unittest.TestCase):
    def test_universal_base_serves_everything(self):
        # test fakes without provider_id: never routed away
        base = object()
        router = run_real.ExecClientRouter(base)
        self.assertIs(router.for_model("CPA/glm-5.3"), base)
        self.assertIs(router.for_model("provider1/glm-5.3"), base)
        self.assertIs(router.for_model("no-prefix-model"), base)

    def test_prefix_routing_to_injected_extra(self):
        class FakeBase:
            provider_id = "CPA"
        class FakeProv1:
            provider_id = "provider1"
        base, prov1 = FakeBase(), FakeProv1()
        router = run_real.ExecClientRouter(base, extra={"provider1": prov1})
        self.assertIs(router.for_model("CPA/tokendance-v4.1-flash"), base)
        self.assertIs(router.for_model("provider1/deepseek-v4-flash-vision-exp"),
                      prov1)
        # unprefixed and same-prefix stay on base
        self.assertIs(router.for_model("bare-model"), base)

    def test_lazy_build_uses_build_hook(self):
        class FakeBase:
            provider_id = "CPA"
        built = []

        def fake_build(pid, config_path):
            built.append((pid, config_path))
            return object()

        orig = run_real._build_client_for
        run_real._build_client_for = fake_build
        try:
            router = run_real.ExecClientRouter(FakeBase(), config_path="cfg.yaml")
            c1 = router.for_model("provider1/glm-5.3")
            c2 = router.for_model("provider1/deepseek-v4-flash-vision-exp")
        finally:
            run_real._build_client_for = orig
        self.assertEqual(built, [("provider1", "cfg.yaml")])  # built once
        self.assertIs(c1, c2)


# ------------------------------------------------------- integration via run()
MT_TASKS = [
    {"id": "MT1", "traj_id": "MTR1", "turn_index": 0, "family": "code_gen",
     "gt_tier": "simple", "message": "写一个两数相加的函数",
     "quality_checklist": ["正确实现"]},
    {"id": "MT2", "traj_id": "MTR1", "turn_index": 1, "family": "code_gen",
     "gt_tier": "reasoning", "message": "审查它的边界场景并修复问题",
     "quality_checklist": ["识别边界"]},
]

TRAJ_JSON = json.dumps({
    "episode": {"success": True, "quality_score": 4, "summary": "完成"},
    "steps": [
        {"turn_index": 0, "routing_verdict": "appropriate", "local_quality": 5,
         "recommended_tier": "simple", "failure_tags": [],
         "feedback": "ok", "certainty": 0.9},
        {"turn_index": 1, "routing_verdict": "appropriate", "local_quality": 4,
         "recommended_tier": "reasoning", "failure_tags": [],
         "feedback": "ok", "certainty": 0.9},
    ],
}, ensure_ascii=False)


class TaggedFakeCPA:
    """Fake with a provider_id; records calls; serves any model."""

    def __init__(self, provider_id):
        self.provider_id = provider_id
        self.calls: list[tuple] = []

    def completion(self, model, messages, *, max_tokens=512,
                   temperature=0.7, extra_body=None):
        self.calls.append(model)
        api = model.split("/")[-1]
        if api == "gpt-5.6-sol":
            return CompletionResult(content=TRAJ_JSON, usage=usage(1500, 300),
                                    model=api, finish_reason="stop",
                                    latency_ms=5.0)
        return CompletionResult(content=f"回复 by {self.provider_id}",
                                usage=usage(), model=api,
                                finish_reason="stop", latency_ms=5.0)


class TestSwitchedMapRun(unittest.TestCase):
    def test_exec_routed_by_provider_and_evaluator_on_base(self):
        base = TaggedFakeCPA("CPA")
        prov1 = TaggedFakeCPA("provider1")
        with tempfile.TemporaryDirectory() as tmp:
            tasks = Path(tmp) / "t.jsonl"
            with open(tasks, "w", encoding="utf-8") as f:
                for t in MT_TASKS:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
            out = Path(tmp) / "out.jsonl"
            summary = run_real.run(
                mode="C", tasks_path=tasks, judge_kind="mock",
                out_path=out, memory_out_path=Path(tmp) / "m.jsonl",
                episodes=1, cpa_client=base, arm="T2",
                tier_map="switched",
                exec_clients={"provider1": prov1},
            )
            records = [json.loads(l) for l in
                       out.read_text(encoding="utf-8").splitlines()
                       if "_summary" not in l]
        # mock judge routes both turns deterministically; exec calls were
        # routed by the switched map's provider prefix
        exec_models = sorted({m for m in base.calls + prov1.calls
                              if "gpt-5.6-sol" not in m})
        self.assertTrue(exec_models)
        for m in exec_models:
            pid = m.split("/")[0]
            fake = prov1 if pid == "provider1" else base
            self.assertIn(m, fake.calls)
        # records carry the full switched-map model ids
        for r in records:
            self.assertIn(r["exec_model"],
                          set(SWITCHED_TIER_TO_MODEL.values()))
        # evaluator (terminal + trajectory-level) ran on the BASE client only
        self.assertIn("CPA/gpt-5.6-sol", base.calls)
        self.assertNotIn("provider1/gpt-5.6-sol", prov1.calls)
        # trajectory succeeded end-to-end (rewards present)
        self.assertTrue(all(r["reward"] is not None for r in records))

    def test_spec_map_run_unchanged_by_router(self):
        # default tier map + universal fake: everything through the base
        base = TaggedFakeCPA.__new__(TaggedFakeCPA)   # no provider_id
        base.calls = []
        base.completion = TaggedFakeCPA.completion.__get__(base)
        with tempfile.TemporaryDirectory() as tmp:
            tasks = Path(tmp) / "t.jsonl"
            with open(tasks, "w", encoding="utf-8") as f:
                f.write(json.dumps(MT_TASKS[0], ensure_ascii=False) + "\n")
            summary = run_real.run(
                mode="C", tasks_path=tasks, judge_kind="mock",
                out_path=Path(tmp) / "o.jsonl",
                memory_out_path=Path(tmp) / "m.jsonl",
                episodes=1, cpa_client=base, arm="T1",
            )
        self.assertTrue(base.calls)   # exec + eval all on the base fake


if __name__ == "__main__":
    unittest.main()
