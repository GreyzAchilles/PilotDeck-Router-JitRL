"""Unit tests for the CPA execution client (config, key, usage, retries, cost)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from harness.cpa_client import (
    CPAConfigError,
    CPATransportError,
    CPAExecClient,
    HTTPFailure,
    NetworkFailure,
    ProviderConfig,
    _api_model_name,
    load_cpa_provider,
    normalize_openai_usage,
    provider_id_from_env,
    resolve_api_key,
    resolve_config_path,
    urllib_transport,
)
from harness.pricing import cost_from_usage


# ------------------------------------------------------------------ fixtures
SECRET = "sk-test-secret-DO-NOT-PRINT-000123"


def make_provider(api_key: str = SECRET, url: str = "http://127.0.0.1:9999/v1",
                  protocol: str = "openai") -> ProviderConfig:
    return ProviderConfig(provider_id="CPA", protocol=protocol, url=url,
                          api_key=api_key, models={"glm-5.3": {}, "gpt-5.6-sol": {}})


def chat_response(content: str = "ok", usage: dict | None = None) -> dict:
    r = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
    if usage is not None:
        r["usage"] = usage
    return r


BASIC_USAGE = {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140}


class ScriptedTransport:
    """Replays scripted outcomes; records every call (headers included)."""

    def __init__(self, outcomes: list):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def __call__(self, url, body, headers, timeout_s):
        self.calls.append({"url": url, "body": dict(body),
                           "headers": dict(headers), "timeout": timeout_s})
        outcome = self.outcomes.pop(0) if self.outcomes else chat_response()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_client(transport: ScriptedTransport, provider: ProviderConfig | None = None):
    sleeps: list[float] = []
    client = CPAExecClient(
        provider=provider or make_provider(),
        transport=transport,
        sleep_fn=sleeps.append,
    )
    return client, sleeps


def write_yaml(tmp: str, text: str) -> Path:
    p = Path(tmp) / "pilotdeck.yaml"
    p.write_text(text, encoding="utf-8")
    return p


VALID_YAML = f"""
model:
  providers:
    CPA:
      protocol: openai
      url: http://127.0.0.1:8317/v1
      apiKey: {SECRET}
      models:
        glm-5.3:
          capabilities: {{maxOutputTokens: 32768}}
"""


# ------------------------------------------------------------- config path
class TestConfigResolution(unittest.TestCase):
    def test_explicit_env_path_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write_yaml(tmp, VALID_YAML)
            env = {"PILOTDECK_CONFIG_PATH": str(p),
                   "PILOT_HOME": "Z:/nowhere"}
            self.assertEqual(resolve_config_path(env), p)
            prov = load_cpa_provider(env=env)
            self.assertEqual(prov.url, "http://127.0.0.1:8317/v1")
            self.assertEqual(prov.api_key, SECRET)

    def test_pilot_home_next(self):
        env = {"PILOT_HOME": "D:/fakehome"}
        expected = Path("D:/fakehome") / "pilotdeck.yaml"
        self.assertEqual(resolve_config_path(env), expected)

    def test_default_home(self):
        expected = Path("~/.pilotdeck").expanduser() / "pilotdeck.yaml"
        self.assertEqual(resolve_config_path({}), expected)

    def test_missing_file_raises(self):
        with self.assertRaises(CPAConfigError):
            load_cpa_provider(config_path="Z:/no/such/file.yaml")

    def test_missing_provider_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write_yaml(tmp, "model:\n  providers:\n    other: {}\n")
            with self.assertRaises(CPAConfigError):
                load_cpa_provider(config_path=p)

    def test_provider_id_env_override(self):
        """JITRL_PROVIDER_ID selects any provider; default stays "CPA"."""
        with tempfile.TemporaryDirectory() as tmp:
            p = write_yaml(tmp, VALID_YAML.replace("    CPA:", "    myrelay:"))
            env = {"PILOTDECK_CONFIG_PATH": str(p),
                   "JITRL_PROVIDER_ID": "myrelay"}
            prov = load_cpa_provider(env=env)
            self.assertEqual(prov.provider_id, "myrelay")
            self.assertEqual(prov.api_key, SECRET)
            # explicit argument wins over the env var
            prov2 = load_cpa_provider(env=env, provider_id="myrelay")
            self.assertEqual(prov2.provider_id, "myrelay")
            # without the override the default "CPA" lookup fails here
            with self.assertRaises(CPAConfigError):
                load_cpa_provider(env={"PILOTDECK_CONFIG_PATH": str(p)})

    def test_default_provider_id_is_cpa(self):
        self.assertEqual(provider_id_from_env({}), "CPA")
        self.assertEqual(provider_id_from_env({"JITRL_PROVIDER_ID": "  "}), "CPA")
        self.assertEqual(provider_id_from_env({"JITRL_PROVIDER_ID": "x"}), "x")

    def test_model_prefix_is_provider_agnostic(self):
        self.assertEqual(_api_model_name("CPA/glm-5.3"), "glm-5.3")
        self.assertEqual(_api_model_name("myrelay/glm-5.3"), "glm-5.3")
        self.assertEqual(_api_model_name("glm-5.3"), "glm-5.3")

    def test_wrong_protocol_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write_yaml(tmp, VALID_YAML.replace("protocol: openai",
                                                   "protocol: anthropic"))
            with self.assertRaises(CPAConfigError):
                load_cpa_provider(config_path=p)

    def test_missing_url_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write_yaml(tmp, VALID_YAML.replace(
                "url: http://127.0.0.1:8317/v1\n", ""))
            with self.assertRaises(CPAConfigError):
                load_cpa_provider(config_path=p)


class TestApiKeyResolution(unittest.TestCase):
    def test_literal_trimmed(self):
        self.assertEqual(resolve_api_key("  abc123  "), "abc123")

    def test_env_reference_resolved(self):
        env = {"CPA_KEY": "  resolved-key  \n"}
        self.assertEqual(resolve_api_key("${CPA_KEY}", env), "resolved-key")

    def test_env_reference_missing(self):
        with self.assertRaises(CPAConfigError) as cm:
            resolve_api_key("${CPA_KEY_UNSET}", {})
        self.assertNotIn("resolved-key", str(cm.exception))

    def test_empty_rejected(self):
        for bad in ("", "   ", None, 123):
            with self.assertRaises(CPAConfigError):
                resolve_api_key(bad)

    def test_yaml_env_reference_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write_yaml(tmp, VALID_YAML.replace(SECRET, "${MY_CPA_KEY}"))
            prov = load_cpa_provider(config_path=p,
                                     env={"MY_CPA_KEY": SECRET})
            self.assertEqual(prov.api_key, SECRET)


# ------------------------------------------------------------ usage handling
class TestUsageNormalization(unittest.TestCase):
    def test_openai_shape_with_cache(self):
        u = normalize_openai_usage({
            "prompt_tokens": 1000, "completion_tokens": 200,
            "total_tokens": 1200,
            "prompt_tokens_details": {"cached_tokens": 300},
        })
        self.assertEqual(u["input_tokens"], 700)      # net of cache read
        self.assertEqual(u["cache_read_tokens"], 300)
        self.assertEqual(u["cache_write_tokens"], 0)
        self.assertEqual(u["output_tokens"], 200)
        self.assertEqual(u["total_tokens"], 1200)
        self.assertIsNone(u["native_cost"])

    def test_anthropic_style_names(self):
        u = normalize_openai_usage({
            "input_tokens": 500, "output_tokens": 100,
            "cache_read_input_tokens": 200,
            "cache_creation_input_tokens": 50, "cost": 0.001,
        })
        self.assertEqual(u["input_tokens"], 250)
        self.assertEqual(u["cache_read_tokens"], 200)
        self.assertEqual(u["cache_write_tokens"], 50)
        self.assertEqual(u["output_tokens"], 100)
        self.assertEqual(u["total_tokens"], 600)
        self.assertEqual(u["native_cost"], 0.001)

    def test_missing_usage_is_none(self):
        self.assertIsNone(normalize_openai_usage(None))
        self.assertIsNone(normalize_openai_usage({}))
        self.assertIsNone(normalize_openai_usage("n/a"))

    def test_input_tokens_details_variant(self):
        u = normalize_openai_usage({
            "input_tokens": 100, "output_tokens": 10,
            "input_tokens_details": {"cached_tokens": 40},
        })
        self.assertEqual(u["input_tokens"], 60)
        self.assertEqual(u["cache_read_tokens"], 40)

    def test_native_cost_fallbacks(self):
        for key in ("cost", "total_cost", "estimated_cost"):
            u = normalize_openai_usage({"prompt_tokens": 10,
                                        "completion_tokens": 5, key: 0.5})
            self.assertEqual(u["native_cost"], 0.5)


# ------------------------------------------------------------------- retries
class TestRetries(unittest.TestCase):
    def test_429_honors_retry_after_then_succeeds(self):
        tr = ScriptedTransport([
            HTTPFailure(429, detail="rate limited", retry_after=0.7),
            chat_response(usage=BASIC_USAGE),
        ])
        client, sleeps = make_client(tr)
        res = client.completion("CPA/glm-5.3", [{"role": "user", "content": "hi"}])
        self.assertEqual(res.content, "ok")
        self.assertEqual(len(tr.calls), 2)
        self.assertEqual(sleeps, [0.7])
        self.assertEqual(res.usage["input_tokens"], 100)

    def test_500_retried_with_backoff(self):
        tr = ScriptedTransport([
            HTTPFailure(500, detail="err"),
            HTTPFailure(500, detail="err"),
            chat_response(usage=BASIC_USAGE),
        ])
        client, sleeps = make_client(tr)
        res = client.completion("glm-5.3", [{"role": "user", "content": "hi"}])
        self.assertEqual(len(tr.calls), 3)
        self.assertEqual(sleeps, [1.0, 2.0])          # 1s, 2s backoff
        self.assertEqual(res.model, "glm-5.3")

    def test_network_error_retried(self):
        tr = ScriptedTransport([
            NetworkFailure("connection refused"),
            chat_response(usage=BASIC_USAGE),
        ])
        client, sleeps = make_client(tr)
        client.completion("glm-5.3", [{"role": "user", "content": "hi"}])
        self.assertEqual(len(tr.calls), 2)
        self.assertEqual(sleeps, [1.0])

    def test_400_fails_immediately(self):
        tr = ScriptedTransport([HTTPFailure(400, detail="bad request")])
        client, sleeps = make_client(tr)
        with self.assertRaises(CPATransportError) as cm:
            client.completion("glm-5.3", [{"role": "user", "content": "hi"}])
        self.assertEqual(len(tr.calls), 1)            # no retry
        self.assertEqual(sleeps, [])
        self.assertIn("HTTP 400", str(cm.exception))

    def test_retryable_exhausted_raises(self):
        tr = ScriptedTransport([HTTPFailure(503, detail="down")] * 10)
        client, sleeps = make_client(tr)
        with self.assertRaises(CPATransportError):
            client.completion("glm-5.3", [{"role": "user", "content": "hi"}])
        self.assertEqual(len(tr.calls), client.max_attempts)

    def test_missing_usage_retried_then_success(self):
        tr = ScriptedTransport([
            chat_response(content="a"),               # no usage
            chat_response(content="b"),               # no usage
            chat_response(content="c", usage=BASIC_USAGE),
        ])
        client, sleeps = make_client(tr)
        res = client.completion("glm-5.3", [{"role": "user", "content": "hi"}])
        self.assertEqual(res.content, "c")
        self.assertEqual(len(tr.calls), 3)            # 1 + 2 usage retries

    def test_missing_usage_exhausted(self):
        tr = ScriptedTransport([chat_response(content="x")] * 10)
        client, _ = make_client(tr)
        with self.assertRaises(CPATransportError) as cm:
            client.completion("glm-5.3", [{"role": "user", "content": "hi"}])
        self.assertIn("missing usage", str(cm.exception))
        self.assertEqual(len(tr.calls), 3)            # 1 + 2 retries


class TestClientSecurity(unittest.TestCase):
    def test_auth_header_sent_but_never_leaked(self):
        tr = ScriptedTransport([chat_response(usage=BASIC_USAGE)])
        client, _ = make_client(tr)
        client.completion("glm-5.3", [{"role": "user", "content": "hi"}])
        self.assertEqual(tr.calls[0]["headers"]["Authorization"],
                         f"Bearer {SECRET}")
        # url/body carry no key either
        self.assertNotIn(SECRET, tr.calls[0]["url"])
        self.assertNotIn(SECRET, json.dumps(tr.calls[0]["body"]))

    def test_error_messages_and_repr_exclude_key(self):
        tr = ScriptedTransport([HTTPFailure(401, detail="Invalid API key")])
        client, _ = make_client(tr)
        with self.assertRaises(CPATransportError) as cm:
            client.completion("glm-5.3", [{"role": "user", "content": "hi"}])
        self.assertNotIn(SECRET, str(cm.exception))
        self.assertNotIn(SECRET, repr(make_provider()))
        self.assertNotIn(SECRET, repr(client))

    def test_request_body_non_streaming_and_capped(self):
        tr = ScriptedTransport([chat_response(usage=BASIC_USAGE)])
        client, _ = make_client(tr)
        client.completion("CPA/gpt-5.6-sol",
                          [{"role": "user", "content": "eval"}],
                          max_tokens=280, temperature=0.2)
        body = tr.calls[0]["body"]
        self.assertEqual(body["model"], "gpt-5.6-sol")  # prefix stripped
        self.assertFalse(body["stream"])
        self.assertEqual(body["max_tokens"], 280)


# ---------------------------------------------------------------- cost math
class TestCostFromUsage(unittest.TestCase):
    def test_directive_formula(self):
        # glm-5.3-flash: 0.15 in / 0.03 cache / 0.50 out ($/Mtok)
        usage = {"input_tokens": 1000, "cache_read_tokens": 200,
                 "cache_write_tokens": 0, "output_tokens": 300}
        self.assertAlmostEqual(
            cost_from_usage("CPA/glm-5.3-flash", usage),
            (1000 * 0.15 + 200 * 0.03 + 300 * 0.50) / 1e6)

    def test_cache_write_billed_at_input_rate(self):
        usage = {"input_tokens": 800, "cache_read_tokens": 100,
                 "cache_write_tokens": 100, "output_tokens": 200}
        self.assertAlmostEqual(
            cost_from_usage("CPA/glm-5.3", usage),
            (900 * 1.40 + 100 * 0.26 + 200 * 4.40) / 1e6)

    def test_zero_usage(self):
        usage = {"input_tokens": 0, "cache_read_tokens": 0,
                 "cache_write_tokens": 0, "output_tokens": 0}
        self.assertEqual(cost_from_usage("CPA/gpt-5.6-sol", usage), 0.0)


if __name__ == "__main__":
    unittest.main()
