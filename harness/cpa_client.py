"""CPA execution client (Session 2 / Workstream A).

Calls the CPA model provider declared in the PilotDeck config
(`model.providers.CPA`, protocol=openai) via its OpenAI-compatible
`POST {url}/chat/completions` endpoint (non-streaming), and returns the
generated content plus normalized token usage.

Config resolution order (mirrors PilotDeck-src/src/pilot/config/loadPilotConfig.ts
+ src/pilot/paths.ts):
  1. PILOTDECK_CONFIG_PATH env var (explicit file path)
  2. PILOT_HOME env var -> {PILOT_HOME}/pilotdeck.yaml
  3. default            -> ~/.pilotdeck/pilotdeck.yaml

apiKey resolution (mirrors PilotDeck-src/src/model/config/resolveCredentials.ts):
  - value is trimmed; a value of the exact form ${ENV_VAR_NAME} resolves to
    the (trimmed) environment variable; empty/missing -> error.
  - The resolved key is used only in the Authorization header. It is never
    printed, logged, written to JSONL, or included in exception messages.

Usage normalization (mirrors PilotDeck-src/src/model/response/normalizeUsage.ts,
normalizeOpenAIUsage):
  prompt         = prompt_tokens ?? input_tokens
  output         = completion_tokens ?? output_tokens
  cache_read     = prompt_tokens_details.cached_tokens ?? input_tokens_details.cached_tokens
                   ?? cache_read_input_tokens
  cache_write    = prompt_tokens_details.cache_write_tokens ?? cache_creation_input_tokens
  net input      = prompt - cache_read - cache_write
  total          = total_tokens ?? prompt + output
  native cost    = cost ?? total_cost ?? estimated_cost   (provider-reported, may be None)

Retry policy:
  - HTTP 429 / 5xx / transport-level network errors: retried up to
    MAX_ATTEMPTS with backoff; an explicit Retry-After header (seconds)
    is honored (capped at RETRY_AFTER_CAP_S).
  - Other 4xx: fail immediately (not transient).
  - Response missing the usage block: re-requested up to
    MISSING_USAGE_RETRIES additional times; still missing -> error.

Standard library only (PyYAML is used for config parsing).
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypedDict

CONFIG_FILE_NAME = "pilotdeck.yaml"
ENV_CONFIG_PATH = "PILOTDECK_CONFIG_PATH"
ENV_PILOT_HOME = "PILOT_HOME"
DEFAULT_PILOT_HOME = "~/.pilotdeck"

PROVIDER_ID = "CPA"
SUPPORTED_PROTOCOLS = ("openai",)

MAX_ATTEMPTS = 4                 # retryable errors: 1 initial + 3 retries
BACKOFF_BASE_S = 1.0             # exponential backoff base: 1s, 2s, 4s
RETRY_AFTER_CAP_S = 30.0         # never sleep longer than this even if asked
MISSING_USAGE_RETRIES = 2        # fresh requests when usage block is absent
ERROR_MSG_MAX_CHARS = 300

_ENV_REF_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


# --------------------------------------------------------------------- errors
class CPAConfigError(RuntimeError):
    """Config file / provider entry / apiKey problem (never contains the key)."""


class CPATransportError(RuntimeError):
    """Request failed after retries (never contains the Authorization header)."""


@dataclass
class HTTPFailure(Exception):
    """Transport-level HTTP failure; retryability decided by the status code."""
    status: int
    detail: str = ""
    retry_after: float | None = None

    @property
    def retryable(self) -> bool:
        return self.status == 429 or self.status >= 500


@dataclass
class NetworkFailure(Exception):
    """Connection-level failure (DNS, refused, timeout, reset)."""
    message: str = ""

    def __str__(self) -> str:  # keep repr(e) compact in error paths
        return self.message or self.__class__.__name__


# ----------------------------------------------------------------- usage type
class UsageResult(TypedDict):
    input_tokens: int          # NET input (prompt minus cached portions)
    cache_read_tokens: int
    cache_write_tokens: int
    output_tokens: int
    total_tokens: int
    native_cost: float | None  # provider-reported cost in $ (may be absent)


@dataclass
class CompletionResult:
    content: str
    usage: UsageResult
    model: str
    finish_reason: str | None
    latency_ms: float
    native_cost: float | None = None


def normalize_openai_usage(raw: Any) -> UsageResult | None:
    """Normalize an OpenAI-compatible usage block; None when absent/empty.

    Mirrors normalizeOpenAIUsage (see module docstring). Net input excludes
    cached read/write portions, which are accounted separately.
    """
    if not isinstance(raw, dict):
        return None

    def num(v: Any) -> float | None:
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) \
            and float(v) == v and float(v) >= 0 else None

    prompt = num(raw.get("prompt_tokens"))
    if prompt is None:
        prompt = num(raw.get("input_tokens"))
    output = num(raw.get("completion_tokens"))
    if output is None:
        output = num(raw.get("output_tokens"))
    native_cost = num(raw.get("cost"))
    if native_cost is None:
        native_cost = num(raw.get("total_cost"))
    if native_cost is None:
        native_cost = num(raw.get("estimated_cost"))

    details = raw.get("prompt_tokens_details")
    if not isinstance(details, dict):
        details = raw.get("input_tokens_details")
    details = details if isinstance(details, dict) else {}
    cache_read = num(details.get("cached_tokens"))
    if cache_read is None:
        cache_read = num(raw.get("cache_read_input_tokens"))
    cache_write = num(details.get("cache_write_tokens"))
    if cache_write is None:
        cache_write = num(raw.get("cache_creation_input_tokens"))

    if prompt is None and output is None and cache_read is None \
            and cache_write is None and native_cost is None:
        return None

    prompt = prompt or 0
    output = output or 0
    cache_read = min(cache_read or 0, prompt)
    cache_write = min(cache_write or 0, prompt - cache_read)
    net_input = prompt - cache_read - cache_write
    total = num(raw.get("total_tokens"))
    if total is None:
        total = prompt + output

    return UsageResult(
        input_tokens=int(net_input),
        cache_read_tokens=int(cache_read),
        cache_write_tokens=int(cache_write),
        output_tokens=int(output),
        total_tokens=int(total),
        native_cost=float(native_cost) if native_cost is not None else None,
    )


# ------------------------------------------------------------------- config
@dataclass
class ProviderConfig:
    provider_id: str
    protocol: str
    url: str
    api_key: str                  # secret: never logged / serialized
    models: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # keep the key out of logs/debuggers
        return (f"ProviderConfig(provider_id={self.provider_id!r}, "
                f"protocol={self.protocol!r}, url={self.url!r}, "
                f"api_key=<redacted>, models={sorted(self.models)!r})")


def resolve_api_key(value: Any, env: dict[str, str] | None = None) -> str:
    """Resolve a provider apiKey (trimmed; ${ENV_VAR} supported). Never leaks it."""
    if not isinstance(value, str) or not value.strip():
        raise CPAConfigError(
            f"provider {PROVIDER_ID}: apiKey must be a non-empty string")
    env_map = os.environ if env is None else env
    trimmed = value.strip()
    m = _ENV_REF_RE.match(trimmed)
    if not m:
        return trimmed
    name = m.group(1)
    resolved = env_map.get(name)
    resolved = resolved.strip() if isinstance(resolved, str) else ""
    if not resolved:
        raise CPAConfigError(
            f"provider {PROVIDER_ID}: apiKey references environment variable "
            f"{name}, which is not set")
    return resolved


def resolve_config_path(env: dict[str, str] | None = None) -> Path:
    """PILOTDECK_CONFIG_PATH -> PILOT_HOME -> ~/.pilotdeck/pilotdeck.yaml."""
    env_map = os.environ if env is None else env
    explicit = (env_map.get(ENV_CONFIG_PATH) or "").strip()
    if explicit:
        return Path(explicit).expanduser()
    home = (env_map.get(ENV_PILOT_HOME) or "").strip()
    if home:
        return Path(home).expanduser() / CONFIG_FILE_NAME
    return Path(DEFAULT_PILOT_HOME).expanduser() / CONFIG_FILE_NAME


def load_cpa_provider(
    config_path: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> ProviderConfig:
    """Load model.providers.CPA from the PilotDeck config. Validated, key resolved."""
    import yaml  # local import: only needed for config parsing

    path = Path(config_path).expanduser() if config_path else resolve_config_path(env)
    if not path.is_file():
        raise CPAConfigError(f"PilotDeck config not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:  # noqa: BLE001 - surfaced as config error
        raise CPAConfigError(f"failed to parse {path}: {e}") from e
    if not isinstance(cfg, dict):
        raise CPAConfigError(f"{path}: config must be a mapping")

    model_cfg = cfg.get("model")
    providers = model_cfg.get("providers") if isinstance(model_cfg, dict) else None
    if not isinstance(providers, dict) or PROVIDER_ID not in providers:
        raise CPAConfigError(
            f"{path}: model.providers.{PROVIDER_ID} not found")
    prov = providers[PROVIDER_ID]
    if not isinstance(prov, dict):
        raise CPAConfigError(f"model.providers.{PROVIDER_ID} must be a mapping")

    protocol = prov.get("protocol")
    if protocol not in SUPPORTED_PROTOCOLS:
        raise CPAConfigError(
            f"model.providers.{PROVIDER_ID}: unsupported protocol {protocol!r} "
            f"(supported: {', '.join(SUPPORTED_PROTOCOLS)})")
    url = prov.get("url")
    if not isinstance(url, str) or not url.strip():
        raise CPAConfigError(
            f"model.providers.{PROVIDER_ID}: url must be a non-empty string")
    api_key = resolve_api_key(prov.get("apiKey"), env)
    models = prov.get("models")
    models = models if isinstance(models, dict) else {}
    return ProviderConfig(
        provider_id=PROVIDER_ID, protocol=protocol,
        url=url.strip().rstrip("/"), api_key=api_key, models=models,
    )


# ------------------------------------------------------------------ transport
#: (url, body, headers, timeout_s) -> parsed JSON dict. Raises
#: HTTPFailure / NetworkFailure. Injectable for tests.
TransportFn = Callable[[str, dict, dict, float], dict]


def urllib_transport(url: str, body: dict, headers: dict, timeout_s: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", "replace")
        except Exception:
            detail = ""
        retry_after: float | None = None
        ra = e.headers.get("Retry-After") if e.headers else None
        if ra:
            try:
                retry_after = float(ra)
            except ValueError:
                retry_after = None
        raise HTTPFailure(e.code, detail=detail, retry_after=retry_after) from None
    except Exception as e:  # noqa: BLE001 - any socket-level failure
        raise NetworkFailure(repr(e)) from None


def _sleep(monotonic: Callable[[float], None], seconds: float) -> None:
    if seconds > 0:
        monotonic(seconds)


# -------------------------------------------------------------------- client
class CPAExecClient:
    """OpenAI-protocol chat-completions client for the CPA provider.

    The API key lives only in the Authorization header of outgoing requests;
    it never appears in results, logs, or exceptions.
    """

    def __init__(
        self,
        provider: ProviderConfig | None = None,
        *,
        config_path: str | Path | None = None,
        transport: TransportFn = urllib_transport,
        timeout_s: float = 120.0,
        max_attempts: int = MAX_ATTEMPTS,
        sleep_fn: Callable[[float], None] = time.sleep,
        env: dict[str, str] | None = None,
    ):
        self._provider = provider or load_cpa_provider(config_path, env=env)
        self._chat_url = self._provider.url + "/chat/completions"
        self._transport = transport
        self.timeout_s = timeout_s
        self.max_attempts = max(1, int(max_attempts))
        self._sleep_fn = sleep_fn

    # ------------------------------------------------------------------ API
    @property
    def provider_id(self) -> str:
        return self._provider.provider_id

    @property
    def url(self) -> str:
        return self._provider.url

    def has_model(self, model: str) -> bool:
        return _api_model_name(model) in self._provider.models

    def completion(
        self,
        model: str,
        messages: list[dict],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        extra_body: dict | None = None,
    ) -> CompletionResult:
        """Non-streaming chat completion; returns content + normalized usage."""
        api_model = _api_model_name(model)
        body: dict[str, Any] = {
            "model": api_model,
            "messages": messages,
            "stream": False,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        if extra_body:
            body.update(extra_body)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._provider.api_key}",
        }

        t0 = time.perf_counter()
        usage_missing = 0
        attempt = 0
        last_err: Exception | None = None
        while True:
            attempt += 1
            try:
                resp = self._transport(self._chat_url, body, headers, self.timeout_s)
            except HTTPFailure as e:
                last_err = e
                if e.retryable and attempt < self.max_attempts:
                    delay = e.retry_after if e.retry_after is not None \
                        else BACKOFF_BASE_S * (2 ** (attempt - 1))
                    _sleep(self._sleep_fn, min(delay, RETRY_AFTER_CAP_S))
                    continue
                raise CPATransportError(
                    f"CPA request failed (HTTP {e.status}) after {attempt} "
                    f"attempt(s): {_clip(e.detail or 'no body')}") from None
            except NetworkFailure as e:
                last_err = e
                if attempt < self.max_attempts:
                    _sleep(self._sleep_fn, BACKOFF_BASE_S * (2 ** (attempt - 1)))
                    continue
                raise CPATransportError(
                    f"CPA request failed (network) after {attempt} "
                    f"attempt(s): {_clip(str(e))}") from None

            try:
                content = resp["choices"][0]["message"]["content"]
                content = content if isinstance(content, str) else json.dumps(
                    content, ensure_ascii=False)
            except Exception as e:  # noqa: BLE001 - malformed provider payload
                last_err = e
                if attempt < self.max_attempts:
                    _sleep(self._sleep_fn, BACKOFF_BASE_S * (2 ** (attempt - 1)))
                    continue
                raise CPATransportError(
                    f"CPA response malformed after {attempt} attempt(s): "
                    f"{_clip(repr(e))}") from None
            finish = None
            try:
                finish = resp["choices"][0].get("finish_reason")
            except Exception:  # noqa: BLE001
                finish = None

            usage = normalize_openai_usage(resp.get("usage"))
            if usage is None:
                last_err = CPATransportError("CPA response missing usage block")
                if usage_missing < MISSING_USAGE_RETRIES:
                    usage_missing += 1
                    _sleep(self._sleep_fn, BACKOFF_BASE_S)
                    continue
                raise CPATransportError(
                    "CPA response missing usage block after "
                    f"{MISSING_USAGE_RETRIES + 1} attempt(s)")

            return CompletionResult(
                content=content,
                usage=usage,
                model=api_model,
                finish_reason=finish,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                native_cost=usage.get("native_cost"),
            )


def _api_model_name(model: str) -> str:
    """CPA/glm-5.3 -> glm-5.3 (provider model catalog is unprefixed)."""
    return model.split("/", 1)[1] if "/" in model else model


def _clip(s: str) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= ERROR_MSG_MAX_CHARS else s[:ERROR_MSG_MAX_CHARS] + "…"
