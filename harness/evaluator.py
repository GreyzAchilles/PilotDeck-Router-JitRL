"""Blind quality evaluator via CPA/gpt-5.6-sol (Session 2 / Workstream A).

Evaluates an (assistant response, user task) pair WITHOUT knowing which
execution model produced the response, and returns a structured verdict:

    quality_score           int 1-5 (5 = fully satisfies the task)
    verdict                 "useful" | "harmful" | "neutral"
    certainty               float 0.0-1.0
    capability_insufficient bool  (response failed because the executing
                                    model lacked the required capability)
    checklist               {hits: [...], misses: [...], notes: str}
                            judged against the task's quality_checklist

JSON extraction is robust: code fences stripped, first balanced {...}
object parsed (the prompt demands JSON-only output, but models sometimes
add prose). Any validation failure -> evaluate() returns None; callers
MUST NOT write C memory for that episode (directive).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from harness.cpa_client import CPAExecClient, CPATransportError

EVALUATOR_MODEL = "CPA/gpt-5.6-sol"          # D8 default
FALLBACK_MODEL = "CPA/OpenBMB-5.3"           # D8 alternative
VERDICTS = ("useful", "harmful", "neutral")
DEFAULT_MAX_TOKENS = 300
DEFAULT_TEMPERATURE = 0.2                    # stability for calibration

PROMPT_TMPL = """你是一个严格、盲评的AI助手回复质量评审（你不知道也不需要知道是哪个模型生成的回复）。

## 用户任务
\"\"\"
{TASK}
\"\"\"

## 待评审回复
\"\"\"
{RESPONSE}
\"\"\"

## 评分要点（quality_checklist）
{CHECKLIST}

## 评分规则
- quality_score：1-5 整数。5=完全满足任务且质量出色；4=满足任务，有小瑕疵；3=基本可用但不完整；2=明显不足或跑题；1=错误、有害或无意义。
- verdict：useful（对完成任务有实际帮助）/ harmful（错误、误导或有损用户利益）/ neutral（既无帮助也无损害）。
- certainty：你对本次评分的确信程度，0.0-1.0。
- capability_insufficient：回复失败是否主要因为执行模型能力不足（如无法推理、无法遵循指令、无法完成该复杂度的任务），而不是任务本身不合理。
- checklist：逐条核对上面的评分要点，hits=满足的条目，misses=未满足的条目（保留原文），notes=一句话补充说明。

只输出一个 JSON 对象，不要输出任何其他文字：
{{"quality_score": <1-5>, "verdict": "useful|harmful|neutral", "certainty": <0.0-1.0>, "capability_insufficient": <true|false>, "checklist": {{"hits": ["..."], "misses": ["..."], "notes": "..."}}}}"""


@dataclass
class EvaluationResult:
    quality_score: int
    verdict: str
    certainty: float
    capability_insufficient: bool
    checklist_hits: list[str] = field(default_factory=list)
    checklist_misses: list[str] = field(default_factory=list)
    checklist_notes: str = ""
    raw_text: str = ""
    latency_ms: float = 0.0
    usage: dict | None = None
    cost_usd: float | None = None
    model: str = ""

    def to_record(self) -> dict:
        return {
            "quality_score": self.quality_score,
            "verdict": self.verdict,
            "certainty": round(self.certainty, 3),
            "capability_insufficient": self.capability_insufficient,
            "checklist_hits": self.checklist_hits,
            "checklist_misses": self.checklist_misses,
            "checklist_notes": self.checklist_notes,
            "evaluator_model": self.model,
            "evaluator_latency_ms": round(self.latency_ms, 1),
        }


# ------------------------------------------------------------ JSON extraction
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict | None:
    """Extract the first valid JSON object from model output.

    Handles: bare JSON, ```json fences, prose around the object, trailing
    commas are NOT tolerated (strict JSON). Returns None when no valid
    object is found.
    """
    if not text or not isinstance(text, str):
        return None
    candidates = [text]
    for m in _FENCE_RE.finditer(text):
        candidates.append(m.group(1))
    for cand in candidates:
        cand = cand.strip()
        if cand.startswith("{"):
            try:
                obj = json.loads(cand)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
        obj = _first_balanced_object(cand)
        if obj is not None:
            return obj
    return None


def _first_balanced_object(text: str) -> dict | None:
    """Scan for the first balanced {...} block and try to parse it."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break  # malformed block; resume after this opener
                    break
        start = text.find("{", start + 1)
    return None


# ---------------------------------------------------------------- validation
def _as_int(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and float(v).is_integer():
        return int(v)
    if isinstance(v, str) and v.strip().isdigit():
        return int(v.strip())
    return None


def _as_float(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return None
    return None


def _as_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        low = v.strip().lower()
        if low in ("true", "yes"):
            return True
        if low in ("false", "no"):
            return False
    return None


def _as_str_list(v: Any) -> list[str]:
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []


def validate_evaluation(obj: dict) -> EvaluationResult | None:
    """Validate/normalize a parsed evaluation object; None when unusable."""
    score = _as_int(obj.get("quality_score"))
    if score is None or not (1 <= score <= 5):
        return None
    verdict = obj.get("verdict")
    if isinstance(verdict, str) and verdict.strip().lower() in VERDICTS:
        verdict = verdict.strip().lower()
    else:
        return None
    certainty = _as_float(obj.get("certainty"))
    if certainty is None or not (0.0 <= certainty <= 1.0):
        certainty = min(max(certainty if certainty is not None else 0.5, 0.0), 1.0)
    cap = _as_bool(obj.get("capability_insufficient"))
    if cap is None:
        cap = False
    checklist = obj.get("checklist")
    hits: list[str] = []
    misses: list[str] = []
    notes = ""
    if isinstance(checklist, dict):
        hits = _as_str_list(checklist.get("hits"))
        misses = _as_str_list(checklist.get("misses"))
        n = checklist.get("notes")
        notes = n if isinstance(n, str) else ""
    return EvaluationResult(
        quality_score=score,
        verdict=verdict,
        certainty=certainty,
        capability_insufficient=cap,
        checklist_hits=hits,
        checklist_misses=misses,
        checklist_notes=notes,
    )


# ----------------------------------------------------------------- evaluator
class Evaluator:
    """Blind evaluator on CPA/gpt-5.6-sol (single call per episode end)."""

    def __init__(
        self,
        client: CPAExecClient | None = None,
        *,
        model: str = EVALUATOR_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        price_of_call: Any = None,   # callable(model, usage) -> $ (optional)
    ):
        self._client = client
        self.model = model
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self._price_of_call = price_of_call

    def bind_client(self, client: CPAExecClient) -> None:
        self._client = client

    def build_prompt(self, task: dict, response: str) -> str:
        checklist = task.get("quality_checklist")
        if isinstance(checklist, list) and checklist:
            cl = "\n".join(f"- {c}" for c in checklist)
        elif isinstance(checklist, str) and checklist.strip():
            cl = f"- {checklist}"
        else:
            cl = "-（任务未提供评分要点，按任务描述自行判断是否完成）"
        return (PROMPT_TMPL
                .replace("{TASK}", str(task.get("message", "")))
                .replace("{RESPONSE}", response or "")
                .replace("{CHECKLIST}", cl))

    def evaluate(self, task: dict, response: str) -> EvaluationResult | None:
        """Blind-evaluate one response. Returns None on ANY failure
        (transport error, unparseable or invalid JSON) — callers must not
        write C memory in that case."""
        if self._client is None:
            raise RuntimeError("Evaluator: no CPA client bound")
        prompt = self.build_prompt(task, response)
        t0 = time.perf_counter()
        try:
            comp = self._client.completion(
                self.model,
                [{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except CPATransportError:
            return None
        latency = (time.perf_counter() - t0) * 1000.0
        obj = extract_json(comp.content)
        if obj is None:
            return None
        result = validate_evaluation(obj)
        if result is None:
            return None
        result.raw_text = comp.content
        result.latency_ms = latency
        result.usage = dict(comp.usage)
        result.model = comp.model
        if self._price_of_call is not None:
            try:
                result.cost_usd = float(self._price_of_call(self.model, comp.usage))
            except Exception:  # noqa: BLE001 - cost accounting is best-effort
                result.cost_usd = None
        return result
