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


# ------------------------------------------- M2 trajectory-level evaluator
#: Blindness boundary (MEMORY-LOOP-DECISIONS.md v1.1): the trajectory
#: evaluator sees the routed TIER of each turn (required to judge
#: under/over-routing) but never the executing MODEL names, prices, or
#: usage — the D8 model-blindness principle is preserved.
ROUTING_VERDICTS = ("appropriate", "under_routed", "over_routed")
TRAJECTORY_TURN_PREVIEW_CHARS = 800
TRAJECTORY_DEFAULT_MAX_TOKENS = 900

TRAJECTORY_PROMPT_TMPL = """你是一个严格、盲评的AI路由复盘评审。你看到的是一段多轮对话的完整轨迹：每一轮都由一个路由器选择了执行档位（simple/medium/complex/reasoning，由低到高），然后由该档位的模型（身份未知，也不需要知道）生成了回复。你的任务是在轨迹结束后一次性复盘：
1) 逐步评价每个路由决策是否恰当（under_routed=档位不足导致回复失败，over_routed=档位过高造成浪费，appropriate=恰当）；
2) 给出每轮回复的本地质量分（只看该轮回复本身对当轮用户消息的满足程度）；
3) 给出更合理的推荐档位；
4) 标注失败标签并用一句话给出反馈。

## 会话初始目标
\"\"\"
{GOAL}
\"\"\"

## 目标评分要点
{CHECKLIST}

## 完整轨迹（按轮次）
{TURNS}

## 评分规则
- routing_verdict：appropriate | under_routed | over_routed（针对该轮用户消息与所需能力，对照所选档位）。
- local_quality：1-5 整数。5=该轮回复完全满足当轮消息；3=基本可用但不完整；1=错误或无意义。
- recommended_tier：simple | medium | complex | reasoning（该轮更合理的档位）。
- failure_tags：从 ["capability_mismatch","late_escalation","over_provisioning","context_loss","instruction_violation","factual_error","other"] 中选择，可为空。
- feedback：一句话，说明该轮路由决策的问题或正确性。
- certainty：你对本轮判断的确信程度，0.0-1.0。
- episode：success=整段会话最终是否达成初始目标；quality_score=1-5 整数（终局交付质量）；summary=一句话总体评价。

只输出一个 JSON 对象，不要输出任何其他文字：
{{"episode": {{"success": <true|false>, "quality_score": <1-5>, "summary": "..."}}, "steps": [{{"turn_index": <0-based>, "routing_verdict": "appropriate|under_routed|over_routed", "local_quality": <1-5>, "recommended_tier": "simple|medium|complex|reasoning", "failure_tags": ["..."], "feedback": "...", "certainty": <0.0-1.0>}}]}}"""


@dataclass
class StepVerdict:
    turn_index: int
    routing_verdict: str
    local_quality: int
    recommended_tier: str | None
    failure_tags: list[str] = field(default_factory=list)
    feedback: str = ""
    certainty: float = 0.5

    def to_record(self) -> dict:
        return {
            "turn_index": self.turn_index,
            "routing_verdict": self.routing_verdict,
            "quality_score": self.local_quality,
            "local_quality": self.local_quality,
            "recommended_tier": self.recommended_tier,
            "failure_tags": list(self.failure_tags),
            "feedback": self.feedback,
            "certainty": round(self.certainty, 3),
        }


@dataclass
class TrajectoryLevelResult:
    episode_success: bool
    episode_quality: int
    episode_summary: str
    steps: list[StepVerdict] = field(default_factory=list)
    raw_text: str = ""
    latency_ms: float = 0.0
    usage: dict | None = None
    cost_usd: float | None = None
    model: str = ""


def _as_verdict(v: Any) -> str | None:
    if isinstance(v, str) and v.strip().lower() in ROUTING_VERDICTS:
        return v.strip().lower()
    return None


def _as_tier(v: Any) -> str | None:
    if isinstance(v, str) and v.strip().lower() in ("simple", "medium",
                                                    "complex", "reasoning"):
        return v.strip().lower()
    return None


def validate_trajectory_evaluation(obj: dict,
                                   n_turns: int) -> TrajectoryLevelResult | None:
    """Validate/normalize a trajectory review object; None when unusable.
    steps must cover every turn exactly (length match, order by list
    position); turn_index, when present, must match the position."""
    episode = obj.get("episode")
    if not isinstance(episode, dict):
        return None
    success = _as_bool(episode.get("success"))
    if success is None:
        return None
    quality = _as_int(episode.get("quality_score"))
    if quality is None or not (1 <= quality <= 5):
        return None
    summary = episode.get("summary")
    summary = summary if isinstance(summary, str) else ""
    raw_steps = obj.get("steps")
    if not isinstance(raw_steps, list) or len(raw_steps) != n_turns:
        return None
    steps: list[StepVerdict] = []
    for i, s in enumerate(raw_steps):
        if not isinstance(s, dict):
            return None
        idx = _as_int(s.get("turn_index"))
        if idx is not None and idx != i:
            return None
        verdict = _as_verdict(s.get("routing_verdict"))
        if verdict is None:
            return None
        lq = _as_int(s.get("local_quality"))
        if lq is None or not (1 <= lq <= 5):
            return None
        certainty = _as_float(s.get("certainty"))
        if certainty is None:
            certainty = 0.5
        certainty = min(max(certainty, 0.0), 1.0)
        steps.append(StepVerdict(
            turn_index=i,
            routing_verdict=verdict,
            local_quality=lq,
            recommended_tier=_as_tier(s.get("recommended_tier")),
            failure_tags=_as_str_list(s.get("failure_tags")),
            feedback=(s.get("feedback") if isinstance(s.get("feedback"), str)
                      else ""),
            certainty=certainty,
        ))
    return TrajectoryLevelResult(
        episode_success=success,
        episode_quality=quality,
        episode_summary=summary,
        steps=steps,
    )


class TrajectoryLevelEvaluator:
    """One blind trajectory review per trajectory (M2): episode verdict +
    per-turn routing verdicts. Sees tiers, never model identities."""

    def __init__(
        self,
        client: CPAExecClient | None = None,
        *,
        model: str = EVALUATOR_MODEL,
        max_tokens: int = TRAJECTORY_DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        price_of_call: Any = None,
    ):
        self._client = client
        self.model = model
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self._price_of_call = price_of_call

    def bind_client(self, client: CPAExecClient) -> None:
        self._client = client

    def build_prompt(self, goal_task: dict,
                     turns: list[dict]) -> str:
        checklist = goal_task.get("quality_checklist")
        if isinstance(checklist, list) and checklist:
            cl = "\n".join(f"- {c}" for c in checklist)
        elif isinstance(checklist, str) and checklist.strip():
            cl = f"- {checklist}"
        else:
            cl = "-（未提供目标评分要点，按初始消息自行判断）"
        rendered = []
        for i, t in enumerate(turns):
            resp = t.get("response") or ""
            if len(resp) > TRAJECTORY_TURN_PREVIEW_CHARS:
                resp = resp[:TRAJECTORY_TURN_PREVIEW_CHARS] + "…（截断）"
            rendered.append(
                f"### 第 {i} 轮（turn_index={i}）\n"
                f"用户：{t.get('message', '')}\n"
                f"路由档位：{t.get('tier', '')}\n"
                f"回复：{resp}")
        return (TRAJECTORY_PROMPT_TMPL
                .replace("{GOAL}", str(goal_task.get("message", "")))
                .replace("{CHECKLIST}", cl)
                .replace("{TURNS}", "\n\n".join(rendered)))

    def evaluate_trajectory(self, goal_task: dict,
                            turns: list[dict]) -> TrajectoryLevelResult | None:
        """Review one full trajectory. Returns None on ANY failure —
        callers must not write memory in that case."""
        if self._client is None:
            raise RuntimeError("TrajectoryLevelEvaluator: no CPA client bound")
        if not turns:
            return None
        prompt = self.build_prompt(goal_task, turns)
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
        result = validate_trajectory_evaluation(obj, n_turns=len(turns))
        if result is None:
            return None
        result.raw_text = comp.content
        result.latency_ms = latency
        result.usage = dict(comp.usage)
        result.model = comp.model
        if self._price_of_call is not None:
            try:
                result.cost_usd = float(self._price_of_call(self.model, comp.usage))
            except Exception:  # noqa: BLE001 - best-effort cost accounting
                result.cost_usd = None
        return result
