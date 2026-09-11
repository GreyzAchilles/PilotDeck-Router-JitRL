"""State abstraction: intent classification + task signature (spec section 2).

- intent_class(message): keyword rules over an ORDERED config dict,
  specificity descending, first match wins; fallback "other".
- task_signature(message): normalization pipeline
    1. fenced code blocks -> <code>; 4+ space indented blocks -> <code>
    2. URL -> <url>; file path -> <path>; number -> <num>;
       quoted segment longer than 8 chars -> <str>
    3. lowercase; strip punctuation but keep placeholder angle brackets
    4. tokenize: latin/digit runs are tokens; CJK runs -> char bigrams
  (Deviation note: URL is replaced BEFORE the path regex, otherwise every
  URL would be consumed by `[\w./\\-]+\.\w{1,4}` and <url> unreachable.)
- Context (turn number, previous_tier) lives in the harness layer and is
  deliberately NOT part of the retrieval key.
"""
from __future__ import annotations

import re

# Ordered dict: specificity descending, first match wins (spec examples verbatim,
# plus English equivalents).
INTENT_RULES: dict[str, tuple[str, ...]] = {
    "code_gen": (
        "写测试", "单元测试", "实现一个", "写一个函数",
        "write tests", "unit test", "write a function", "implement", "test case",
    ),
    "data_analysis": (
        "分析", "数据", "统计", "对比",
        "analy", "statistic", "compare",
    ),
    "refactor": (
        "重构", "优化这段", "改写",
        "refactor", "optimize this", "rewrite",
    ),
    "chat_qa": (
        "你好", "谢谢", "记住", "嗨",
        "hello", "thanks", "hi", "remember",
    ),
    "continuation": (
        "继续", "好的", "嗯", "可以", "接着",
        "continue", "ok", "go on",
    ),
    "info_retrieval": (
        "检索", "查找", "综合",
        "retriev", "find", "search", "look up",
    ),
    "doc_writing": (
        "写一封", "文档", "报告", "通知",
        "doc", "report", "letter", "notice", "memo",
    ),
}

INTENT_CLASSES: tuple[str, ...] = tuple(INTENT_RULES) + ("other",)

_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_PATH_RE = re.compile(r"[\w./\\-]+\.\w{1,4}")
_QUOTED_RE = re.compile(r"\"([^\"]{9,})\"|“([^”]{9,})”")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_TOKEN_RE = re.compile(r"<[a-z]+>|[a-z0-9_]+|[\u4e00-\u9fff]+")

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def intent_class(message: str) -> str:
    """Bucket a message via ordered keyword rules (first match wins)."""
    low = message.lower()
    for bucket, keywords in INTENT_RULES.items():
        for kw in keywords:
            if kw in low:
                return bucket
    return "other"


def _replace_indented_code(text: str) -> str:
    """Collapse blocks where >=1 line starts with 4+ spaces (or a tab) into <code>."""
    lines = text.split("\n")
    out: list[str] = []
    in_block = False
    block_opened = False
    for line in lines:
        if re.match(r"^( {4}|\t)", line):
            if not in_block:
                out.append("<code>")
                in_block = True
                block_opened = True
            continue
        if in_block and line.strip() == "":
            # blank line may belong to the block; keep absorbing
            continue
        in_block = False
        out.append(line)
    if not block_opened:
        return text
    return "\n".join(out)


def task_signature(message: str) -> list[str]:
    """Normalized signature tokens (document order; use set() for Jaccard)."""
    text = _CODE_FENCE_RE.sub(" <code> ", message)
    text = _replace_indented_code(text)
    text = _URL_RE.sub(" <url> ", text)
    text = _PATH_RE.sub(" <path> ", text)
    text = _NUM_RE.sub(" <num> ", text)
    text = _QUOTED_RE.sub(lambda m: " <str> ", text)
    text = text.lower()
    tokens: list[str] = []
    for tok in _TOKEN_RE.findall(text):
        if _CJK_RE.fullmatch(tok):
            if len(tok) == 1:
                tokens.append(tok)
            else:
                tokens.extend(tok[i:i + 2] for i in range(len(tok) - 1))
        else:
            tokens.append(tok)
    return tokens


def signature_token_set(message: str) -> frozenset[str]:
    return frozenset(task_signature(message))


def jaccard(a: frozenset[str] | set[str], b: frozenset[str] | set[str]) -> float:
    """Jaccard similarity of two token sets; empty-vs-empty is 1.0, one-empty 0.0."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union
