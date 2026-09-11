"""Production client for the PilotDeck TokenSaver judge (S1 deliverable).

Classifies a user message into one of the 4 cost tiers
{simple, medium, complex, reasoning} against a local llama.cpp server
(llama-server, OpenAI-compatible API), and extracts the base-policy
logprob vector z over all 4 tier names for downstream JitRL modulation
(z' = z + beta * A).

z extraction — S1 verdict: Method B, hardened (see S1 report).
  Per tier T one chat request force-emits `<tier>T</tier>` via a
  single-tier GBNF grammar; z(T) = sum of the raw logprobs of the tokens
  that make up the tier name at the name position. Grammar masking does
  not change logits, so this measures the true base-policy logprob.

  Why not Method A (single request + top_logprobs): "reasoning" is two
  tokens in this vocab (reason + ing, verified via /tokenize), so its
  full-name logprob is unreachable from a single position; and unlikely
  single-token names can fall outside top-50.

  Hardening (probe-observed failure mode): GBNF constrains the emitted
  TEXT, not the tokenization path — sampling may take a degenerate split
  (observed: "complex" -> "co"+"m"+"plex", corrupting z by up to ~27
  nats). Mitigation ladder per tier:
    1. retry (up to CANONICAL_ATTEMPTS) until the emitted token path
       equals the canonical tokenization (discovered once via
       POST /tokenize; hardcoded fallback map on failure);
    2. single-token names: read the canonical token's raw logprob from
       the top-logprobs list at the name position (top_logprobs=100);
    3. multi-token names: accept the last path sum — observed re-splits
       of "reasoning" are benign BPE splits whose path sum matches the
       canonical one within ~0.02 nats;
    4. otherwise clamp to Z_MIN = -10.0 (documented APPROXIMATION: the
       true logprob is <= Z_MIN).

  Under Method B there is no multi-tier request, so:
    chosen_tier = argmax_z (consistent with grammar-masked sampling,
    whose tier distribution is monotone in the canonical path logprobs)
    raw_output  = "<tier>{chosen}</tier>" (synthesized)

Transport errors are retried TRANSPORT_RETRIES times with RETRY_SLEEP_S
pause; final failure raises RuntimeError with a compact (<=300 char)
message. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TypedDict

from . import config
from .config import ENDPOINT, MODEL, TIMEOUT_S

TIERS = config.TIERS

# --- Verbatim copy of local_judge/probe.py prompt templates (which replicate
# --- PilotDeck generateJudgePrompt). Do not edit without mirroring probe.py.
PROMPT_TMPL = """You are a model-tier classifier for the PilotDeck router. Given the following user message, return exactly one tier wrapped in <tier>...</tier>.

Available tiers:
- simple: Simple greetings, confirmations, single-step Q&A, trivial file writes, remembering rules
- medium: Single tool call, short text generation, 1-2 file read/write, code generation
- complex: Needs sub-agent orchestration: parallel workstreams, delegation to specialized agents
- reasoning: Deep single-agent work: multi-file operations, data analysis, multi-step workflows, web research, structured reports from many sources

Routing rules:
- complex is ONLY for tasks that need sub-agent orchestration or parallel delegation — do NOT use it for single-agent multi-step work
- Multi-file operations, data analysis, and multi-step workflows without orchestration should be reasoning
- Simple file creation (1-2 files) or single code generation is medium
- Trivial greetings, confirmations, remembering rules, or reading one file and answering a short question is simple
{CONTINUATION}
User message:
\"\"\"
{USER_MESSAGE}
\"\"\"

Default tier when uncertain: medium.
Respond with only <tier>NAME</tier>."""

CONT_TMPL = """## CRITICAL RULE — Continuation messages
The previous turn was classified as: **{PT}**.
Short messages like "go", "continue", "ok", "yes", "好的", "继续", "开始", "冲" etc. are continuations of the previous task. They are NOT new simple requests.
For ANY message that is clearly a continuation or acknowledgment of the previous task, you MUST return <tier>{PT}</tier>.
Only reclassify if the user message introduces a genuinely NEW task with different complexity.

"""

# Same 7 test messages as probe.py (kept in sync for the CLI smoke table).
TEST_MESSAGES = [
    {"id": "M1", "msg": "你好，今天心情不错", "expect": "simple", "prev": None},
    {"id": "M2", "msg": "帮我记住：以后所有回复都用中文", "expect": "simple", "prev": None},
    {"id": "M3", "msg": "给下面这个函数写一个单元测试：\ndef add(a, b):\n    return a + b", "expect": "medium", "prev": None},
    {"id": "M4", "msg": "分析下面这组销售数据，找出增长最快的产品并写一段总结：\n一月 A产品 120万 B产品 300万\n二月 A产品 150万 B产品 290万\n三月 A产品 210万 B产品 310万", "expect": "reasoning", "prev": None},
    {"id": "M5", "msg": "请同时完成三件独立的事：1) 把这段文字翻译成英文：今天天气很好 2) 给它起三个标题 3) 列出五个关键词", "expect": "complex", "prev": None},
    {"id": "M6", "msg": "继续", "expect": "medium", "prev": "medium"},
    {"id": "M7", "msg": "hi", "expect": "simple", "prev": None},
]


class JudgeResult(TypedDict):
    tier_logits: dict[str, float]   # all 4 tiers: {"simple": z, "medium": z, "complex": z, "reasoning": z}
    chosen_tier: str                # tier actually emitted under the multi-tier grammar (A) / argmax z (B)
    raw_output: str
    latency_ms: float               # total wall time of the z-extraction requests (1 or 4 requests)


def build_prompt(user_message: str, previous_tier: str | None = None,
                 memory_block: str | None = None) -> str:
    """Assemble the judge prompt; the continuation block is inserted only
    when previous_tier is given.

    memory_block (S3-T8 ablation, prereg §3.2/§4): optional extra text
    appended after the continuation block and before "User message:".
    It must be a non-empty, already-sanitized block: any occurrence of the
    {USER_MESSAGE} / {CONTINUATION} / {PT} placeholders raises ValueError
    because build_prompt substitutes {CONTINUATION} BEFORE {USER_MESSAGE}
    and a placeholder inside the block would trigger a second replacement.
    With memory_block=None the returned prompt is byte-identical to the
    pre-ablation version of this function."""
    cont = CONT_TMPL.replace("{PT}", previous_tier) if previous_tier else ""
    if memory_block is not None:
        if not memory_block.strip():
            raise ValueError("memory_block must be non-empty when provided")
        for placeholder in ("{USER_MESSAGE}", "{CONTINUATION}", "{PT}"):
            if placeholder in memory_block:
                raise ValueError(
                    f"memory_block contains prompt placeholder {placeholder!r}")
        cont = cont + memory_block
    return PROMPT_TMPL.replace("{CONTINUATION}", cont).replace("{USER_MESSAGE}", user_message)


class JudgeClient:
    """Client for the local llama.cpp judge endpoint."""

    def __init__(self, endpoint: str = ENDPOINT, model: str = MODEL,
                 timeout_s: float = TIMEOUT_S, z_method: str = "auto"):
        if z_method not in ("auto", "A", "B"):
            raise ValueError(f"z_method must be 'auto', 'A' or 'B', got {z_method!r}")
        self._base = endpoint.rstrip("/")
        self._chat_url = self._base + "/v1/chat/completions"
        self.model = model
        self.timeout_s = timeout_s
        self._z_method_cfg = z_method
        self._canonical: dict[str, list[str]] | None = None
        self._grammar_multi: str | None = None

    # ------------------------------------------------------------------ API
    def effective_z_method(self) -> str:
        """Resolve 'auto' to the S1 verdict: Method B.

        Rationale: "reasoning" is not a single token in this vocab, so Method A
        can never recover its full-name logprob from one position, and the
        single-token names are not reliably inside top-50 either.
        """
        if self._z_method_cfg == "auto":
            return "B"
        return self._z_method_cfg

    def judge(self, user_message: str, previous_tier: str | None = None,
              memory_block: str | None = None) -> JudgeResult:
        """Classify user_message; return the 4-tier z vector plus the chosen tier.

        memory_block (S3-T8 ablation): optional whitelist memory block
        injected into the prompt (see build_prompt). None (default) keeps
        the prompt — and therefore the request stream — byte-identical to
        the pre-ablation behavior."""
        prompt = build_prompt(user_message, previous_tier, memory_block)
        t0 = time.perf_counter()
        if self.effective_z_method() == "A":
            result = self._judge_method_a(prompt)
        else:
            result = self._judge_method_b(prompt)
        result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        return result

    # ------------------------------------------------------------- Method A
    def _judge_method_a(self, prompt: str) -> JudgeResult:
        """Single multi-tier grammar request; z read from the name-position
        top_logprobs. Missing tiers are clamped to Z_MIN (approximation:
        their true logprob is below the reported cutoff)."""
        resp = self._chat(prompt, self._multi_grammar(), config.TOP_LOGPROBS_A,
                          temperature=1.0)
        content = self._emitted_text(resp)
        m = re.search(r"<tier>\s*(\w+)\s*</tier>", content)
        if not m:
            raise RuntimeError(f"judge: unparseable output under grammar: {content[:120]!r}")
        entries = resp["choices"][0]["logprobs"]["content"]
        i = self._skip_opener(entries)
        top = {e["token"]: e["logprob"] for e in entries[i].get("top_logprobs", [])}
        z = {t: float(top.get(t, config.Z_MIN)) for t in TIERS}
        return {"tier_logits": z, "chosen_tier": m.group(1), "raw_output": content,
                "latency_ms": 0.0}

    # ------------------------------------------------------------- Method B
    def _judge_method_b(self, prompt: str) -> JudgeResult:
        """4 forced-emission requests (one per tier), z = canonical name-token
        path logprob sum. chosen = argmax z; raw_output synthesized."""
        canonical = self._canonical_tokens()
        z: dict[str, float] = {}
        for t in TIERS:
            z[t] = self._forced_tier_z(prompt, t, canonical[t])
        chosen = max(TIERS, key=lambda t: z[t])
        raw = f"<tier>{chosen}</tier>"
        return {"tier_logits": {t: float(z[t]) for t in TIERS},
                "chosen_tier": chosen, "raw_output": raw, "latency_ms": 0.0}

    def _forced_tier_z(self, prompt: str, tier: str, canon_pieces: list[str]) -> float:
        """z(tier) via single-tier grammar force-emission with canonical-path
        verification. Fallback ladder documented in the module docstring."""
        grammar = f'root ::= "<tier>" ("{tier}") "</tier>"'
        last_sum: float | None = None
        last_top: dict[str, float] = {}
        for _ in range(config.CANONICAL_ATTEMPTS):
            resp = self._chat(prompt, grammar, config.TOP_LOGPROBS_B,
                              temperature=config.Z_TEMPERATURE)
            parsed = self._parse_forced(resp, tier)
            if parsed is None:
                continue  # content sanity check failed; retry
            toks, total, top = parsed
            last_sum, last_top = total, top
            if toks == canon_pieces:
                return total
        # Canonical path not sampled within budget.
        if len(canon_pieces) == 1 and canon_pieces[0] in last_top:
            return last_top[canon_pieces[0]]            # raw single-position read
        if last_sum is not None:
            return last_sum                              # benign re-split tolerance
        return config.Z_MIN                              # documented approximation

    # -------------------------------------------------------------- plumbing
    @staticmethod
    def _emitted_text(resp: dict) -> str:
        """Assistant text of a chat completion response.

        Newer llama-server builds (observed system_fingerprint b10903-*)
        running a thinking-capable chat template route grammar-forced
        output into `message.reasoning_content` and leave `message.content`
        empty: the template opens a <think> block that the constrained
        output never closes, so the whole emission is classified as
        reasoning. Prefer `content`; fall back to `reasoning_content` when
        it is empty so z extraction works on both server generations. The
        logprob token path is identical in both cases (verified against
        the b10903 server: tokens '<','t','ier','>','medium','</',... with
        unchanged logprobs), so this is a pure read-location fix.
        """
        msg = resp["choices"][0]["message"]
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content
        reasoning = msg.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning
        return content if isinstance(content, str) else ""

    def _parse_forced(self, resp: dict, tier: str):
        """Return (name_tokens, logprob_sum, top_logprobs_at_first_name_pos)
        or None when the content is not exactly <tier>NAME</tier>."""
        content = self._emitted_text(resp)
        if content.strip() != f"<tier>{tier}</tier>":
            return None
        entries = resp["choices"][0]["logprobs"]["content"]
        i = self._skip_opener(entries)
        first = entries[i]
        toks, total = [first["token"]], first["logprob"]
        j = i + 1
        while j < len(entries) and not entries[j]["token"].startswith("</"):
            toks.append(entries[j]["token"])
            total += entries[j]["logprob"]
            j += 1
        top = {e["token"]: e["logprob"] for e in first.get("top_logprobs", [])}
        return toks, total, top

    @staticmethod
    def _skip_opener(entries: list[dict]) -> int:
        """Index of the first name token: skip tokens that make up '<tier>'."""
        acc, i = "", 0
        while i < len(entries) and not acc.endswith(">"):
            acc += entries[i]["token"]
            i += 1
        if i >= len(entries):
            raise RuntimeError("judge: malformed logprobs (no <tier> opener found)")
        return i

    def _multi_grammar(self) -> str:
        if self._grammar_multi is None:
            text = (Path(__file__).parent / "grammar.gbnf").read_text(encoding="utf-8")
            self._grammar_multi = "\n".join(
                ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
            ).strip()
        return self._grammar_multi

    def _canonical_tokens(self) -> dict[str, list[str]]:
        """Canonical tokenization per tier, discovered once via /tokenize
        (authoritative); falls back to the S1-measured map on any failure."""
        if self._canonical is None:
            try:
                cand = {t: self._tokenize(t) for t in TIERS}
                for t, pieces in cand.items():
                    if not pieces or "".join(pieces) != t:
                        raise ValueError(f"tokenize mismatch for {t!r}: {pieces}")
                self._canonical = cand
            except Exception:
                self._canonical = {t: list(p) for t, p in config.CANONICAL_TOKENS_FALLBACK.items()}
        return self._canonical

    def _tokenize(self, text: str) -> list[str]:
        req = urllib.request.Request(
            self._base + "/tokenize",
            data=json.dumps({"content": text, "add_special": False}).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        ids = json.loads(urllib.request.urlopen(req, timeout=self.timeout_s).read())["tokens"]
        pieces = []
        for tid in ids:
            dreq = urllib.request.Request(
                self._base + "/detokenize",
                data=json.dumps({"tokens": [tid]}).encode("utf-8"),
                headers={"Content-Type": "application/json"})
            pieces.append(json.loads(urllib.request.urlopen(dreq, timeout=self.timeout_s).read())["content"])
        return pieces

    def _chat(self, prompt: str, grammar: str | None, top_logprobs: int,
              temperature: float) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 24,
            "logprobs": True,
            "top_logprobs": top_logprobs,
            "stream": False,
        }
        if grammar:
            body["grammar"] = grammar
        data = json.dumps(body).encode("utf-8")
        last_err: Exception | None = None
        for attempt in range(config.TRANSPORT_RETRIES):
            try:
                req = urllib.request.Request(self._chat_url, data=data,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                    return json.loads(r.read())
            except Exception as e:  # noqa: BLE001 - compact re-raise below
                last_err = e
                if attempt < config.TRANSPORT_RETRIES - 1:
                    time.sleep(config.RETRY_SLEEP_S)
        raise RuntimeError(
            f"judge request failed after {config.TRANSPORT_RETRIES} attempts: "
            f"{self._fmt_err(last_err)}")

    @staticmethod
    def _fmt_err(e: Exception | None) -> str:
        if isinstance(e, urllib.error.HTTPError):
            try:
                detail = e.read().decode("utf-8", "replace")
            except Exception:
                detail = ""
            s = f"HTTP {e.code}: {detail}"
        else:
            s = repr(e)
        return s[:config.ERROR_MSG_MAX_CHARS]


# ------------------------------------------------------------------- CLI
def _truncate(s: str, n: int) -> str:
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[: n - 1] + "…"


def _smoke() -> None:
    client = JudgeClient()
    print(f"endpoint={client._base} model={client.model} "
          f"z_method={client.effective_z_method()}")
    print(f"{'id':4s} {'message':30s} {'expect':9s} {'chosen':9s} {'hit':4s} "
          f"{' '.join(f'{t:>9s}' for t in TIERS)} {'lat_ms':>7s}")
    hits = 0
    for m in TEST_MESSAGES:
        r = client.judge(m["msg"], m["prev"])
        hit = r["chosen_tier"] == m["expect"]
        hits += hit
        zs = " ".join(f"{r['tier_logits'][t]:9.3f}" for t in TIERS)
        print(f"{m['id']:4s} {_truncate(m['msg'], 30):30s} {m['expect']:9s} "
              f"{r['chosen_tier']:9s} {'HIT' if hit else 'MISS':4s} {zs} "
              f"{r['latency_ms']:7.0f}")
    print(f"chosen vs expect: {hits}/{len(TEST_MESSAGES)}")


def _stability(n: int) -> None:
    client = JudgeClient()
    runs = [client.judge(TEST_MESSAGES[2]["msg"]) for _ in range(n)]
    print(f"stability: M3 x{n} via JudgeClient ({client.effective_z_method()})")
    for t in TIERS:
        vals = [r["tier_logits"][t] for r in runs]
        print(f"  {t:9s} mean={statistics.fmean(vals):8.4f} "
              f"pstdev={statistics.pstdev(vals):.4f} vals={[round(v, 3) for v in vals]}")


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="local_judge smoke / stability CLI")
    ap.add_argument("--stability", type=int, default=0, metavar="N",
                    help="run message M3 N times and print per-tier mean/std "
                         "instead of the 7-message smoke table")
    args = ap.parse_args(argv)
    if args.stability > 0:
        _stability(args.stability)
    else:
        _smoke()


if __name__ == "__main__":
    main()
