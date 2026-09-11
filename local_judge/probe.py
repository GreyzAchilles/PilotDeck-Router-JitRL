"""S1 probe — determine tier-logit extraction method for the JitRL judge.

Usage: python local_judge/probe.py [--out logs/s1_probe.json]

Answers (per docs: JitRL-Router D7 S1 verification points):
  T1  tier-name token counts (from single-tier force-emission)
  T2  Method A viability: multi-tier grammar + top_logprobs=50 -> all 4 names
      visible as complete tokens at the name position?
  T3  Method B z-vectors: per-tier force-emission, summed name-token logprobs
  T4  temperature invariance of reported logprobs (0.1 vs 1.0)
  T5  z tables for 7 test messages + chosen tier + latencies (A and B)
  T6  stability: message 3 repeated 5x -> per-tier z std
"""
import argparse
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

EP = "http://127.0.0.1:18080/v1/chat/completions"
TIERS = ["simple", "medium", "complex", "reasoning"]
GRAMMAR_MULTI = 'root ::= "<tier>" ("simple" | "medium" | "complex" | "reasoning") "</tier>"'
Z_MIN = -10.0

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

MESSAGES = [
    {"id": "M1", "msg": "你好，今天心情不错", "expect": "simple", "prev": None},
    {"id": "M2", "msg": "帮我记住：以后所有回复都用中文", "expect": "simple", "prev": None},
    {"id": "M3", "msg": "给下面这个函数写一个单元测试：\ndef add(a, b):\n    return a + b", "expect": "medium", "prev": None},
    {"id": "M4", "msg": "分析下面这组销售数据，找出增长最快的产品并写一段总结：\n一月 A产品 120万 B产品 300万\n二月 A产品 150万 B产品 290万\n三月 A产品 210万 B产品 310万", "expect": "reasoning", "prev": None},
    {"id": "M5", "msg": "请同时完成三件独立的事：1) 把这段文字翻译成英文：今天天气很好 2) 给它起三个标题 3) 列出五个关键词", "expect": "complex", "prev": None},
    {"id": "M6", "msg": "继续", "expect": "medium", "prev": "medium"},
    {"id": "M7", "msg": "hi", "expect": "simple", "prev": None},
]


def build_prompt(user_message: str, previous_tier: str | None = None) -> str:
    cont = CONT_TMPL.replace("{PT}", previous_tier) if previous_tier else ""
    return PROMPT_TMPL.replace("{CONTINUATION}", cont).replace("{USER_MESSAGE}", user_message)


def chat(prompt: str, grammar: str | None, top_logprobs: int, temperature: float = 1.0,
         max_tokens: int = 24, timeout: float = 120.0):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "stream": False,
    }
    if grammar:
        body["grammar"] = grammar
    req = urllib.request.Request(EP, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return resp, (time.time() - t0) * 1000.0


def _find_name_entries(entries):
    """Skip the '<tier>' opener tokens; return (index_of_name_start, opener_len)."""
    acc = ""
    i = 0
    while i < len(entries) and not acc.endswith(">"):
        acc += entries[i]["token"]
        i += 1
    return i


def _emitted_text(resp: dict) -> str:
    """Assistant text of a chat completion response.

    Verbatim mirror of JudgeClient._emitted_text (local_judge/client.py) —
    do not edit without mirroring. Newer llama-server builds (observed
    system_fingerprint b10903-*) running a thinking-capable chat template
    route grammar-forced output into `message.reasoning_content` and leave
    `message.content` empty; prefer `content` and fall back to
    `reasoning_content` when it is empty so z extraction works on both
    server generations. Without this, method B raises on the empty-content
    sanity check and method A reports chosen='??' — i.e. the probe artifact
    would misreport the z table (all-None / Z_MIN-clamped entries)."""
    msg = resp["choices"][0]["message"]
    content = msg.get("content")
    if isinstance(content, str) and content.strip():
        return content
    reasoning = msg.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    return content if isinstance(content, str) else ""


def name_tokens_info(resp, tier: str):
    """Single-tier response -> (count, sum_logprob, tokens) of the tier-name tokens."""
    content = _emitted_text(resp)
    expected = f"<tier>{tier}</tier>"
    if content.strip() != expected:
        raise AssertionError(f"unexpected content: {content!r} (expected {expected!r})")
    entries = resp["choices"][0]["logprobs"]["content"]
    i = _find_name_entries(entries)
    toks, total = [], 0.0
    while i < len(entries) and not entries[i]["token"].startswith("</"):
        toks.append(entries[i]["token"])
        total += entries[i]["logprob"]
        i += 1
    return len(toks), total, toks


def method_a(prompt: str, top_logprobs: int = 50):
    """Multi-tier grammar, read tier tokens from top_logprobs at the name position."""
    try:
        resp, dt = chat(prompt, GRAMMAR_MULTI, top_logprobs)
    except urllib.error.HTTPError as e:
        if top_logprobs != 20:
            resp, dt = chat(prompt, GRAMMAR_MULTI, 20)
            top_logprobs = 20
        else:
            raise
    content = _emitted_text(resp)
    m = re.search(r"<tier>\s*(\w+)\s*</tier>", content)
    chosen = m.group(1) if m else "??"
    entries = resp["choices"][0]["logprobs"]["content"]
    i = _find_name_entries(entries)
    pos = entries[i]
    top = {t["token"]: t["logprob"] for t in pos.get("top_logprobs", [])}
    zs = {t: top.get(t) for t in TIERS}  # None = not visible within cutoff
    raw_top1 = pos["top_logprobs"][0]["token"] if pos.get("top_logprobs") else None
    return {
        "chosen": chosen,
        "z": zs,
        "raw_top1_at_name_pos": raw_top1,
        "top10_at_name_pos": [(t["token"], round(t["logprob"], 4))
                              for t in pos.get("top_logprobs", [])[:10]],
        "emitted_name_logprob": pos["logprob"],
        "top_logprobs_used": top_logprobs,
        "latency_ms": dt,
    }


def method_b(prompt: str, temperature: float = 1.0):
    """4x single-tier force-emission; z = sum of name-token logprobs."""
    zs, counts, tokens, lats = {}, {}, {}, {}
    for t in TIERS:
        g = f'root ::= "<tier>" ("{t}") "</tier>"'
        resp, dt = chat(prompt, g, 20, temperature)
        c, s, toks = name_tokens_info(resp, t)
        zs[t], counts[t], tokens[t], lats[t] = s, c, toks, dt
    return {"z": zs, "name_token_counts": counts, "name_tokens": tokens,
            "latencies_ms": lats, "total_latency_ms": sum(lats.values())}


def diag(base="http://127.0.0.1:18080"):
    """Definitive tokenization via /tokenize + top-100 fallback viability."""
    out = {"tokenize": {}, "forced_top100": []}
    for t in TIERS:
        body = json.dumps({"content": t}).encode("utf-8")
        req = urllib.request.Request(base + "/tokenize", data=body,
                                     headers={"Content-Type": "application/json"})
        tok = json.loads(urllib.request.urlopen(req, timeout=30).read())
        ids = tok.get("tokens", [])
        pieces = []
        for i in ids:
            dr = urllib.request.Request(
                base + "/detokenize",
                data=json.dumps({"tokens": [i]}).encode("utf-8"),
                headers={"Content-Type": "application/json"})
            pieces.append(json.loads(urllib.request.urlopen(dr, timeout=30).read())["content"])
        out["tokenize"][t] = {"n": len(ids), "pieces": pieces}
        print(f"tokenize {t}: n={len(ids)} pieces={pieces}")

    # Forced per-tier with top_logprobs=100 on M3 & M4: canonical-path rate +
    # whether the canonical first token is visible in the raw top-100 list.
    for m in (MESSAGES[2], MESSAGES[3]):
        prompt = build_prompt(m["msg"], m["prev"])
        for t in TIERS:
            g = f'root ::= "<tier>" ("{t}") "</tier>"'
            for attempt in range(3):
                resp, dt = chat(prompt, g, 100, 0.1)
                entries = resp["choices"][0]["logprobs"]["content"]
                i = _find_name_entries(entries)
                pos = entries[i]
                toks = [pos["token"]]
                total = pos["logprob"]
                j = i + 1
                while j < len(entries) and not entries[j]["token"].startswith("</"):
                    toks.append(entries[j]["token"])
                    total += entries[j]["logprob"]
                    j += 1
                top = {e["token"]: e["logprob"] for e in pos.get("top_logprobs", [])}
                rec = {
                    "msg": m["id"], "tier": t, "attempt": attempt,
                    "emitted_path": toks, "z_path_sum": round(total, 4),
                    "top100_size": len(pos.get("top_logprobs", [])),
                    "latency_ms": round(dt),
                }
                # canonical check: is the *canonical* (single-/tokenize) first
                # piece the emitted one? record its raw logprob when visible
                can_first = out["tokenize"][t]["pieces"][0] if out["tokenize"][t]["pieces"] else t
                rec["canonical_first_piece"] = can_first
                rec["canonical_first_emitted"] = (toks[0] == can_first)
                rec["canonical_first_in_top100"] = top.get(can_first)
                out["forced_top100"].append(rec)
                print(json.dumps(rec, ensure_ascii=False))
                if rec["canonical_first_emitted"]:
                    break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/s1_probe.json")
    ap.add_argument("--diag", action="store_true", help="tokenization + top-100 diagnostics only")
    args = ap.parse_args()

    if args.diag:
        d = diag()
        out = Path(args.out.replace(".json", "_diag.json"))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"saved -> {out}")
        return

    report = {"messages": [], "stability": None, "temp_invariance": None}

    print("== T5: 7 messages, Method A + B ==")
    for m in MESSAGES:
        prompt = build_prompt(m["msg"], m["prev"])
        a = method_a(prompt)
        b = method_b(prompt)
        rec = {
            "id": m["id"], "expect": m["expect"],
            "method_a": a, "method_b": b,
            "chosen": a["chosen"],
            "hit": a["chosen"] == m["expect"],
        }
        report["messages"].append(rec)
        za = {k: (round(v, 3) if v is not None else None) for k, v in a["z"].items()}
        print(f"{m['id']} expect={m['expect']:9s} chosen={a['chosen']:9s} hit={rec['hit']} "
              f"| A: {za} rawtop1={a['raw_top1_at_name_pos']!r} ({a['latency_ms']:.0f}ms) "
              f"| B: { {k: round(v,3) for k,v in b['z'].items()} } counts={b['name_token_counts']} ({b['total_latency_ms']:.0f}ms)")

    print("== T6: stability (M3 x5, method B) ==")
    prompt3 = build_prompt(MESSAGES[2]["msg"])
    runs = [method_b(prompt3) for _ in range(5)]
    std = {t: statistics.pstdev([r["z"][t] for r in runs]) for t in TIERS}
    report["stability"] = {"std_per_tier": std,
                           "z_runs": [{k: round(v, 3) for k, v in r["z"].items()} for r in runs]}
    print("std:", {k: round(v, 4) for k, v in std.items()})

    print("== T4: temperature invariance (M3, method B, 0.1 vs 1.0) ==")
    b01 = method_b(prompt3, temperature=0.1)
    b10 = runs[0]
    same = all(abs(b01["z"][t] - b10["z"][t]) < 1e-6 for t in TIERS)
    report["temp_invariance"] = {"z_temp01": {k: round(v, 4) for k, v in b01["z"].items()},
                                 "z_temp10": {k: round(v, 4) for k, v in b10["z"].items()},
                                 "identical": same}
    print("identical:", same, "| t0.1:", {k: round(v, 3) for k, v in b01["z"].items()},
          "| t1.0:", {k: round(v, 3) for k, v in b10["z"].items()})

    # Aggregate verdicts
    all_single = all(all(c == 1 for c in mb["name_token_counts"].values()) for mb in
                     [m["method_b"] for m in report["messages"]])
    a_complete = all(all(v is not None for v in m["method_a"]["z"].values()) for m in report["messages"])
    hits = sum(1 for m in report["messages"] if m["hit"])
    report["verdicts"] = {
        "tier_names_all_single_token": all_single,
        "method_a_all_four_visible": a_complete,
        "chosen_vs_expect_hits": f"{hits}/7",
        "raw_distribution_evidence": report["messages"][0]["method_a"]["raw_top1_at_name_pos"],
    }
    print("== VERDICTS ==")
    print(json.dumps(report["verdicts"], ensure_ascii=False, indent=1))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
