"""Generate the four JitRL Router technical diagrams as self-contained SVGs.

Run:
    python diagrams/make_diagrams.py

Outputs (deterministic, Python stdlib only, no external dependencies):
    diagrams/architecture.svg          end-to-end C-arm decision path
    diagrams/arms.svg                  A / B / C0 / C1 / T8-PM comparison
    diagrams/online-offline.svg        online live path vs offline frozen data
    diagrams/pilotdeck-integration.svg upstream PilotDeck integration points

Style: hand-rolled SVG (no mermaid/graphviz dependency) so the files are
version-controllable, diff-able and render in plain <img> tags.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
FONT = "ui-sans-serif, system-ui, 'Segoe UI', sans-serif"

# palette
INK = "#0f172a"
MUTED = "#475569"
NEUTRAL_FILL, NEUTRAL_STROKE = "#f1f5f9", "#64748b"
JUDGE_FILL, JUDGE_STROKE = "#e0f2fe", "#0369a1"
JITRL_FILL, JITRL_STROKE = "#dcfce7", "#15803d"
MEM_FILL, MEM_STROKE = "#ede9fe", "#7c3aed"
EXEC_FILL, EXEC_STROKE = "#fef3c7", "#b45309"
WARN_FILL, WARN_STROKE = "#fee2e2", "#dc2626"
INFO_FILL, INFO_STROKE = "#cffafe", "#0e7490"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    """Minimal deterministic SVG builder."""

    def __init__(self, width: int, height: int, title: str):
        self.w, self.h = width, height
        self.buf: list[str] = []
        self.title = title
        self.buf.append(
            f'<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{MUTED}"/></marker></defs>')
        self.buf.append(
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')

    def box(self, x: float, y: float, w: float, h: float, fill: str,
            stroke: str, rx: float = 10, dash: str | None = None,
            width_px: float = 1.5) -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.buf.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{width_px}"{d}/>')

    def text(self, x: float, y: float, s: str, size: int = 12,
             anchor: str = "start", fill: str = INK, weight: str = "normal",
             spacing: float | None = None) -> None:
        sp = f' letter-spacing="{spacing}"' if spacing else ""
        self.buf.append(
            f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'text-anchor="{anchor}" fill="{fill}" font-weight="{weight}"{sp}>'
            f'{esc(s)}</text>')

    def lines(self, x: float, y0: float, rows: list[str], size: int = 11,
              anchor: str = "start", fill: str = INK, lh: float = 15,
              weight: str = "normal") -> None:
        for i, row in enumerate(rows):
            self.text(x, y0 + i * lh, row, size=size, anchor=anchor,
                      fill=fill, weight=weight)

    def arrow(self, x1: float, y1: float, x2: float, y2: float,
              color: str = MUTED, width_px: float = 1.6) -> None:
        self.buf.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{width_px}" marker-end="url(#arr)"/>')

    def render(self) -> str:
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" '
                f'height="{self.h}" viewBox="0 0 {self.w} {self.h}">'
                f'<title>{esc(self.title)}</title>')
        return head + "".join(self.buf) + "</svg>"

    def write(self, name: str) -> Path:
        p = OUT_DIR / name
        p.write_text(self.render(), encoding="utf-8")
        ET.fromstring(p.read_text(encoding="utf-8"))  # well-formedness gate
        return p


def badge(s: Svg, x: float, y: float, label: str, fill: str, stroke: str) -> None:
    s.box(x, y, 96, 24, fill, stroke, rx=12, width_px=1.2)
    s.text(x + 48, y + 16, label, size=11, anchor="middle", fill=stroke,
           weight="bold")


def header(s: Svg, title: str, subtitle: str, top: float = 26) -> None:
    s.text(24, top + 6, title, size=19, weight="bold")
    s.text(24, top + 28, subtitle, size=12, fill=MUTED)


# ----------------------------------------------------------------------
# 1. architecture.svg — end-to-end C-arm decision path
# ----------------------------------------------------------------------
def architecture() -> Path:
    s = Svg(1060, 1420,
            "JitRL Router end-to-end decision path (arm C)")
    header(s, "JitRL Router — End-to-End Decision Path (Arm C)",
           "frozen local judge · non-parametric memory · closed-form logit update")

    cx, cw = 90, 520          # main pipeline column
    mx, mw = 700, 330         # memory sidebar column
    y = 92
    step = 104

    def pipeline(idx: int, title: str, body: list[str], fill: str,
                 stroke: str, note: str | None = None) -> float:
        nonlocal y
        top = y + idx * step
        h = 72
        s.box(cx, top, cw, h, fill, stroke)
        s.text(cx + 16, top + 26, title, size=13, weight="bold")
        s.lines(cx + 16, top + 45, body, size=11, fill=MUTED)
        if note:
            s.text(cx + cw - 12, top + 26, note, size=10.5, anchor="end",
                   fill=stroke, weight="bold")
        if idx > 0:
            s.arrow(cx + cw / 2, top - step + 72 + 2, cx + cw / 2, top - 3)
        return top

    t0 = pipeline(0, "User message", ["raw conversation turn (Chinese/English)"],
                  NEUTRAL_FILL, NEUTRAL_STROKE)
    t1 = pipeline(1, "State extraction",
                  ["intent_class · rule-normalized task signature · CJK bigrams"],
                  NEUTRAL_FILL, NEUTRAL_STROKE)
    t2 = pipeline(2, "Local Judge — frozen, no weight updates",
                  ["MiniCPM5-1B Q4_K_M via llama.cpp (:18080)",
                   "Method B: 4 grammar-forced requests → 4-tier base logits z"],
                  JUDGE_FILL, JUDGE_STROKE, note="≈ 1–3 s")
    t3 = pipeline(3, "Memory retrieval",
                  ["same intent_class · token-set Jaccard ≥ 0.5 · top-k = 10",
                   "entries {intent, signature, tier, G}"],
                  MEM_FILL, MEM_STROKE)
    t4 = pipeline(4, "Value estimation",
                  ["V = mean G · Q(tier) from neighbors · λ-exploration bonus",
                   "normalized advantage Â per tier"],
                  JITRL_FILL, JITRL_STROKE)
    t5 = pipeline(5, "Min-neighbor gate",
                  ["C0: min_neighbors = 1 · C1: min_neighbors = 3",
                   "n_retrieved < min → pass-through (z' = z)"],
                  "#ffffff", JITRL_STROKE)
    t6 = pipeline(6, "Closed-form logit update",
                  ["z'(tier) = z(tier) + β·Â(tier) · β = 5 · clamp z_min = −10"],
                  JITRL_FILL, JITRL_STROKE, note="mean 0.24 ms / max 0.67 ms")
    t7 = pipeline(7, "Tier decision", ["argmax(z') over simple / medium / complex / reasoning"],
                  NEUTRAL_FILL, NEUTRAL_STROKE)
    t8 = pipeline(8, "Tier → model static map",
                  ["simple → CPA/glm-5.3-flash · medium → CPA/opencode-v4-flash",
                   "complex → CPA/OpenBMB-5.3 · reasoning → CPA/glm-5.3"],
                  EXEC_FILL, EXEC_STROKE)
    t9 = pipeline(9, "Execution",
                  ["CPA completion · 1024-token budget · temperature 0.7"],
                  EXEC_FILL, EXEC_STROKE)
    t10 = pipeline(10, "Blind Evaluator",
                   ["CPA/gpt-5.6-sol · structured JSON · not in the tier map"],
                   JUDGE_FILL, JUDGE_STROKE)
    t11 = y + 12 * step - 12
    s.box(cx, t11, cw, 72, NEUTRAL_FILL, NEUTRAL_STROKE)
    s.text(cx + 16, t11 + 26, "Reward", size=13, weight="bold")
    s.lines(cx + 16, t11 + 45,
            ["r = 0.6·quality + 0.3·cost_saving  (denominator: glm-5.3)"],
            size=11, fill=MUTED)
    s.arrow(cx + cw / 2, t10 + 72 + 2, cx + cw / 2, t11 - 3)

    # memory sidebar
    m_top, m_h = t3 - 8, (t11 + 72) - t3 + 16
    s.box(mx, m_top, mw, m_h, MEM_FILL, MEM_STROKE, rx=14, dash="7 5")
    s.text(mx + 18, m_top + 30, "Experience memory (non-parametric)",
           size=13, weight="bold", fill=MEM_STROKE)
    s.lines(mx + 18, m_top + 54,
            ["stores (state, tier, reward-G) triples",
             "written only after a successful evaluation",
             "session-isolated in the live demo (RAM only)",
             "frozen C0 trace replayed read-only in the T8 ablation"],
            size=11, fill=MUTED, lh=20)
    s.arrow(mx, t3 + 36, cx + cw + 3, t3 + 36, color=MEM_STROKE)
    s.text(mx - 8, t3 + 28, "read", size=10.5, anchor="end", fill=MEM_STROKE)
    s.arrow(cx + cw, t11 + 36, mx - 3, t11 + 36, color=MEM_STROKE)
    s.text(cx + cw + 10, t11 + 28, "write-back (state, tier, G)", size=10.5,
           fill=MEM_STROKE)

    s.lines(90, 1372,
            ["Latency split: Judge ≈ 1–3 s (4 llama.cpp requests) vs JitRL post-processing 0.24–0.67 ms.",
             "Evaluator failure ⇒ no memory write (fail-safe: never store an unmeasured outcome)."],
            size=11, fill=MUTED)
    return s.write("architecture.svg")


# ----------------------------------------------------------------------
# 2. arms.svg — A / B / C0 / C1 / T8 comparison grid
# ----------------------------------------------------------------------
def arms() -> Path:
    s = Svg(1240, 460, "JitRL Router arm comparison")
    header(s, "Arm Comparison — what each arm adds",
           "A static · B judge · C0 JitRL · C1 gated · T8-PM prompt injection (ablation)")

    cols = ["Arm", "Judge call", "Memory retrieval", "V/Q/Â", "Logit modulation",
            "Prompt injection", "Tier decision", "Exec + blind eval", "Memory write"]
    widths = [120, 150, 140, 70, 140, 120, 130, 150, 110]
    rows = [
        ("A Static", ["— (skipped)", "—", "—", "—", "—", "fixed: reasoning",
                      "CPA/glm-5.3 · eval", "—"], NEUTRAL_FILL, NEUTRAL_STROKE),
        ("B Judge", ["Method B → z", "—", "—", "—", "—", "argmax z",
                     "tier→model · eval", "—"], JUDGE_FILL, JUDGE_STROKE),
        ("C0 JitRL", ["Method B → z", "top-k · Jaccard ≥ 0.5", "✓",
                      "z' = z + β·Â", "—", "argmax z'", "tier→model · eval",
                      "✓ online learn"], JITRL_FILL, JITRL_STROKE),
        ("C1 gated", ["Method B → z", "top-k · Jaccard ≥ 0.5", "✓",
                      "only if n ≥ 3 (gate)", "—", "argmax z'",
                      "tier→model · eval", "✓ online learn"], INFO_FILL, INFO_STROKE),
        ("T8-PM ablation", ["injected prompt → z_prompt", "frozen C0 prefix (read-only)",
                            "—", "—", "✓ whitelist block", "argmax z_prompt",
                            "tier→model · eval", "— (frozen)"], MEM_FILL, MEM_STROKE),
    ]

    x0, y0, rh, hh = 24, 96, 56, 46
    xs: list[float] = []
    acc = x0
    for w in widths:
        xs.append(acc)
        acc += w
    total = acc - x0

    # header band
    s.box(x0, y0, total, hh, "#e2e8f0", NEUTRAL_STROKE, rx=8, width_px=1.2)
    for i, c in enumerate(cols):
        s.text(xs[i] + widths[i] / 2, y0 + 28, c, size=11.5, anchor="middle",
               weight="bold")

    for r, (name, cells, fill, stroke) in enumerate(rows):
        ry = y0 + hh + r * rh
        s.box(x0, ry, total, rh - 4, "#ffffff", "#cbd5e1", rx=8, width_px=1.0)
        s.box(xs[0], ry + 3, widths[0] - 8, rh - 10, fill, stroke, rx=8,
              width_px=1.2)
        s.text(xs[0] + (widths[0] - 8) / 2 + 4, ry + rh / 2 + 4, name, size=12,
               anchor="middle", weight="bold")
        for i, cell in enumerate(cells, start=1):
            s.text(xs[i] + widths[i] / 2, ry + rh / 2 + 4, cell, size=10.5,
                   anchor="middle",
                   fill=INK if cell not in ("—",) else "#94a3b8")

    note_y = y0 + hh + len(rows) * rh + 22
    s.lines(x0, note_y,
            ["T8-LM (the ablation's other arm) = exact C0 math replayed against the frozen C0 prefix memory:",
             "read-only, no online write-back — isolates the memory mechanism from the learning trajectory.",
             "C1's gate suppressed C0's T06/T19 flips but its own memory writes cascaded into a new T20 flip."],
            size=11, fill=MUTED)
    return s.write("arms.svg")


# ----------------------------------------------------------------------
# 3. online-offline.svg — two lanes
# ----------------------------------------------------------------------
def online_offline() -> Path:
    s = Svg(1180, 620, "Online live path vs offline frozen data path")
    header(s, "Online Live Path vs Offline Frozen Data",
           "the demo defaults to OFFLINE (zero network); ONLINE is optional and clearly marked")

    # ---- online lane
    oy = 96
    s.box(24, oy, 1132, 220, "#f0fdf4", JITRL_STROKE, rx=14, width_px=1.6)
    s.text(44, oy + 34, "ONLINE — live decision loop (optional)", size=14,
           weight="bold", fill=JITRL_STROKE)
    boxes = [
        ("Browser UI", ["preset task or free input", "arm switch A/B/C0/C1/T8"]),
        ("Demo backend", ["stdlib server · proxies calls", "keys never reach the browser"]),
        ("Local Judge", ["llama.cpp · Method B", "base logits z"]),
        ("JitRL engine", ["retrieve → V/Q/Â", "z' = z + β·Â (gate-aware)"]),
        ("CPA execution", ["tier→model · 1024 tokens", "blind gpt-5.6-sol eval"]),
        ("Live trace panel", ["z, memories, Â, z', tier", "cost · memory count"]),
    ]
    bw, gap = 158, 26
    bx = 44
    for i, (t, body) in enumerate(boxes):
        s.box(bx, oy + 56, bw, 96, "#ffffff", JITRL_STROKE, rx=10, width_px=1.2)
        s.text(bx + bw / 2, oy + 80, t, size=12, anchor="middle", weight="bold")
        s.lines(bx + bw / 2, oy + 100, body, size=10.5, anchor="middle",
                fill=MUTED, lh=15)
        if i < len(boxes) - 1:
            s.arrow(bx + bw + 2, oy + 104, bx + bw + gap - 3, oy + 104,
                    color=JITRL_STROKE)
        bx += bw + gap
    s.box(44, oy + 168, 1112, 34, "#ffffff", JITRL_STROKE, rx=8, width_px=1.0,
          dash="5 4")
    s.text(600, oy + 190,
           "session memory: RAM only · reset per session · visitor text never persisted",
           size=11, anchor="middle", fill=JITRL_STROKE)

    # ---- offline lane
    fy = oy + 250
    s.box(24, fy, 1132, 200, "#f8fafc", NEUTRAL_STROKE, rx=14, width_px=1.6)
    s.text(44, fy + 34, "OFFLINE — frozen experiment assets (default mode)",
           size=14, weight="bold", fill=NEUTRAL_STROKE)
    fboxes = [
        ("Frozen logs", ["logs/s2_full_* (A/B/C0/C1)", "logs/s3_ablation_T8_*"]),
        ("Analyzers", ["analyze_results.py", "analyze_c1.py · analyze_ablation.py"]),
        ("Aggregates", ["results-summary.json · CSV", "eval/c1/ · eval/ablation/"]),
        ("SVG charts", ["5 + 4 + 4 charts", "cost / quality / gates / latency"]),
        ("README + Demo panels", ["A/B/C0/C1/T8 tables", "flips · cascade · 0/5 probes"]),
    ]
    fw, fgap = 190, 32
    fx = 44
    for i, (t, body) in enumerate(fboxes):
        s.box(fx, fy + 56, fw, 96, "#ffffff", NEUTRAL_STROKE, rx=10, width_px=1.2)
        s.text(fx + fw / 2, fy + 80, t, size=12, anchor="middle", weight="bold")
        s.lines(fx + fw / 2, fy + 100, body, size=10.5, anchor="middle",
                fill=MUTED, lh=15)
        if i < len(fboxes) - 1:
            s.arrow(fx + fw + 2, fy + 104, fx + fw + fgap - 3, fy + 104)
        fx += fw + fgap

    s.box(24, fy + 218, 1132, 36, WARN_FILL, WARN_STROKE, rx=8, width_px=1.2)
    s.text(590, fy + 242,
           "network failure in ONLINE mode ⇒ graceful fallback to these OFFLINE panels, never a blank screen",
           size=11.5, anchor="middle", fill=WARN_STROKE, weight="bold")
    return s.write("online-offline.svg")


# ----------------------------------------------------------------------
# 4. pilotdeck-integration.svg — upstream integration points
# ----------------------------------------------------------------------
def pilotdeck_integration() -> Path:
    s = Svg(1180, 640, "PilotDeck upstream integration points (future work)")
    header(s, "PilotDeck Upstream Integration — future work",
           "harness replicates the Router decision path today; upstream PR is out of the 22h budget")
    badge(s, 1010, 26, "FUTURE WORK", WARN_FILL, WARN_STROKE)

    # left: current runtime
    lx, lw = 24, 500
    s.box(lx, 96, lw, 470, "#f8fafc", NEUTRAL_STROKE, rx=14, width_px=1.6)
    s.text(lx + 20, 128, "PilotDeck Router runtime (today)", size=14,
           weight="bold")
    items = [
        ("RouterRuntime", ["invalidateSticky() at every user turn",
                           "sticky covers only the intra-turn tool loop"], JUDGE_FILL, JUDGE_STROKE),
        ("generateJudgePrompt", ["4-tier prompt · previousTier as TEXT hint",
                                 "no outcome / cost / quality feedback flows back"], JUDGE_FILL, JUDGE_STROKE),
        ("Judge LLM → tier → model", ["cross-turn experience vacuum:",
                                      "the judge re-decides each turn from scratch"], NEUTRAL_FILL, NEUTRAL_STROKE),
    ]
    iy = 150
    for t, body, fill, stroke in items:
        s.box(lx + 20, iy, lw - 40, 84, fill, stroke, rx=10, width_px=1.2)
        s.text(lx + 36, iy + 28, t, size=12.5, weight="bold")
        s.lines(lx + 36, iy + 48, body, size=11, fill=MUTED)
        iy += 106
    s.text(lx + 20, iy + 18,
           "⇒ between turns the router re-makes similar mistakes",
           size=11.5, fill=NEUTRAL_STROKE, weight="bold")

    # right: integration points
    rx_, rw = 560, 596
    s.box(rx_, 96, rw, 470, "#f0fdf4", JITRL_STROKE, rx=14, width_px=1.6)
    s.text(rx_ + 20, 128, "JitRL integration via extension points", size=14,
           weight="bold", fill=JITRL_STROKE)

    s.box(rx_ + 20, 150, rw - 40, 96, "#ffffff", JITRL_STROKE, rx=10,
          width_px=1.4)
    s.text(rx_ + 36, 178, "① CustomRouter.decide()", size=12.5, weight="bold")
    s.lines(rx_ + 36, 198,
            ["already-pluggable: may return provider/model and fully take over routing",
             "⇒ hosts the JitRL engine with ZERO Router-core changes"],
            size=11, fill=MUTED)

    missing = [
        ("② logprob passthrough (missing upstream)",
         ["the judge LLM call must expose per-tier logits",
          "today PilotDeck only surfaces the chosen tier"]),
        ("③ episode outcome callback (missing upstream)",
         ["quality / cost after the episode → engine.learn()",
          "feeds the (state, tier, reward) memory"]),
    ]
    my = 262
    for t, body in missing:
        s.box(rx_ + 20, my, rw - 40, 96, WARN_FILL, WARN_STROKE, rx=10,
              width_px=1.4)
        s.text(rx_ + 36, my + 28, t, size=12.5, weight="bold", fill=WARN_STROKE)
        s.lines(rx_ + 36, my + 48, body, size=11, fill=MUTED)
        my += 112

    s.box(rx_ + 20, my, rw - 40, 60, "#ffffff", JITRL_STROKE, rx=10,
          width_px=1.2, dash="6 4")
    s.lines(rx_ + 36, my + 24,
            ["meanwhile: this repo's independent Python harness replicates the",
             "Router decision path (same prompt format · 4 tiers · previousTier rules)"],
            size=11, fill=MUTED)

    s.arrow(lx + lw, 330, rx_ - 3, 330, color=NEUTRAL_STROKE)
    s.text((lx + lw + rx_) / 2, 318, "extend", size=10.5, anchor="middle",
           fill=MUTED)
    return s.write("pilotdeck-integration.svg")


def main() -> None:
    outputs = [architecture(), arms(), online_offline(), pilotdeck_integration()]
    for p in outputs:
        print(f"[make_diagrams] wrote {p.name}  ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
