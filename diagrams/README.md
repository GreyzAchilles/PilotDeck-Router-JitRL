# Technical diagrams

Hand-rolled, stdlib-generated SVGs (no mermaid / graphviz dependency). Referenced by `README.md` and the Web Demo.

| File | Content |
|---|---|
| `architecture.svg` | End-to-end arm-C decision path: user message → state extraction → frozen local Judge (Method B, 4 grammar-forced requests) → memory retrieval → V/Q/Â → min-neighbor gate → closed-form logit update → tier→model → execution → blind evaluator → reward → memory write-back, with the Judge-vs-JitRL latency split |
| `arms.svg` | What each arm adds: A static / B judge / C0 JitRL / C1 gated / T8-PM prompt-injection ablation (+ T8-LM note) |
| `online-offline.svg` | Online live decision loop (optional, session-isolated memory) vs offline frozen-experiment data path (default demo mode, graceful fallback) |
| `pilotdeck-integration.svg` | Upstream integration points: CustomRouter.decide() extension point + the two missing upstream channels (logprob passthrough, episode outcome callback) — future work |

Regenerate deterministically:

```bash
python diagrams/make_diagrams.py
```

Each SVG is self-contained (system font stack, no external references) and renders in plain `<img>` tags.
