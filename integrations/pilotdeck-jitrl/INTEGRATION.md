# PilotDeck JitRL Integration

## Baseline

- Upstream repository: `https://github.com/OpenBMB/PilotDeck.git`
- Upstream commit: `cfc4d1779228f91fececc5d6705c14dab5b7ef2f`
- Upstream tag: `v2026.09.10`
- Integration type: builtin `PilotDeckCustomRouter` / `RouterContribution`

## Contents

- `pilotdeck-jitrl.patch`: complete Git patch against the baseline commit, including source changes and tests.
- `overlay/`: the resulting source and test files laid out with paths relative to the PilotDeck repository root.

The patch is the preferred way to apply the integration because it preserves changes to existing PilotDeck files. The overlay is included for review and archival.

## Implemented behavior

- Native TypeScript port of the Python JitRL engine.
- Four canonical tiers: `simple`, `medium`, `complex`, `reasoning`.
- Ordered intent classification and normalized task signatures.
- Intent-filtered top-k Jaccard experience retrieval.
- V/Q/normalized-advantage estimation.
- Closed-form logit modulation: `z' = z + beta * A_norm`.
- `minNeighbors` modulation gate and deterministic tier tie-breaking.
- Four parallel direct judge-model scoring calls through `ModelRuntime.complete`; deterministic fallback when unavailable or invalid.
- Optional `PilotDeckCustomRouter.onTurnOutcome` callback after custom-routed execution.
- Reward: `0.6 * quality + 0.3 * costSaving`.
- Direct evaluator-model quality review without recursively entering Router.
- No memory write after evaluator failure or model execution failure.
- Atomic, throttled memory persistence at `<pilotHome>/router/jitrl-memory.json` by default.
- Builtin `jitrl` plugin registration and preservation of programmatic contributions across plugin refresh.
- Backward compatibility with existing `{ extensionId }` custom-router configuration.

## Apply

From a PilotDeck checkout at the baseline commit:

```bash
git apply --check /path/to/pilotdeck-jitrl.patch
git apply /path/to/pilotdeck-jitrl.patch
```

If the upstream checkout uses sparse-checkout, ensure `tests/` is included before applying, or apply the patch with an index/worktree configuration that permits those paths.

## Configuration example

```yaml
router:
  enabled: true
  customRouter:
    extensionId: jitrl
    judge: CPA/minicpm5-1b
    evaluator: CPA/gpt-5.6-sol
    tiers:
      simple: CPA/glm-5.3-flash
      medium: CPA/gpt-5.4-mini
      complex: CPA/gpt-5.6-sol
      reasoning: CPA/glm-5.3
    hyperparams:
      k: 10
      beta: 5.0
      lam: 0.05
      alpha: 5.0
      jaccardThreshold: 0.5
      zMin: -10.0
      memoryCap: 5000
      seed: 42
      minNeighbors: 3
    judgeTimeoutMs: 15000
    evalTimeoutMs: 15000
```

All provider/model references must already exist in the target PilotDeck model configuration.

## Verification performed

From the local upstream checkout:

```text
npx tsc -p tsconfig.json
PASS

node --test --test-force-exit --test-timeout 60000 "dist/tests/router/jitrl/*.test.js"
59 passed, 0 failed

python -m pytest harness jitrl_core local_judge eval -q
261 passed

git diff --check
PASS (line-ending warning only)
```

The upstream snapshot is partial and does not include `scripts/check-node-runtime.mjs`, so its standard `npm test` command stops in the pre-existing `prebuild` step. The TypeScript compiler and generated JitRL test suite were run directly instead.

## Known limitations

- The TypeScript seeded RNG uses Mulberry32 rather than Python's MT19937. Algorithm semantics are equivalent, but exploration draws are not bit-identical by default.
- Judge classification uses four model scoring requests because PilotDeck's canonical response protocol does not currently expose token log probabilities.
- A successful routed turn adds one evaluator request and therefore additional latency and cost.
- Execution failures are reported through `onTurnOutcome` but are not learned because there is no reliable quality score.
