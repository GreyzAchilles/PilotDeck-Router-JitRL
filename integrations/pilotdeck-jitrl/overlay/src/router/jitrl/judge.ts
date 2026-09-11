/**
 * JitRL judge: produces a 4-tier logit vector z over {simple, medium,
 * complex, reasoning} for a user message.
 *
 * The Python reference (local_judge, Method B) extracts z per tier via a
 * force-emission request against a llama.cpp endpoint and reads the tier
 * name-token logprobs. The PilotDeck canonical ModelRuntime response carries
 * no logprobs, so this MVP implements the sanctioned conservative
 * equivalent: FOUR parallel scoring requests (one per tier) through the
 * injected ModelRuntime.complete — never routed through the Router — each
 * asking the judge model to rate the tier fit 0..10, converted to a logit
 * via the logit of the midpoint bin probability.
 *
 * If the judge is not configured, or ANY per-tier request fails / times out
 * / cannot be parsed, we fall back to a fully DETERMINISTIC heuristic judge
 * (a port of jitrl_core.mock_judge.MockJudgeClient) so callers always
 * receive a complete 4-tier numeric vector.
 */
import type {
  CanonicalModelRequest,
  CanonicalModelResponse,
  ModelRuntime,
} from "../../model/index.js";
import type { RouterModelRef } from "../config/schema.js";
import { chooseTier, type TierLogits } from "./policy.js";
import { intentClass } from "./state.js";
import { createSeededRandom, TIERS, type JitRLTier } from "./tiers.js";

export type JudgeResult = {
  /** all 4 tiers */
  tierLogits: TierLogits;
  chosenTier: string;
  rawOutput: string;
  latencyMs: number;
  /** "model": four scoring requests; "fallback": deterministic heuristic. */
  source: "model" | "fallback";
  /** Human-readable reason when source === "fallback". */
  fallbackReason?: string;
};

export type JitRLJudgeInput = {
  userMessage: string;
  /** Tier from the previous turn (continuation context for the judge). */
  previousTier?: string;
};

export type JitRLJudgeDeps = {
  /** Injected by RouterRuntime; the JitRL judge must NOT recurse through the Router. */
  judgeRuntime?: ModelRuntime;
  judge?: RouterModelRef;
  timeoutMs?: number;
  abortSignal?: AbortSignal;
  /** Seed for the deterministic fallback noise (default 42). */
  fallbackSeed?: number;
};

export const JITRL_JUDGE_DEFAULT_TIMEOUT_MS = 15_000;

const SCORE_MIN = 0;
const SCORE_MAX = 10;

const CODE_CUES: readonly string[] = [
  "```", "def ", "函数", "单元测试", "测试用例", "实现", "重构",
  "function", "implement", "unit test",
];
const DOC_CUES: readonly string[] = [
  "写一封", "文档", "报告", "通知", "公告", "周报", "邮件",
  "documentation", "report", "notice", "memo", "email",
];
const LONG_MESSAGE_CHARS = 300;
const CUE_MASS = 8.0;
const MEDIUM_BIAS = 1.0;
const PREV_TIER_MASS = 6.0;
const NOISE = 0.05;

export class JitRLJudgeClient {
  private readonly fallbackRng: () => number;

  constructor(private readonly deps: JitRLJudgeDeps = {}) {
    this.fallbackRng = createSeededRandom(deps.fallbackSeed ?? 42);
  }

  async judge(input: JitRLJudgeInput): Promise<JudgeResult> {
    const started = Date.now();
    const modelResult = await this.judgeViaModel(input, started);
    if (modelResult) {
      return modelResult;
    }
    return this.deterministicFallback(input, started);
  }

  // ------------------------------------------------------------- model path
  private async judgeViaModel(
    input: JitRLJudgeInput,
    started: number,
  ): Promise<JudgeResult | undefined> {
    const { judgeRuntime, judge } = this.deps;
    if (!judgeRuntime || !judge) {
      return undefined;
    }

    const tierResults = await Promise.all(
      TIERS.map((tier) =>
        scoreTierFit({
          judgeRuntime,
          judge,
          tier,
          userMessage: input.userMessage,
          previousTier: input.previousTier,
          timeoutMs: this.deps.timeoutMs ?? JITRL_JUDGE_DEFAULT_TIMEOUT_MS,
          abortSignal: this.deps.abortSignal,
        }),
      ),
    );

    const tierLogits: TierLogits = {};
    for (let index = 0; index < TIERS.length; index += 1) {
      const score = tierResults[index];
      if (score === undefined) {
        // Any per-tier failure degrades the WHOLE vector to the deterministic
        // fallback — mixing model scores with heuristic scores would produce
        // a vector with two different scales.
        return undefined;
      }
      tierLogits[TIERS[index]] = scoreToLogit(score);
    }

    const chosenTier = chooseTier(tierLogits, TIERS);
    return {
      tierLogits,
      chosenTier,
      rawOutput: `<tier>${chosenTier}</tier>`,
      latencyMs: Date.now() - started,
      source: "model",
    };
  }

  // ------------------------------------------------- deterministic fallback
  private deterministicFallback(input: JitRLJudgeInput, started: number): JudgeResult {
    const scores: TierLogits = {};
    for (const tier of TIERS) {
      scores[tier] = 0.0;
    }

    const previousTier = input.previousTier;
    if (intentClass(input.userMessage) === "continuation" && isTier(previousTier)) {
      scores[previousTier] += PREV_TIER_MASS;
    } else {
      // systematic surface bias: always extra preference for medium
      scores.medium += MEDIUM_BIAS;
      const low = input.userMessage.toLowerCase();
      if (CODE_CUES.some((cue) => low.includes(cue))) {
        scores.complex += CUE_MASS;
      } else if (DOC_CUES.some((cue) => low.includes(cue))) {
        scores.simple += CUE_MASS;
      }
      if (input.userMessage.length >= LONG_MESSAGE_CHARS) {
        scores.reasoning += CUE_MASS;
      }
    }

    for (const tier of TIERS) {
      scores[tier] += (this.fallbackRng() * 2 - 1) * NOISE;
    }

    const chosenTier = chooseTier(scores, TIERS);
    return {
      tierLogits: scores,
      chosenTier,
      rawOutput: `<tier>${chosenTier}</tier>`,
      latencyMs: Date.now() - started,
      source: "fallback",
      fallbackReason: this.deps.judgeRuntime && this.deps.judge
        ? "judge request failed or unparsable"
        : "no judge model configured",
    };
  }
}

function isTier(value: unknown): value is JitRLTier {
  return typeof value === "string" && (TIERS as readonly string[]).includes(value);
}

/** logit of the score's midpoint bin: p = (score + 0.5) / 11, z = ln(p / (1-p)). */
export function scoreToLogit(score: number): number {
  const clamped = Math.min(SCORE_MAX, Math.max(SCORE_MIN, Math.round(score)));
  const p = (clamped + 0.5) / (SCORE_MAX + 1);
  const safeP = Math.min(0.98, Math.max(0.02, p));
  return Math.log(safeP / (1 - safeP));
}

function buildTierScorePrompt(input: {
  tier: string;
  userMessage: string;
  previousTier?: string;
}): string {
  const continuation = input.previousTier
    ? `\nNote: the previous turn was classified as "${input.previousTier}". Short continuation or acknowledgment messages ("go", "continue", "ok", "好的", "继续") should keep that tier; only a genuinely new task justifies a different tier.\n`
    : "";
  return `You are a model-tier classifier for the PilotDeck router. Rate how well the tier "${input.tier}" fits the user message below.

Tiers:
- simple: simple greetings, confirmations, single-step Q&A, trivial file writes, remembering rules
- medium: single tool call, short text generation, 1-2 file read/write, code generation
- complex: needs sub-agent orchestration: parallel workstreams, delegation to specialized agents
- reasoning: deep single-agent work: multi-file operations, data analysis, multi-step workflows, web research, structured reports from many sources
${continuation}
Respond with ONLY a single integer score from 0 (terrible fit) to 10 (perfect fit). No other text.

User message:
"""
${input.userMessage}
"""`;
}

async function scoreTierFit(input: {
  judgeRuntime: ModelRuntime;
  judge: RouterModelRef;
  tier: string;
  userMessage: string;
  previousTier?: string;
  timeoutMs: number;
  abortSignal?: AbortSignal;
}): Promise<number | undefined> {
  const request: CanonicalModelRequest = {
    provider: input.judge.provider,
    model: input.judge.model,
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: buildTierScorePrompt({
              tier: input.tier,
              userMessage: input.userMessage,
              previousTier: input.previousTier,
            }),
          },
        ],
      },
    ],
    maxOutputTokens: 32,
    thinking: { enabled: false },
    stream: false,
  };

  let timeout: NodeJS.Timeout | undefined;
  const controller = new AbortController();
  const forwardAbort = () => controller.abort(input.abortSignal?.reason);
  input.abortSignal?.addEventListener("abort", forwardAbort, { once: true });
  try {
    if (input.abortSignal?.aborted) {
      return undefined;
    }
    const response = await Promise.race([
      input.judgeRuntime.complete(request, { signal: controller.signal }),
      new Promise<never>((_, reject) => {
        timeout = setTimeout(() => {
          controller.abort(new Error("jitrl judge timeout"));
          reject(new Error("jitrl judge timeout"));
        }, Math.max(500, input.timeoutMs));
      }),
    ]);
    return parseScore(response);
  } catch {
    return undefined;
  } finally {
    if (timeout) {
      clearTimeout(timeout);
    }
    input.abortSignal?.removeEventListener("abort", forwardAbort);
  }
}

function parseScore(response: CanonicalModelResponse): number | undefined {
  const text = response.content
    .filter((block) => block.type === "text")
    .map((block) => (block.type === "text" ? block.text : ""))
    .join("")
    .trim();
  const match = /-?\d+(?:\.\d+)?/.exec(text.replace(/```[a-z]*\n?/g, "").replace(/```/g, ""));
  if (!match) {
    return undefined;
  }
  const value = Number(match[0]);
  if (!Number.isFinite(value)) {
    return undefined;
  }
  if (value < SCORE_MIN || value > SCORE_MAX) {
    return undefined;
  }
  return value;
}
