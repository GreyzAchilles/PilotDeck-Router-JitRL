/**
 * State abstraction: intent classification + task signature.
 * TypeScript port of `jitrl_core/state.py` (spec section 2, verbatim semantics).
 *
 * - intentClass(message): keyword rules over an ORDERED map, specificity
 *   descending, first match wins; fallback "other".
 * - taskSignature(message): normalization pipeline
 *     1. fenced code blocks -> <code>; 4+ space indented blocks -> <code>
 *     2. URL -> <url>; file path -> <path>; number -> <num>;
 *        quoted segment longer than 8 chars -> <str>
 *     3. lowercase; strip punctuation but keep placeholder angle brackets
 *     4. tokenize: latin/digit runs are tokens; CJK runs -> char bigrams
 * - Context (turn number, previous tier) lives in the router layer and is
 *   deliberately NOT part of the retrieval key.
 */

/** Ordered rules: specificity descending, first match wins (spec examples verbatim, plus English equivalents). */
export const INTENT_RULES: ReadonlyArray<readonly [string, readonly string[]]> = [
  ["code_gen", [
    "写测试", "单元测试", "实现一个", "写一个函数",
    "write tests", "unit test", "write a function", "implement", "test case",
  ]],
  ["data_analysis", [
    "分析", "数据", "统计", "对比",
    "analy", "statistic", "compare",
  ]],
  ["refactor", [
    "重构", "优化这段", "改写",
    "refactor", "optimize this", "rewrite",
  ]],
  ["chat_qa", [
    "你好", "谢谢", "记住", "嗨",
    "hello", "thanks", "hi", "remember",
  ]],
  ["continuation", [
    "继续", "好的", "嗯", "可以", "接着",
    "continue", "ok", "go on",
  ]],
  ["info_retrieval", [
    "检索", "查找", "综合",
    "retriev", "find", "search", "look up",
  ]],
  ["doc_writing", [
    "写一封", "文档", "报告", "通知",
    "doc", "report", "letter", "notice", "memo",
  ]],
];

export const INTENT_CLASSES: readonly string[] = [...INTENT_RULES.map(([name]) => name), "other"];

const CODE_FENCE_RE = /```[\s\S]*?```/g;
const URL_RE = /\b(?:https?:\/\/|www\.)\S+/gi;
const PATH_RE = /[\w./\\-]+\.\w{1,4}/g;
const QUOTED_RE = /"([^"]{9,})"|“([^”]{9,})”/g;
const NUM_RE = /\d+(?:\.\d+)?/g;
const TOKEN_RE = /<[a-z]+>|[a-z0-9_]+|[\u4e00-\u9fff]+/g;
const CJK_FULL_RE = /^[\u4e00-\u9fff]+$/;
const INDENTED_LINE_RE = /^(?: {4}|\t)/;

export function intentClass(message: string): string {
  const low = message.toLowerCase();
  for (const [bucket, keywords] of INTENT_RULES) {
    for (const keyword of keywords) {
      if (low.includes(keyword)) {
        return bucket;
      }
    }
  }
  return "other";
}

function replaceIndentedCode(text: string): string {
  /** Collapse blocks where >=1 line starts with 4+ spaces (or a tab) into <code>. */
  const lines = text.split("\n");
  const out: string[] = [];
  let inBlock = false;
  let blockOpened = false;
  for (const line of lines) {
    if (INDENTED_LINE_RE.test(line)) {
      if (!inBlock) {
        out.push("<code>");
        inBlock = true;
        blockOpened = true;
      }
      continue;
    }
    if (inBlock && line.trim() === "") {
      // blank line may belong to the block; keep absorbing
      continue;
    }
    inBlock = false;
    out.push(line);
  }
  if (!blockOpened) {
    return text;
  }
  return out.join("\n");
}

export function taskSignature(message: string): string[] {
  let text = message.replace(CODE_FENCE_RE, " <code> ");
  text = replaceIndentedCode(text);
  text = text.replace(URL_RE, " <url> ");
  text = text.replace(PATH_RE, " <path> ");
  text = text.replace(NUM_RE, " <num> ");
  text = text.replace(QUOTED_RE, " <str> ");
  text = text.toLowerCase();
  const tokens: string[] = [];
  for (const token of text.match(TOKEN_RE) ?? []) {
    if (CJK_FULL_RE.test(token)) {
      if (token.length === 1) {
        tokens.push(token);
      } else {
        for (let i = 0; i < token.length - 1; i += 1) {
          tokens.push(token.slice(i, i + 2));
        }
      }
    } else {
      tokens.push(token);
    }
  }
  return tokens;
}

export function signatureTokenSet(message: string): Set<string> {
  return new Set(taskSignature(message));
}

export function jaccard(a: ReadonlySet<string>, b: ReadonlySet<string>): number {
  /** Jaccard similarity of two token sets; empty-vs-empty is 1.0, one-empty 0.0. */
  if (a.size === 0 && b.size === 0) {
    return 1.0;
  }
  if (a.size === 0 || b.size === 0) {
    return 0.0;
  }
  let intersection = 0;
  for (const token of a) {
    if (b.has(token)) {
      intersection += 1;
    }
  }
  const union = a.size + b.size - intersection;
  return union === 0 ? 0.0 : intersection / union;
}
