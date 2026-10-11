import type { AgentMessage } from "./agent";

/**
 * One question and everything said after it until the next question.
 *
 * The turn, not the message, is what a reader navigates by: "the third thing I
 * asked". `index` is its position in the conversation, which is stable because
 * messages are only ever appended — unlike a message id, which changes when the
 * locally echoed question is replaced by the stored one.
 */
export interface Turn {
  index: number;
  /** The run the question started; "" for a question still being sent. */
  runId: string;
  question: AgentMessage | null;
  replies: AgentMessage[];
}

export function groupTurns(messages: AgentMessage[]): Turn[] {
  const turns: Turn[] = [];
  for (const message of messages) {
    if (message.role === "user") {
      turns.push({ index: turns.length, runId: message.run_id, question: message, replies: [] });
      continue;
    }
    let last = turns.at(-1);
    if (!last) {
      // An answer with no question before it: keep it, as a turn of its own.
      last = { index: 0, runId: message.run_id, question: null, replies: [] };
      turns.push(last);
    }
    last.replies.push(message);
    if (!last.runId) last.runId = message.run_id;
  }
  return turns;
}

export type DateBucket = "today" | "week" | "earlier";

/** Which heading a conversation last touched at `ms` belongs under, in local time. */
export function dateBucket(ms: number, now: number): DateBucket {
  if (!Number.isFinite(ms)) return "earlier";
  const startOfToday = new Date(now);
  startOfToday.setHours(0, 0, 0, 0);
  if (ms >= startOfToday.getTime()) return "today";
  if (ms >= startOfToday.getTime() - 6 * 24 * 3600 * 1000) return "week";
  return "earlier";
}

const EVIDENCE_ID = "ev_[0-9a-f]{20}";
// What weaker models write instead of `[ev_a][ev_b]`: several ids in one
// bracket, full-width brackets, padding, or the id in backticks.
const LOOSE_CITATION_RE = new RegExp(
  `[\\[【［]\\s*(\`?${EVIDENCE_ID}\`?(?:(?:\\s*[,，、;；]\\s*|\\s+)\`?${EVIDENCE_ID}\`?)*)\\s*[\\]】］]`,
  "g",
);
const EVIDENCE_ID_RE = new RegExp(EVIDENCE_ID, "g");

/**
 * Rewrite `[ev_a, ev_b]` and its relatives as `[ev_a][ev_b]`, the form
 * everything else reads. The backend counts those as citations either way (it
 * scans for known ids), so without this the page showed cited ids as raw text.
 */
export function normalizeCitations(content: string): string {
  return content.replace(LOOSE_CITATION_RE, (_match, ids: string) =>
    (ids.match(EVIDENCE_ID_RE) ?? []).map((id) => `[${id}]`).join(""),
  );
}

/**
 * Citation ids in an answer, numbered by first appearance, so one source
 * keeps one number throughout the answer and in the list under it.
 */
export function citationNumbers(content: string): Map<string, number> {
  const map = new Map<string, number>();
  const pattern = /\[(ev_[0-9a-f]{20})\]/g;
  const normalized = normalizeCitations(content);
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(normalized)) !== null) {
    if (!map.has(match[1])) map.set(match[1], map.size + 1);
  }
  return map;
}
