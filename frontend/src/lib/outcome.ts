/**
 * How a turn ended, when it did not simply finish (P13).
 *
 * An answer can now be all there is of a turn that failed, was stopped, was
 * cut at the model's output ceiling, or was written because a budget ran out.
 * The server says which on the message (`completion`, `stop_reason`); this
 * turns that into the one line shown under the answer and what the reader can
 * do about it. Pure, so the live view and a reopened session cannot disagree.
 */

/** Codes a failed turn is closed with. The server sends the code, never its prose. */
const ERROR_CODES = new Set([
  "rate_limited",
  "context_too_long",
  "timeout",
  "content_filtered",
  "empty_response",
  "upstream_error",
  "internal_error",
]);

/** Ceilings that end a turn's gathering. */
const LIMITS = new Set([
  "max_tool_calls",
  "max_llm_calls",
  "max_tokens",
  "max_wall_seconds",
  "tool_failures",
]);

/** The i18n key for a failure code; an unknown code says nothing rather than something wrong. */
export function errorTextKey(code: string | null | undefined): string {
  return ERROR_CODES.has(code || "") ? `chat.error.${code}` : "chat.error.unknown";
}

export function limitTextKey(reason: string | null | undefined): string {
  return LIMITS.has(reason || "") ? `chat.limit.${reason}` : "chat.limit.unknown";
}

export interface AnswerOutcome {
  tone: "error" | "muted";
  /** i18n key of the sentence. */
  key: string;
  /** i18n key of what fills the sentence's `{reason}`, when it has one. */
  reasonKey?: string;
  action: "retry" | "continue" | null;
}

export function answerOutcome(message: {
  completion?: string | null;
  stop_reason?: string | null;
}): AnswerOutcome | null {
  const completion = message.completion || "complete";
  const reason = message.stop_reason || "";

  if (completion === "failed") {
    return { tone: "error", key: "chat.answer.failed", reasonKey: errorTextKey(reason), action: "retry" };
  }
  if (completion === "truncated") {
    return { tone: "muted", key: "chat.answer.truncated", action: "continue" };
  }
  if (completion === "stopped") {
    return reason === "cancelled"
      ? { tone: "muted", key: "chat.answer.stopped", action: "retry" }
      : { tone: "muted", key: "chat.answer.cut_off", reasonKey: limitTextKey(reason), action: "retry" };
  }
  // A whole answer, written because the turn ran out of room to look further.
  if (reason) {
    return { tone: "muted", key: "chat.answer.wrapped_up", reasonKey: limitTextKey(reason), action: null };
  }
  return null;
}
