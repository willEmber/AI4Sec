import type { AgentEvent } from "@/lib/agent";

/** One tool call as the reader sees it: what ran, how it ended, how much it found. */
export interface ToolActivity {
  callId: string;
  tool: string;
  status: "running" | "ok" | "partial" | "unavailable" | "error";
  evidenceCount: number;
  note?: string;
  error?: { code: string; message: string };
  summary?: Record<string, unknown>;
}

/**
 * Fold one event into the tool list.
 *
 * This is deliberately the *only* place that turns events into activity. The
 * live view applies it event by event as the stream arrives; a reopened session
 * applies it to the events replayed from the server. Two implementations of the
 * same fold would eventually disagree, and the disagreement would show up as a
 * turn that looks like it did different work depending on when you looked at it.
 */
export function applyToolEvent(
  prev: ToolActivity[],
  event: AgentEvent,
): ToolActivity[] {
  const payload = event.payload || {};

  if (event.type === "tool.started") {
    return [
      ...prev,
      {
        callId: String(payload.call_id || `${payload.tool}-${event.seq}`),
        tool: String(payload.tool || ""),
        status: "running",
        evidenceCount: 0,
      },
    ];
  }

  if (event.type !== "tool.completed" && event.type !== "tool.failed") {
    return prev;
  }

  const toolName = String(payload.tool || "");
  // Match the most recent still-running call of this tool: the backend does not
  // correlate a result back to its call id, and a tool can be called more than
  // once in a turn.
  const index = prev.findLastIndex(
    (t) => t.tool === toolName && t.status === "running",
  );
  const entry: ToolActivity = {
    callId: index >= 0 ? prev[index].callId : `${toolName}-${event.seq}`,
    tool: toolName,
    status: (payload.status as ToolActivity["status"]) || "ok",
    evidenceCount: Number(payload.evidence_count || 0),
    note: payload.note ? String(payload.note) : undefined,
    error: payload.error as ToolActivity["error"],
    summary: payload.summary as Record<string, unknown> | undefined,
  };
  if (index < 0) return [...prev, entry];
  const next = [...prev];
  next[index] = entry;
  return next;
}

/** Rebuild a finished turn's tool activity from its replayed events. */
export function foldToolActivity(events: AgentEvent[]): ToolActivity[] {
  return events.reduce<ToolActivity[]>(applyToolEvent, []);
}

/**
 * A tool left `running` in a replayed log never reported back.
 *
 * That is not a rendering artefact — it is what an interrupted turn looks like,
 * and saying so is more useful than showing a spinner that will never stop.
 */
export function sealActivity(tools: ToolActivity[]): ToolActivity[] {
  return tools.map((t) =>
    t.status === "running" ? { ...t, status: "unavailable" as const } : t,
  );
}
