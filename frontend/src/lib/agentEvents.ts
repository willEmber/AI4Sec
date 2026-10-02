import type { AgentEvent, SessionArtifact } from "@/lib/agent";

/** One intermediate step a long tool reported (a mode report, a parse). */
export interface ToolStep {
  step: string;
  status: string;
}

/** One tool call as the reader sees it: what ran, how it ended, how much it found. */
export interface ToolActivity {
  callId: string;
  tool: string;
  status: "running" | "ok" | "partial" | "unavailable" | "error";
  evidenceCount: number;
  note?: string;
  error?: { code: string; message: string };
  summary?: Record<string, unknown>;
  /** Progress steps, latest status per step, in the order they first appeared. */
  steps?: ToolStep[];
  /** Event timestamps, for how long the call took. */
  startedAt?: string;
  endedAt?: string;
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
        startedAt: event.timestamp,
      },
    ];
  }

  if (event.type === "tool.progress") {
    const toolName = String(payload.tool || "");
    const index = prev.findLastIndex(
      (t) => t.tool === toolName && t.status === "running",
    );
    if (index < 0) return prev;
    const step = String(payload.step || "");
    if (!step) return prev;
    const status = String(payload.status || "running");
    const steps = [...(prev[index].steps ?? [])];
    const existing = steps.findIndex((s) => s.step === step);
    if (existing >= 0) steps[existing] = { step, status };
    else steps.push({ step, status });
    const next = [...prev];
    next[index] = { ...prev[index], steps };
    return next;
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
    // Steps are kept: once a report is done, the reader may still want to see
    // what it went through.
    steps: index >= 0 ? prev[index].steps : undefined,
    startedAt: index >= 0 ? prev[index].startedAt : undefined,
    endedAt: event.timestamp,
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

/** The report a turn produced, as announced by its `artifact.created` event. */
export function artifactFromEvent(event: AgentEvent): SessionArtifact | null {
  if (event.type !== "artifact.created") return null;
  const p = event.payload || {};
  if (!p.run_id || !p.paper_id) return null;
  return {
    run_id: String(p.run_id),
    agent_run_id: event.run_id,
    paper_id: String(p.paper_id),
    paper_title: String(p.title || ""),
    mode: String(p.mode || ""),
    language: String(p.language || "en"),
    status: "done",
    created_at: event.timestamp,
  };
}

/** What a compaction event says, in the shape the divider renders. */
export interface CompactionNotice {
  seq: number;
  summarizedMessages: number;
  keptMessages: number;
  tokensBefore: number;
}

export function compactionFromEvent(event: AgentEvent): CompactionNotice | null {
  if (event.type !== "context.compacted") return null;
  const p = event.payload || {};
  return {
    seq: event.seq,
    summarizedMessages: Number(p.summarized_messages || 0),
    keptMessages: Number(p.kept_messages || 0),
    tokensBefore: Number(p.tokens_before || 0),
  };
}

/**
 * One entry of a live turn, in the order it happened.
 *
 * A turn is rarely "read, then answer": the model says what it is about to
 * check, calls a tool, says what it found, calls another. Keeping the order is
 * what lets the reader follow that, instead of seeing every tool first and all
 * the prose glued together after them. Tool entries point at `tools` by call
 * id, so `applyToolEvent` stays the one place a tool's state is decided.
 */
export type TimelineItem =
  | { kind: "text"; key: string; text: string }
  | { kind: "tool"; key: string; callId: string }
  | { kind: "artifact"; key: string; artifact: SessionArtifact }
  | { kind: "compaction"; key: string; notice: CompactionNotice };

/** Everything one turn has shown so far. */
export interface TurnState {
  /** All answer text, concatenated — what the server will persist. */
  answer: string;
  tools: ToolActivity[];
  timeline: TimelineItem[];
  artifacts: SessionArtifact[];
  compactions: CompactionNotice[];
}

export const EMPTY_TURN: TurnState = {
  answer: "",
  tools: [],
  timeline: [],
  artifacts: [],
  compactions: [],
};

/** Fold one event into a turn. Events that do not change what is shown return `state`. */
export function applyTurnEvent(state: TurnState, event: AgentEvent): TurnState {
  const payload = event.payload || {};

  switch (event.type) {
    case "message.delta": {
      const text = String(payload.text || "");
      if (!text) return state;
      const last = state.timeline.at(-1);
      const timeline =
        last?.kind === "text"
          ? [...state.timeline.slice(0, -1), { ...last, text: last.text + text }]
          : [...state.timeline, { kind: "text" as const, key: `text-${event.seq}`, text }];
      return { ...state, answer: state.answer + text, timeline };
    }

    case "tool.started":
    case "tool.progress":
    case "tool.completed":
    case "tool.failed": {
      const tools = applyToolEvent(state.tools, event);
      if (tools === state.tools) return state;
      // A call enters the timeline when it is first seen — normally at
      // `tool.started`, or at its result when the start never arrived.
      const placed = new Set(
        state.timeline.flatMap((item) => (item.kind === "tool" ? [item.callId] : [])),
      );
      const arrivals = tools
        .filter((tool) => !placed.has(tool.callId))
        .map((tool) => ({ kind: "tool" as const, key: `tool-${tool.callId}`, callId: tool.callId }));
      return {
        ...state,
        tools,
        timeline: arrivals.length ? [...state.timeline, ...arrivals] : state.timeline,
      };
    }

    case "artifact.created": {
      const artifact = artifactFromEvent(event);
      if (!artifact) return state;
      return {
        ...state,
        artifacts: [...state.artifacts, artifact],
        timeline: [...state.timeline, { kind: "artifact", key: `artifact-${event.seq}`, artifact }],
      };
    }

    case "context.compacted": {
      const notice = compactionFromEvent(event);
      if (!notice) return state;
      return {
        ...state,
        compactions: [...state.compactions, notice],
        timeline: [...state.timeline, { kind: "compaction", key: `compaction-${event.seq}`, notice }],
      };
    }

    default:
      return state;
  }
}

/**
 * Milliseconds since the epoch for a server timestamp, or NaN.
 *
 * Replayed events carry the database's UTC `YYYY-MM-DD HH:MM:SS`, which
 * `Date.parse` would read as local time; live ones may carry ISO 8601.
 */
export function parseServerTime(value: string | null | undefined): number {
  if (!value) return NaN;
  let iso = value.trim().replace(" ", "T");
  if (!/(Z|[+-]\d{2}:?\d{2})$/.test(iso)) iso += "Z";
  return Date.parse(iso);
}

/** How long something took, in ms, when both ends are known. */
export function elapsedMs(start?: string | null, end?: string | null): number | null {
  const a = parseServerTime(start);
  const b = parseServerTime(end);
  if (Number.isNaN(a) || Number.isNaN(b) || b < a) return null;
  return b - a;
}

/** `850ms` → "<1s", `4200` → "4s", `135000` → "2m 15s". */
export function formatDuration(ms: number | null): string {
  if (ms == null) return "";
  if (ms < 1000) return "<1s";
  const seconds = Math.round(ms / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  return rest ? `${minutes}m ${rest}s` : `${minutes}m`;
}
