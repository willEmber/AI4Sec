/**
 * Client for the agent session API.
 *
 * Identity is a credential the *server* signs, unlike the browser-generated
 * `owner_token` used elsewhere: anyone can send any owner_token, so it can
 * scope a list but cannot decide who may read a session. Creating a session
 * mints the credential and returns it in the `X-Agent-Token` response header;
 * every later call sends it back.
 */

import { getOwnerToken } from "./owner";

const API_BASE = "/api";

// EventSource must reach the backend directly — the Next.js rewrite proxy
// buffers streaming responses, so SSE would arrive only at the end.
const BACKEND_SSE_BASE =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8001";

const AGENT_TOKEN_KEY = "scholar_agent_token";
export const AGENT_TOKEN_HEADER = "X-Agent-Token";

export type EventType =
  | "run.started"
  | "message.delta"
  | "tool.started"
  | "tool.progress"
  | "tool.completed"
  | "tool.failed"
  | "paper.added"
  | "artifact.created"
  | "context.compacted"
  | "memory.saved"
  | "run.completed"
  | "run.failed"
  | "run.cancelled";

/** Every named event the stream can carry; the hook subscribes to all of them. */
export const ALL_EVENT_TYPES: EventType[] = [
  "run.started",
  "message.delta",
  "tool.started",
  "tool.progress",
  "tool.completed",
  "tool.failed",
  "paper.added",
  "artifact.created",
  "context.compacted",
  "memory.saved",
  "run.completed",
  "run.failed",
  "run.cancelled",
];

/** What the reader can pick in the composer. `auto` leaves it to the agent. */
export type AgentMode = "auto" | "snap" | "lens" | "sphere";
export const AGENT_MODES: AgentMode[] = ["auto", "snap", "lens", "sphere"];

/** A mode report produced inside a conversation — the same row the report page renders. */
export interface SessionArtifact {
  run_id: string;
  agent_run_id: string;
  paper_id: string;
  paper_title: string;
  mode: string;
  language: string;
  status: string;
  created_at: string;
}

export type MemoryKind = "preference" | "fact" | "project" | "instruction";

export interface AgentMemory {
  memory_id: string;
  kind: MemoryKind;
  content: string;
  source_session_id: string;
  source_run_id: string;
  active: boolean;
  created_at: string;
  updated_at: string;
}

export interface SessionContextStats {
  compactions: number;
  last_turn_tokens: number | null;
}

export interface AgentEvent {
  schema_version: number;
  session_id: string;
  run_id: string;
  seq: number;
  type: EventType;
  timestamp: string;
  payload: Record<string, unknown>;
}

export interface AgentMessage {
  message_id: string;
  session_id: string;
  run_id: string;
  role: "user" | "assistant" | "system";
  content: string;
  citations: string[];
  seq: number;
  created_at: string;
}

export interface SessionPaper {
  session_id: string;
  literature_id: string;
  paper_id: string;
  availability:
    | "candidate"
    | "downloading"
    | "pdf_ready"
    | "parsed"
    | "unavailable";
  added_by: string;
  note: string;
  title: string;
  year: number;
  year_known: boolean;
  venue: string;
  doi: string;
  arxiv_id: string;
  updated_at: string;
}

export interface AgentRun {
  run_id: string;
  session_id: string;
  status: "pending" | "running" | "done" | "failed" | "cancelled";
  cancel_requested: boolean;
  error_code: string;
  error_msg: string;
  usage: Record<string, unknown>;
  started_at: string;
  finished_at: string | null;
  /** The worker executing this turn, and when it last reported in. */
  worker_id?: string;
  heartbeat_at?: string | null;
}

export interface AgentSession {
  session_id: string;
  thread_id: string;
  title: string;
  language: "zh" | "en";
  llm_model: string;
  config?: Record<string, unknown>;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface SessionDetail {
  session: AgentSession;
  messages: AgentMessage[];
  papers: SessionPaper[];
  runs: AgentRun[];
  artifacts: SessionArtifact[];
  context: SessionContextStats;
  last_event_seq: number;
}

export interface EvidenceLocator {
  section_id: string;
  section_path: string;
  page_index: number | null;
  page_end_index: number | null;
  block_start: number | null;
  block_end: number | null;
  node_id: string;
  bbox: number[];
  /** 1-based, for display. The backend owns the conversion. */
  page_label: number | null;
  page_end_label: number | null;
}

export interface Evidence {
  evidence_id: string;
  session_id: string;
  literature_id: string;
  paper_id: string;
  parse_version: string;
  source_level: "fulltext" | "abstract" | "metadata" | "external_web";
  locator: EvidenceLocator;
  quote: string;
  content_hash: string;
  source_url: string;
  provider: string;
  retrieved_at: string;
  literature?: {
    literature_id: string;
    title: string;
    year: number | null;
    year_known: boolean;
    venue: string;
    doi: string;
    arxiv_id: string;
  };
  parse_status?: {
    evidence_id: string;
    parse_version: string;
    is_current_version: boolean;
    current_version: string;
    still_present: boolean | null;
  };
}

/** Read the stored credential. Returns "" during SSR or when storage is blocked. */
export function getAgentToken(): string {
  if (typeof window === "undefined") return "";
  try {
    return window.localStorage.getItem(AGENT_TOKEN_KEY) || "";
  } catch {
    return "";
  }
}

function storeAgentToken(token: string): void {
  if (!token || typeof window === "undefined") return;
  try {
    window.localStorage.setItem(AGENT_TOKEN_KEY, token);
  } catch {
    // Private mode: the credential lives for this page only. The session still
    // works; it just will not be recoverable after a reload.
  }
}

async function agentRequest<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const headers = new Headers(init.headers);
  const token = getAgentToken();
  if (token) headers.set(AGENT_TOKEN_HEADER, token);
  if (init.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const res = await fetch(`${API_BASE}${path}`, { ...init, headers });

  // A minted credential arrives on the response, not in the body.
  const issued = res.headers.get(AGENT_TOKEN_HEADER);
  if (issued) storeAgentToken(issued);

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json();
}

export async function createSession(input: {
  title?: string;
  language?: "zh" | "en";
  llm_model?: string;
  paper_ids?: string[];
}): Promise<{ session_id: string; thread_id: string; created_at: string }> {
  return agentRequest("/agent/sessions", {
    method: "POST",
    body: JSON.stringify({
      title: input.title || "",
      language: input.language || "zh",
      llm_model: input.llm_model || "",
      paper_ids: input.paper_ids || [],
      // Lets mode reports made in this conversation appear in the compare
      // matrix beside reports made from the classic upload page.
      owner_token: getOwnerToken(),
    }),
  });
}

export async function attachPapers(
  sessionId: string,
  paperIds: string[],
): Promise<{ session_id: string; attached: string[]; papers: SessionPaper[] }> {
  return agentRequest(`/agent/sessions/${sessionId}/papers`, {
    method: "POST",
    body: JSON.stringify({ paper_ids: paperIds }),
  });
}

export async function listMemories(): Promise<{ memories: AgentMemory[] }> {
  return agentRequest("/agent/memories");
}

export async function createMemory(input: {
  content: string;
  kind?: MemoryKind;
}): Promise<AgentMemory> {
  return agentRequest("/agent/memories", {
    method: "POST",
    body: JSON.stringify({ content: input.content, kind: input.kind || "preference" }),
  });
}

export async function deleteMemory(memoryId: string): Promise<void> {
  await agentRequest(`/agent/memories/${memoryId}`, { method: "DELETE" });
}

export async function listSessions(): Promise<{ sessions: AgentSession[] }> {
  return agentRequest("/agent/sessions");
}

export async function getSession(sessionId: string): Promise<SessionDetail> {
  return agentRequest(`/agent/sessions/${sessionId}`);
}

export async function postMessage(
  sessionId: string,
  input: {
    content: string;
    clientRequestId: string;
    paperIds?: string[];
    mode?: AgentMode;
  },
): Promise<{
  run_id: string;
  session_id: string;
  status: string;
  deduplicated: boolean;
}> {
  return agentRequest(`/agent/sessions/${sessionId}/messages`, {
    method: "POST",
    body: JSON.stringify({
      content: input.content,
      client_request_id: input.clientRequestId,
      paper_ids: input.paperIds || [],
      mode: input.mode || "auto",
      owner_token: getOwnerToken(),
    }),
  });
}

export async function cancelRun(runId: string): Promise<void> {
  await agentRequest(`/agent/runs/${runId}/cancel`, { method: "POST" });
}

export interface RunActivity {
  run_id: string;
  status: AgentRun["status"];
  error_code: string;
  events: AgentEvent[];
}

/**
 * A past turn's events, so reopening a session shows what it read.
 *
 * `message.delta` is excluded server-side — the answer is already a stored
 * message, and replaying its fragments would just rebuild text that is on
 * screen.
 */
export async function getRunActivity(runId: string): Promise<RunActivity> {
  return agentRequest(`/agent/runs/${runId}/activity`);
}

export async function getEvidence(evidenceId: string): Promise<Evidence> {
  return agentRequest(`/agent/evidence/${evidenceId}`);
}

/**
 * URL for a run's event stream.
 *
 * `EventSource` cannot set headers, so the credential goes in the query string
 * and `after` carries the resume point. Both are what the backend expects.
 */
export function getEventStreamUrl(runId: string, afterSeq = 0): string {
  const params = new URLSearchParams({
    after: String(afterSeq),
    token: getAgentToken(),
  });
  return `${BACKEND_SSE_BASE}/api/agent/runs/${runId}/events?${params}`;
}

/** Ids the backend handed the model, as they appear in an answer: `[ev_xxx]`. */
export const CITATION_PATTERN = /\[(ev_[0-9a-f]{20})\]/g;
