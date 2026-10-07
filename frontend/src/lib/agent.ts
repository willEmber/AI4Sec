/**
 * Client for the agent session API.
 *
 * Identity is a credential the *server* issues, unlike the browser-generated
 * `owner_token` used elsewhere: anyone can send any owner_token, so it can
 * scope a list but cannot decide who may read a session. The credential lives
 * in an HttpOnly cookie (`scholar_auth`) that every same-origin `/api` call
 * carries — a login session, or an anonymous one minted on first use.
 *
 * Before accounts the credential was kept in localStorage and sent as the
 * `X-Agent-Token` header. A browser that still has one sends it until the
 * server has copied it into the cookie, then forgets it.
 */

import { getOwnerToken } from "./owner";

const API_BASE = "/api";

// EventSource must reach the backend directly — the Next.js rewrite proxy
// buffers streaming responses, so SSE would arrive only at the end.
const BACKEND_SSE_BASE =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8001";

const AGENT_TOKEN_KEY = "scholar_agent_token";
export const AGENT_TOKEN_HEADER = "X-Agent-Token";

/** An API failure with the server's error code, when it sent one. */
export class AgentApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly detail: Record<string, unknown> = {},
  ) {
    super(message);
    this.name = "AgentApiError";
  }
}

/** Turn a failed response into an `AgentApiError`, keeping `detail.code`. */
export async function apiErrorFrom(res: Response): Promise<AgentApiError> {
  const text = await res.text();
  let detail: unknown = text;
  try {
    detail = (JSON.parse(text) as { detail?: unknown }).detail ?? text;
  } catch {
    // Not JSON; keep the text.
  }
  if (detail && typeof detail === "object") {
    const d = detail as Record<string, unknown>;
    return new AgentApiError(
      res.status,
      String(d.code || ""),
      String(d.message || `API ${res.status}`),
      d,
    );
  }
  return new AgentApiError(res.status, "", `API ${res.status}: ${String(detail)}`);
}

/** A failure in words the reader can act on: quota and login get their own. */
export function describeApiError(err: unknown, t: (key: string) => string): string {
  if (err instanceof AgentApiError) {
    if (err.code === "quota_exceeded") {
      return t(err.detail.kind === "anonymous" ? "auth.quotaExceededAnon" : "auth.quotaExceeded");
    }
    if (err.code === "login_required") return t("auth.loginRequired");
    return err.message;
  }
  return err instanceof Error ? err.message : String(err);
}

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
  /** "" for a global memory; otherwise only that project's conversations see it. */
  project_id: string;
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
  /** The research project the session belongs to; "" when none. */
  project_id: string;
  created_at: string;
  updated_at: string;
}

/** A group of conversations about one research question (P8). */
export interface AgentProject {
  project_id: string;
  title: string;
  description: string;
  status: "active" | "archived";
  session_count: number;
  created_at: string;
  updated_at: string;
}

export interface ProjectDetail {
  project: AgentProject;
  sessions: AgentSession[];
  /** Derived from the project's sessions, one row per work. */
  papers: SessionPaper[];
  memories: AgentMemory[];
}

/** One earlier question and answer, with the evidence the answer cited. */
export interface RecalledTurn {
  session_id: string;
  session_title: string;
  project_id: string;
  run_id: string;
  asked_at: string;
  question: string;
  answer_excerpt: string;
  evidence: {
    evidence_id: string;
    source_level: string;
    paper_title: string;
    page: number | null;
    section: string;
    quote: string;
  }[];
}

export interface SessionDetail {
  session: AgentSession;
  messages: AgentMessage[];
  papers: SessionPaper[];
  runs: AgentRun[];
  artifacts: SessionArtifact[];
  context: SessionContextStats;
  last_event_seq: number;
  /** The session's project, and that project's papers this session lacks. */
  project: AgentProject | null;
  project_papers: SessionPaper[];
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

/**
 * The pre-accounts credential, if this browser still holds one. Returns ""
 * during SSR or when storage is blocked. Nothing new is ever stored here.
 */
export function getAgentToken(): string {
  if (typeof window === "undefined") return "";
  try {
    return window.localStorage.getItem(AGENT_TOKEN_KEY) || "";
  } catch {
    return "";
  }
}

/** Drop the pre-accounts credential once the cookie carries the identity. */
export function forgetLegacyAgentToken(): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(AGENT_TOKEN_KEY);
  } catch {
    // Storage blocked: there was nothing stored either.
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

  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers,
    credentials: "same-origin",
  });
  if (!res.ok) throw await apiErrorFrom(res);
  return res.json();
}

export async function createSession(input: {
  title?: string;
  language?: "zh" | "en";
  llm_model?: string;
  paper_ids?: string[];
  project_id?: string;
}): Promise<{ session_id: string; thread_id: string; created_at: string }> {
  return agentRequest("/agent/sessions", {
    method: "POST",
    body: JSON.stringify({
      title: input.title || "",
      language: input.language || "zh",
      llm_model: input.llm_model || "",
      paper_ids: input.paper_ids || [],
      project_id: input.project_id || "",
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
  project_id?: string;
}): Promise<AgentMemory> {
  return agentRequest("/agent/memories", {
    method: "POST",
    body: JSON.stringify({
      content: input.content,
      kind: input.kind || "preference",
      project_id: input.project_id || "",
    }),
  });
}

export async function deleteMemory(memoryId: string): Promise<void> {
  await agentRequest(`/agent/memories/${memoryId}`, { method: "DELETE" });
}

/** Recent sessions; `projectId` narrows to one project ("" = sessions in none). */
export async function listSessions(projectId?: string): Promise<{ sessions: AgentSession[] }> {
  const query = projectId === undefined ? "" : `?project_id=${encodeURIComponent(projectId)}`;
  return agentRequest(`/agent/sessions${query}`);
}

/** Rename a session, or move it into a project (`project_id: ""` takes it out). */
export async function updateSession(
  sessionId: string,
  input: { project_id?: string; title?: string },
): Promise<{ session: AgentSession }> {
  return agentRequest(`/agent/sessions/${sessionId}`, {
    method: "PATCH",
    body: JSON.stringify(input),
  });
}

export async function listProjects(includeArchived = false): Promise<{ projects: AgentProject[] }> {
  return agentRequest(`/agent/projects${includeArchived ? "?include_archived=true" : ""}`);
}

export async function createProject(input: {
  title: string;
  description?: string;
}): Promise<{ project: AgentProject }> {
  return agentRequest("/agent/projects", {
    method: "POST",
    body: JSON.stringify({ title: input.title, description: input.description || "" }),
  });
}

export async function getProject(projectId: string): Promise<ProjectDetail> {
  return agentRequest(`/agent/projects/${projectId}`);
}

export async function updateProject(
  projectId: string,
  input: { title?: string; description?: string; status?: AgentProject["status"] },
): Promise<{ project: AgentProject }> {
  return agentRequest(`/agent/projects/${projectId}`, {
    method: "PATCH",
    body: JSON.stringify(input),
  });
}

/** Delete a conversation with its messages, evidence and the reports made in it. */
export async function deleteSession(sessionId: string): Promise<void> {
  await agentRequest(`/agent/sessions/${sessionId}`, { method: "DELETE" });
}

/** Delete a project; its conversations are moved out of it, or deleted with it. */
export async function deleteProject(
  projectId: string,
  sessions: "keep" | "delete",
): Promise<{ sessions_deleted: number }> {
  return agentRequest(`/agent/projects/${projectId}?sessions=${sessions}`, { method: "DELETE" });
}

/** The same search the agent's recall uses, over the caller's conversations. */
export async function searchConversations(
  query: string,
  projectId?: string,
): Promise<{ query: string; turns: RecalledTurn[] }> {
  const params = new URLSearchParams({ q: query });
  if (projectId) params.set("project_id", projectId);
  return agentRequest(`/agent/search?${params}`);
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
 * The stream goes straight to the backend, where the same-origin cookie does
 * not reach, and `EventSource` cannot set headers. So each (re)connect first
 * asks — through the proxy, with the cookie — for a ticket that opens this one
 * run for a few minutes; `after` carries the resume point.
 */
export async function getEventStreamUrl(runId: string, afterSeq = 0): Promise<string> {
  const { ticket } = await agentRequest<{ ticket: string }>(
    `/agent/runs/${runId}/stream-ticket`,
    { method: "POST" },
  );
  const params = new URLSearchParams({ after: String(afterSeq), ticket });
  return `${BACKEND_SSE_BASE}/api/agent/runs/${runId}/events?${params}`;
}

/** Ids the backend handed the model, as they appear in an answer: `[ev_xxx]`. */
export const CITATION_PATTERN = /\[(ev_[0-9a-f]{20})\]/g;
