import type {
  ModelListResponse,
  PaperResponse,
  PaperUploadResponse,
  RecentRunResponse,
  RunCreate,
  RunResponse,
  RunOutputResponse,
  LibraryDatasetsResponse,
  LibraryDocumentsResponse,
  LibraryMarkdownResponse,
  LibrarySearchRequest,
  LibrarySearchResponse,
  LibraryAskRequest,
  LibraryAskResponse,
} from "./types";
import { getOwnerToken } from "./owner";

const API_BASE = "/api";

// SSE/EventSource must connect directly to backend — Next.js rewrite proxy
// buffers streaming responses, preventing real-time SSE delivery.
const BACKEND_SSE_BASE =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8001";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json();
}

export async function listModels(): Promise<ModelListResponse> {
  return request("/models");
}

export async function uploadPaper(file: File): Promise<PaperUploadResponse> {
  const form = new FormData();
  form.append("file", file);
  return request("/papers/upload", { method: "POST", body: form });
}

export async function getPaper(paperId: string): Promise<PaperResponse> {
  return request(`/papers/${paperId}`);
}

export async function createRun(body: RunCreate): Promise<RunResponse> {
  return request("/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...body, owner_token: getOwnerToken() }),
  });
}

export async function getRun(runId: string): Promise<RunResponse> {
  return request(`/runs/${runId}`);
}

export async function getRunOutput(runId: string): Promise<RunOutputResponse> {
  return request(`/runs/${runId}/output`);
}

export async function listPaperRuns(paperId: string): Promise<RunResponse[]> {
  return request(`/papers/${paperId}/runs`);
}

export async function listRecentRuns(
  limit = 20,
  activeOnly = false,
): Promise<RecentRunResponse[]> {
  const qs = new URLSearchParams({
    limit: String(limit),
    active_only: activeOnly ? "true" : "false",
    owner_token: getOwnerToken(),
  });
  return request(`/runs/recent?${qs.toString()}`);
}

export async function dismissRun(runId: string): Promise<RunResponse> {
  const qs = new URLSearchParams({ owner_token: getOwnerToken() });
  return request(`/runs/${runId}/dismiss?${qs.toString()}`, { method: "POST" });
}

export function getPaperPdfUrl(paperId: string): string {
  return `${API_BASE}/papers/${paperId}/pdf`;
}

export function getRunStreamUrl(runId: string): string {
  return `${BACKEND_SSE_BASE}/api/runs/${runId}/stream`;
}

// --- Knowledge base (Dify) ---

export async function listLibraryDatasets(
  page = 1,
  limit = 20,
): Promise<LibraryDatasetsResponse> {
  const qs = new URLSearchParams({ page: String(page), limit: String(limit) });
  return request(`/library/datasets?${qs.toString()}`);
}

export async function listLibraryDocuments(opts: {
  datasetId?: string;
  page?: number;
  limit?: number;
} = {}): Promise<LibraryDocumentsResponse> {
  const qs = new URLSearchParams({
    page: String(opts.page ?? 1),
    limit: String(opts.limit ?? 20),
  });
  if (opts.datasetId) qs.set("dataset_id", opts.datasetId);
  return request(`/library/documents?${qs.toString()}`);
}

export async function getLibraryMarkdown(
  documentId: string,
  datasetId?: string,
): Promise<LibraryMarkdownResponse> {
  const qs = new URLSearchParams();
  if (datasetId) qs.set("dataset_id", datasetId);
  const suffix = qs.toString() ? `?${qs.toString()}` : "";
  return request(`/library/documents/${encodeURIComponent(documentId)}/markdown${suffix}`);
}

export async function searchLibrary(
  body: LibrarySearchRequest,
): Promise<LibrarySearchResponse> {
  return request("/library/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function askLibrary(
  body: LibraryAskRequest,
): Promise<LibraryAskResponse> {
  return request("/library/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
