import type {
  ModelListResponse,
  PaperResponse,
  PaperUploadResponse,
  RecentRunResponse,
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
import { apiErrorFrom } from "./agent";
import { getOwnerToken } from "./owner";

const API_BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) throw await apiErrorFrom(res);
  return res.json();
}

export async function listModels(): Promise<ModelListResponse> {
  return request("/models");
}

export async function recordVisit(path: string): Promise<void> {
  const ownerToken = getOwnerToken();
  if (!ownerToken) return;
  await request<{ recorded: boolean }>("/traffic/visit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ owner_token: ownerToken, path }),
  });
}

export async function uploadPaper(file: File): Promise<PaperUploadResponse> {
  const form = new FormData();
  form.append("file", file);
  return request("/papers/upload", { method: "POST", body: form });
}

export async function getPaper(paperId: string): Promise<PaperResponse> {
  return request(`/papers/${paperId}`);
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
  days?: number,
): Promise<RecentRunResponse[]> {
  const qs = new URLSearchParams({
    limit: String(limit),
    active_only: activeOnly ? "true" : "false",
    owner_token: getOwnerToken(),
  });
  if (days !== undefined) qs.set("days", String(days));
  return request(`/runs/recent?${qs.toString()}`);
}

/** Delete a run and its report. Links to it stop working. */
export async function deleteRun(runId: string): Promise<void> {
  const qs = new URLSearchParams({ owner_token: getOwnerToken() });
  await request(`/runs/${runId}?${qs.toString()}`, { method: "DELETE" });
}

export function getPaperPdfUrl(paperId: string): string {
  return `${API_BASE}/papers/${paperId}/pdf`;
}

// Zotero RDF import bundle (.zip: item + report note + PDF) for a run.
export function getZoteroBundleUrl(runId: string): string {
  return `${API_BASE}/runs/${runId}/zotero-bundle`;
}

// Portable Markdown export. The backend rewrites figure links to their public
// object-storage URLs (uploading on demand) when R2 is configured.
export function getMarkdownExportUrl(runId: string): string {
  return `${API_BASE}/runs/${runId}/markdown`;
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
