import type {
  PaperResponse,
  PaperUploadResponse,
  RunCreate,
  RunResponse,
  RunOutputResponse,
} from "./types";

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

export async function uploadPaper(file: File): Promise<PaperUploadResponse> {
  const form = new FormData();
  form.append("file", file);
  return request("/papers/upload", { method: "POST", body: form });
}

export async function getPaper(paperId: string): Promise<PaperResponse> {
  return request(`/papers/${paperId}`);
}

export async function listPapers(): Promise<PaperResponse[]> {
  return request("/papers");
}

export async function createRun(body: RunCreate): Promise<RunResponse> {
  return request("/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
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

export function getPaperPdfUrl(paperId: string): string {
  return `${API_BASE}/papers/${paperId}/pdf`;
}

export function getRunStreamUrl(runId: string): string {
  return `${BACKEND_SSE_BASE}/api/runs/${runId}/stream`;
}
