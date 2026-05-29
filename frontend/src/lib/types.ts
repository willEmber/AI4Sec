export interface PaperResponse {
  paper_id: string;
  title: string;
  doi: string;
  venue: string;
  year: number;
  sci_rank: string;
  ccf_rank: string;
  created_at: string;
}

export interface PaperUploadResponse {
  paper_id: string;
  message: string;
}

export interface RunResponse {
  run_id: string;
  paper_id: string;
  mode: string;
  llm_model: string;
  status: string;
  error_msg: string;
  started_at: string;
  finished_at: string | null;
  user_question?: string;
  detected_intent?: string;
  current_step?: string;
  progress_json?: string;
}

export interface RecentRunResponse {
  run_id: string;
  paper_id: string;
  paper_title: string;
  mode: string;
  status: string;
  started_at: string;
  finished_at: string | null;
  current_step: string;
  user_question: string;
}

export interface ProgressEntry {
  step: string;
  status: string;
  [key: string]: unknown;
}

export interface RunOutputResponse {
  run_id: string;
  markdown: string;
  json_data: string;
}

export interface SSEEvent {
  event: string;
  data: Record<string, unknown>;
}

export type ReadingMode = "snap" | "lens" | "sphere" | "auto";

export interface RunCreate {
  paper_id: string;
  mode: ReadingMode;
  llm_model: string;
  language?: string;  // "en" | "zh"
  question?: string;  // non-empty only when mode === "auto"
  owner_token?: string;  // injected by the API client; scopes recent-runs visibility
}
