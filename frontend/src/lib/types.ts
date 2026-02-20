export interface PaperResponse {
  paper_id: string;
  title: string;
  doi: string;
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

export type ReadingMode = "snap" | "lens" | "sphere";

export interface RunCreate {
  paper_id: string;
  mode: ReadingMode;
  llm_model: string;
}
