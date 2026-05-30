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

export interface ModelListResponse {
  models: string[];
  default: string;
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

// --- Knowledge base (Dify) ---

export type SearchMethod = "semantic_search" | "full_text_search" | "hybrid_search";

export interface LibraryDataset {
  id: string;
  name: string;
  description?: string;
  document_count?: number;
  word_count?: number;
  created_at?: number;
  [key: string]: unknown;
}

export interface LibraryDatasetsResponse {
  data: LibraryDataset[];
  has_more?: boolean;
  total?: number;
  page?: number;
  limit?: number;
}

export interface LibraryDocument {
  id: string;
  name: string;
  word_count?: number;
  tokens?: number;
  indexing_status?: string;
  display_status?: string;
  enabled?: boolean;
  created_at?: number;
  [key: string]: unknown;
}

export interface LibraryDocumentsResponse {
  data: LibraryDocument[];
  has_more?: boolean;
  total?: number;
  page?: number;
  limit?: number;
}

export interface LibrarySearchRecord {
  document_id: string;
  document_name: string;
  segment_id: string;
  content: string;
  score: number | null;
  metadata?: unknown;
}

export interface LibrarySearchResponse {
  query: string;
  records: LibrarySearchRecord[];
}

export interface LibraryMarkdownResponse {
  document_id: string;
  document_name: string;
  markdown_file?: string;
  content: string;
}

export interface LibrarySource {
  idx: number;
  document_id: string;
  document_name: string;
  segment_id: string;
  score: number | null;
}

export interface LibraryAskResponse {
  markdown: string;
  sources: LibrarySource[];
  blocks_used: number;
  search_method: string;
  question: string;
}

export interface LibrarySearchRequest {
  query: string;
  top_k?: number;
  score_threshold?: number | null;
  search_method?: SearchMethod | null;
  dataset_id?: string;
}

export interface LibraryAskRequest {
  question: string;
  top_k?: number;
  search_method?: SearchMethod | null;
  language?: string;
  llm_model?: string;
  dataset_id?: string;
}
