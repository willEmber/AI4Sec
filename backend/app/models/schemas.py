from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


# --- Request models ---

class RunCreate(BaseModel):
    paper_id: str
    mode: Literal["snap", "lens", "sphere", "auto"] = "snap"
    llm_model: str = ""
    language: str = "en"        # en | zh
    question: str = ""          # required (non-empty) when mode == "auto"
    owner_token: str = ""       # per-browser token; scopes which runs the client sees


SearchMethod = Literal["semantic_search", "full_text_search", "hybrid_search"]


class LibrarySearchRequest(BaseModel):
    query: str
    top_k: int = 10
    score_threshold: float | None = None
    search_method: SearchMethod | None = None   # defaults to DIFY_SEARCH_METHOD
    dataset_id: str = ""                          # empty → project/proxy default dataset


class LibraryAskRequest(BaseModel):
    question: str
    top_k: int = 10
    search_method: SearchMethod | None = None   # defaults to DIFY_SEARCH_METHOD
    language: str = "en"        # en | zh
    llm_model: str = ""
    dataset_id: str = ""


# --- Response models ---

class PaperResponse(BaseModel):
    paper_id: str
    title: str
    doi: str
    venue: str = ""
    year: int = 0
    sci_rank: str = ""
    ccf_rank: str = ""
    created_at: str


class PaperUploadResponse(BaseModel):
    paper_id: str
    message: str


class ModelListResponse(BaseModel):
    """Selectable LLM models offered to the frontend dropdown."""
    models: list[str]
    default: str


class RunResponse(BaseModel):
    run_id: str
    paper_id: str
    mode: str                          # post-classification mode (may be snap|lens|sphere|qa)
    llm_model: str
    language: str = "en"
    status: str
    error_msg: str
    started_at: str
    finished_at: str | None
    user_question: str = ""
    detected_intent: str = ""
    current_step: str = ""
    progress_json: str = "[]"


class RecentRunResponse(BaseModel):
    run_id: str
    paper_id: str
    paper_title: str = ""
    mode: str
    status: str
    started_at: str
    finished_at: str | None
    current_step: str = ""
    user_question: str = ""


class RunOutputResponse(BaseModel):
    run_id: str
    markdown: str
    json_data: str


class BlockResponse(BaseModel):
    block_id: int
    paper_id: str
    type: str
    sub_type: str
    page_idx: int
    bbox_json: str
    text: str
    section_path: str
    order_idx: int
