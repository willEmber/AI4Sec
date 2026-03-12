from __future__ import annotations

from pydantic import BaseModel


# --- Request models ---

class RunCreate(BaseModel):
    paper_id: str
    mode: str = "snap"          # snap | lens | sphere
    llm_model: str = ""


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


class RunResponse(BaseModel):
    run_id: str
    paper_id: str
    mode: str
    llm_model: str
    status: str
    error_msg: str
    started_at: str
    finished_at: str | None


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
