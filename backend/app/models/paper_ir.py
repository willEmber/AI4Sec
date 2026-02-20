from __future__ import annotations

from pydantic import BaseModel


class Block(BaseModel):
    type: str                       # text | title | table | image | equation | code | list | ref_text
    sub_type: str = ""
    page_idx: int = 0
    bbox: list[float] = []          # [x0, y0, x1, y1]
    text: str = ""
    section_path: str = ""
    order_idx: int = 0


class Section(BaseModel):
    path: str                       # e.g. "2.Method/2.1 Model"
    title: str
    level: int = 0
    blocks: list[Block] = []


class PaperIR(BaseModel):
    paper_id: str
    title: str = ""
    sections: list[Section] = []
    blocks: list[Block] = []
