from __future__ import annotations

from typing import TypedDict


class MainGraphState(TypedDict, total=False):
    """What the literature map's steps read: the paper and the reader's choices.

    Named for the upload pipeline's graph, which is gone (P16); the
    `build_literature_map` tool builds it (`agents/tools/literature_map.py`).
    """

    paper_id: str
    llm_model: str
    language: str            # en | zh
    paper_ir_json: str       # serialized PaperIR
    persist_sphere: bool     # write the map's nodes to the report's `runs` row
