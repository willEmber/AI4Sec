from __future__ import annotations

from typing import Any, TypedDict


class MainGraphState(TypedDict, total=False):
    """The state a mode subgraph reads and writes.

    Named for the upload pipeline's graph, which is gone (P16); the mode tools
    build it in `agents/tools/modes.py` and pass it to one subgraph.
    """

    # Input
    paper_id: str
    run_id: str
    mode: str               # snap | lens | sphere
    llm_model: str
    language: str            # en | zh
    user_question: str       # the reader's focus, if any

    # Pipeline progress
    pdf_path: str
    parse_id: str
    output_dir: str
    paper_ir_json: str      # serialized PaperIR
    pub_rank_json: str      # serialized {"venue":"...","year":2024,"sci":"Q1","ccf":"A"}

    # Output
    final_markdown: str
    final_json: str
    analysis_language: str   # language the subgraph natively produced final_markdown in;
                             # when it equals `language`, translate_output is skipped
    error: str

    # Sphere internal state (debugging)
    sphere_state_json: str

    # Progress tracking
    progress: list[dict[str, Any]]
