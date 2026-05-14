from __future__ import annotations

from typing import Any, TypedDict


class MainGraphState(TypedDict, total=False):
    # Input
    paper_id: str
    run_id: str
    mode: str               # snap | lens | sphere | auto (input) ; mutated by classify_intent to snap|lens|sphere|qa
    llm_model: str
    language: str            # en | zh
    user_question: str       # free-text question; required when mode == "auto"

    # Pipeline progress
    pdf_path: str
    parse_id: str
    output_dir: str
    paper_ir_json: str      # serialized PaperIR
    pub_rank_json: str      # serialized {"venue":"...","year":2024,"sci":"Q1","ccf":"A"}
    detected_intent: str    # set by classify_intent — one of snap | lens | sphere | qa

    # Output
    final_markdown: str
    final_json: str
    error: str

    # Sphere internal state (debugging)
    sphere_state_json: str

    # Progress tracking
    progress: list[dict[str, Any]]
