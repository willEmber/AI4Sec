from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.models.paper_ir import PaperIR
from app.services.qa_retrieval import retrieve_qa_context, retrieve_qa_context_for_paper
from app.services.llm_service import get_llm_service
from app.workflows.state import MainGraphState

logger = logging.getLogger("scholar.graph")


def _assemble_context(paper_ir: PaperIR, question: str, max_chars: int = 20000) -> tuple[str, int]:
    """Compatibility wrapper for tests and callers that expect synchronous QA context assembly."""
    return retrieve_qa_context(paper_ir, question, max_chars=max_chars)


_SYSTEM_PROMPT = """You are answering a specific user question about a single academic paper.

Rules:
1. Use ONLY the provided paper context. Each block is prefixed with `[p.X]` indicating its page number.
2. The context is assembled from hierarchical paper nodes. Blocks may include a section path after the page marker.
3. Every factual claim MUST be followed by a `[p.X]` citation copied from the relevant block.
4. If the retrieved paper context does NOT contain the answer, say so explicitly and (if possible) suggest which section likely covers it.
5. Do not describe the context as "abstract only" or "provided excerpts" unless that is literally all that was provided.
6. Use LaTeX for math: `$inline$` or `$$display$$`.
7. Be concise and direct. This is a focused answer, not a full report. Markdown is fine; avoid restating the question.
8. Do not fabricate. Quote sparingly; paraphrase mostly.
"""


async def run_qa(state: MainGraphState) -> dict[str, Any]:
    """Direct Q&A: keyword-retrieve relevant blocks, then LLM answers with citations."""
    paper_id = state["paper_id"]
    t0 = time.perf_counter()

    question = (state.get("user_question") or "").strip()
    if not question:
        return {
            "final_markdown": "# Error\n\nNo question was provided for Q&A mode.",
            "final_json": json.dumps({"error": "no_question", "mode": "qa"}),
            "progress": state.get("progress", []) + [{"step": "run_qa", "status": "failed", "error": "no_question"}],
        }

    try:
        paper_ir = PaperIR.model_validate_json(state["paper_ir_json"])
    except Exception as e:
        logger.error(f"[{paper_id}] qa: failed to parse PaperIR — {e}")
        return {
            "final_markdown": "# Error\n\nFailed to load paper content for Q&A.",
            "final_json": json.dumps({"error": "ir_parse_failed", "mode": "qa"}),
            "progress": state.get("progress", []) + [{"step": "run_qa", "status": "failed", "error": str(e)}],
        }

    logger.info(f"[{paper_id}] qa: parsed PaperIR — {len(paper_ir.blocks)} blocks; question={question[:120]!r}")

    t_ctx = time.perf_counter()
    context, n_blocks = await retrieve_qa_context_for_paper(paper_id, paper_ir, question)
    logger.info(f"[{paper_id}] qa: assembled context in {time.perf_counter()-t_ctx:.3f}s — {n_blocks} blocks, {len(context)} chars")

    if not context.strip():
        return {
            "final_markdown": "# No relevant content\n\nThe paper text could not be searched for this question.",
            "final_json": json.dumps({"error": "no_content", "mode": "qa", "question": question}),
            "progress": state.get("progress", []) + [{"step": "run_qa", "status": "failed", "error": "no_content"}],
        }

    llm = get_llm_service()
    model = state.get("llm_model", "")
    logger.info(f"[{paper_id}] qa: calling LLM (model={model or '(default)'})")

    t_llm = time.perf_counter()
    markdown = await llm.chat(
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nPaper context:\n{context}"},
        ],
        model=model,
        temperature=0.2,
        max_tokens=2048,
    )
    logger.info(f"[{paper_id}] qa: LLM returned in {time.perf_counter()-t_llm:.1f}s — {len(markdown)} chars")

    logger.info(f"[{paper_id}] qa: TOTAL {time.perf_counter()-t0:.1f}s")
    return {
        "final_markdown": markdown,
        "final_json": json.dumps({
            "mode": "qa",
            "paper_id": paper_id,
            "title": paper_ir.title,
            "question": question,
            "blocks_used": n_blocks,
        }),
        "progress": state.get("progress", []) + [{"step": "run_qa", "status": "done"}],
    }
