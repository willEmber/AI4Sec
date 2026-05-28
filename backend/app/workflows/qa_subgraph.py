from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from app.models.paper_ir import PaperIR
from app.services.llm_service import get_llm_service
from app.workflows.state import MainGraphState

logger = logging.getLogger("scholar.graph")


_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "on", "at", "is", "are", "was", "were",
    "to", "for", "and", "or", "but", "with", "as", "by", "from", "this",
    "that", "these", "those", "it", "its", "what", "how", "why", "which",
    "who", "where", "when", "do", "does", "did", "can", "could", "should",
    "would", "will", "be", "been", "being", "have", "has", "had", "you",
    "your", "they", "their", "paper", "authors",
})

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_SEARCHABLE_BLOCK_TYPES = frozenset({"text", "title", "list", "equation", "table", "image"})

_CJK_QUERY_EXPANSIONS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("指标", "数据", "性能", "表现", "评估", "实验", "结果", "对比", "准确率", "鲁棒", "保真", "攻击"),
        (
            "metric", "metrics", "performance", "evaluation", "experiment", "experiments",
            "result", "results", "accuracy", "robust", "robustness", "fidelity",
            "psnr", "ssim", "lpips", "fid", "ablation", "comparison", "baseline",
            "table",
        ),
    ),
    (
        ("方法", "方案", "算法", "模型", "框架"),
        ("method", "approach", "proposed", "model", "framework", "architecture"),
    ),
)


def _tokenize(question: str) -> list[str]:
    """Lowercase tokens, drop stopwords + short words. Keeps CJK characters as their own tokens."""
    tokens: list[str] = []
    question_l = question.lower()
    for m in _TOKEN_RE.findall(question_l):
        if m not in _STOPWORDS:
            tokens.append(m)
    # Also include CJK runs as tokens (each 2-4 char run is treated as a phrase)
    for run in re.findall(r"[一-鿿]{2,4}", question):
        tokens.append(run)
    for triggers, expansions in _CJK_QUERY_EXPANSIONS:
        if any(trigger in question for trigger in triggers):
            tokens.extend(expansions)
    # Deduplicate while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered


def _score_blocks(paper_ir: PaperIR, tokens: list[str]) -> list[tuple[int, Any]]:
    """Return [(score, block), ...] sorted by descending score."""
    if not tokens:
        return []
    scored: list[tuple[int, Any]] = []
    for b in paper_ir.blocks:
        if b.type not in _SEARCHABLE_BLOCK_TYPES:
            continue
        text_l = " ".join(
            part for part in (b.text, b.section_path, b.sub_type, b.type) if part
        ).lower()
        if not text_l:
            continue
        score = 0
        for t in tokens:
            if t in text_l:
                score += 1
        if b.type == "table" and any(
            t in tokens for t in ("metric", "metrics", "table", "performance")
        ):
            score += 2
        if score > 0:
            scored.append((score, b))
    scored.sort(key=lambda x: (-x[0], x[1].page_idx, x[1].order_idx))
    return scored


def _expand_with_neighbors(paper_ir: PaperIR, blocks: list[Any], window: int = 1) -> list[Any]:
    """Include nearby blocks so captions, section titles, and tables stay together."""
    by_order = {b.order_idx: b for b in paper_ir.blocks}
    expanded: list[Any] = []
    seen: set[tuple[int, int]] = set()

    for block in blocks:
        for order_idx in range(block.order_idx - window, block.order_idx + window + 1):
            neighbor = by_order.get(order_idx)
            if neighbor is None:
                continue
            if neighbor.type not in _SEARCHABLE_BLOCK_TYPES:
                continue
            key = (neighbor.page_idx, neighbor.order_idx)
            if key in seen:
                continue
            seen.add(key)
            expanded.append(neighbor)

    return expanded


def _assemble_context(paper_ir: PaperIR, question: str, max_chars: int = 20000) -> tuple[str, int]:
    """Pick top-K relevant blocks + always-include early-page anchors, format with [p.X] markers."""
    tokens = _tokenize(question)
    scored = _score_blocks(paper_ir, tokens)
    top = _expand_with_neighbors(paper_ir, [b for _, b in scored[:30]])

    # Always include first-page title/text blocks for grounding
    anchors = [
        b for b in paper_ir.blocks
        if b.page_idx == 0 and b.type in ("title", "text")
    ][:5]

    seen: set[tuple[int, int]] = set()
    chosen: list[Any] = []
    for b in anchors + top:
        key = (b.page_idx, b.order_idx)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(b)
    chosen.sort(key=lambda b: (b.page_idx, b.order_idx))

    parts = [f"[p.{b.page_idx + 1}] {(b.text or '').strip()}" for b in chosen if (b.text or "").strip()]
    context = "\n\n".join(parts)
    if len(context) > max_chars:
        context = context[:max_chars] + "\n\n[... truncated for length ...]"
    return context, len(chosen)


_SYSTEM_PROMPT = """You are answering a specific user question about a single academic paper.

Rules:
1. Use ONLY the provided paper context. Each block is prefixed with `[p.X]` indicating its page number.
2. The context is assembled from parsed paper blocks and may include tables or section-local snippets.
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
    context, n_blocks = _assemble_context(paper_ir, question)
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
