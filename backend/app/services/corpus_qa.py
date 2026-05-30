"""Corpus-wide RAG Q&A over the Dify knowledge base.

Unlike single-paper Q&A (``workflows/qa_subgraph.py``), this retrieves passages
across the whole library via the Dify proxy and asks the LLM to synthesise an
answer. Because library documents are markdown (no page index), citations use a
document-level ``[L#]`` scheme instead of ``[p.X]`` — each marker maps to one
retrieved passage (``document_id`` + ``segment_id``) that the frontend links to
the library document view.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.config import get_settings
from app.services import dify_client
from app.services.llm_service import get_llm_service

logger = logging.getLogger("scholar.corpus_qa")

# Per-passage and total context budgets (chars). Mirrors the ~20k budget used by
# single-paper Q&A so prompt sizes stay comparable.
_MAX_PASSAGE_CHARS = 3000
_MAX_CONTEXT_CHARS = 20000


_SYSTEM_PROMPT_EN = """You are answering a user's question using passages retrieved from the user's personal research-paper library (a curated corpus of papers).

Rules:
1. Use ONLY the provided library passages. Each passage is prefixed with an `[L#]` marker and its source document name.
2. Every factual claim MUST be followed by the `[L#]` marker(s) of the passage(s) it came from, copied verbatim (e.g. `[L1]`, `[L2]`). When a claim draws on several passages, list each marker.
3. If the passages do not contain enough to answer, say so explicitly and suggest what to search for next. Do NOT use outside knowledge or fabricate.
4. Passages may come from different papers and may disagree — attribute claims to their source document and note any disagreement.
5. Use LaTeX for math: `$inline$` or `$$display$$`.
6. Be concise and well-structured (Markdown). Do not restate the question.
"""

_SYSTEM_PROMPT_ZH = """你正在使用从用户个人论文知识库（一个已策展的论文语料库）中检索到的片段来回答用户的问题。

规则:
1. 只能使用所提供的知识库片段。每个片段都以 `[L#]` 标记和其来源文档名作为前缀。
2. 每一条事实性陈述之后都必须跟上其所依据片段的 `[L#]` 标记,原样复制(例如 `[L1]`、`[L2]`)。若一条陈述综合了多个片段,逐个列出标记。
3. 若所提供片段不足以回答,要明确说明,并建议下一步可检索的方向。不得使用外部知识,不得编造。
4. 不同片段可能来自不同论文、可能相互矛盾——把陈述归因到来源文档,并指出分歧。
5. 数学使用 LaTeX:行内 `$inline$`,独立公式 `$$display$$`。
6. 简洁、结构清晰(Markdown)。不要复述问题。
7. 输出语言:用简体中文回答。但保留以下内容的英文原样、不得翻译:LaTeX 公式、`[L#]` 标记、论文标题、作者姓名、期刊/会议名称;专有技术术语首次出现时保留英文并在括号内附中文解释。
"""


def _format_context(records: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Build the LLM context string and the parallel `sources` list.

    Records are numbered ``[L1]..[Ln]`` in retrieval order so the markers the LLM
    copies line up with `sources[i]`.
    """
    parts: list[str] = []
    sources: list[dict[str, Any]] = []
    total = 0
    for record in records:
        content = (record.get("content") or "").strip()
        if not content:
            continue
        idx = len(sources) + 1
        name = (record.get("document_name") or "").strip() or f"document {idx}"
        snippet = content[:_MAX_PASSAGE_CHARS]
        block = f"[L{idx}] «{name}»\n{snippet}"
        if total + len(block) > _MAX_CONTEXT_CHARS and parts:
            break
        parts.append(block)
        total += len(block)
        sources.append({
            "idx": idx,
            "document_id": record.get("document_id") or "",
            "document_name": name,
            "segment_id": record.get("segment_id") or "",
            "score": record.get("score"),
        })
    return "\n\n".join(parts), sources


def _no_results_markdown(language: str) -> str:
    if language == "zh":
        return (
            "# 未检索到相关内容\n\n"
            "知识库中没有与该问题匹配的片段。可以尝试换用关键词,"
            "或在检索方式中切换到语义/混合检索。"
        )
    return (
        "# No relevant passages found\n\n"
        "The knowledge base returned no passages for this question. "
        "Try different keywords, or switch the retrieval method to semantic / hybrid search."
    )


async def answer_corpus_question(
    question: str,
    *,
    top_k: int = 10,
    search_method: str | None = None,
    language: str = "en",
    llm_model: str = "",
    dataset_id: str | None = None,
) -> dict[str, Any]:
    """Retrieve passages from the library and synthesise a cited answer.

    Returns ``{markdown, sources, blocks_used, search_method, question}``.
    Raises :class:`dify_client.DifyError` if retrieval fails (the caller maps it
    to an HTTP error).
    """
    question = (question or "").strip()
    method = (search_method or get_settings().dify_search_method or "full_text_search").strip()
    t0 = time.perf_counter()

    records = await dify_client.search_records(
        question, top_k=top_k, search_method=method, dataset_id=dataset_id
    )
    logger.info(
        "corpus_qa: retrieved %d records (method=%s) in %.2fs for q=%r",
        len(records), method, time.perf_counter() - t0, question[:120],
    )

    context, sources = _format_context(records)
    if not context:
        return {
            "markdown": _no_results_markdown(language),
            "sources": [],
            "blocks_used": 0,
            "search_method": method,
            "question": question,
        }

    system_prompt = _SYSTEM_PROMPT_ZH if language == "zh" else _SYSTEM_PROMPT_EN
    if language == "zh":
        user_content = f"问题:{question}\n\n知识库片段:\n{context}"
    else:
        user_content = f"Question: {question}\n\nLibrary passages:\n{context}"

    t_llm = time.perf_counter()
    markdown = await get_llm_service().chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        model=llm_model,
        temperature=0.2,
        max_tokens=2048,
    )
    logger.info(
        "corpus_qa: LLM answered in %.1fs — %d sources, %d chars",
        time.perf_counter() - t_llm, len(sources), len(markdown),
    )

    return {
        "markdown": markdown,
        "sources": sources,
        "blocks_used": len(sources),
        "search_method": method,
        "question": question,
    }
