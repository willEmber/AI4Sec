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


_SYSTEM_PROMPT_EN = """You are answering a specific user question about a single academic paper.

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


_SYSTEM_PROMPT_ZH = """你正在回答用户关于某一篇学术论文的具体问题。

规则:
1. 只能使用所提供的论文上下文。每个文本块都以 `[p.X]` 前缀标明其页码。
2. 上下文由论文的层级节点组装而成,文本块在页码标记之后可能附带章节路径。
3. 每一条事实性陈述之后都必须跟上从相关块中复制来的 `[p.X]` 引用(引用标记保持英文原样)。
4. 若检索到的论文上下文中并不包含答案,要明确说明,并(如可能)指出大概哪一章节会涉及该内容。
5. 除非所提供的确实只有摘要或片段,否则不要把上下文描述为"仅有摘要"或"仅是节选"。
6. 数学使用 LaTeX:行内 `$inline$`,独立公式 `$$display$$`。
7. 简洁直接——这是一个聚焦的回答,不是完整报告。可用 Markdown;不要复述问题。
8. 不得编造。少量直接引用,多用转述。
9. 输出语言:用简体中文回答。但保留以下内容的英文原样、不得翻译:LaTeX 公式、页码引用 [p.X]、论文标题、作者姓名、期刊/会议名称;专有技术术语首次出现时保留英文并在括号内附中文解释。
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
    language = state.get("language", "en")
    system_prompt = _SYSTEM_PROMPT_ZH if language == "zh" else _SYSTEM_PROMPT_EN
    if language == "zh":
        user_content = f"问题:{question}\n\n论文上下文:\n{context}"
    else:
        user_content = f"Question: {question}\n\nPaper context:\n{context}"
    logger.info(f"[{paper_id}] qa: calling LLM (model={model or '(default)'})")

    t_llm = time.perf_counter()
    markdown = await llm.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        model=model,
        temperature=0.2,
        max_tokens=2048,
    )
    logger.info(f"[{paper_id}] qa: LLM returned in {time.perf_counter()-t_llm:.1f}s — {len(markdown)} chars")

    logger.info(f"[{paper_id}] qa: TOTAL {time.perf_counter()-t0:.1f}s")
    return {
        "final_markdown": markdown,
        "analysis_language": language,
        "final_json": json.dumps({
            "mode": "qa",
            "paper_id": paper_id,
            "title": paper_ir.title,
            "question": question,
            "blocks_used": n_blocks,
        }),
        "progress": state.get("progress", []) + [{"step": "run_qa", "status": "done"}],
    }
