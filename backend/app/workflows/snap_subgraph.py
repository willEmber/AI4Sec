from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.models.paper_ir import PaperIR
from app.services.citation_validator import (
    format_coverage_summary,
    validate_citation_coverage,
)
from app.services.evidence_extractor import extract_evidence_pool
from app.services.llm_service import get_llm_service
from app.workflows.state import MainGraphState

logger = logging.getLogger("scholar.graph")

# Slot names exposed to the evidence extractor. Kept aligned with the section
# headings the Snap prompt produces so the second LLM call can map quotes back.
SNAP_SLOTS: list[str] = [
    "problem",
    "method",
    "result",
    "evidence_strength",
    "limitation",
    "verdict",
]


def _extract_key_sections(paper_ir: PaperIR) -> str:
    """Extract title, abstract, introduction, conclusion, and contribution sections."""
    key_section_names = {
        "abstract", "introduction", "conclusion", "conclusions",
        "contributions", "summary", "results",
    }

    parts: list[str] = []

    if paper_ir.title:
        parts.append(f"# {paper_ir.title}\n")

    for section in paper_ir.sections:
        section_lower = section.title.lower().strip()
        is_key = any(kw in section_lower for kw in key_section_names)
        # Also include sections with no title (pre-title content) or level 0
        if is_key or section.level == 0:
            section_text_parts: list[str] = []
            for block in section.blocks:
                if block.type in ("text", "title", "list"):
                    text = block.text.strip()
                    if text:
                        page_ref = f" [p.{block.page_idx + 1}]"
                        section_text_parts.append(f"{text}{page_ref}")
            if section_text_parts:
                header = f"## {section.title}\n" if section.title else ""
                parts.append(header + "\n".join(section_text_parts))

    return "\n\n".join(parts)


SNAP_SYSTEM_PROMPT_EN = """You are a research paper analysis assistant. Your task is to provide a quick "Insight Snap" - a 30-second triage summary to help a researcher decide if a paper is worth reading in depth.

You MUST format your output as structured Markdown with the following sections exactly:

## One-Sentence Summary
(Problem + Method + Key Result in one sentence)

## Core Contributions
- (3-5 bullet points, each with a page citation like [p.X])

## Key Experimental Findings
- (Main metrics, improvements, comparisons, with page citations)

## Applicability & Limitations
- Suitable for: ...
- Limitations: ... (with page citations)

## Worth Reading?
(Yes/No with brief justification. Consider the publication venue's reputation if available.)

IMPORTANT RULES:
1. Every factual claim MUST include a page citation in the format [p.X] where X is the page number.
2. Be concise but precise. This is a triage tool.
3. Use LaTeX for any mathematical notation: $inline$ or $$display$$.
4. Do not fabricate information. Only cite what is in the paper.
5. If a slot has no supporting evidence in the extracted text (e.g. the paper omits limitations or experimental setup), write `_Not reported in extracted text._` for that bullet instead of guessing or paraphrasing from general knowledge.
"""


SNAP_SYSTEM_PROMPT_ZH = """你是一名科研论文分析助手。你的任务是给出一份快速的"洞察速览"(Insight Snap)——一个 30 秒分诊式摘要,帮助研究者判断这篇论文是否值得深入精读。

你必须输出结构化 Markdown,且严格使用以下小节标题:

## 一句话总结
(用一句话概括:问题 + 方法 + 关键结果)

## 核心贡献
- (3-5 个要点,每条都要带形如 [p.X] 的页码引用)

## 关键实验发现
- (主要指标、提升幅度、对比结果,并附页码引用)

## 适用性与局限
- 适用于:……
- 局限:……(附页码引用)

## 是否值得精读?
(给出 是/否 并简要说明理由。如有期刊/会议声誉信息,可纳入考量。)

重要规则:
1. 每一条事实性陈述都必须带形如 [p.X] 的页码引用(X 为页码),引用标记本身保持英文原样,不要翻译或改写。
2. 简洁而精确——这是一个分诊工具。
3. 所有数学记号一律使用 LaTeX:行内用 $inline$,独立公式用 $$display$$。
4. 不得编造信息,只能引用论文中确有的内容。
5. 若某一小节在提取文本中没有支撑证据(例如论文未给出局限或实验设置),该条写 `_提取文本中未提及。_`,不要凭常识猜测或臆造。
6. 输出语言:整篇内容必须用简体中文撰写。但请保留以下内容的英文原样、不得翻译:LaTeX 公式、页码引用 [p.X]、论文标题、作者姓名、期刊/会议名称;专有技术术语首次出现时保留英文并在括号内给出中文解释(如 "Transformer(变换器)")。
"""


async def run_insight_snap(state: MainGraphState) -> dict[str, Any]:
    """Run Insight Snap analysis on the paper."""
    paper_id = state["paper_id"]
    t0 = time.perf_counter()

    paper_ir = PaperIR.model_validate_json(state["paper_ir_json"])
    logger.info(f"[{paper_id}] snap: Parsed PaperIR — {len(paper_ir.sections)} sections, {len(paper_ir.blocks)} blocks")

    t_extract = time.perf_counter()
    key_content = _extract_key_sections(paper_ir)
    logger.info(f"[{paper_id}] snap: Extracted key sections in {time.perf_counter()-t_extract:.3f}s — {len(key_content)} chars")

    if not key_content.strip():
        return {
            "final_markdown": "# Error\n\nNo content could be extracted from the paper.",
            "final_json": json.dumps({"error": "no_content"}),
            "progress": state.get("progress", []) + [{"step": "run_snap", "status": "failed"}],
        }

    # Inject publication rank context if available
    pub_rank_json = state.get("pub_rank_json", "")
    if pub_rank_json:
        try:
            pub_rank = json.loads(pub_rank_json)
            meta_parts: list[str] = []
            if pub_rank.get("venue"):
                meta_parts.append(f"Published in: {pub_rank['venue']}")
            if pub_rank.get("sci"):
                meta_parts.append(f"SCI Tier: {pub_rank['sci']}")
            if pub_rank.get("ccf"):
                meta_parts.append(f"CCF Rating: {pub_rank['ccf']}")
            if meta_parts:
                key_content = f"[Publication Info: {' | '.join(meta_parts)}]\n\n" + key_content
        except (json.JSONDecodeError, TypeError):
            pass

    # Truncate if too long (rough token estimate: ~4 chars per token)
    max_chars = 12000
    if len(key_content) > max_chars:
        key_content = key_content[:max_chars] + "\n\n[... truncated for length ...]"
        logger.info(f"[{paper_id}] snap: Content truncated to {max_chars} chars")

    language = state.get("language", "en")
    system_prompt = SNAP_SYSTEM_PROMPT_ZH if language == "zh" else SNAP_SYSTEM_PROMPT_EN
    user_prefix = "请分析这篇论文:" if language == "zh" else "Analyze this paper:"

    llm = get_llm_service()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{user_prefix}\n\n{key_content}"},
    ]

    model = state.get("llm_model", "")
    logger.info(f"[{paper_id}] snap: Calling LLM (model={model or '(default)'})...")
    t_llm = time.perf_counter()
    markdown = await llm.chat(messages, model=model, temperature=0.3, max_tokens=4096)
    logger.info(f"[{paper_id}] snap: LLM returned in {time.perf_counter()-t_llm:.1f}s — {len(markdown)} chars response")

    audit = validate_citation_coverage(markdown)
    logger.info(f"[{paper_id}] snap: {format_coverage_summary(audit)}")
    if audit["claims_uncited"] > 0 and audit["uncited_samples"]:
        logger.warning(
            f"[{paper_id}] snap: {audit['claims_uncited']} uncited claims — samples: {audit['uncited_samples'][:3]}"
        )

    t_evidence = time.perf_counter()
    evidence_pool = await extract_evidence_pool(
        llm,
        context=key_content,
        slots=SNAP_SLOTS,
        model=model,
        log_label=f"[{paper_id}] snap",
    )
    logger.info(
        f"[{paper_id}] snap: evidence extraction in {time.perf_counter()-t_evidence:.1f}s "
        f"— {len(evidence_pool)} cards"
    )

    logger.info(f"[{paper_id}] snap: TOTAL {time.perf_counter()-t0:.1f}s")
    return {
        "final_markdown": markdown,
        "analysis_language": language,
        "final_json": json.dumps({
            "mode": "snap",
            "paper_id": state["paper_id"],
            "title": paper_ir.title,
            "sections_used": [s.title for s in paper_ir.sections if s.title],
            "citation_audit": audit,
            "evidence_pool": evidence_pool,
        }),
        "progress": state.get("progress", []) + [{"step": "run_snap", "status": "done"}],
    }
