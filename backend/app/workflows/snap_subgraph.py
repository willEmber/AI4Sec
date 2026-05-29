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


SNAP_SYSTEM_PROMPT = """You are a research paper analysis assistant. Your task is to provide a quick "Insight Snap" - a 30-second triage summary to help a researcher decide if a paper is worth reading in depth.

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

    llm = get_llm_service()
    messages = [
        {"role": "system", "content": SNAP_SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze this paper:\n\n{key_content}"},
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
