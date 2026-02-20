from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.models.paper_ir import PaperIR
from app.services.llm_service import get_llm_service
from app.workflows.state import MainGraphState

logger = logging.getLogger("scholar.graph")


def _extract_equations(paper_ir: PaperIR) -> list[dict[str, Any]]:
    """Extract equation blocks with context."""
    equations = []
    for block in paper_ir.blocks:
        if block.type in ("equation", "isolate_formula") or "formula" in block.sub_type.lower():
            equations.append({
                "text": block.text,
                "page": block.page_idx + 1,
                "section": block.section_path,
                "bbox": block.bbox,
            })
    return equations


def _extract_algorithms(paper_ir: PaperIR) -> list[dict[str, Any]]:
    """Extract algorithm/code blocks."""
    algos = []
    for block in paper_ir.blocks:
        if block.type in ("code", "algorithm") or "algorithm" in block.sub_type.lower():
            algos.append({
                "text": block.text,
                "page": block.page_idx + 1,
                "section": block.section_path,
                "bbox": block.bbox,
            })
    return algos


def _extract_tables(paper_ir: PaperIR) -> list[dict[str, Any]]:
    """Extract table blocks."""
    tables = []
    for block in paper_ir.blocks:
        if block.type == "table":
            tables.append({
                "text": block.text,
                "page": block.page_idx + 1,
                "section": block.section_path,
                "bbox": block.bbox,
            })
    return tables


def _extract_method_section(paper_ir: PaperIR) -> str:
    """Extract text from method-related sections."""
    method_keywords = {"method", "approach", "model", "framework", "architecture", "proposed"}
    parts = []
    for section in paper_ir.sections:
        if any(kw in section.title.lower() for kw in method_keywords):
            for block in section.blocks:
                if block.type in ("text", "title", "list", "equation"):
                    parts.append(f"{block.text.strip()} [p.{block.page_idx + 1}]")
    return "\n".join(parts)


def _extract_experiment_section(paper_ir: PaperIR) -> str:
    """Extract text from experiment-related sections."""
    exp_keywords = {"experiment", "evaluation", "result", "empirical", "ablation", "setup"}
    parts = []
    for section in paper_ir.sections:
        if any(kw in section.title.lower() for kw in exp_keywords):
            for block in section.blocks:
                if block.type in ("text", "title", "list", "table"):
                    parts.append(f"{block.text.strip()} [p.{block.page_idx + 1}]")
    return "\n".join(parts)


LENS_SYSTEM_PROMPT = """You are a deep technical paper analysis assistant. Produce a "Logic Lens" analysis with exactly these 6 sections:

## 1. Problem Setup
- Input/output specification
- Core assumptions
- Symbol table (if applicable, in LaTeX)

## 2. Method Overview
- High-level pipeline description
- Key module responsibilities
- Flow description

## 3. Formula Analysis
For each key formula:
- The formula in LaTeX ($$...$$)
- Variable meanings
- Derivation logic
- Difference from baselines

## 4. Algorithm Analysis
- Step-by-step pseudocode/procedure explanation
- Per-line or per-step annotation
- Complexity discussion if relevant

## 5. Experiment Reproduction Checklist
- Datasets used
- Preprocessing steps
- Training details (optimizer, LR, schedule, batch size)
- Key hyperparameters
- Evaluation metrics
- Hardware requirements

## 6. Reliability Assessment
- Are ablations sufficient?
- Statistical significance?
- Potential confounds?
- Missing comparisons?

RULES:
1. Every claim MUST have a page citation [p.X].
2. Use LaTeX for ALL mathematical content.
3. Be thorough but structured.
4. Don't fabricate - only cite what's in the paper.
"""


async def run_logic_lens(state: MainGraphState) -> dict[str, Any]:
    """Run Logic Lens deep analysis."""
    paper_id = state["paper_id"]
    t0 = time.perf_counter()

    paper_ir = PaperIR.model_validate_json(state["paper_ir_json"])
    logger.info(f"[{paper_id}] lens: Parsed PaperIR — {len(paper_ir.sections)} sections, {len(paper_ir.blocks)} blocks")

    equations = _extract_equations(paper_ir)
    algorithms = _extract_algorithms(paper_ir)
    tables = _extract_tables(paper_ir)
    method_text = _extract_method_section(paper_ir)
    experiment_text = _extract_experiment_section(paper_ir)
    logger.info(
        f"[{paper_id}] lens: Extracted — equations={len(equations)} algorithms={len(algorithms)} "
        f"tables={len(tables)} method={len(method_text)} chars experiment={len(experiment_text)} chars"
    )

    # Build context for LLM
    context_parts = []

    if paper_ir.title:
        context_parts.append(f"# Paper: {paper_ir.title}")

    # Full paper text for comprehensive analysis (truncated)
    all_text = []
    for block in paper_ir.blocks:
        if block.type in ("text", "title", "list"):
            all_text.append(f"{block.text.strip()} [p.{block.page_idx + 1}]")
    full_text = "\n".join(all_text)
    if len(full_text) > 15000:
        full_text = full_text[:15000] + "\n[...truncated...]"
    context_parts.append(f"## Full Paper Text\n{full_text}")

    if equations:
        eq_text = "\n".join(f"- {e['text']} [p.{e['page']}]" for e in equations)
        context_parts.append(f"## Extracted Equations\n{eq_text}")

    if algorithms:
        algo_text = "\n".join(f"- {a['text']} [p.{a['page']}]" for a in algorithms)
        context_parts.append(f"## Extracted Algorithms\n{algo_text}")

    if tables:
        tbl_text = "\n".join(f"- [p.{t['page']}] {t['text'][:500]}" for t in tables)
        context_parts.append(f"## Extracted Tables\n{tbl_text}")

    context = "\n\n".join(context_parts)
    logger.info(f"[{paper_id}] lens: Built LLM context — {len(context)} chars total")

    llm = get_llm_service()
    messages = [
        {"role": "system", "content": LENS_SYSTEM_PROMPT},
        {"role": "user", "content": f"Perform a deep Logic Lens analysis:\n\n{context}"},
    ]

    model = state.get("llm_model", "")
    logger.info(f"[{paper_id}] lens: Calling LLM (model={model or '(default)'})...")
    t_llm = time.perf_counter()
    markdown = await llm.chat(messages, model=model, temperature=0.3, max_tokens=16384)
    logger.info(f"[{paper_id}] lens: LLM returned in {time.perf_counter()-t_llm:.1f}s — {len(markdown)} chars response")

    logger.info(f"[{paper_id}] lens: TOTAL {time.perf_counter()-t0:.1f}s")
    return {
        "final_markdown": markdown,
        "final_json": json.dumps({
            "mode": "lens",
            "paper_id": state["paper_id"],
            "title": paper_ir.title,
            "num_equations": len(equations),
            "num_algorithms": len(algorithms),
            "num_tables": len(tables),
        }),
        "progress": state.get("progress", []) + [{"step": "run_lens", "status": "done"}],
    }
