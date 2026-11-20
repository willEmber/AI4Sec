"""`get_paper_elements`: a paper's equations, algorithms, tables and figures, citable (P16).

A section read returns equations and tables among the prose, cut to what fits
a passage. A report about the method needs them as a list — every equation
with its number, every table whole, every figure with an image it can embed —
and each one citable, so a formula in the report points at the formula in the
paper. Logic Lens extracted these into its prompt; written as an answer, the
report asks for them here.

Elements come from the stored parse (`PaperIR` blocks, as `ir_extract` reads
them); nothing is sent anywhere. Each is recorded as `fulltext` evidence whose
quote is the element itself: LaTeX for an equation, a Markdown table for a
table, the caption for a figure.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from langchain.tools import ToolRuntime, tool

from app.agents.context import AgentContext
from app.agents.tools.reading import _authorize, _paper_of_section
from app.agents.tools.sections import find_section, section_views
from app.models.agent_models import ErrorCode, ToolResult
from app.models.evidence_models import Locator
from app.models.paper_ir import Block
from app.services import evidence_service
from app.services.ir_extract import (
    compact_table_text,
    figure_embed_url,
    select_framework_figures,
)
from app.services.paper_ir_loader import PaperNotParsed, load_paper_ir
from app.services.qa_retrieval import load_paper_nodes

Kind = Literal["equation", "algorithm", "table", "figure"]
KINDS: tuple[str, ...] = ("equation", "algorithm", "table", "figure")

# Elements returned per kind. A long paper has a hundred equations; past these
# the model is told how many there are and to narrow by section.
MAX_PER_KIND = {"equation": 40, "algorithm": 8, "table": 16, "figure": 16}
# The longest quote each kind keeps. A table needs room for its rows.
QUOTE_LIMIT = {"equation": 1500, "algorithm": 2500, "table": 1800, "figure": 600}

_TAG_RE = re.compile(r"\\tag\s*\{\s*([^}]*?)\s*\}")
_DISPLAY_RE = re.compile(r"^\s*\$\$\s*|\s*\$\$\s*$")
_LABEL_RE = {
    "table": re.compile(r"\bTable\s+([A-Z]?\d+[a-z]?)", re.IGNORECASE),
    "figure": re.compile(r"\b(?:Figure|Fig\.)\s*([A-Z]?\d+[a-z]?)", re.IGNORECASE),
    "algorithm": re.compile(r"\bAlgorithm\s+([A-Z]?\d+)", re.IGNORECASE),
}


def _kind_of(block: Block) -> str:
    """Which element a block is, as `ir_extract.extract_*` classify them."""
    sub = block.sub_type.lower()
    if block.type in ("equation", "isolate_formula") or "formula" in sub:
        return "equation"
    if block.type in ("code", "algorithm") or "algorithm" in sub:
        return "algorithm"
    if block.type == "table":
        return "table"
    if block.type == "image":
        caption = block.text.strip()
        return "figure" if caption and caption != "[image]" else ""
    return ""


def _label(kind: str, text: str) -> str:
    if kind == "equation":
        found = _TAG_RE.search(text)
        return f"Eq. {found.group(1)}" if found else ""
    pattern = _LABEL_RE.get(kind)
    found = pattern.search(text) if pattern else None
    if not found:
        return ""
    name = {"table": "Table", "figure": "Figure", "algorithm": "Algorithm"}[kind]
    return f"{name} {found.group(1)}"


def _content(kind: str, block: Block) -> str:
    text = block.text.strip()
    if kind == "equation":
        return _DISPLAY_RE.sub("", text).strip()
    if kind == "table":
        return compact_table_text(text, max_chars=QUOTE_LIMIT["table"])
    return text


@tool(parse_docstring=False)
async def get_paper_elements(
    runtime: ToolRuntime[AgentContext],
    paper_id: str = "",
    kinds: list[Kind] | None = None,
    section_id: str = "",
) -> str:
    """List a parsed paper's equations, algorithms, tables and figures, each citable.

    Use it to write about a paper's formulas, procedures and results exactly:
    an equation comes back as its LaTeX with its number, a table as a Markdown
    table, an algorithm as its text, a figure as its caption plus an
    `image_url` a report may embed as `![caption](image_url)` (copied
    verbatim). Figures marked `framework` are the paper's architecture or
    overview diagrams. Every element carries an evidence number to cite.
    `kinds` narrows the list (default: all four); `section_id` (from
    get_paper_outline) keeps one section's elements, its subsections included.
    Omit paper_id when the session holds one paper.
    """
    ctx = runtime.context
    if not paper_id and section_id:
        paper_id = await _paper_of_section(ctx, section_id)
    paper_id, denial = await _authorize(ctx, paper_id)
    if denial is not None:
        return denial.to_json()
    try:
        paper_ir = await load_paper_ir(paper_id)
    except PaperNotParsed:
        return ToolResult.unavailable(
            ErrorCode.NOT_PARSED, f"Paper {paper_id} is not parsed; call ensure_paper_parsed."
        ).to_json()

    wanted = [k for k in KINDS if not kinds or k in kinds]
    first, last, section_title = 0, None, ""
    if section_id:
        views = section_views(await load_paper_nodes(paper_id))
        view = find_section(views, section_id=section_id)
        if view is None:
            available = [f"{v.short_id} {v.title}" for v in views if not v.folded]
            return ToolResult.failed(
                ErrorCode.SECTION_NOT_FOUND,
                "No section matched. Available sections (id, title): " + "; ".join(available),
                retryable=False,
            ).to_json()
        first, last, section_title = view.node.block_start, view.block_end, view.title

    framework = {
        id(f["block"])
        for f in select_framework_figures(
            [
                {"text": b.text, "img_path": b.img_path, "block": b}
                for b in paper_ir.blocks
                if _kind_of(b) == "figure"
            ]
        )[0]
    }

    found: dict[str, list[Block]] = {k: [] for k in wanted}
    for block in paper_ir.blocks:
        kind = _kind_of(block)
        if kind not in found:
            continue
        if block.order_idx < first or (last is not None and block.order_idx > last):
            continue
        found[kind].append(block)

    shown = [(k, b) for k in wanted for b in found[k][: MAX_PER_KIND[k]]]
    passages: list[tuple[str, Locator]] = []
    limits: list[int] = []
    for kind, block in shown:
        passages.append(
            (
                _content(kind, block),
                Locator(
                    section_path=block.section_path,
                    page_index=block.page_idx,
                    page_end_index=block.page_idx,
                    block_start=block.order_idx,
                    block_end=block.order_idx,
                    bbox=block.bbox,
                ),
            )
        )
        limits.append(QUOTE_LIMIT[kind])
    evidence_ids = await evidence_service.record_fulltext_evidence_many(
        paper_id=paper_id,
        passages=passages,
        owner_id=ctx.owner_id,
        session_id=ctx.session_id,
        quote_limits=limits,
    )

    field = {"equation": "latex", "algorithm": "text", "table": "table", "figure": "caption"}
    elements: list[dict[str, Any]] = []
    for (kind, block), (content, _locator), evidence_id in zip(shown, passages, evidence_ids):
        element: dict[str, Any] = {
            "evidence_id": evidence_id,
            "kind": kind,
            "label": _label(kind, block.text),
            "section": block.section_path,
            "page": block.page_idx + 1,
            field[kind]: content[: QUOTE_LIMIT[kind]],
        }
        if kind == "figure":
            url = figure_embed_url(paper_id, block.img_path)
            if url:
                element["image_url"] = url
            if id(block) in framework:
                element["framework"] = True
        elements.append(element)

    counts = {k: len(found[k]) for k in wanted}
    cut = [f"{MAX_PER_KIND[k]} of {n} {k}s" for k, n in counts.items() if n > MAX_PER_KIND[k]]
    data = {
        "paper_id": paper_id,
        "title": paper_ir.title,
        "section": section_title,
        "counts": counts,
        "elements": elements,
    }
    if not elements:
        return ToolResult.ok(
            data,
            note="No such elements in "
            + (f"section {section_title!r}." if section_title else "this paper's parse."),
        ).to_json()
    note = (
        f"Showing the first {', '.join(cut)}; pass section_id for the rest." if cut else ""
    )
    return ToolResult.ok(data, evidence_ids=evidence_ids, note=note).to_json()


ELEMENT_TOOLS = [get_paper_elements]
