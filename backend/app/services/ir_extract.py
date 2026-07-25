"""Shared ``PaperIR`` extraction primitives.

Logic Lens and Insight Snap need the same handful of operations over a
``PaperIR``: find the sections whose (ancestor-aware) title matches a keyword
set, pull blocks of a given type, and turn MinerU's HTML table bodies into
something compact enough to put in a prompt. These used to be private helpers
inside ``lens_subgraph``; Snap needs them too, so they live here instead of
being copied.

MinerU stores a table's caption and its HTML body in one ``text`` field
(``caption\\n<table>…``), so a raw table block is both verbose and hard for a
model to read. ``compact_table_text`` rewrites the HTML as a pipe table, which
costs roughly a third of the tokens and renders directly in the report.
"""

from __future__ import annotations

import html as _html
import re
from typing import Iterable

from app.models.paper_ir import Block, PaperIR, Section

# ──────────────────────────────────────────────────────────────────────
# Section keyword sets (shared by the mode subgraphs)
# ──────────────────────────────────────────────────────────────────────

FRAMING_KEYWORDS = frozenset({
    "abstract", "introduction", "related work", "related", "background",
    "motivation", "conclusion", "conclusions", "summary", "discussion",
})

METHOD_KEYWORDS = frozenset({
    "method", "approach", "model", "framework", "architecture",
    "proposed", "algorithm", "implementation", "design",
})

EXPERIMENT_KEYWORDS = frozenset({
    "experiment", "evaluation", "result", "empirical", "ablation",
    "setup", "training", "implementation", "dataset", "benchmark",
})

# Narrower than FRAMING_KEYWORDS: the sections a triage read must never miss.
ABSTRACT_KEYWORDS = frozenset({"abstract", "summary"})
INTRO_KEYWORDS = frozenset({"introduction", "motivation", "contribution", "contributions"})
CONCLUSION_KEYWORDS = frozenset({"conclusion", "conclusions", "concluding", "discussion", "outlook"})

_SECTION_NUM_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)")

TEXT_BLOCK_TYPES = ("text", "title", "list")


# ──────────────────────────────────────────────────────────────────────
# Section matching
# ──────────────────────────────────────────────────────────────────────

def section_match_keys(paper_ir: PaperIR) -> dict[str, str]:
    """Map each ``section.title`` to a lowercased match string that also includes
    the titles of its numeric ancestors.

    MinerU sometimes flattens heading levels, collapsing ``section_path`` to the
    leaf (e.g. ``5.3 Optimizer`` loses its ``5 Training`` parent). We rebuild the
    ancestor chain from section numbering so a sub-section inherits its parent's
    keyword matches — e.g. ``5.3 Optimizer`` then matches the ``training`` keyword
    and its optimizer/batch/hardware details are no longer dropped.
    """
    num_to_title: dict[str, str] = {}
    for section in paper_ir.sections:
        m = _SECTION_NUM_RE.match(section.title)
        if m:
            num_to_title[m.group(1)] = section.title

    keys: dict[str, str] = {}
    for section in paper_ir.sections:
        titles = [section.title]
        m = _SECTION_NUM_RE.match(section.title)
        if m:
            parts = m.group(1).split(".")
            for i in range(1, len(parts)):
                anc_title = num_to_title.get(".".join(parts[:i]))
                if anc_title:
                    titles.append(anc_title)
        keys[section.title] = " ".join(titles).lower()
    return keys


def sections_matching(
    paper_ir: PaperIR,
    keywords: Iterable[str],
    *,
    match_keys: dict[str, str] | None = None,
) -> list[Section]:
    """Sections whose ancestor-aware title key contains any of ``keywords``."""
    keys = match_keys if match_keys is not None else section_match_keys(paper_ir)
    kws = tuple(keywords)
    return [
        section
        for section in paper_ir.sections
        if any(kw in keys.get(section.title, section.title.lower()) for kw in kws)
    ]


def section_text(
    paper_ir: PaperIR,
    keywords: Iterable[str],
    *,
    block_types: tuple[str, ...] = TEXT_BLOCK_TYPES,
    match_keys: dict[str, str] | None = None,
) -> str:
    """Concatenate the text of matching sections, one block per line, page-tagged."""
    parts: list[str] = []
    for section in sections_matching(paper_ir, keywords, match_keys=match_keys):
        for block in section.blocks:
            if block.type not in block_types:
                continue
            text = block.text.strip()
            if text:
                parts.append(f"{text} [p.{block.page_idx + 1}]")
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────
# Typed block extraction
# ──────────────────────────────────────────────────────────────────────

def _block_record(block: Block) -> dict:
    return {
        "text": block.text,
        "page": block.page_idx + 1,
        "section": block.section_path,
        "bbox": block.bbox,
    }


def extract_equations(paper_ir: PaperIR) -> list[dict]:
    """Equation blocks with page/section context."""
    return [
        _block_record(b)
        for b in paper_ir.blocks
        if b.type in ("equation", "isolate_formula") or "formula" in b.sub_type.lower()
    ]


def extract_algorithms(paper_ir: PaperIR) -> list[dict]:
    """Algorithm / pseudocode blocks with page/section context."""
    return [
        _block_record(b)
        for b in paper_ir.blocks
        if b.type in ("code", "algorithm") or "algorithm" in b.sub_type.lower()
    ]


def extract_tables(paper_ir: PaperIR) -> list[dict]:
    """Table blocks with page/section context (``text`` is caption + HTML body)."""
    return [_block_record(b) for b in paper_ir.blocks if b.type == "table"]


def extract_figures(paper_ir: PaperIR) -> list[dict]:
    """Figure blocks with their captions.

    MinerU stores each figure's caption as the image block's ``text``; empty or
    placeholder captions are skipped so the LLM only sees figures it can
    actually describe to the reader.
    """
    figures: list[dict] = []
    for block in paper_ir.blocks:
        if block.type != "image":
            continue
        caption = block.text.strip()
        if not caption or caption == "[image]":
            continue
        record = _block_record(block)
        record["img_path"] = block.img_path
        figures.append(record)
    return figures


# ──────────────────────────────────────────────────────────────────────
# HTML table → pipe table
# ──────────────────────────────────────────────────────────────────────

_TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_CELL_RE = re.compile(r"<t[hd][^>]*>(.*?)</t[hd]>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _clean_cell(raw: str, max_chars: int) -> str:
    text = _html.unescape(_TAG_RE.sub(" ", raw))
    text = _WS_RE.sub(" ", text).strip().replace("|", "\\|")
    return text[:max_chars]


def html_table_to_pipe(
    html_text: str,
    *,
    max_rows: int = 14,
    max_cols: int = 10,
    max_cell_chars: int = 60,
) -> str:
    """Rewrite an HTML ``<table>`` as a Markdown pipe table.

    Returns ``""`` when the input has no parseable rows, so callers can fall
    back to the raw text. ``colspan``/``rowspan`` are ignored — the cell text is
    kept in place, which is enough for a model to read the numbers even if the
    header alignment of a multi-level table is imperfect.
    """
    rows: list[list[str]] = []
    for row_html in _TR_RE.findall(html_text):
        cells = [_clean_cell(c, max_cell_chars) for c in _CELL_RE.findall(row_html)]
        if any(cells):
            rows.append(cells[:max_cols])
        if len(rows) >= max_rows:
            break

    if not rows:
        return ""

    width = max(len(r) for r in rows)
    lines = ["| " + " | ".join(r + [""] * (width - len(r))) + " |" for r in rows]
    # Insert a header separator after the first row so the result is valid
    # Markdown; MinerU's first <tr> is the header row in practice.
    lines.insert(1, "|" + "|".join([" --- "] * width) + "|")
    return "\n".join(lines)


def compact_table_text(text: str, *, max_chars: int = 1600, **kwargs) -> str:
    """Best-effort compaction of a MinerU table block for prompt inclusion.

    The HTML body becomes a pipe table when parseable; otherwise the raw text is
    tag-stripped. Either way the result is hard-capped at ``max_chars``.
    """
    if not text:
        return ""
    split_at = text.find("<table")
    if split_at == -1:
        split_at = text.find("<tr")
    caption = text[:split_at].strip() if split_at > 0 else ""
    body_html = text[split_at:] if split_at >= 0 else ""

    pipe = html_table_to_pipe(body_html, **kwargs) if body_html else ""
    if not pipe:
        stripped = _WS_RE.sub(" ", _html.unescape(_TAG_RE.sub(" ", text))).strip()
        return stripped[:max_chars]

    out = f"{caption}\n{pipe}" if caption else pipe
    return out[:max_chars]
