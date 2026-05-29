"""Lightweight citation-coverage audit for generated reports.

Looks at a rendered Markdown report and flags "fact-like" sentences that do not
carry a ``[p.X]`` citation. Intended for logging only — it does not block the
pipeline or mutate the report.

Heuristics are intentionally conservative: code blocks, headings, table
separator rows, standalone bold labels, and lines that already declare
``_Not reported in extracted text._`` are excluded. Short fragments (< 20
chars) are dropped because they are usually bullet headers, not claims.
"""

from __future__ import annotations

import re
from typing import Any


_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_MATH_BLOCK_RE = re.compile(r"\$\$.+?\$\$", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]+`")
_MATH_INLINE_RE = re.compile(r"\$[^$\n]+\$")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}.*$")
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_BOLD_LABEL_RE = re.compile(r"^\*\*[^*]{1,60}\*\*\s*[:：]?\s*$")
_CITATION_RE = re.compile(r"\[p\.\d+\]", re.IGNORECASE)
_FALLBACK_RE = re.compile(r"_?\s*not\s+reported", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+(?=[\(\[\"'A-Z一-鿿])")

_MIN_CLAIM_CHARS = 20
_TABLE_CELL_MIN_CHARS = 25


def _strip_blocks(md: str) -> str:
    md = _CODE_FENCE_RE.sub(" ", md)
    md = _MATH_BLOCK_RE.sub(" ", md)
    return md


def _line_is_skippable(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if _HEADING_RE.match(stripped):
        return True
    if _TABLE_SEP_RE.match(stripped):
        return True
    if _BOLD_LABEL_RE.match(stripped):
        return True
    return False


def _line_to_claim_text(line: str) -> str:
    text = _LIST_PREFIX_RE.sub("", line)
    text = _INLINE_CODE_RE.sub(" ", text)
    text = _MATH_INLINE_RE.sub(" ", text)
    return text.strip()


def validate_citation_coverage(
    markdown: str,
    *,
    max_samples: int = 5,
) -> dict[str, Any]:
    """Return a coverage audit for ``[p.X]`` citations in ``markdown``.

    The result has::

        {
            "claims_total": int,         # candidate fact sentences/cells
            "claims_uncited": int,       # subset missing any [p.X] citation
            "coverage": float,           # 1.0 means everything cited
            "uncited_samples": list[str] # up to ``max_samples`` example snippets
        }

    Sentences explicitly tagged ``_Not reported in extracted text._`` and very
    short bullet fragments are excluded from the total.
    """
    if not markdown or not markdown.strip():
        return {
            "claims_total": 0,
            "claims_uncited": 0,
            "coverage": 1.0,
            "uncited_samples": [],
        }

    cleaned = _strip_blocks(markdown)

    claims_total = 0
    claims_uncited = 0
    samples: list[str] = []

    for raw_line in cleaned.splitlines():
        line = raw_line.rstrip()
        if _line_is_skippable(line):
            continue

        if _TABLE_ROW_RE.match(line):
            for cell in (c.strip() for c in line.strip().strip("|").split("|")):
                if len(cell) < _TABLE_CELL_MIN_CHARS:
                    continue
                if _FALLBACK_RE.search(cell):
                    continue
                claims_total += 1
                if not _CITATION_RE.search(cell):
                    claims_uncited += 1
                    if len(samples) < max_samples:
                        samples.append(cell[:160])
            continue

        text = _line_to_claim_text(line)
        if not text:
            continue
        for sent in _SENTENCE_SPLIT_RE.split(text):
            sent = sent.strip()
            if len(sent) < _MIN_CLAIM_CHARS:
                continue
            if _FALLBACK_RE.search(sent):
                continue
            claims_total += 1
            if not _CITATION_RE.search(sent):
                claims_uncited += 1
                if len(samples) < max_samples:
                    samples.append(sent[:160])

    coverage = (
        (claims_total - claims_uncited) / claims_total
        if claims_total
        else 1.0
    )
    return {
        "claims_total": claims_total,
        "claims_uncited": claims_uncited,
        "coverage": round(coverage, 3),
        "uncited_samples": samples,
    }


def format_coverage_summary(audit: dict[str, Any]) -> str:
    """One-line log-friendly summary."""
    total = audit.get("claims_total", 0)
    uncited = audit.get("claims_uncited", 0)
    coverage = audit.get("coverage", 0.0)
    return f"citation_coverage={coverage:.2%} ({total - uncited}/{total} cited)"
