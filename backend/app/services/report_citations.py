"""Citations of a report written as an answer: frozen when it is saved, converted on export (P16).

A report written by the conversation cites evidence the way any answer does,
stored as `[ev_…]`. Two things follow that an answer never had to deal with:

*A report is shared.* Anyone holding its run id may read it, but evidence is
owner-checked (`GET /agent/evidence/{id}`), so a reader of a shared link would
see numbers that open nothing. The snapshot therefore freezes the passages the
report cites into its own `json_data.citations`, numbered in order of first
appearance, and the report page renders from that table alone.

*A report leaves the application.* A Markdown file or a Zotero note has no
evidence API either, so the export replaces each id with its number and adds a
references section listing what each number quotes. Both exports go through
:func:`to_numbered`, so the two cannot number differently.

Old reports cite pages as `[p.X]` and have no table; they pass through
unchanged.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from app.db import agent_repository as repo
from app.services import evidence_service

# The longest quote a frozen citation keeps. A passage is cited for a sentence
# or two; the whole section stays a click away in the conversation.
QUOTE_LIMIT = 600

_ID = r"ev_[0-9a-f]{20}(?![0-9a-f])"
_ID_RE = re.compile(_ID)
# One bracket holding several ids, as a model sometimes writes them:
# `[ev_a, ev_b]`, `【ev_a、ev_b】`. The rewriter emits `[ev_a][ev_b]`, but an id
# typed by the model itself is kept as written.
_GROUP_RE = re.compile(rf"[\[【]\s*{_ID}(?:\s*[,，、;；]\s*{_ID})*\s*[\]】]")
_BRACKETED_RE = re.compile(rf"\[({_ID})\]")
_REPEATED_PAGE_RE = re.compile(r"(\[p\.[^\]]+\])(?:\s*\1)+")


def normalise(markdown: str) -> str:
    """Every citation as `[ev_…]`, one id per bracket."""
    return _GROUP_RE.sub(
        lambda m: "".join(f"[{eid}]" for eid in _ID_RE.findall(m.group(0))), markdown
    )


def cited_ids(markdown: str) -> list[str]:
    """The evidence ids a text cites, in order of first appearance."""
    return list(dict.fromkeys(_ID_RE.findall(markdown)))


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").removeprefix("www.")
    except ValueError:
        return ""


def _one_line(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


async def freeze(markdown: str, *, owner_id: str) -> tuple[str, dict[str, dict[str, Any]]]:
    """The report's text and its citation table.

    Ids are resolved as the report's owner, so a passage only its owner could
    open is what everyone holding the link now sees — the passages the report
    cites and nothing else from the conversation. An id that resolves to
    nothing is taken out of the text: the runner's filter normally makes that
    impossible, and a number that opens nothing is worse than none.
    """
    text = normalise(markdown)
    ids = cited_ids(text)
    found = {
        e.evidence_id: e for e in await evidence_service.resolve_many(ids, owner_id=owner_id)
    }
    titles = await repo.literature_titles([e.literature_id for e in found.values()])
    table: dict[str, dict[str, Any]] = {}
    for eid in ids:
        evidence = found.get(eid)
        if evidence is None:
            continue
        table[eid] = {
            "n": len(table) + 1,
            "source_level": getattr(evidence.source_level, "value", evidence.source_level),
            "paper_id": evidence.paper_id,
            "literature_id": evidence.literature_id,
            "title": titles.get(evidence.literature_id, "") or _host(evidence.source_url),
            "page_label": evidence.locator.page_label,
            "section": evidence.locator.section_path,
            "quote": evidence.quote[:QUOTE_LIMIT],
            "source_url": evidence.source_url,
            "provider": evidence.provider,
        }
    if len(table) < len(ids):
        text = _BRACKETED_RE.sub(lambda m: m.group(0) if m.group(1) in table else "", text)
        text = _ID_RE.sub(lambda m: m.group(0) if m.group(0) in table else "", text)
    return text, table


def _reference(entry: dict[str, Any], *, paper_id: str, zh: bool) -> str:
    """One line of the references section, after its number."""
    open_q, close_q = ("「", "」") if zh else ("“", "”")
    quote = _one_line(entry.get("quote", ""), 300)
    quoted = f"{open_q}{quote}{close_q}" if quote else ""
    page = entry.get("page_label")
    where = f"p.{page}" if page else ""
    level = entry.get("source_level", "")
    title = _one_line(entry.get("title", ""), 160)
    url = entry.get("source_url", "")

    if level == "fulltext" and entry.get("paper_id") == paper_id:
        head = where or (title if not zh else "本文")
    elif level in ("fulltext", "abstract"):
        label = "摘要" if zh else "abstract"
        head = ", ".join(p for p in (title, where if level == "fulltext" else label) if p)
    elif level == "metadata":
        head = " · ".join(p for p in (entry.get("provider", ""), title) if p)
    else:
        head = title or _host(url)
    # 「」 carry their own spacing; an English quote needs it.
    line = (" —" if zh else " — ").join(p for p in (head, quoted) if p)
    if url and level in ("metadata", "external_web", "abstract"):
        line += f" — {url}"
    return line


def to_pages(markdown: str, citations: dict[str, dict[str, Any]], *, paper_id: str) -> str:
    """The report with this paper's passages cited as `[p.X]` and the rest left out.

    For what reads a report by its pages: the Lens digest copies `[p.X]` into
    the cards, whose badges jump into the PDF. Repeats of one page in a row
    collapse, as `[ev_a][ev_b]` on one page would otherwise read `[p.5][p.5]`.
    """

    def page(eid: str) -> str:
        entry = citations.get(eid) or {}
        label = entry.get("page_label")
        if entry.get("paper_id") == paper_id and entry.get("source_level") == "fulltext" and label:
            return f"[p.{label}]"
        return ""

    text = normalise(markdown)
    text = _BRACKETED_RE.sub(lambda m: page(m.group(1)), text)
    text = _ID_RE.sub(lambda m: page(m.group(0)), text)
    return _REPEATED_PAGE_RE.sub(r"\1", text)


def to_numbered(
    markdown: str, citations: dict[str, dict[str, Any]], *, paper_id: str, language: str
) -> str:
    """The report as a file outside the application: `[n]` and a references section.

    Without a table (a report from before P16) the text is returned as it is.
    """
    if not citations:
        return markdown

    def number(eid: str) -> str:
        entry = citations.get(eid)
        return f"[{entry['n']}]" if entry else ""

    text = normalise(markdown)
    text = _BRACKETED_RE.sub(lambda m: number(m.group(1)), text)
    text = _ID_RE.sub(lambda m: number(m.group(0)), text)

    zh = language == "zh"
    lines = [
        f"{entry['n']}. {_reference(entry, paper_id=paper_id, zh=zh)}"
        for entry in sorted(citations.values(), key=lambda e: e["n"])
    ]
    heading = "## 引用" if zh else "## References"
    return text.rstrip() + f"\n\n{heading}\n\n" + "\n".join(lines) + "\n"
