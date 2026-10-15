"""A paper's sections as the reading tools present them.

The stored section tree is only as good as the parse. MinerU marks most
headings `text_level=1`, so for some papers every section sits at depth 1:
"3 Methodology" then covers nothing but its own heading, and a read of it
returns one line. A NeurIPS checklist appended to the paper adds twenty-odd
more "sections" (`1. Claims`, `Answer: [Yes]`, `Guidelines:`) that crowd the
outline. Rebuilding stored parses is a separate job; this works on what is
stored, at read time:

* a section's level is the deeper of its stored depth and its numbering
  (`3.1`, `3.1.`, `A.1` → 2), and its body runs through the deeper sections
  that follow it, so reading "3 Methodology" returns 3.1 to 3.4;
* everything after a heading that names a checklist is folded under it, up
  to an appendix heading;
* each section gets a short id, `<first 8 of paper_id>:s<n>`, instead of the
  stored `<paper_id>:section:<n>:<title_with_underscores>` — a dozen
  characters rather than ninety, and nothing for the model to retype wrongly.

Ids are resolved generously: the short id, the stored node id, a stored id
whose title part was mistyped (its index still names the section), and last
the title itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.qa_retrieval import PaperNode

# "3 Method", "3.1 Overview", "3.1. Overview", "A Scheme details", "A.1 Design".
_NUMBERED_RE = re.compile(r"^\s*((?:\d{1,2}|[A-Z])(?:\.\d{1,2})*)\.?\s+\S")
_CHECKLIST_RE = re.compile(r"\bchecklist\b", re.IGNORECASE)
# What ends a checklist trailer: an appendix after it ("Appendix", "A Proofs").
_APPENDIX_RE = re.compile(r"^\s*(?:appendix|supplementary|[A-Z](?:\.\d{1,2})*\.?\s)", re.IGNORECASE)
_STORED_INDEX_RE = re.compile(r":section:(\d+)(?::|$)")
_SHORT_RE = re.compile(r"^([0-9A-Za-z]{4,40}):s(\d+)$")
SHORT_PAPER_CHARS = 8


@dataclass
class SectionView:
    node: PaperNode
    title: str
    level: int
    # Last block of the section including the deeper sections under it.
    block_end: int
    short_id: str
    # Under a checklist-like trailer: not listed in the outline.
    folded: bool = False
    # For a trailer heading, how many sections are folded under it.
    folded_count: int = 0


def numbered_level(title: str) -> int:
    """How deep a heading's numbering puts it; 0 when it has none."""
    match = _NUMBERED_RE.match(title or "")
    return match.group(1).count(".") + 1 if match else 0


def stored_index(node_id: str) -> str:
    match = _STORED_INDEX_RE.search(node_id or "")
    return match.group(1) if match else ""


def short_id_of(node: PaperNode) -> str:
    index = stored_index(node.node_id)
    if not index:
        return node.node_id   # not the stored format (fixtures); already short
    return f"{node.paper_id[:SHORT_PAPER_CHARS]}:s{index}"


def section_views(nodes: list[PaperNode]) -> list[SectionView]:
    """The paper's sections in reading order, with levels and full extents."""
    sections = sorted(
        (n for n in nodes if n.node_type == "section"),
        key=lambda n: (n.block_start, n.order_idx),
    )
    views: list[SectionView] = []
    trailer: SectionView | None = None
    for node in sections:
        title = (node.title or node.title_path.split("/")[-1]).strip()
        level = max(node.depth, numbered_level(title), 1)
        view = SectionView(
            node=node, title=title, level=level, block_end=node.block_end,
            short_id=short_id_of(node),
        )
        if trailer is not None and _APPENDIX_RE.match(title) and not title[:1].isdigit():
            trailer = None
        if trailer is not None:
            view.level = max(view.level, trailer.level + 1)
            view.folded = True
            trailer.folded_count += 1
        elif _CHECKLIST_RE.search(title):
            view.level = 1
            trailer = view
        views.append(view)

    for i, view in enumerate(views):
        for later in views[i + 1:]:
            if later.level <= view.level:
                break
            view.block_end = max(view.block_end, later.node.block_end)
    return views


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text or "").strip().lower()


def find_section(
    views: list[SectionView], *, section_id: str = "", section_title: str = ""
) -> SectionView | None:
    """The section an id or title names, tolerating the usual mistypes."""
    wanted = (section_id or "").strip()
    if wanted:
        for view in views:
            if wanted in (view.node.node_id, view.short_id):
                return view
        # An id of another paper names nothing here: its index or title part
        # would otherwise pick an unrelated section of this one.
        head = wanted.partition(":")[0] if ":" in wanted else ""
        paper_id = views[0].node.paper_id if views else ""
        if head and not (len(head) >= 4 and paper_id.startswith(head)):
            return None
        index = ""
        short = _SHORT_RE.match(wanted)
        if short:
            index = short.group(2)
        else:
            index = stored_index(wanted)
        if index:
            for view in views:
                if stored_index(view.node.node_id) == index:
                    return view

    needle = section_title or ""
    if not needle and wanted:
        # A stored id's title part, or whatever follows the last colon.
        needle = re.split(r":section:\d+:", wanted)[-1].rsplit(":", 1)[-1]
    needle = _norm(needle)
    if not needle:
        return None
    for view in views:
        if needle in (_norm(view.title), _norm(view.node.title_path)):
            return view
    # Loose last: the model often writes "Method" for "3 Method".
    for view in views:
        if needle in _norm(view.node.title_path):
            return view
    return None


def paper_of_short_id(section_id: str, paper_ids: list[str]) -> str:
    """The paper a section id's prefix names, among `paper_ids`, or ""."""
    head, sep, _rest = (section_id or "").partition(":")
    if not sep or not head:
        return ""
    if head in paper_ids:
        return head
    if len(head) >= 6:
        matches = [pid for pid in paper_ids if pid.startswith(head)]
        if len(matches) == 1:
            return matches[0]
    return ""
