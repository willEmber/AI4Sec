"""Reading tools: outline, search, section read.

Each one registers what it returns as evidence, so the answer has something to
cite and a citation can be resolved back to a page. A tool that returned prose
without evidence would let the model write unfalsifiable claims.

Access is checked against the session, not against the argument: `paper_id`
comes from the model, so a tool must confirm the paper is actually attached to
the caller's session before reading it. Otherwise guessing an id would be
enough to read someone else's upload (acceptance case A16).
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.tools import ToolRuntime, tool

from app.agents.context import AgentContext
from app.db import agent_repository as repo
from app.db import database as db
from app.models.agent_models import ErrorCode, ToolResult
from app.services import evidence_service, paper_catalog
from app.services.ir_extract import compact_table_text
from app.services.qa_retrieval import (
    PaperNode,
    load_paper_nodes,
    load_section_nodes,
    rank_paper_chunks,
)

logger = logging.getLogger("scholar.agents.tools.reading")

# A section read returns the whole section, which can be most of a paper. This
# caps what reaches the model per call. At 12k an experiments section came
# back cut in half and the model had to find its way to the rest through the
# outline; 40k (~10k tokens) returns nearly any section whole, and a paper in
# a handful of reads.
MAX_SECTION_CHARS = 40_000
# Table markup is verbose out of all proportion to what it says.
MAX_TABLE_CHARS = 1_600


async def _authorize(
    ctx: AgentContext, paper_id: str
) -> tuple[str, ToolResult | None]:
    """Resolve a paper the session may read, or explain why it may not.

    Returns `(paper_id, None)` on success, `("", ToolResult)` otherwise. When
    the session holds exactly one paper, an omitted id resolves to it — the
    common single-paper case, where making the model repeat an opaque SHA-1
    buys nothing.
    """
    papers = await repo.list_session_papers(ctx.session_id)
    readable = [p for p in papers if p.paper_id]

    if not paper_id:
        if len(readable) == 1:
            return readable[0].paper_id, None
        if not readable:
            return "", ToolResult.unavailable(
                ErrorCode.PAPER_NOT_FOUND,
                "This session has no readable paper yet.",
            )
        return "", ToolResult.failed(
            ErrorCode.INVALID_ARGUMENT,
            "This session has several papers; pass the paper_id you mean. "
            f"Available: {', '.join(p.paper_id for p in readable)}",
            retryable=False,
        )

    if not await repo.session_owns_paper(ctx.session_id, paper_id):
        # Deliberately the same answer whether the paper does not exist or
        # belongs to someone else: distinguishing them leaks their existence.
        return "", ToolResult.unavailable(
            ErrorCode.PAPER_NOT_FOUND,
            f"No paper {paper_id} in this session.",
        )
    return paper_id, None


async def _authorize_many(
    ctx: AgentContext, paper_ids: list[str] | None
) -> tuple[list[str], ToolResult | None]:
    """Resolve a set of papers the session may read.

    An omitted list means every readable paper in the session — which is what
    makes a comparison question work without the model first having to list
    them. Each id is still checked individually; a set is not a way around the
    per-paper check.
    """
    papers = await repo.list_session_papers(ctx.session_id)
    readable = [p.paper_id for p in papers if p.paper_id]

    if not paper_ids:
        if not readable:
            return [], ToolResult.unavailable(
                ErrorCode.PAPER_NOT_FOUND,
                "This session has no readable paper yet.",
            )
        return readable, None

    allowed = [pid for pid in paper_ids if pid in readable]
    rejected = [pid for pid in paper_ids if pid not in readable]
    if not allowed:
        return [], ToolResult.unavailable(
            ErrorCode.PAPER_NOT_FOUND,
            f"None of those papers are in this session: {', '.join(paper_ids)}",
        )
    if rejected:
        logger.info("Dropping papers outside session %s: %s", ctx.session_id, rejected)
    return allowed, None


def _node_payload(node: PaperNode, *, evidence_id: str = "") -> dict[str, Any]:
    text = node.text.strip()
    if node.block_type == "table":
        text = compact_table_text(text, max_chars=MAX_TABLE_CHARS)
    payload: dict[str, Any] = {
        "section": node.title_path,
        "page": node.page_start + 1,   # 1-based for the model, as a reader sees it
        "type": node.block_type or node.node_type,
        "text": text,
    }
    if evidence_id:
        payload["evidence_id"] = evidence_id
    return payload


@tool(parse_docstring=False)
async def get_paper_outline(
    runtime: ToolRuntime[AgentContext],
    paper_id: str = "",
    max_depth: int = 3,
) -> str:
    """Return a parsed paper's section outline with page numbers.

    Use this to find out what a paper contains before deciding what to read.
    It returns section ids you can pass to read_paper_section. Omit paper_id
    when the session holds a single paper.
    """
    ctx = runtime.context
    paper_id, denial = await _authorize(ctx, paper_id)
    if denial is not None:
        return denial.to_json()

    nodes = await load_section_nodes(paper_id)
    sections = [n for n in nodes if n.node_type == "section" and n.depth <= max_depth]
    if not sections:
        return ToolResult.unavailable(
            ErrorCode.NOT_PARSED,
            f"Paper {paper_id} has no parsed sections yet.",
        ).to_json()

    root = next((n for n in nodes if n.node_type == "paper"), None)
    outline = [
        {
            "section_id": n.node_id,
            "title": n.title or n.title_path,
            "path": n.title_path,
            "depth": n.depth,
            "page": n.page_start + 1,
            "page_end": n.page_end + 1,
        }
        for n in sorted(sections, key=lambda n: (n.page_start, n.order_idx))
    ]
    return ToolResult.ok(
        {
            "paper_id": paper_id,
            "title": root.title if root else "",
            "page_count": (root.page_end + 1) if root else 0,
            "sections": outline,
        }
    ).to_json()


async def _search_one(
    ctx: AgentContext, paper_id: str, question: str, limit: int
) -> tuple[list[dict[str, Any]], list[str]]:
    """Rank one paper's passages and register each as evidence."""
    ranked = await rank_paper_chunks(paper_id, question, limit=limit)
    if not ranked:
        return [], []

    literature_id = await paper_catalog.literature_for_paper(paper_id)
    hits: list[dict[str, Any]] = []
    evidence_ids: list[str] = []
    for _score, node in ranked:
        evidence = await evidence_service.record_fulltext_evidence(
            paper_id=paper_id,
            quote=node.text,
            owner_id=ctx.owner_id,
            session_id=ctx.session_id,
            locator=evidence_service.locator_from_node(
                {
                    "node_id": node.node_id,
                    "title_path": node.title_path,
                    "page_start": node.page_start,
                    "page_end": node.page_end,
                    "block_start": node.block_start,
                    "block_end": node.block_end,
                }
            ),
            literature_id=literature_id,
        )
        evidence_ids.append(evidence.evidence_id)
        hits.append(_node_payload(node, evidence_id=evidence.evidence_id))
    return hits, evidence_ids


@tool(parse_docstring=False)
async def search_paper_content(
    question: str,
    runtime: ToolRuntime[AgentContext],
    paper_id: str = "",
    paper_ids: list[str] | None = None,
    limit: int = 6,
) -> str:
    """Find passages bearing on a question, in one paper or across several.

    Returns short excerpts with page numbers and an evidence_id for each, which
    you cite in your answer. Good for locating where something is discussed;
    when an excerpt is too thin to settle the question, follow it with
    read_paper_section. Omit paper_id when the session holds a single paper.
    To compare papers, pass paper_ids — the same question is run against each
    and the results come back grouped by paper, so differences in setup and
    results line up.
    """
    ctx = runtime.context

    question = (question or "").strip()
    if not question:
        return ToolResult.failed(
            ErrorCode.INVALID_ARGUMENT, "question must not be empty."
        ).to_json()

    limit = max(1, min(int(limit or 6), 12))

    if paper_ids:
        return await _search_across(ctx, paper_ids, question, limit)

    paper_id, denial = await _authorize(ctx, paper_id)
    if denial is not None:
        return denial.to_json()

    hits, evidence_ids = await _search_one(ctx, paper_id, question, limit)
    if not hits:
        # Not an error: "the paper does not discuss this" is a real finding, and
        # reporting it as a failure would invite a pointless retry.
        return ToolResult.ok(
            {"paper_id": paper_id, "question": question, "hits": []},
            note="No passage matched. The paper may not discuss this; consider "
            "the outline to check which sections exist.",
        ).to_json()

    return ToolResult.ok(
        {"paper_id": paper_id, "question": question, "hits": hits},
        evidence_ids=evidence_ids,
    ).to_json()


async def _search_across(
    ctx: AgentContext, paper_ids: list[str], question: str, limit: int
) -> str:
    """Run one question against several papers and group the answers by paper.

    Grouped rather than merged into one ranked list: a comparison needs to know
    which paper each passage came from, and a global ranking would quietly let
    the paper with the better-matching wording supply most of the evidence.
    """
    allowed, denial = await _authorize_many(ctx, paper_ids)
    if denial is not None:
        return denial.to_json()

    # Per paper, so one verbose paper cannot crowd the others out of the answer.
    per_paper = max(2, min(limit, 6))
    groups: list[dict[str, Any]] = []
    evidence_ids: list[str] = []
    silent: list[str] = []

    for pid in allowed:
        hits, ids = await _search_one(ctx, pid, question, per_paper)
        title = await _paper_title(pid)
        groups.append(
            {"paper_id": pid, "title": title, "hits": hits}
        )
        evidence_ids.extend(ids)
        if not hits:
            silent.append(title or pid)

    data = {"question": question, "papers": groups}
    if not evidence_ids:
        return ToolResult.ok(
            data,
            note="None of these papers discuss this. Check their outlines, or say "
            "the comparison cannot be made on this point.",
        ).to_json()
    if silent:
        return ToolResult.partial(
            data,
            note=(
                "No passage matched in: "
                + "; ".join(silent)
                + ". Do not treat silence as disagreement — say which papers the "
                "comparison could not cover."
            ),
            evidence_ids=evidence_ids,
        ).to_json()
    return ToolResult.ok(data, evidence_ids=evidence_ids).to_json()


async def _paper_title(paper_id: str) -> str:
    row = await db.fetch_one("SELECT title FROM papers WHERE paper_id = ?", (paper_id,))
    return (row or {}).get("title") or ""


@tool(parse_docstring=False)
async def read_paper_section(
    runtime: ToolRuntime[AgentContext],
    section_id: str = "",
    paper_id: str = "",
    section_title: str = "",
) -> str:
    """Read one section of a parsed paper in full, including its tables and equations.

    Use this when a search excerpt is not enough — for exact numbers, an
    equation, an experimental setup, or an argument that spans paragraphs.
    Identify the section by the section_id from get_paper_outline, or by
    section_title when you only know its name.
    """
    ctx = runtime.context
    paper_id, denial = await _authorize(ctx, paper_id)
    if denial is not None:
        return denial.to_json()

    nodes = await load_paper_nodes(paper_id)
    if not nodes:
        return ToolResult.unavailable(
            ErrorCode.NOT_PARSED, f"Paper {paper_id} is not parsed."
        ).to_json()

    section = _find_section(nodes, section_id=section_id, section_title=section_title)
    if section is None:
        available = [
            n.title_path for n in nodes if n.node_type == "section"
        ][:20]
        return ToolResult.failed(
            ErrorCode.SECTION_NOT_FOUND,
            f"No section matched. Available sections: {'; '.join(available)}",
            retryable=False,
        ).to_json()

    # A section node covers its descendants' blocks, so range containment is
    # what selects the body — matching on section_path alone would drop the
    # blocks that sit under a subsection.
    children = [
        n
        for n in nodes
        if n.node_type == "chunk"
        and section.block_start <= n.block_start <= section.block_end
        and n.text.strip()
    ]
    children.sort(key=lambda n: n.order_idx)
    if not children:
        return ToolResult.ok(
            {"paper_id": paper_id, "section": section.title_path, "blocks": []},
            note="The section is present in the outline but parsed with no body text.",
        ).to_json()

    literature_id = await paper_catalog.literature_for_paper(paper_id)
    blocks: list[dict[str, Any]] = []
    evidence_ids: list[str] = []
    used = 0
    truncated = False
    for node in children:
        payload = _node_payload(node)
        used += len(payload["text"])
        if used > MAX_SECTION_CHARS:
            truncated = True
            break
        evidence = await evidence_service.record_fulltext_evidence(
            paper_id=paper_id,
            quote=node.text,
            owner_id=ctx.owner_id,
            session_id=ctx.session_id,
            locator=evidence_service.locator_from_node(
                {
                    "node_id": node.node_id,
                    "title_path": node.title_path,
                    "page_start": node.page_start,
                    "page_end": node.page_end,
                    "block_start": node.block_start,
                    "block_end": node.block_end,
                }
            ),
            literature_id=literature_id,
        )
        payload["evidence_id"] = evidence.evidence_id
        evidence_ids.append(evidence.evidence_id)
        blocks.append(payload)

    data = {
        "paper_id": paper_id,
        "section": section.title_path,
        "section_id": section.node_id,
        "page": section.page_start + 1,
        "page_end": section.page_end + 1,
        "blocks": blocks,
    }
    if truncated:
        return ToolResult.partial(
            data,
            note=(
                f"Section truncated at {MAX_SECTION_CHARS} characters. Read a "
                "subsection from the outline if you need the rest."
            ),
            evidence_ids=evidence_ids,
        ).to_json()
    return ToolResult.ok(data, evidence_ids=evidence_ids).to_json()


def _find_section(
    nodes: list[PaperNode], *, section_id: str, section_title: str
) -> PaperNode | None:
    """Locate a section by id, then by title, then by a loose title match."""
    sections = [n for n in nodes if n.node_type == "section"]
    if section_id:
        for node in sections:
            if node.node_id == section_id:
                return node
    needle = (section_title or section_id or "").strip().lower()
    if not needle:
        return None
    for node in sections:
        if node.title.strip().lower() == needle or node.title_path.strip().lower() == needle:
            return node
    # Loose match last: the model often writes "Method" for "3 Method".
    for node in sections:
        if needle in node.title_path.lower():
            return node
    return None


READING_TOOLS = [get_paper_outline, search_paper_content, read_paper_section]
