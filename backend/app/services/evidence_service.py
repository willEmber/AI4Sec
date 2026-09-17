"""Creating, resolving and checking evidence.

The point of the snapshot. An evidence row stores the quote itself, not a
pointer into `blocks`. `build_and_store_paper_ir` deletes and reinserts every
block on a re-parse, so a pointer would dangle; the snapshot plus the parse
version means an answer given before the re-parse still resolves to the text
that was actually read (acceptance case A17). What the *current* parse says is
a separate question, answered by `check_against_current_parse`.

What is checked here is mechanical only: the citation exists, the caller may
read it, and the stored quote still hashes to what was recorded. Whether the
quote supports the sentence it was attached to is a judgement for evaluation,
not for this module (development plan §5 rule 6).
"""

from __future__ import annotations

import logging
from typing import Any

from app.db import agent_repository as repo
from app.db import database as db
from app.models.evidence_models import (
    CitationCheck,
    Evidence,
    Locator,
    SourceLevel,
    compute_evidence_id,
    content_hash,
)
from app.services import paper_catalog

logger = logging.getLogger("scholar.evidence")

# Quotes are snapshots, not whole sections: long enough to be checkable, short
# enough that the evidence table does not become a second copy of the corpus.
MAX_QUOTE_CHARS = 2000


def _truncate(quote: str) -> str:
    quote = (quote or "").strip()
    if len(quote) <= MAX_QUOTE_CHARS:
        return quote
    return quote[:MAX_QUOTE_CHARS].rstrip() + " …"


async def _store(
    *,
    source_level: SourceLevel,
    quote: str,
    owner_id: str,
    session_id: str,
    literature_id: str = "",
    paper_id: str = "",
    parse_version: str = "",
    locator: Locator | None = None,
    source_url: str = "",
    provider: str = "",
) -> Evidence:
    quote = _truncate(quote)
    locator = locator or Locator()
    evidence = Evidence(
        evidence_id=compute_evidence_id(
            source_level=source_level,
            quote=quote,
            literature_id=literature_id,
            paper_id=paper_id,
            parse_version=parse_version,
            locator=locator,
            source_url=source_url,
        ),
        owner_id=owner_id,
        session_id=session_id,
        literature_id=literature_id,
        paper_id=paper_id,
        parse_version=parse_version,
        source_level=source_level,
        locator=locator,
        quote=quote,
        content_hash=content_hash(quote),
        source_url=source_url,
        provider=provider,
    )
    return await repo.insert_evidence(evidence)


async def record_fulltext_evidence(
    *,
    paper_id: str,
    quote: str,
    owner_id: str,
    session_id: str,
    locator: Locator | None = None,
    parse_version: str = "",
    literature_id: str = "",
) -> Evidence:
    """Record an excerpt read out of a parsed PDF.

    `parse_version` defaults to the paper's current version, so a caller that
    does not track versions still produces evidence pinned to a specific parse
    rather than to "whatever is in the table right now".
    """
    if not parse_version:
        current = await paper_catalog.get_current_version(paper_id)
        parse_version = current["version_id"] if current else ""
    if not literature_id:
        literature_id = await paper_catalog.literature_for_paper(paper_id)
    return await _store(
        source_level=SourceLevel.FULLTEXT,
        quote=quote,
        owner_id=owner_id,
        session_id=session_id,
        literature_id=literature_id,
        paper_id=paper_id,
        parse_version=parse_version,
        locator=locator,
        provider="mineru",
    )


async def record_abstract_evidence(
    *,
    literature_id: str,
    quote: str,
    owner_id: str,
    session_id: str,
    provider: str = "",
    source_url: str = "",
) -> Evidence:
    """Record an abstract-level excerpt — used when the full text is unreachable.

    Kept as its own source level so an answer can say it is working from the
    abstract rather than the paper (acceptance case A08).
    """
    return await _store(
        source_level=SourceLevel.ABSTRACT,
        quote=quote,
        owner_id=owner_id,
        session_id=session_id,
        literature_id=literature_id,
        provider=provider,
        source_url=source_url,
    )


async def record_metadata_evidence(
    *,
    literature_id: str,
    quote: str,
    owner_id: str,
    session_id: str,
    provider: str,
    source_url: str = "",
) -> Evidence:
    """Record a bibliographic fact (venue rank, citation count, year).

    Deliberately carries no page: forcing a page number onto a record that has
    none would make a metadata claim look like a reading of the paper
    (development plan §5 rule 5).
    """
    return await _store(
        source_level=SourceLevel.METADATA,
        quote=quote,
        owner_id=owner_id,
        session_id=session_id,
        literature_id=literature_id,
        provider=provider,
        source_url=source_url,
    )


async def record_web_evidence(
    *,
    quote: str,
    source_url: str,
    owner_id: str,
    session_id: str,
    provider: str = "web",
    literature_id: str = "",
) -> Evidence:
    """Record an excerpt from a web page or API response."""
    return await _store(
        source_level=SourceLevel.EXTERNAL_WEB,
        quote=quote,
        owner_id=owner_id,
        session_id=session_id,
        literature_id=literature_id,
        source_url=source_url,
        provider=provider,
    )


def locator_from_node(node: dict[str, Any]) -> Locator:
    """Build a locator from a `paper_nodes` row.

    `block_start` / `block_end` are positions in the parse's block order, not
    `blocks.block_id`; they stay meaningful for the version they were taken
    from, which the autoincrement key does not.
    """
    return Locator(
        node_id=node.get("node_id", "") or "",
        section_path=node.get("title_path", "") or "",
        section_id=node.get("node_id", "") or "",
        page_index=node.get("page_start"),
        page_end_index=node.get("page_end"),
        block_start=node.get("block_start"),
        block_end=node.get("block_end"),
    )


def locator_from_block(block: dict[str, Any]) -> Locator:
    """Build a locator from a `blocks` row."""
    return Locator(
        section_path=block.get("section_path", "") or "",
        page_index=block.get("page_idx"),
        page_end_index=block.get("page_idx"),
        block_start=block.get("order_idx"),
        block_end=block.get("order_idx"),
    )


# ── Resolution and checking ─────────────────────────────────────────────────


async def resolve(evidence_id: str, *, owner_id: str = "") -> Evidence:
    """Fetch one evidence row, enforcing ownership. Raises `EvidenceNotFound`."""
    return await repo.get_evidence(evidence_id, owner_id=owner_id)


async def resolve_many(evidence_ids: list[str], *, owner_id: str = "") -> list[Evidence]:
    return await repo.get_evidence_many(evidence_ids, owner_id=owner_id)


async def validate_citations(
    evidence_ids: list[str], *, owner_id: str = ""
) -> list[CitationCheck]:
    """Check each citation mechanically: exists, readable, internally consistent."""
    checks: list[CitationCheck] = []
    found = {e.evidence_id: e for e in await repo.get_evidence_many(evidence_ids)}
    for evidence_id in evidence_ids:
        evidence = found.get(evidence_id)
        if evidence is None:
            checks.append(
                CitationCheck(evidence_id=evidence_id, reason="no such evidence")
            )
            continue
        authorized = not owner_id or not evidence.owner_id or evidence.owner_id == owner_id
        if not authorized:
            checks.append(
                CitationCheck(
                    evidence_id=evidence_id,
                    exists=True,
                    reason="evidence belongs to another session owner",
                )
            )
            continue
        matches = content_hash(evidence.quote) == evidence.content_hash
        checks.append(
            CitationCheck(
                evidence_id=evidence_id,
                exists=True,
                authorized=True,
                quote_matches=matches,
                reason="" if matches else "stored quote no longer matches its hash",
            )
        )
    return checks


async def check_against_current_parse(evidence: Evidence) -> dict[str, Any]:
    """Report how an old citation relates to the paper's current parse.

    Separate from `validate_citations` on purpose: evidence taken from a
    superseded version is still valid evidence of what was read. This only says
    whether the newest parse still contains the same text, which is what a UI
    needs to decide between "jump to page" and "this quote predates a re-parse".
    """
    result: dict[str, Any] = {
        "evidence_id": evidence.evidence_id,
        "parse_version": evidence.parse_version,
        "is_current_version": False,
        "current_version": "",
        "still_present": None,
    }
    if not evidence.paper_id:
        return result

    current = await paper_catalog.get_current_version(evidence.paper_id)
    if current is None:
        return result
    result["current_version"] = current["version_id"]
    result["is_current_version"] = current["version_id"] == evidence.parse_version

    needle = evidence.quote.strip().rstrip(" …")
    if not needle:
        return result
    # Compare against a distinctive head of the quote: MinerU re-runs often
    # differ in trailing whitespace or hyphenation, and a whole-quote equality
    # test would report a spurious mismatch for text that plainly survived.
    probe = needle[:120]
    row = await db.fetch_one(
        "SELECT 1 AS hit FROM blocks WHERE paper_id = ? AND text LIKE ? LIMIT 1",
        (evidence.paper_id, f"%{probe}%"),
    )
    result["still_present"] = row is not None
    return result


async def evidence_to_api(evidence: Evidence) -> dict[str, Any]:
    """Client-facing form: the citation, its work, and its parse standing."""
    payload = evidence.to_api()
    if evidence.literature_id:
        item = await paper_catalog.get_literature_item(evidence.literature_id)
        if item:
            payload["literature"] = {
                "literature_id": item["literature_id"],
                "title": item["title"],
                "year": item["year"] if item["year_known"] else None,
                "year_known": bool(item["year_known"]),
                "venue": item["venue"],
                "doi": item["doi"],
                "arxiv_id": item["arxiv_id"],
            }
    payload["parse_status"] = await check_against_current_parse(evidence)
    return payload
