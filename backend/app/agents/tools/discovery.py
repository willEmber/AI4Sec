"""Finding papers, pinning down what they are, and checking where they appeared.

These four tools are what separates "read this PDF" from "research this
question". They share one rule the reading tools also follow: whatever they
return is registered as evidence, at the level it was actually obtained. A
search result yields `abstract` evidence, never `fulltext` — the difference is
what stops an answer built on abstracts from reading as though the papers had
been read (acceptance case A08).

The second rule is about dates and sources. A missing publication year is not
year zero, and a venue rank taken from a web search is not the same claim as
one from a ranking database. Both are carried explicitly rather than smoothed
over, because a reader asking "papers since 2022" or "is this a Q1 journal"
is entitled to know which answers are verified (acceptance cases A04, A10).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from langchain.tools import ToolRuntime, tool

from app.agents.context import AgentContext
from app.db import agent_repository as repo
from app.models.agent_models import ErrorCode, ToolResult
from app.services import citation_graph, evidence_service, paper_catalog
from app.services.citation_graph import PaperMetadata
from app.services.paper_search import Settings as SearchSettings
from app.services.paper_search import search_papers as run_search
from app.services.paper_search.config import load_env_file
from app.services.publication_rank import query_publication_rank as lookup_rank

logger = logging.getLogger("scholar.agents.tools.discovery")

DEFAULT_PLATFORMS = ["OpenAlex", "arXiv", "SemanticScholar"]
KNOWN_PLATFORMS = {"openalex", "arxiv", "semanticscholar", "crossref", "ieeexplore"}

MAX_SEARCH_RESULTS = 20
MAX_RELATED_RESULTS = 25
# Enough for the model to judge relevance; the full abstract is in the evidence
# record and reaches the reader through the citation, not through the context.
ABSTRACT_EXCERPT_CHARS = 700

RELATIONS = {
    "references": "papers this one cites",
    "citations": "papers that cite this one",
    "related": "papers on the same topic",
    "recommendations": "papers recommended from this one",
}


def _http_client(timeout: float = 30.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"User-Agent": "scholar-agent/1.0 (mailto:noreply@example.org)"},
        timeout=timeout,
    )


async def _register_candidate(
    meta: PaperMetadata | dict[str, Any], *, source: str
) -> str:
    """Record a discovered paper as a work, whether or not it has a file.

    Everything the agent finds goes through here, so a later `download_paper`
    or `get_related_papers` can name it by `literature_id` instead of passing
    metadata back and forth through the model.
    """
    if isinstance(meta, PaperMetadata):
        return await paper_catalog.upsert_literature_item(
            title=meta.title,
            doi=meta.doi,
            arxiv_id=meta.arxiv_id,
            openalex_id=meta.openalex_id,
            s2_paper_id=meta.s2_paper_id,
            authors=[a.strip() for a in (meta.authors or "").split(";") if a.strip()],
            year=meta.year or None,
            venue=meta.venue,
            abstract=meta.abstract_text,
            source=source,
        )
    return await paper_catalog.upsert_literature_item(
        title=meta.get("title", ""),
        doi=meta.get("doi", ""),
        arxiv_id=meta.get("arxiv_id", ""),
        authors=[a.strip() for a in (meta.get("authors") or "").split(";") if a.strip()],
        year=meta.get("year") or None,
        venue=meta.get("venue", ""),
        abstract=meta.get("abstract", ""),
        url=meta.get("url", ""),
        source=source,
    )


async def _candidate_payload(
    literature_id: str, ctx: AgentContext, *, provider: str, source_url: str = ""
) -> tuple[dict[str, Any], str]:
    """Shape one candidate for the model and register its evidence.

    Evidence level follows what is actually held: an abstract when there is
    one, bare metadata when there is not. Nothing here is `fulltext`.
    """
    item = await paper_catalog.get_literature_item(literature_id) or {}
    abstract = (item.get("abstract") or "").strip()
    url = source_url or item.get("url") or ""

    if abstract:
        evidence = await evidence_service.record_abstract_evidence(
            literature_id=literature_id,
            quote=abstract,
            owner_id=ctx.owner_id,
            session_id=ctx.session_id,
            source_url=url,
            provider=provider,
        )
    else:
        summary = " · ".join(
            part
            for part in (item.get("title"), item.get("venue"), item.get("doi"))
            if part
        )
        evidence = await evidence_service.record_metadata_evidence(
            literature_id=literature_id,
            quote=summary or literature_id,
            owner_id=ctx.owner_id,
            session_id=ctx.session_id,
            source_url=url,
            provider=provider,
        )

    payload = {
        "literature_id": literature_id,
        "title": item.get("title") or "",
        "authors": json.loads(item.get("authors_json") or "[]"),
        # Never a bare integer: "unknown" and "year 0" must stay distinguishable
        # for a date-filtered question to be answerable honestly.
        "year": item.get("year") if item.get("year_known") else None,
        "year_known": bool(item.get("year_known")),
        "venue": item.get("venue") or "",
        "doi": item.get("doi") or "",
        "arxiv_id": item.get("arxiv_id") or "",
        "url": url,
        "abstract_excerpt": abstract[:ABSTRACT_EXCERPT_CHARS],
        "evidence_id": evidence.evidence_id,
        "evidence_level": evidence.source_level.value,
    }
    return payload, evidence.evidence_id


@tool(parse_docstring=False)
async def search_papers(
    query: str,
    runtime: ToolRuntime[AgentContext],
    year_from: int = 0,
    year_to: int = 0,
    limit: int = 8,
    platforms: list[str] | None = None,
) -> str:
    """Search academic databases for papers on a topic.

    Returns candidates with titles, abstracts and identifiers — not full text.
    Use year_from / year_to to restrict publication dates; papers whose year is
    unknown are reported separately rather than silently dropped, because an
    unknown year cannot be checked against a date filter. To read one of the
    results, pass its literature_id to download_paper.
    """
    ctx = runtime.context
    query = (query or "").strip()
    if not query:
        return ToolResult.failed(
            ErrorCode.INVALID_ARGUMENT, "query must not be empty.", retryable=False
        ).to_json()

    limit = max(1, min(int(limit or 8), MAX_SEARCH_RESULTS))
    chosen = [p for p in (platforms or DEFAULT_PLATFORMS) if p.lower() in KNOWN_PLATFORMS]
    if not chosen:
        chosen = DEFAULT_PLATFORMS

    try:
        load_env_file(".env")
        settings = SearchSettings.from_env()
        # Over-fetch: a date filter applied after ranking would otherwise
        # return far fewer than the caller asked for.
        fetch = limit * 3 if (year_from or year_to) else limit
        raw = await run_search(query, chosen, final_limit=fetch, settings=settings)
        items = json.loads(raw) if raw else []
    except Exception as exc:  # noqa: BLE001 — every provider failure looks alike here
        logger.warning("search_papers failed for %r: %s", query, exc)
        return ToolResult.failed(
            ErrorCode.UPSTREAM_ERROR, f"Search failed: {exc}"[:300]
        ).to_json()

    if isinstance(items, dict):
        items = items.get("results") or items.get("papers") or []
    if not isinstance(items, list):
        items = []

    matched: list[dict[str, Any]] = []
    unknown_year: list[dict[str, Any]] = []
    filtered_out = 0
    evidence_ids: list[str] = []

    for item in items:
        if not isinstance(item, dict) or not (item.get("title") or "").strip():
            continue
        if len(matched) >= limit and len(unknown_year) >= 3:
            break

        year = int(item.get("year") or 0)
        known = year > 0
        if known and year_from and year < int(year_from):
            filtered_out += 1
            continue
        if known and year_to and year > int(year_to):
            filtered_out += 1
            continue

        literature_id = await _register_candidate(item, source="search")
        payload, evidence_id = await _candidate_payload(
            literature_id,
            ctx,
            provider=item.get("source_platform") or "search",
            source_url=item.get("url") or "",
        )
        evidence_ids.append(evidence_id)

        if not known and (year_from or year_to):
            if len(unknown_year) < 3:
                unknown_year.append(payload)
        elif len(matched) < limit:
            matched.append(payload)

    data = {
        "query": query,
        "platforms": chosen,
        "filter": {"year_from": year_from or None, "year_to": year_to or None},
        "results": matched,
        "unknown_year": unknown_year,
        "counts": {
            "returned": len(matched),
            "unknown_year": len(unknown_year),
            "excluded_by_year": filtered_out,
        },
    }

    if not matched and not unknown_year:
        return ToolResult.ok(
            data,
            note="No candidate matched. Try different terms, or widen the year range.",
        ).to_json()
    if unknown_year:
        return ToolResult.partial(
            data,
            note=(
                f"{len(unknown_year)} candidate(s) have no publication year on record, so "
                "the date filter could not be checked against them. They are listed under "
                "`unknown_year`; say so if you use one."
            ),
            evidence_ids=evidence_ids,
        ).to_json()
    return ToolResult.ok(data, evidence_ids=evidence_ids).to_json()


@tool(parse_docstring=False)
async def resolve_paper(
    runtime: ToolRuntime[AgentContext],
    doi: str = "",
    arxiv_id: str = "",
    title: str = "",
) -> str:
    """Identify a paper from a DOI, an arXiv id or a title, and return its canonical record.

    Use this when a citation or a user's phrasing names a paper you then need to
    act on — it returns the literature_id that download_paper and
    get_related_papers take, plus the identifiers and the publication year when
    one is on record.
    """
    ctx = runtime.context
    doi, arxiv_id, title = doi.strip(), arxiv_id.strip(), title.strip()
    if not (doi or arxiv_id or title):
        return ToolResult.failed(
            ErrorCode.INVALID_ARGUMENT,
            "Pass at least one of doi, arxiv_id or title.",
            retryable=False,
        ).to_json()

    meta: PaperMetadata | None = None
    provider = ""
    try:
        async with _http_client() as client:
            oa_id = await citation_graph.openalex_resolve_id(
                client, doi=doi, title=title, arxiv_id=arxiv_id
            )
            if oa_id:
                meta = await citation_graph.openalex_get_metadata(client, oa_id)
                provider = "openalex"
            if meta is None and title:
                meta = await citation_graph.openalex_search_one(client, title)
                provider = "openalex"
            if meta is None:
                # Semantic Scholar indexes preprints OpenAlex has not picked up
                # yet. Resolving an id is itself the confirmation that the paper
                # exists there; the descriptive fields stay as the caller gave
                # them, and the `partial` below says they are unverified.
                s2_id = await citation_graph.s2_resolve_id(
                    client, doi=doi, arxiv_id=arxiv_id, title=title
                )
                if s2_id:
                    meta = PaperMetadata(
                        title=title, doi=doi, arxiv_id=arxiv_id, s2_paper_id=s2_id
                    )
                    provider = "semanticscholar"
    except Exception as exc:  # noqa: BLE001
        logger.warning("resolve_paper failed (%s/%s/%s): %s", doi, arxiv_id, title, exc)
        return ToolResult.failed(
            ErrorCode.UPSTREAM_ERROR, f"Metadata lookup failed: {exc}"[:300]
        ).to_json()

    if meta is None:
        # The identifiers the caller gave are still worth recording: a paper
        # with no index entry can still be downloaded from arXiv.
        if doi or arxiv_id:
            literature_id = await paper_catalog.upsert_literature_item(
                title=title, doi=doi, arxiv_id=arxiv_id, source="user"
            )
            payload, evidence_id = await _candidate_payload(
                literature_id, ctx, provider="user"
            )
            return ToolResult.partial(
                payload,
                note="No metadata index recognised this paper; only the identifier "
                "you supplied is known. Year and venue are unverified.",
                evidence_ids=[evidence_id],
            ).to_json()
        return ToolResult.unavailable(
            ErrorCode.METADATA_INCOMPLETE,
            "No paper matched. Check the title, or supply a DOI or arXiv id.",
        ).to_json()

    literature_id = await _register_candidate(meta, source=provider)
    payload, evidence_id = await _candidate_payload(
        literature_id, ctx, provider=provider
    )
    payload["cited_by_count"] = meta.cited_by_count
    payload["provider"] = provider

    gaps: list[str] = []
    if not payload["year_known"]:
        gaps.append("no publication year is on record — do not assert one")
    if not payload["venue"]:
        gaps.append("no venue is on record")
    if gaps:
        return ToolResult.partial(
            payload, note=f"Incomplete record: {'; '.join(gaps)}.",
            evidence_ids=[evidence_id],
        ).to_json()
    return ToolResult.ok(payload, evidence_ids=[evidence_id]).to_json()


@tool(parse_docstring=False)
async def get_related_papers(
    runtime: ToolRuntime[AgentContext],
    literature_id: str = "",
    paper_id: str = "",
    relation: str = "citations",
    limit: int = 10,
) -> str:
    """Walk the citation graph around a paper.

    relation is one of: references (what it cites), citations (what cites it),
    related, recommendations. Identify the paper by literature_id, or by the
    paper_id of a paper in this session. Results are candidates with abstracts;
    use download_paper to read one in full.
    """
    ctx = runtime.context
    relation = (relation or "citations").strip().lower()
    if relation not in RELATIONS:
        return ToolResult.failed(
            ErrorCode.INVALID_ARGUMENT,
            f"relation must be one of: {', '.join(RELATIONS)}",
            retryable=False,
        ).to_json()

    if not literature_id and paper_id:
        if not await repo.session_owns_paper(ctx.session_id, paper_id):
            return ToolResult.unavailable(
                ErrorCode.PAPER_NOT_FOUND, f"No paper {paper_id} in this session."
            ).to_json()
        literature_id = await paper_catalog.literature_for_paper(paper_id)
    if not literature_id:
        return ToolResult.failed(
            ErrorCode.INVALID_ARGUMENT,
            "Pass literature_id, or the paper_id of a paper in this session.",
            retryable=False,
        ).to_json()

    item = await paper_catalog.get_literature_item(literature_id)
    if item is None:
        return ToolResult.unavailable(
            ErrorCode.PAPER_NOT_FOUND, f"No such work: {literature_id}"
        ).to_json()

    limit = max(1, min(int(limit or 10), MAX_RELATED_RESULTS))
    neighbours: list[PaperMetadata] = []
    provider = ""
    try:
        async with _http_client(timeout=45.0) as client:
            oa_id = item.get("openalex_id") or await citation_graph.openalex_resolve_id(
                client,
                doi=item.get("doi") or "",
                title=item.get("title") or "",
                arxiv_id=item.get("arxiv_id") or "",
            )
            if oa_id and relation in {"references", "citations", "related"}:
                provider = "openalex"
                if relation == "references":
                    neighbours = await citation_graph.openalex_get_referenced_works(client, oa_id)
                elif relation == "citations":
                    neighbours = await citation_graph.openalex_get_cited_by(client, oa_id, limit=limit)
                else:
                    neighbours = await citation_graph.openalex_get_related_works(client, oa_id)

            if not neighbours:
                s2_id = item.get("s2_paper_id") or await citation_graph.s2_resolve_id(
                    client,
                    doi=item.get("doi") or "",
                    arxiv_id=item.get("arxiv_id") or "",
                    title=item.get("title") or "",
                )
                if s2_id:
                    provider = "semanticscholar"
                    if relation == "references":
                        neighbours = await citation_graph.s2_get_references(client, s2_id, limit=limit)
                    elif relation == "citations":
                        neighbours = await citation_graph.s2_get_citations(client, s2_id, limit=limit)
                    else:
                        neighbours = await citation_graph.s2_get_recommendations(client, s2_id, limit=limit)
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_related_papers failed for %s: %s", literature_id, exc)
        return ToolResult.failed(
            ErrorCode.UPSTREAM_ERROR, f"Citation lookup failed: {exc}"[:300]
        ).to_json()

    if not neighbours:
        return ToolResult.unavailable(
            ErrorCode.METADATA_INCOMPLETE,
            f"No {relation} are indexed for this paper. It may be too recent, or "
            "not covered by OpenAlex or Semantic Scholar.",
            data={"literature_id": literature_id, "relation": relation, "papers": []},
        ).to_json()

    papers: list[dict[str, Any]] = []
    evidence_ids: list[str] = []
    for meta in neighbours[:limit]:
        if not (meta.title or "").strip():
            continue
        neighbour_id = await _register_candidate(meta, source=provider)
        payload, evidence_id = await _candidate_payload(
            neighbour_id, ctx, provider=provider
        )
        payload["cited_by_count"] = meta.cited_by_count
        papers.append(payload)
        evidence_ids.append(evidence_id)

    return ToolResult.ok(
        {
            "literature_id": literature_id,
            "of_paper": item.get("title") or "",
            "relation": relation,
            "provider": provider,
            "papers": papers,
        },
        evidence_ids=evidence_ids,
    ).to_json()


# Keys EasyScholar uses for the 中科院 (CAS) division, most specific first.
_CAS_KEYS = ("sciUpSmall", "sciUp", "sciBase")


def _ranking_entries(result: Any) -> list[dict[str, Any]]:
    """Split one lookup into the separate systems it actually spans.

    JCR quartiles, CAS divisions and CCF tiers are different classifications
    with different editions and different meanings. Reporting them as one
    "rank" field is the mistake this exists to prevent.
    """
    extra = getattr(result, "extra", None) or {}
    entries: list[dict[str, Any]] = []

    if result.sci:
        entry = {
            "ranking_system": "JCR",
            "rank": result.sci,
            "category": "all",
            "edition_year": "",
        }
        if extra.get("sciif"):
            entry["impact_factor"] = extra["sciif"]
        entries.append(entry)

    for key in _CAS_KEYS:
        value = extra.get(key)
        if value:
            entries.append(
                {
                    "ranking_system": "CAS",   # 中科院分区
                    "rank": value,
                    "category": "subject" if key == "sciUpSmall" else "broad",
                    "edition_year": "",
                    "top": bool(extra.get("sciUpTop")),
                }
            )
            break

    if result.ccf:
        entries.append(
            {
                "ranking_system": "CCF",
                "rank": result.ccf,
                "category": "all",
                "edition_year": "",
            }
        )
    return entries


@tool(parse_docstring=False)
async def query_publication_rank(venue: str, runtime: ToolRuntime[AgentContext]) -> str:
    """Look up a journal's or conference's rank.

    Returns each ranking system separately — JCR quartile, 中科院 (CAS) division,
    CCF tier — with the source it came from and whether that source is a ranking
    database or a web search. Never merge these into a single "rank"; a CCF A
    conference and a JCR Q1 journal are different claims.
    """
    ctx = runtime.context
    venue = (venue or "").strip()
    if not venue:
        return ToolResult.failed(
            ErrorCode.INVALID_ARGUMENT, "venue must not be empty.", retryable=False
        ).to_json()

    try:
        result = await lookup_rank(venue)
    except Exception as exc:  # noqa: BLE001
        logger.warning("query_publication_rank failed for %r: %s", venue, exc)
        return ToolResult.failed(
            ErrorCode.UPSTREAM_ERROR, f"Rank lookup failed: {exc}"[:300]
        ).to_json()

    source = getattr(result, "source", "") or "unknown"
    # A ranking database is authoritative; an LLM reading web-search results is
    # a lead, and the answer must say which one it had.
    verification_status = "verified" if source == "easyscholar" else "unverified"

    if not result.success:
        return ToolResult.unavailable(
            ErrorCode.METADATA_INCOMPLETE,
            f"No ranking found for {venue!r}: {result.error or 'not indexed'}",
            data={"venue": venue, "rankings": [], "verification_status": "unknown"},
        ).to_json()

    entries = _ranking_entries(result)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    data = {
        "venue": result.name or venue,
        "rankings": entries,
        "source": source,
        "verification_status": verification_status,
        "retrieved_at": retrieved_at,
    }

    if not entries:
        return ToolResult.unavailable(
            ErrorCode.METADATA_INCOMPLETE,
            f"{venue!r} is not listed in JCR, the CAS divisions or CCF.",
            data=data,
        ).to_json()

    evidence = await evidence_service.record_metadata_evidence(
        literature_id="",
        quote="; ".join(
            f"{e['ranking_system']} {e['rank']}" for e in entries
        ),
        owner_id=ctx.owner_id,
        session_id=ctx.session_id,
        provider=source,
    )

    if verification_status == "unverified":
        return ToolResult.partial(
            data,
            note="This ranking came from a web search, not a ranking database. "
            "State that it is unverified.",
            evidence_ids=[evidence.evidence_id],
        ).to_json()
    return ToolResult.ok(data, evidence_ids=[evidence.evidence_id]).to_json()


DISCOVERY_TOOLS = [
    search_papers,
    resolve_paper,
    get_related_papers,
    query_publication_rank,
]
