"""Finding papers, pinning down what they are, and checking where they appeared.

These tools are what separates "read this PDF" from "research this question".
They share one rule the reading tools also follow: whatever they return is
registered as evidence, at the level it was actually obtained. A search result
yields `abstract` evidence, never `fulltext` — the difference is what stops an
answer built on abstracts from reading as though the papers had been read
(acceptance case A08). A passage Semantic Scholar extracted from a paper we
never parsed is `abstract`-level too: we did not read it in context and cannot
place it on a page.

The second rule is about dates and sources. A missing publication year is not
year zero, a citation count belongs to the index that counted it, and a venue
rank taken from a web search is not the same claim as one from a ranking
database. All of these are carried explicitly rather than smoothed over,
because a reader asking "papers since 2022", "the most cited ones" or "is this
a Q1 journal" is entitled to know which answers are verified (acceptance cases
A04, A10).

The third is about coverage. A search fans out over several platforms, and a
platform that was rate-limited, out of quota or refused the key has not
searched anything. That is reported per platform and turns the result
`partial`, so "nothing found" is never said when the truth is "not looked".
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

import httpx
from langchain.tools import ToolRuntime, tool

from app.agents.context import AgentContext
from app.db import agent_repository as repo
from app.models.agent_models import ErrorCode, ToolResult
from app.services import citation_graph, evidence_service, paper_catalog
from app.services.citation_graph import PaperMetadata
from app.services.paper_search import Paper, SearchFilters, SearchOutcome, jaccard_similarity
from app.services.paper_search import search_papers_detailed as run_search
from app.services.paper_search.http_client import HTTPClient, HTTPStatusError
from app.services.paper_search.models import PUBLICATION_TYPES, SORTS
from app.services.paper_search.platforms import canonical_platform
from app.services.paper_search.platforms import openreview as openreview_api
from app.services.paper_search.platforms.semanticscholar import (
    paper_from_s2,
    s2_batch,
    s2_get,
)
from app.services.paper_search.ratelimit import OPENALEX_BUDGET
from app.services.publication_rank import query_publication_rank as lookup_rank
from app.services.search_settings import backend_search_settings

logger = logging.getLogger("scholar.agents.tools.discovery")

DEFAULT_PLATFORMS = ["SemanticScholar", "OpenAlex", "arXiv"]
KNOWN_PLATFORMS = {
    "SemanticScholar", "OpenAlex", "arXiv", "Crossref", "IEEE Xplore", "DBLP", "OpenReview",
}

MAX_SEARCH_RESULTS = 20
MAX_SEARCH_PAGE = 5
MAX_RELATED_RESULTS = 25
MAX_METADATA_ITEMS = 20
MAX_SNIPPETS = 15
# Enough for the model to judge relevance; the full abstract is in the evidence
# record and reaches the reader through the citation, not through the context.
ABSTRACT_EXCERPT_CHARS = 700
SNIPPET_EXCERPT_CHARS = 700
REVIEW_EXCERPT_CHARS = 500

# A platform that returned nothing because it could not search, as opposed to
# one that searched and found nothing.
_NOT_SEARCHED = {"failed", "auth_failed", "rate_limited", "quota_exhausted", "skipped_no_key", "unsupported"}

RELATIONS = {
    "references": "papers this one cites",
    "citations": "papers that cite this one",
    "related": "papers on the same topic",
    "recommendations": "papers recommended from this one",
}

_SOURCE_LABELS = {
    "SemanticScholar": "Semantic Scholar",
    "semanticscholar": "Semantic Scholar",
    "OpenAlex": "OpenAlex",
    "openalex": "OpenAlex",
    "Crossref": "Crossref",
    "IEEE Xplore": "IEEE Xplore",
}


def _http_client(timeout: float = 30.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"User-Agent": "scholar-agent/1.0 (mailto:noreply@example.org)"},
        timeout=timeout,
    )


def _search_client(timeout: float = 30.0) -> HTTPClient:
    return HTTPClient(timeout=timeout, headers={"User-Agent": "scholar-agent/1.0"})


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _clean_list(values: list[str] | None) -> tuple[str, ...]:
    return tuple(v.strip() for v in (values or []) if isinstance(v, str) and v.strip())


async def _register_candidate(
    meta: PaperMetadata | Paper | dict[str, Any], *, source: str
) -> str:
    """Record a discovered paper as a work, whether or not it has a file.

    Everything the agent finds goes through here, so a later `download_paper`
    or `get_related_papers` can name it by `literature_id` instead of passing
    metadata back and forth through the model. The arXiv id and open-access
    PDF link are what `download_paper` tries first, so they are never dropped.
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
            oa_pdf_url=meta.oa_pdf_url,
            source=source,
        )
    if isinstance(meta, Paper):
        meta = meta.to_dict()
    return await paper_catalog.upsert_literature_item(
        title=meta.get("title", ""),
        doi=meta.get("doi", ""),
        arxiv_id=meta.get("arxiv_id", ""),
        openalex_id=meta.get("openalex_id", ""),
        s2_paper_id=meta.get("s2_paper_id", ""),
        authors=[a.strip() for a in (meta.get("authors") or "").split(";") if a.strip()],
        year=meta.get("year") or None,
        venue=meta.get("venue", ""),
        abstract=meta.get("abstract", ""),
        url=meta.get("url", ""),
        oa_pdf_url=meta.get("oa_pdf_url", ""),
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


def _citation_line(counts: dict[str, int], *, as_of: str) -> str:
    parts = [f"{n} ({_SOURCE_LABELS.get(src, src)})" for src, n in sorted(counts.items())]
    return f"cited by {', '.join(parts)} as of {as_of}" if parts else ""


async def _record_bibliographic_facts(
    literature_id: str, paper: Paper, ctx: AgentContext, *, as_of: str
) -> str:
    """Metadata evidence for what a search says *about* a paper.

    Citation counts and an OpenReview decision are claims about the paper, not
    from it; citing them against the abstract evidence would attribute them to
    the authors. Returns "" when there is nothing beyond the abstract.
    """
    facts: list[str] = []
    line = _citation_line(paper.citation_counts, as_of=as_of)
    if line:
        facts.append(line)
    if paper.influential_citation_count is not None:
        facts.append(f"{paper.influential_citation_count} influential citations (Semantic Scholar)")
    if paper.acceptance:
        facts.append(f"OpenReview: {paper.acceptance}")
    if paper.is_retracted:
        facts.append("marked as retracted (OpenAlex)")
    if not facts:
        return ""
    evidence = await evidence_service.record_metadata_evidence(
        literature_id=literature_id,
        quote=f"{paper.title} — " + "; ".join(facts),
        owner_id=ctx.owner_id,
        session_id=ctx.session_id,
        source_url=paper.url,
        provider="+".join(paper.sources or [paper.source_platform]),
    )
    return evidence.evidence_id


def _search_extras(paper: Paper) -> dict[str, Any]:
    """Fields a search adds beyond the catalog record, kept compact."""
    extras: dict[str, Any] = {
        "citation_count": paper.citation_count,
        "citation_counts": dict(paper.citation_counts),
        "publication_date": paper.publication_date or None,
        "venue_type": paper.venue_type or None,
        "is_open_access": paper.is_open_access,
        "full_text_route": bool(paper.arxiv_id or paper.oa_pdf_url or paper.openreview_id),
        "sources": list(paper.sources or [paper.source_platform]),
    }
    if paper.influential_citation_count is not None:
        extras["influential_citation_count"] = paper.influential_citation_count
    if paper.tldr:
        extras["tldr"] = paper.tldr
    if paper.acceptance:
        extras["openreview_status"] = paper.acceptance
    if paper.openreview_id:
        extras["openreview_id"] = paper.openreview_id
    if paper.is_retracted:
        extras["is_retracted"] = True
    return extras


def _choose_platforms(
    requested: list[str] | None, *, venues: tuple[str, ...], sort: str
) -> tuple[list[str], list[str]]:
    """The platforms to fan out to, and any requested names that are unknown."""
    if requested:
        chosen: list[str] = []
        unknown: list[str] = []
        for name in requested:
            canonical = canonical_platform(name)
            if canonical in KNOWN_PLATFORMS:
                if canonical not in chosen:
                    chosen.append(canonical)
            else:
                unknown.append(name)
        if chosen:
            return chosen, unknown
        return list(DEFAULT_PLATFORMS), unknown

    chosen = list(DEFAULT_PLATFORMS)
    if sort == "citations":
        # arXiv has no citation data: its results could only sort last.
        chosen.remove("arXiv")
    if venues:
        # A venue-scoped question is where dblp (the venue as the community
        # names it) and OpenReview (accepted or not) earn their latency.
        chosen.append("DBLP")
        if any(openreview_api.known_venue(v) for v in venues):
            chosen.append("OpenReview")
    return chosen, []


def _coverage_note(statuses: list[dict[str, Any]]) -> str:
    missing = [s for s in statuses if s["status"] in _NOT_SEARCHED]
    if not missing:
        return ""
    parts = [f"{s['platform']} ({s['status']}{': ' + s['detail'] if s.get('detail') else ''})" for s in missing]
    return (
        "Not every platform was searched: " + "; ".join(parts) + ". Results cover the "
        "remaining platforms only — do not say a paper does not exist because it is "
        "absent here."
    )


@tool(parse_docstring=False)
async def search_papers(
    query: str,
    runtime: ToolRuntime[AgentContext],
    year_from: int = 0,
    year_to: int = 0,
    min_citations: int = 0,
    venues: list[str] | None = None,
    fields_of_study: list[str] | None = None,
    open_access_only: bool = False,
    publication_types: list[str] | None = None,
    sort: str = "relevance",
    limit: int = 8,
    page: int = 1,
    platforms: list[str] | None = None,
) -> str:
    """Search academic databases for papers on a topic.

    Returns candidates with titles, abstracts, identifiers and citation counts —
    not full text. Filters are applied by each platform that supports them:
    - year_from / year_to: publication year range. Papers whose year is unknown
      are reported separately, since a date filter cannot be checked on them.
    - min_citations: at least this many citations (any index's count).
    - venues: e.g. ["ICLR", "NeurIPS"]; also searches dblp and OpenReview.
    - fields_of_study: e.g. ["Computer Science"].
    - open_access_only: only papers with a free full text.
    - publication_types: any of conference, journal, preprint, review.
    sort is relevance, citations (most cited first) or recent (newest first).
    page fetches the next results for the same query. platforms defaults to
    Semantic Scholar, OpenAlex and arXiv; also available: Crossref, IEEE Xplore,
    DBLP, OpenReview.

    Citation counts are reported per index (they differ between indexes);
    cite `metadata_evidence_id` for a count or an OpenReview decision, and
    `evidence_id` for what the abstract says. To read a result, pass its
    literature_id to download_paper.
    """
    ctx = runtime.context
    query = (query or "").strip()
    if not query:
        return ToolResult.failed(
            ErrorCode.INVALID_ARGUMENT, "query must not be empty.", retryable=False
        ).to_json()
    sort = (sort or "relevance").strip().lower()
    if sort not in SORTS:
        return ToolResult.failed(
            ErrorCode.INVALID_ARGUMENT, f"sort must be one of: {', '.join(SORTS)}", retryable=False
        ).to_json()
    types = tuple(t.strip().lower() for t in (publication_types or []) if t and t.strip())
    bad_types = [t for t in types if t not in PUBLICATION_TYPES]
    if bad_types:
        return ToolResult.failed(
            ErrorCode.INVALID_ARGUMENT,
            f"unknown publication_types {bad_types}; use: {', '.join(PUBLICATION_TYPES)}",
            retryable=False,
        ).to_json()

    limit = max(1, min(int(limit or 8), MAX_SEARCH_RESULTS))
    page = max(1, min(int(page or 1), MAX_SEARCH_PAGE))
    year_from, year_to = int(year_from or 0), int(year_to or 0)
    venue_list = _clean_list(venues)
    filters = SearchFilters(
        year_from=year_from,
        year_to=year_to,
        min_citations=max(int(min_citations or 0), 0),
        venues=venue_list,
        fields_of_study=_clean_list(fields_of_study),
        open_access_only=bool(open_access_only),
        publication_types=types,
        sort=sort,
    )
    chosen, unknown_platforms = _choose_platforms(platforms, venues=venue_list, sort=sort)

    try:
        # Room for the undated candidates reported beside the matches.
        fetch = limit + 3 if (year_from or year_to) else limit
        outcome: SearchOutcome = await run_search(
            query,
            chosen,
            filters=filters,
            final_limit=fetch,
            page=page,
            rerank_mode="auto",
            settings=backend_search_settings(),
        )
    except ValueError as exc:
        return ToolResult.failed(ErrorCode.INVALID_ARGUMENT, str(exc)[:300], retryable=False).to_json()
    except Exception as exc:  # noqa: BLE001 — every provider failure looks alike here
        logger.warning("search_papers failed for %r: %s", query, exc)
        return ToolResult.failed(
            ErrorCode.UPSTREAM_ERROR, f"Search failed: {exc}"[:300]
        ).to_json()

    statuses = [s.to_dict() for s in outcome.platforms]
    statuses.extend(
        {"platform": name, "status": "unsupported", "count": 0, "detail": "unknown platform"}
        for name in unknown_platforms
    )
    if outcome.platforms and not any(s.searched for s in outcome.platforms):
        limited = all(s.status in {"rate_limited", "quota_exhausted"} for s in outcome.platforms)
        return ToolResult.failed(
            ErrorCode.RATE_LIMITED if limited else ErrorCode.UPSTREAM_ERROR,
            "No platform could be searched: "
            + "; ".join(f"{s['platform']} {s['status']}" for s in statuses),
            retryable=True,
        ).to_json()

    matched: list[dict[str, Any]] = []
    unknown_year: list[dict[str, Any]] = []
    evidence_ids: list[str] = []
    as_of = _today()
    excluded = dict(outcome.excluded)
    filtered_out = excluded.pop("year", 0)

    for paper in outcome.papers:
        if not paper.title.strip():
            continue
        known = paper.year > 0
        # The platforms and the orchestrator filter by year already; this is
        # the last check before a result is presented as matching the range.
        if known and ((year_from and paper.year < year_from) or (year_to and paper.year > year_to)):
            filtered_out += 1
            continue
        undated_bucket = not known and bool(year_from or year_to)
        if undated_bucket and len(unknown_year) >= 3:
            continue
        if not undated_bucket and len(matched) >= limit:
            continue

        literature_id = await _register_candidate(paper, source="search")
        payload, evidence_id = await _candidate_payload(
            literature_id,
            ctx,
            provider=paper.source_platform or "search",
            source_url=paper.url,
        )
        payload.update(_search_extras(paper))
        evidence_ids.append(evidence_id)
        facts_id = await _record_bibliographic_facts(literature_id, paper, ctx, as_of=as_of)
        if facts_id:
            payload["metadata_evidence_id"] = facts_id
            evidence_ids.append(facts_id)

        if undated_bucket:
            unknown_year.append(payload)
        else:
            matched.append(payload)

    data = {
        "query": query,
        "platforms": statuses,
        "filter": {
            "year_from": year_from or None,
            "year_to": year_to or None,
            "min_citations": filters.min_citations or None,
            "venues": list(venue_list) or None,
            "fields_of_study": list(filters.fields_of_study) or None,
            "open_access_only": filters.open_access_only or None,
            "publication_types": list(types) or None,
        },
        "sort": sort,
        "page": page,
        "results": matched,
        "unknown_year": unknown_year,
        "counts": {
            "returned": len(matched),
            "unknown_year": len(unknown_year),
            "excluded_by_year": filtered_out,
            "excluded_by_other_filters": excluded,
            "candidates_seen": outcome.candidates,
        },
    }
    if len(matched) >= limit and page < MAX_SEARCH_PAGE:
        data["next_page"] = page + 1

    notes: list[str] = []
    coverage = _coverage_note(statuses)
    if coverage:
        notes.append(coverage)
    if unknown_year:
        notes.append(
            f"{len(unknown_year)} candidate(s) have no publication year on record, so "
            "the date filter could not be checked against them. They are listed under "
            "`unknown_year`; say so if you use one."
        )
    if any(k.endswith("_unknown") for k in excluded):
        notes.append(
            "Some candidates were excluded because the value a filter needs (citation "
            "count or venue) is not on record for them; see counts.excluded_by_other_filters."
        )

    if not matched and not unknown_year:
        note = " ".join(notes) or "No candidate matched. Try different terms, or relax the filters."
        if coverage:
            return ToolResult.partial(data, note=note).to_json()
        return ToolResult.ok(data, note=note).to_json()
    if coverage or unknown_year:
        return ToolResult.partial(data, note=" ".join(notes), evidence_ids=evidence_ids).to_json()
    return ToolResult.ok(data, evidence_ids=evidence_ids, note=" ".join(notes)).to_json()


def _metadata_from_paper(paper: Paper) -> PaperMetadata:
    return PaperMetadata(
        title=paper.title,
        doi=paper.doi,
        arxiv_id=paper.arxiv_id,
        s2_paper_id=paper.s2_paper_id,
        openalex_id=paper.openalex_id,
        year=paper.year,
        venue=paper.venue,
        authors=paper.authors,
        abstract_text=paper.abstract,
        cited_by_count=paper.citation_counts.get("SemanticScholar", 0),
        citation_source="semanticscholar",
        influential_citation_count=paper.influential_citation_count,
        publication_date=paper.publication_date,
        oa_pdf_url=paper.oa_pdf_url,
        tldr=paper.tldr,
    )


async def _s2_metadata(s2_id: str) -> PaperMetadata | None:
    try:
        records = await s2_batch(_search_client(), [s2_id])
    except Exception as exc:  # noqa: BLE001 — the bare id is still usable
        logger.info("S2 metadata fetch for %s failed: %s", s2_id, exc)
        return None
    record = records[0] if records else None
    if not isinstance(record, dict) or not record.get("title"):
        return None
    return _metadata_from_paper(paper_from_s2(record))


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
                # yet, and is the whole fallback when OpenAlex's daily budget is
                # spent. The resolved id is fetched in full; if that fails the
                # descriptive fields stay as the caller gave them, and the
                # `partial` below says they are unverified.
                s2_id = await citation_graph.s2_resolve_id(
                    client, doi=doi, arxiv_id=arxiv_id, title=title
                )
                if s2_id:
                    meta = await _s2_metadata(s2_id) or PaperMetadata(
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
    payload["citation_source"] = meta.citation_source or provider
    payload["provider"] = provider
    if meta.oa_pdf_url or meta.arxiv_id:
        payload["full_text_route"] = True
    if meta.is_retracted:
        payload["is_retracted"] = True

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


# ── Bibliographic facts in bulk ─────────────────────────────────────────────


def _s2_lookup_id(target: dict[str, str]) -> str:
    if target.get("s2_paper_id"):
        return target["s2_paper_id"]
    if target.get("doi"):
        return f"DOI:{target['doi']}"
    if target.get("arxiv_id"):
        return f"ARXIV:{target['arxiv_id']}"
    return ""


async def _metadata_targets(
    literature_ids: list[str] | None, dois: list[str] | None, arxiv_ids: list[str] | None
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """What the caller asked about, each with whatever identifiers are known."""
    targets: list[dict[str, str]] = []
    not_found: list[dict[str, str]] = []
    for lit in _clean_list(literature_ids):
        item = await paper_catalog.get_literature_item(lit)
        if item is None:
            not_found.append({"query": lit, "reason": "unknown literature_id"})
            continue
        targets.append(
            {
                "query": lit,
                "literature_id": lit,
                "doi": item.get("doi") or "",
                "arxiv_id": item.get("arxiv_id") or "",
                "s2_paper_id": item.get("s2_paper_id") or "",
                "title": item.get("title") or "",
            }
        )
    for doi in _clean_list(dois):
        targets.append({"query": doi, "doi": doi})
    for arxiv_id in _clean_list(arxiv_ids):
        targets.append({"query": arxiv_id, "arxiv_id": arxiv_id})
    return targets, not_found


@tool(parse_docstring=False)
async def get_paper_metadata(
    runtime: ToolRuntime[AgentContext],
    literature_ids: list[str] | None = None,
    dois: list[str] | None = None,
    arxiv_ids: list[str] | None = None,
) -> str:
    """Bibliographic facts for up to 20 papers in one call.

    For each paper: citation counts per index (Semantic Scholar and OpenAlex
    count differently — report which), influential citations, reference count,
    publication date, venue and its type, open-access PDF link, retraction
    status, fields of study and a one-sentence TL;DR. Use it to compare how
    cited several papers are, or to check a venue or date, instead of calling
    resolve_paper once per paper. Identify papers by literature_id (from any
    earlier tool), DOI or arXiv id.
    """
    ctx = runtime.context
    targets, not_found = await _metadata_targets(literature_ids, dois, arxiv_ids)
    if not targets and not not_found:
        return ToolResult.failed(
            ErrorCode.INVALID_ARGUMENT,
            "Pass literature_ids, dois or arxiv_ids.",
            retryable=False,
        ).to_json()
    if len(targets) > MAX_METADATA_ITEMS:
        return ToolResult.failed(
            ErrorCode.INVALID_ARGUMENT,
            f"At most {MAX_METADATA_ITEMS} papers per call.",
            retryable=False,
        ).to_json()

    # A title-only work needs one S2 lookup before the batch can name it.
    title_only = [t for t in targets if not _s2_lookup_id(t) and t.get("title")]
    if title_only:
        async with _http_client() as client:
            for t in title_only[:5]:
                s2_id = await citation_graph.s2_resolve_id(client, title=t["title"])
                if s2_id:
                    t["s2_paper_id"] = s2_id

    lookup = [(t, _s2_lookup_id(t)) for t in targets]
    ids = [sid for _, sid in lookup if sid]
    records: dict[str, dict[str, Any]] = {}
    s2_error = ""
    if ids:
        try:
            batch = await s2_batch(_search_client(), ids, fields=citation_graph.S2_METADATA_FIELDS)
            for sid, rec in zip(ids, batch):
                if isinstance(rec, dict) and rec.get("paperId"):
                    records[sid] = rec
        except HTTPStatusError as exc:
            s2_error = f"Semantic Scholar HTTP {exc.status_code}"
        except Exception as exc:  # noqa: BLE001
            s2_error = f"Semantic Scholar: {type(exc).__name__}"

    papers: dict[str, Paper] = {sid: paper_from_s2(rec) for sid, rec in records.items()}
    # The second index: OpenAlex by DOI, when its budget allows.
    openalex: dict[str, PaperMetadata] = {}
    openalex_note = ""
    all_dois = sorted(
        {(p.doi or "").lower() for p in papers.values() if p.doi}
        | {t["doi"].lower() for t, _ in lookup if t.get("doi")}
    )
    if all_dois:
        if OPENALEX_BUDGET.exhausted():
            openalex_note = "OpenAlex skipped: daily quota exhausted."
        else:
            async with _http_client() as client:
                openalex = await citation_graph.openalex_batch_fetch_by_doi(client, all_dois)

    as_of = _today()
    results: list[dict[str, Any]] = []
    evidence_ids: list[str] = []
    for target, sid in lookup:
        paper = papers.get(sid) if sid else None
        doi = ((paper.doi if paper else "") or target.get("doi", "")).lower()
        oa = openalex.get(doi) if doi else None
        if paper is None and oa is None:
            not_found.append(
                {
                    "query": target["query"],
                    "reason": s2_error or ("not indexed by Semantic Scholar or OpenAlex" if sid else "no identifier to look up"),
                }
            )
            continue

        citations: dict[str, int] = {}
        if paper is not None and "SemanticScholar" in paper.citation_counts:
            citations["semanticscholar"] = paper.citation_counts["SemanticScholar"]
        if oa is not None:
            citations["openalex"] = oa.cited_by_count
        record = records.get(sid, {}) if sid else {}

        literature_id = target.get("literature_id") or ""
        registered = await paper_catalog.upsert_literature_item(
            title=(paper.title if paper else oa.title) if (paper or oa) else "",
            doi=doi,
            arxiv_id=(paper.arxiv_id if paper else "") or (oa.arxiv_id if oa else "") or target.get("arxiv_id", ""),
            openalex_id=oa.openalex_id if oa else "",
            s2_paper_id=paper.s2_paper_id if paper else "",
            year=(paper.year if paper else 0) or (oa.year if oa else 0) or None,
            venue=(paper.venue if paper else "") or (oa.venue if oa else ""),
            abstract=(paper.abstract if paper else "") or (oa.abstract_text if oa else ""),
            oa_pdf_url=(paper.oa_pdf_url if paper else "") or (oa.oa_pdf_url if oa else ""),
            source="metadata",
        )
        literature_id = literature_id or registered
        item = await paper_catalog.get_literature_item(literature_id) or {}

        is_retracted = oa.is_retracted if oa is not None else None
        facts = [
            _citation_line({("SemanticScholar" if k == "semanticscholar" else "OpenAlex"): v for k, v in citations.items()}, as_of=as_of),
            f"published {paper.publication_date or paper.year}" if paper and (paper.publication_date or paper.year) else "",
            f"venue {item.get('venue')}" if item.get("venue") else "",
            "retracted (OpenAlex)" if is_retracted else "",
        ]
        evidence = await evidence_service.record_metadata_evidence(
            literature_id=literature_id,
            quote=f"{item.get('title') or target['query']} — " + "; ".join(f for f in facts if f),
            owner_id=ctx.owner_id,
            session_id=ctx.session_id,
            source_url=(paper.url if paper else "") or item.get("url") or "",
            provider="semanticscholar+openalex" if (paper and oa) else ("semanticscholar" if paper else "openalex"),
        )
        evidence_ids.append(evidence.evidence_id)
        results.append(
            {
                "literature_id": literature_id,
                "title": item.get("title") or "",
                "year": item.get("year") if item.get("year_known") else None,
                "publication_date": (paper.publication_date if paper else "") or (oa.publication_date if oa else "") or None,
                "venue": item.get("venue") or "",
                "venue_type": (paper.venue_type if paper else "") or None,
                "venue_aliases": (paper.venue_aliases[:5] if paper else []),
                "citations": citations,
                "influential_citation_count": paper.influential_citation_count if paper else None,
                "reference_count": record.get("referenceCount"),
                "is_open_access": paper.is_open_access if paper else None,
                "oa_pdf_url": item.get("oa_pdf_url") or "",
                "is_retracted": is_retracted,
                "fields_of_study": paper.fields_of_study if paper else [],
                "publication_types": paper.publication_types if paper else [],
                "tldr": paper.tldr if paper else "",
                "ids": {
                    "doi": item.get("doi") or "",
                    "arxiv_id": item.get("arxiv_id") or "",
                    "s2_paper_id": item.get("s2_paper_id") or "",
                    "openalex_id": item.get("openalex_id") or "",
                    "dblp_key": paper.dblp_key if paper else "",
                },
                "evidence_id": evidence.evidence_id,
            }
        )

    data = {"papers": results, "not_found": not_found, "retrieved_at": as_of}
    notes = [n for n in (openalex_note, s2_error and f"{s2_error}; only OpenAlex was consulted.") if n]
    if not results:
        return ToolResult.unavailable(
            ErrorCode.METADATA_INCOMPLETE,
            "None of these papers could be found. " + " ".join(notes),
            data=data,
        ).to_json()
    if not_found or notes:
        note = " ".join(notes + ([f"{len(not_found)} paper(s) not found; see not_found."] if not_found else []))
        return ToolResult.partial(data, note=note, evidence_ids=evidence_ids).to_json()
    return ToolResult.ok(data, evidence_ids=evidence_ids).to_json()


# ── Passages from papers not downloaded ─────────────────────────────────────


@tool(parse_docstring=False)
async def search_paper_snippets(
    query: str,
    runtime: ToolRuntime[AgentContext],
    literature_ids: list[str] | None = None,
    year_from: int = 0,
    year_to: int = 0,
    venues: list[str] | None = None,
    fields_of_study: list[str] | None = None,
    min_citations: int = 0,
    limit: int = 8,
) -> str:
    """Find passages inside papers' text without downloading them (Semantic Scholar).

    Returns short passages — from the body, not only the abstract — that
    answer the query, with the paper and section each came from. Pass
    literature_ids to look only inside those papers (e.g. "what learning rate
    did these use"), or leave it empty to search across the literature. Filters
    as in search_papers.

    Passages are abstract-level evidence: we did not read them in context and
    they carry no page number. For exact numbers, equations or tables in a
    paper that matters, download_paper and read the section instead.
    """
    ctx = runtime.context
    query = (query or "").strip()
    if not query:
        return ToolResult.failed(
            ErrorCode.INVALID_ARGUMENT, "query must not be empty.", retryable=False
        ).to_json()
    limit = max(1, min(int(limit or 8), MAX_SNIPPETS))

    params: dict[str, str] = {"query": query, "limit": str(limit)}
    years = SearchFilters(year_from=int(year_from or 0), year_to=int(year_to or 0)).year_range()
    if years:
        params["year"] = years
    if _clean_list(venues):
        params["venue"] = ",".join(_clean_list(venues))
    if _clean_list(fields_of_study):
        params["fieldsOfStudy"] = ",".join(_clean_list(fields_of_study))
    if int(min_citations or 0) > 0:
        params["minCitationCount"] = str(int(min_citations))

    scope: dict[str, str] = {}
    for lit in _clean_list(literature_ids):
        item = await paper_catalog.get_literature_item(lit) or {}
        sid = _s2_lookup_id(
            {
                "s2_paper_id": item.get("s2_paper_id") or "",
                "doi": item.get("doi") or "",
                "arxiv_id": item.get("arxiv_id") or "",
            }
        )
        if sid:
            scope[sid] = lit
    if literature_ids and not scope:
        return ToolResult.unavailable(
            ErrorCode.METADATA_INCOMPLETE,
            "None of these papers has an identifier Semantic Scholar can look up "
            "(DOI, arXiv id or S2 id). Try resolve_paper first.",
        ).to_json()
    if scope:
        params["paperIds"] = ",".join(scope)

    client = _search_client()
    try:
        data = await s2_get(client, "/snippet/search", params)
    except HTTPStatusError as exc:
        code = ErrorCode.RATE_LIMITED if exc.status_code == 429 else ErrorCode.UPSTREAM_ERROR
        return ToolResult.failed(code, f"Semantic Scholar snippet search: HTTP {exc.status_code}").to_json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("search_paper_snippets failed for %r: %s", query, exc)
        return ToolResult.failed(ErrorCode.UPSTREAM_ERROR, f"Snippet search failed: {exc}"[:300]).to_json()

    hits = [h for h in (data or {}).get("data") or [] if isinstance(h, dict) and (h.get("snippet") or {}).get("text")]
    if not hits:
        return ToolResult.ok(
            {"query": query, "snippets": []},
            note="No passage matched. Try other wording, or search_papers for whole papers.",
        ).to_json()

    # One batch call turns corpus ids into identities the catalog can hold, so
    # a passage's paper can be downloaded or cited like any other result.
    corpus_ids = list(dict.fromkeys(str((h.get("paper") or {}).get("corpusId") or "") for h in hits))
    corpus_ids = [c for c in corpus_ids if c]
    identities: dict[str, Paper] = {}
    try:
        batch = await s2_batch(client, [f"CorpusId:{c}" for c in corpus_ids])
        for cid, rec in zip(corpus_ids, batch):
            if isinstance(rec, dict) and rec.get("title"):
                identities[cid] = paper_from_s2(rec)
    except Exception as exc:  # noqa: BLE001 — fall back to the snippet's own title
        logger.info("snippet identity batch failed: %s", exc)

    snippets: list[dict[str, Any]] = []
    evidence_ids: list[str] = []
    lit_by_corpus: dict[str, str] = {}
    for hit in hits:
        paper_info = hit.get("paper") or {}
        snippet = hit.get("snippet") or {}
        cid = str(paper_info.get("corpusId") or "")
        identity = identities.get(cid)
        if cid not in lit_by_corpus:
            if identity is not None:
                lit_by_corpus[cid] = await _register_candidate(identity, source="semanticscholar")
            else:
                lit_by_corpus[cid] = await paper_catalog.upsert_literature_item(
                    title=paper_info.get("title") or "",
                    authors=[a for a in paper_info.get("authors") or [] if isinstance(a, str)],
                    source="semanticscholar",
                    fallback_key=f"s2corpus:{cid}",
                )
        literature_id = lit_by_corpus[cid]
        text = normalize_snippet(snippet.get("text") or "")
        section = (snippet.get("section") or "").strip()
        url = (identity.url if identity else "") or (f"https://www.semanticscholar.org/p/{cid}" if cid else "")
        evidence = await evidence_service.record_abstract_evidence(
            literature_id=literature_id,
            quote=text,
            owner_id=ctx.owner_id,
            session_id=ctx.session_id,
            provider="semanticscholar_snippet",
            source_url=url,
            section_path=section,
        )
        evidence_ids.append(evidence.evidence_id)
        snippets.append(
            {
                "literature_id": literature_id,
                "title": (identity.title if identity else "") or paper_info.get("title") or "",
                "year": (identity.year if identity and identity.year else None),
                "venue": identity.venue if identity else "",
                "arxiv_id": identity.arxiv_id if identity else "",
                "section": section,
                "kind": snippet.get("snippetKind") or "",
                "text": text[:SNIPPET_EXCERPT_CHARS],
                "score": round(float(hit.get("score") or 0.0), 3),
                "evidence_id": evidence.evidence_id,
                "evidence_level": evidence.source_level.value,
            }
        )

    return ToolResult.ok(
        {"query": query, "scoped_to": list(scope.values()) or None, "snippets": snippets},
        evidence_ids=evidence_ids,
        note="Passages extracted by Semantic Scholar; abstract-level evidence without page numbers.",
    ).to_json()


def normalize_snippet(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


# ── Peer review ─────────────────────────────────────────────────────────────


_RATING_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)")


def _rating_number(value: str) -> float | None:
    m = _RATING_RE.match(value or "")
    return float(m.group(1)) if m else None


def _review_quote(review: dict[str, Any]) -> str:
    scores = ", ".join(f"{k} {review[k]}" for k in ("rating", "confidence") if review.get(k))
    parts = [f"Reviewer ({scores})" if scores else "Reviewer"]
    for key in ("summary", "strengths", "weaknesses", "strengths_and_weaknesses"):
        if review.get(key):
            parts.append(f"{key.replace('_', ' ').capitalize()}: {review[key]}")
    return " ".join(parts)


@tool(parse_docstring=False)
async def get_peer_reviews(
    runtime: ToolRuntime[AgentContext],
    literature_id: str = "",
    openreview_id: str = "",
    title: str = "",
) -> str:
    """Whether a paper was accepted at an OpenReview venue, and what its reviewers said.

    Covers venues that review on OpenReview (ICLR, NeurIPS, ICML, TMLR, COLM…).
    Returns the decision (accepted as oral/spotlight/poster, not accepted,
    withdrawn) and each review's scores, summary, strengths and weaknesses.
    Identify the paper by literature_id, openreview_id (search results carry
    one when known) or title.

    Reviews are opinions of anonymous reviewers, not findings of the paper:
    attribute them ("a reviewer noted…") and cite their evidence_id.
    """
    ctx = runtime.context
    item = await paper_catalog.get_literature_item(literature_id) if literature_id else None
    if literature_id and item is None:
        return ToolResult.unavailable(
            ErrorCode.PAPER_NOT_FOUND, f"No such work: {literature_id}"
        ).to_json()
    title = (title or (item or {}).get("title") or "").strip()
    forum_id = (openreview_id or "").strip()
    if not (forum_id or title):
        return ToolResult.failed(
            ErrorCode.INVALID_ARGUMENT,
            "Pass literature_id, openreview_id or title.",
            retryable=False,
        ).to_json()

    client = _search_client()
    submission: dict[str, Any] | None = None
    try:
        if title:
            # The exact title as a phrase first; OpenReview ANDs loose terms,
            # so the stopword-free words are only the fallback.
            notes = await openreview_api.search_submissions(
                client, openreview_api.title_phrase(title[:200]), limit=10
            )
            if not notes and openreview_api.search_term(title):
                notes = await openreview_api.search_submissions(
                    client, openreview_api.search_term(title), limit=10
                )
            best = max(
                notes,
                key=lambda n: jaccard_similarity(title, openreview_api.note_to_paper(n).title),
                default=None,
            )
            if best is not None:
                best_title = openreview_api.note_to_paper(best).title
                if (forum_id and (best.get("forum") or best.get("id")) == forum_id) or jaccard_similarity(title, best_title) >= 0.85:
                    submission = best
                    forum_id = forum_id or best.get("forum") or best.get("id") or ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenReview search failed for %r: %s", title, exc)
        return ToolResult.failed(ErrorCode.UPSTREAM_ERROR, f"OpenReview search failed: {exc}"[:300]).to_json()

    if not forum_id:
        return ToolResult.unavailable(
            ErrorCode.METADATA_INCOMPLETE,
            "Not found on OpenReview. The paper may not have been reviewed there — "
            "most journals and many conferences do not use it.",
        ).to_json()

    login_needed = False
    forum: dict[str, Any] = {"decision": "", "meta_review": "", "reviews": []}
    try:
        forum_notes = await openreview_api.get_forum_notes(client, forum_id)
        forum = openreview_api.parse_forum_notes(forum_notes)
        if submission is None:
            submission = next(
                (n for n in forum_notes if n.get("id") == forum_id and openreview_api.is_submission(n)),
                None,
            )
    except openreview_api.OpenReviewLoginRequired:
        login_needed = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenReview forum %s failed: %s", forum_id, exc)
        if submission is None:
            return ToolResult.failed(ErrorCode.UPSTREAM_ERROR, f"OpenReview forum fetch failed: {exc}"[:300]).to_json()

    sub_paper = openreview_api.note_to_paper(submission) if submission else None
    venue_line = sub_paper.acceptance if sub_paper else ""
    state, tier = openreview_api.classify_acceptance(venue_line)
    forum_url = f"{openreview_api.OPENREVIEW_WEB}/forum?id={forum_id}"

    if not literature_id:
        literature_id = await paper_catalog.upsert_literature_item(
            title=(sub_paper.title if sub_paper else "") or title,
            authors=[a.strip() for a in (sub_paper.authors if sub_paper else "").split(";") if a.strip()],
            year=(sub_paper.year if sub_paper else 0) or None,
            abstract=sub_paper.abstract if sub_paper else "",
            url=forum_url,
            oa_pdf_url=sub_paper.oa_pdf_url if sub_paper else "",
            source="openreview",
        )

    evidence_ids: list[str] = []
    status_quote = "; ".join(
        p for p in (f"OpenReview: {venue_line}" if venue_line else "", f"decision: {forum['decision']}" if forum["decision"] else "") if p
    )
    status_evidence_id = ""
    if status_quote:
        evidence = await evidence_service.record_metadata_evidence(
            literature_id=literature_id,
            quote=f"{(sub_paper.title if sub_paper else title)} — {status_quote}",
            owner_id=ctx.owner_id,
            session_id=ctx.session_id,
            source_url=forum_url,
            provider="openreview",
        )
        status_evidence_id = evidence.evidence_id
        evidence_ids.append(status_evidence_id)

    reviews: list[dict[str, Any]] = []
    for review in forum["reviews"]:
        evidence = await evidence_service.record_web_evidence(
            quote=_review_quote(review),
            source_url=f"{forum_url}&noteId={review.get('review_id', '')}",
            owner_id=ctx.owner_id,
            session_id=ctx.session_id,
            provider="openreview",
            literature_id=literature_id,
        )
        evidence_ids.append(evidence.evidence_id)
        shaped = {k: v for k, v in review.items() if k != "review_id"}
        for key in ("summary", "strengths", "weaknesses", "strengths_and_weaknesses", "questions", "limitations"):
            if key in shaped:
                shaped[key] = shaped[key][:REVIEW_EXCERPT_CHARS]
        shaped["evidence_id"] = evidence.evidence_id
        reviews.append(shaped)

    ratings = [r for r in (_rating_number(rv.get("rating", "")) for rv in forum["reviews"]) if r is not None]
    data = {
        "literature_id": literature_id,
        "openreview_id": forum_id,
        "url": forum_url,
        "title": (sub_paper.title if sub_paper else title),
        "status": {
            "state": state,
            "tier": tier or None,
            "venue_line": venue_line or None,
            "decision": forum["decision"] or None,
            "evidence_id": status_evidence_id or None,
        },
        "meta_review": forum["meta_review"][:800] or None,
        "reviews": reviews,
        "rating_summary": (
            {"count": len(ratings), "mean": round(sum(ratings) / len(ratings), 2), "min": min(ratings), "max": max(ratings)}
            if ratings
            else None
        ),
    }
    if login_needed:
        refused = openreview_api.login_error() if openreview_api.has_account() else ""
        how = (
            f"OpenReview refused the configured account ({refused}); check OPENREVIEW_USERNAME "
            "and OPENREVIEW_PASSWORD, or whether the profile is activated"
            if refused
            else "Set OPENREVIEW_USERNAME and OPENREVIEW_PASSWORD to read them"
        )
        return ToolResult.partial(
            data,
            note="The decision is known, but OpenReview only shows reviews to a logged-in "
            f"client. {how}; do not describe the reviews.",
            evidence_ids=evidence_ids,
        ).to_json()
    if not reviews:
        return ToolResult.partial(
            data,
            note="No public reviews on this forum (reviews may be hidden until decisions, or the venue does not publish them).",
            evidence_ids=evidence_ids,
        ).to_json()
    return ToolResult.ok(data, evidence_ids=evidence_ids).to_json()


DISCOVERY_TOOLS = [
    search_papers,
    resolve_paper,
    get_paper_metadata,
    search_paper_snippets,
    get_related_papers,
    get_peer_reviews,
    query_publication_rank,
]
