from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from ..config import Settings
from ..credentials import s2_headers
from ..http_client import HTTPClient, HTTPStatusError
from ..models import (
    FILTER_FIELDS_OF_STUDY,
    FILTER_MIN_CITATIONS,
    FILTER_OPEN_ACCESS,
    FILTER_PUBLICATION_TYPES,
    FILTER_VENUES,
    FILTER_YEAR,
    Paper,
    SearchFilters,
)
from ..ratelimit import S2_LIMITER
from ..utils import normalize_doi, normalize_whitespace

S2_GRAPH = "https://api.semanticscholar.org/graph/v1"

PAPER_FIELDS = (
    "title,abstract,authors,url,externalIds,year,venue,publicationVenue,publicationDate,"
    "citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,fieldsOfStudy,"
    "publicationTypes,tldr"
)
# The bulk endpoint returns up to 1000 rows per call and has no `limit`, so it
# is asked for identifiers only; the few rows kept are then fetched in full.
_BULK_FIELDS = "paperId,citationCount,publicationDate"

SUPPORTED_FILTERS = frozenset(
    {
        FILTER_YEAR,
        FILTER_MIN_CITATIONS,
        FILTER_VENUES,
        FILTER_FIELDS_OF_STUDY,
        FILTER_OPEN_ACCESS,
        FILTER_PUBLICATION_TYPES,
    }
)
SUPPORTED_SORTS = frozenset({"relevance", "citations", "recent"})

# S2's publicationTypes vocabulary for each normalised type.
_TYPE_TO_S2 = {
    "conference": "Conference",
    "journal": "JournalArticle",
    "review": "Review",
}
_S2_TO_TYPE = {"Conference": "conference", "JournalArticle": "journal", "Review": "review"}


def filter_params(filters: SearchFilters | None) -> dict[str, str]:
    """S2 query parameters for the filters it can apply itself."""
    if filters is None:
        return {}
    params: dict[str, str] = {}
    years = filters.year_range()
    if years:
        params["year"] = years
    if filters.min_citations > 0:
        params["minCitationCount"] = str(int(filters.min_citations))
    if filters.venues:
        params["venue"] = ",".join(filters.venues)
    if filters.fields_of_study:
        params["fieldsOfStudy"] = ",".join(filters.fields_of_study)
    if filters.open_access_only:
        params["openAccessPdf"] = ""
    types = [_TYPE_TO_S2[t] for t in filters.publication_types if t in _TYPE_TO_S2]
    if types:
        params["publicationTypes"] = ",".join(types)
    return params


def paper_from_s2(item: dict[str, Any], *, source: str = "SemanticScholar") -> Paper:
    """One S2 paper record, as returned by search, batch or the graph endpoints."""
    external_ids = item.get("externalIds") or {}
    authors_raw = item.get("authors") or []
    authors = "; ".join(
        normalize_whitespace(a.get("name") if isinstance(a, dict) else str(a))
        for a in authors_raw
        if normalize_whitespace(a.get("name") if isinstance(a, dict) else str(a))
    )

    year = 0
    raw_year = item.get("year")
    if isinstance(raw_year, int):
        year = raw_year
    elif isinstance(raw_year, str) and raw_year.isdigit():
        year = int(raw_year)

    pub_venue = item.get("publicationVenue") or {}
    venue_type = ""
    vt = (pub_venue.get("type") or "").lower()
    if vt == "conference":
        venue_type = "conference"
    elif vt == "journal":
        venue_type = "journal"
    aliases = [normalize_whitespace(n) for n in (pub_venue.get("alternate_names") or []) if n]
    if pub_venue.get("name"):
        aliases.insert(0, normalize_whitespace(pub_venue["name"]))

    types = [_S2_TO_TYPE[t] for t in (item.get("publicationTypes") or []) if t in _S2_TO_TYPE]
    arxiv_id = normalize_whitespace(external_ids.get("ArXiv") or "")
    if not venue_type and arxiv_id and not (item.get("venue") or "").strip():
        venue_type = "preprint"

    oa = item.get("openAccessPdf") or {}
    oa_url = normalize_whitespace(oa.get("url") or "")
    citation_counts: dict[str, int] = {}
    if isinstance(item.get("citationCount"), int):
        citation_counts[source] = int(item["citationCount"])
    influential = item.get("influentialCitationCount")
    tldr = item.get("tldr") or {}

    is_oa = item.get("isOpenAccess")
    return Paper(
        title=normalize_whitespace(item.get("title") or ""),
        abstract=normalize_whitespace(item.get("abstract") or ""),
        url=normalize_whitespace(item.get("url") or ""),
        doi=normalize_doi(external_ids.get("DOI") or ""),
        authors=authors,
        source_platform=source,
        year=year,
        venue=normalize_whitespace(item.get("venue") or ""),
        arxiv_id=arxiv_id,
        s2_paper_id=normalize_whitespace(item.get("paperId") or ""),
        dblp_key=normalize_whitespace(external_ids.get("DBLP") or ""),
        oa_pdf_url=oa_url,
        publication_date=normalize_whitespace(item.get("publicationDate") or ""),
        venue_type=venue_type,
        publication_types=types,
        venue_aliases=aliases,
        fields_of_study=[f for f in (item.get("fieldsOfStudy") or []) if isinstance(f, str)],
        citation_counts=citation_counts,
        influential_citation_count=int(influential) if isinstance(influential, int) else None,
        is_open_access=bool(is_oa or oa_url) if (is_oa is not None or oa_url) else None,
        tldr=normalize_whitespace(tldr.get("text") or "") if isinstance(tldr, dict) else "",
    )


_S2_RETRIES = 2


async def _paced(call: Callable[[], Awaitable[Any]]) -> Any:
    """One S2 request in the shared 1 req/s lane, retried on 429.

    Pacing our own requests does not make S2's window agree with ours: it
    still answers 429 now and then, and a short wait is what it asks for.
    """
    for attempt in range(_S2_RETRIES + 1):
        await S2_LIMITER.wait()
        try:
            return await call()
        except HTTPStatusError as exc:
            if exc.status_code != 429 or attempt >= _S2_RETRIES:
                raise
            await asyncio.sleep(min(exc.retry_after or 1.5 * (attempt + 1), 10.0))
    raise RuntimeError("unreachable")


async def s2_get(client: HTTPClient, path: str, params: dict[str, str]) -> Any:
    return await _paced(
        lambda: client.get_json(f"{S2_GRAPH}{path}", params=params, headers=s2_headers())
    )


async def s2_batch(
    client: HTTPClient, ids: list[str], *, fields: str = PAPER_FIELDS
) -> list[dict[str, Any] | None]:
    """Full records for up to 500 ids in one request; unknown ids come back as None."""
    if not ids:
        return []
    out: list[dict[str, Any] | None] = []
    for i in range(0, len(ids), 500):
        chunk = ids[i : i + 500]
        data = await _paced(
            lambda chunk=chunk: client.post_json(
                f"{S2_GRAPH}/paper/batch?fields={fields}",
                json_body={"ids": chunk},
                headers=s2_headers(),
            )
        )
        out.extend(data if isinstance(data, list) else [])
    return out


async def search_semanticscholar(
    client: HTTPClient,
    *,
    query: str,
    limit: int,
    settings: Settings,
    filters: SearchFilters | None = None,
    offset: int = 0,
) -> list[Paper]:
    _ = settings
    limit = max(1, min(int(limit), 100))
    params = filter_params(filters)
    sort = (filters.sort if filters else "relevance") or "relevance"

    if sort == "relevance":
        params.update(
            {"query": query, "limit": str(limit), "offset": str(max(int(offset), 0)), "fields": PAPER_FIELDS}
        )
        data = await s2_get(client, "/paper/search", params)
        items = data.get("data") or []
        return [paper_from_s2(item) for item in items[:limit] if isinstance(item, dict)]

    # Citation or recency order: only the bulk endpoint sorts server-side. It
    # matches every query term (AND), which is also what "the most cited papers
    # on X" needs — a relevance search re-sorted locally would only reorder
    # its own top page.
    params.update(
        {
            "query": query,
            "fields": _BULK_FIELDS,
            "sort": "citationCount:desc" if sort == "citations" else "publicationDate:desc",
        }
    )
    data = await s2_get(client, "/paper/search/bulk", params)
    rows = [r for r in (data.get("data") or []) if isinstance(r, dict) and r.get("paperId")]
    picked = rows[max(int(offset), 0) : max(int(offset), 0) + limit]
    if not picked:
        return []
    records = await s2_batch(client, [r["paperId"] for r in picked])
    return [paper_from_s2(rec) for rec in records if isinstance(rec, dict) and rec.get("title")]
