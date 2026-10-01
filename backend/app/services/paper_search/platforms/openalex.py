from __future__ import annotations

import re
from typing import Any

from ..config import Settings
from ..credentials import openalex_auth_params
from ..http_client import HTTPClient, HTTPStatusError
from ..models import (
    FILTER_MIN_CITATIONS,
    FILTER_OPEN_ACCESS,
    FILTER_PUBLICATION_TYPES,
    FILTER_YEAR,
    Paper,
    SearchFilters,
)
from ..ratelimit import OPENALEX_BUDGET, QuotaExhaustedError, is_quota_response
from ..utils import normalize_doi, normalize_whitespace, openalex_abstract_from_inverted_index

OPENALEX_WORKS = "https://api.openalex.org/works"

SELECT = (
    "id,doi,title,publication_year,publication_date,type,primary_location,locations,"
    "best_oa_location,open_access,authorships,cited_by_count,abstract_inverted_index,"
    "ids,is_retracted"
)

SUPPORTED_FILTERS = frozenset(
    {FILTER_YEAR, FILTER_MIN_CITATIONS, FILTER_OPEN_ACCESS, FILTER_PUBLICATION_TYPES}
)
SUPPORTED_SORTS = frozenset({"relevance", "citations", "recent"})

# Preprint servers: never the venue we want when a published version exists.
PREPRINT_RE = re.compile(
    r"arxiv|preprint|biorxiv|medrxiv|ssrn|research square|repec", re.IGNORECASE
)
_ARXIV_URL_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf)/([a-z\-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})", re.IGNORECASE
)
_ARXIV_DOI_RE = re.compile(r"^10\.48550/arxiv\.(.+)$", re.IGNORECASE)

_TYPE_TO_OA = {"preprint": "preprint", "review": "review"}


def extract_venue(work: dict[str, Any]) -> str:
    """Venue of the published version, not a preprint server or repository.

    For arXiv-first papers OpenAlex's primary_location is arXiv, which used
    to make every top-conference paper score as a preprint. But "any
    non-preprint location" is not enough either: OpenAlex lists institutional
    repositories (HAL, LA Referencia, Apollo, UvA-DARE, …) as locations, and
    those beat the real journal/conference by list order. OpenAlex marks all
    of them — arXiv included — with source.type == "repository", so filter by
    type first and keep the name regex as a backstop for untyped sources.
    """
    primary_name, primary_type = _loc_source(work.get("primary_location"))
    if _is_published_outlet(primary_name, primary_type):
        return primary_name
    for loc in work.get("locations") or []:
        name, stype = _loc_source(loc)
        if _is_published_outlet(name, stype):
            return name
    return primary_name


def extract_venue_type(work: dict[str, Any]) -> str:
    for loc in [work.get("primary_location"), *(work.get("locations") or [])]:
        name, stype = _loc_source(loc)
        if not _is_published_outlet(name, stype):
            continue
        if stype == "conference":
            return "conference"
        if stype == "journal":
            return "journal"
    if (work.get("type") or "") == "preprint":
        return "preprint"
    return ""


def extract_arxiv_id(work: dict[str, Any]) -> str:
    """OpenAlex has no arXiv id field; it shows up as a DOI or a location URL."""
    doi = normalize_doi(work.get("doi") or (work.get("ids") or {}).get("doi") or "")
    m = _ARXIV_DOI_RE.match(doi)
    if m:
        return m.group(1)
    for loc in [work.get("primary_location"), *(work.get("locations") or [])]:
        loc = loc or {}
        for key in ("landing_page_url", "pdf_url"):
            m = _ARXIV_URL_RE.search(loc.get(key) or "")
            if m:
                return m.group(1)
    return ""


def extract_oa_pdf_url(work: dict[str, Any]) -> str:
    best = work.get("best_oa_location") or {}
    url = normalize_whitespace(best.get("pdf_url") or "")
    if url:
        return url
    return normalize_whitespace(((work.get("open_access") or {}).get("oa_url")) or "")


def _loc_source(loc: dict[str, Any] | None) -> tuple[str, str]:
    source = (loc or {}).get("source") or {}
    name = normalize_whitespace(source.get("display_name", "") or "")
    stype = (source.get("type") or "").strip().lower()
    return name, stype


def _is_published_outlet(name: str, stype: str) -> bool:
    if not name or stype == "repository":
        return False
    return not PREPRINT_RE.search(name)


def work_to_paper(item: dict[str, Any]) -> Paper:
    doi = normalize_doi(item.get("doi") or "")
    primary = item.get("primary_location") or {}
    url = (primary.get("landing_page_url") or "").strip() or (item.get("id") or "").strip()
    if not url and doi:
        url = f"https://doi.org/{doi}"

    author_names: list[str] = []
    for a in item.get("authorships") or []:
        name = (((a or {}).get("author") or {}).get("display_name") or "").strip()
        if name:
            author_names.append(name)

    year = 0
    raw_year = item.get("publication_year")
    if isinstance(raw_year, int):
        year = raw_year
    elif isinstance(raw_year, str) and raw_year.isdigit():
        year = int(raw_year)

    citation_counts: dict[str, int] = {}
    if isinstance(item.get("cited_by_count"), int):
        citation_counts["OpenAlex"] = int(item["cited_by_count"])
    oa = item.get("open_access") or {}
    is_oa = oa.get("is_oa")
    retracted = item.get("is_retracted")
    work_type = (item.get("type") or "").lower()

    return Paper(
        title=normalize_whitespace(item.get("title") or ""),
        abstract=openalex_abstract_from_inverted_index(item.get("abstract_inverted_index")),
        url=url,
        doi=doi,
        authors="; ".join(author_names),
        source_platform="OpenAlex",
        year=year,
        venue=extract_venue(item),
        arxiv_id=extract_arxiv_id(item),
        openalex_id=(item.get("id") or "").replace("https://openalex.org/", ""),
        oa_pdf_url=extract_oa_pdf_url(item),
        publication_date=normalize_whitespace(item.get("publication_date") or ""),
        venue_type=extract_venue_type(item),
        publication_types=["review"] if work_type == "review" else [],
        citation_counts=citation_counts,
        is_open_access=bool(is_oa) if is_oa is not None else None,
        is_retracted=bool(retracted) if retracted is not None else None,
    )


def filter_clauses(filters: SearchFilters | None) -> list[str]:
    if filters is None:
        return []
    clauses: list[str] = []
    years = filters.year_range()
    if years:
        clauses.append(f"publication_year:{years}")
    if filters.min_citations > 0:
        # OpenAlex has strict comparisons only.
        clauses.append(f"cited_by_count:>{int(filters.min_citations) - 1}")
    if filters.open_access_only:
        clauses.append("open_access.is_oa:true")
    types = sorted({_TYPE_TO_OA[t] for t in filters.publication_types if t in _TYPE_TO_OA})
    if types:
        clauses.append("type:" + "|".join(types))
    return clauses


def _filter_text(query: str) -> str:
    # Inside `filter=` a comma separates clauses and a colon separates a key
    # from its value, so neither may appear in the search text.
    return normalize_whitespace(re.sub(r"[,:|]", " ", query))


async def openalex_get(client: HTTPClient, url: str, params: dict[str, str]) -> Any:
    """GET against OpenAlex that keeps the shared daily budget up to date."""
    if OPENALEX_BUDGET.exhausted():
        raise QuotaExhaustedError("OpenAlex", OPENALEX_BUDGET.reset_in())
    try:
        data, headers = await client.get_json_with_headers(url, params=params)
    except HTTPStatusError as exc:
        if is_quota_response(exc.status_code, exc.headers):
            reset = OPENALEX_BUDGET.mark_exhausted(exc.headers)
            raise QuotaExhaustedError("OpenAlex", reset) from exc
        raise
    OPENALEX_BUDGET.observe(headers)
    return data


async def search_openalex(
    client: HTTPClient,
    *,
    query: str,
    limit: int,
    settings: Settings,
    filters: SearchFilters | None = None,
    offset: int = 0,
) -> list[Paper]:
    limit = max(1, min(int(limit), 200))
    params: dict[str, str] = {
        **openalex_auth_params(settings.pick_openalex_mailto()),
        "per_page": str(limit),
        "page": str(max(int(offset), 0) // limit + 1),
        "select": SELECT,
    }
    clauses = filter_clauses(filters)
    sort = (filters.sort if filters else "relevance") or "relevance"
    # `search=` now matches full text as well, which lets a paper that mentions
    # the words once in its body outrank the ones about them — and sorted by
    # citations, a highly cited paper from another field wins outright.
    # Title-and-abstract matching keeps every order on topic.
    clauses.insert(0, f"title_and_abstract.search:{_filter_text(query)}")
    params["sort"] = {
        "citations": "cited_by_count:desc",
        "recent": "publication_date:desc",
    }.get(sort, "relevance_score:desc")
    if clauses:
        params["filter"] = ",".join(clauses)

    data = await openalex_get(client, OPENALEX_WORKS, params)
    results = data.get("results") or []
    return [work_to_paper(item) for item in results[:limit] if isinstance(item, dict)]
