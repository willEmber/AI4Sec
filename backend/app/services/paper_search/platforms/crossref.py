from __future__ import annotations

from ..config import Settings
from ..http_client import HTTPClient
from ..models import FILTER_PUBLICATION_TYPES, FILTER_YEAR, Paper, SearchFilters
from ..utils import jaccard_similarity, normalize_doi, normalize_whitespace, strip_html

SUPPORTED_FILTERS = frozenset({FILTER_YEAR, FILTER_PUBLICATION_TYPES})
SUPPORTED_SORTS = frozenset({"relevance", "citations", "recent"})

_TYPE_TO_CROSSREF = {
    "conference": "proceedings-article",
    "journal": "journal-article",
    "preprint": "posted-content",
}
_CROSSREF_TO_TYPE = {v: k for k, v in _TYPE_TO_CROSSREF.items()}


def filter_param(filters: SearchFilters | None) -> str:
    if filters is None:
        return ""
    clauses: list[str] = []
    if filters.year_from:
        clauses.append(f"from-pub-date:{filters.year_from}")
    if filters.year_to:
        clauses.append(f"until-pub-date:{filters.year_to}")
    for t in filters.publication_types:
        if t in _TYPE_TO_CROSSREF:
            clauses.append(f"type:{_TYPE_TO_CROSSREF[t]}")
    return ",".join(clauses)


async def search_crossref(
    client: HTTPClient,
    *,
    query: str,
    limit: int,
    settings: Settings,
    filters: SearchFilters | None = None,
    offset: int = 0,
) -> list[Paper]:
    params = {"query": query, "rows": str(limit)}
    if offset:
        params["offset"] = str(int(offset))
    mailto = settings.pick_crossref_mailto()
    if mailto:
        params["mailto"] = mailto
    flt = filter_param(filters)
    if flt:
        params["filter"] = flt
    sort = (filters.sort if filters else "relevance") or "relevance"
    if sort == "citations":
        params.update({"sort": "is-referenced-by-count", "order": "desc"})
    elif sort == "recent":
        params.update({"sort": "published", "order": "desc"})

    data = await client.get_json("https://api.crossref.org/works", params=params)
    items = ((data.get("message") or {}).get("items")) or []

    papers: list[Paper] = []
    for item in items[:limit]:
        titles = item.get("title") or []
        title = normalize_whitespace((titles[0] if titles else "") or "")
        abstract = strip_html(item.get("abstract") or "")

        doi = normalize_doi(item.get("DOI") or "")
        url = (item.get("URL") or "").strip()
        if not url and doi:
            url = f"https://doi.org/{doi}"

        author_parts: list[str] = []
        for a in item.get("author") or []:
            given = (a.get("given") or "").strip()
            family = (a.get("family") or "").strip()
            name = normalize_whitespace(f"{given} {family}".strip())
            if name:
                author_parts.append(name)
        authors = "; ".join(author_parts)

        # year: try published-print → published-online → issued
        year = 0
        publication_date = ""
        for date_key in ("published-print", "published-online", "issued"):
            date_parts = (item.get(date_key) or {}).get("date-parts")
            if date_parts and isinstance(date_parts, list) and date_parts[0]:
                try:
                    year = int(date_parts[0][0])
                    publication_date = "-".join(f"{int(x):02d}" for x in date_parts[0][:3])
                    break
                except (IndexError, ValueError, TypeError):
                    continue

        # venue: container-title (journal/conference name)
        venue = ""
        container = item.get("container-title") or []
        if container and isinstance(container, list):
            venue = normalize_whitespace(container[0] or "")

        citation_counts: dict[str, int] = {}
        if isinstance(item.get("is-referenced-by-count"), int):
            citation_counts["Crossref"] = int(item["is-referenced-by-count"])
        venue_type = _CROSSREF_TO_TYPE.get(item.get("type") or "", "")

        papers.append(
            Paper(
                title=title,
                abstract=abstract,
                url=url,
                doi=doi,
                authors=authors,
                source_platform="Crossref",
                year=year,
                venue=venue,
                publication_date=publication_date if len(publication_date) >= 7 else "",
                venue_type=venue_type,
                citation_counts=citation_counts,
            )
        )
    return papers


async def guess_doi_from_crossref(
    client: HTTPClient, *, title: str, authors: str, settings: Settings
) -> tuple[str, str] | None:
    """Best-effort DOI enrichment using Crossref.

    Returns ``(doi, url)`` if a confident match is found, otherwise None.
    """
    title = normalize_whitespace(title)
    if not title:
        return None

    params = {"query.bibliographic": title, "rows": "3"}
    mailto = settings.pick_crossref_mailto()
    if mailto:
        params["mailto"] = mailto

    try:
        data = await client.get_json("https://api.crossref.org/works", params=params)
    except Exception:
        return None

    items = (((data or {}).get("message") or {}).get("items")) or []
    if not items:
        return None

    author_hint = ""
    if authors:
        author_hint = normalize_whitespace(authors.split(";", 1)[0])

    best: tuple[float, str, str] | None = None
    for item in items:
        titles = item.get("title") or []
        cand_title = normalize_whitespace((titles[0] if titles else "") or "")
        if not cand_title:
            continue
        score = jaccard_similarity(title, cand_title)
        if score < 0.90:
            continue
        doi = normalize_doi(item.get("DOI") or "")
        if not doi:
            continue
        url = normalize_whitespace(item.get("URL") or "") or f"https://doi.org/{doi}"

        if author_hint:
            cand_authors = item.get("author") or []
            cand_str = " ".join(
                normalize_whitespace(f"{(a.get('given') or '').strip()} {(a.get('family') or '').strip()}")
                for a in cand_authors[:3]
            )
            if author_hint and author_hint in cand_str:
                score += 0.02

        if best is None or score > best[0]:
            best = (score, doi, url)

    if best is None:
        return None
    _, doi, url = best
    return doi, url
