from __future__ import annotations

from ..config import Settings
from ..http_client import HTTPClient
from ..models import Paper
from ..utils import jaccard_similarity, normalize_doi, normalize_whitespace, strip_html


async def search_crossref(
    client: HTTPClient, *, query: str, limit: int, settings: Settings
) -> list[Paper]:
    params = {"query": query, "rows": str(limit)}
    mailto = settings.pick_crossref_mailto()
    if mailto:
        params["mailto"] = mailto

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

        papers.append(
            Paper(
                title=title,
                abstract=abstract,
                url=url,
                doi=doi,
                authors=authors,
                source_platform="Crossref",
            )
        )
    return papers


async def guess_doi_from_crossref(
    client: HTTPClient, *, title: str, authors: str, settings: Settings
) -> tuple[str, str] | None:
    """
    Best-effort DOI enrichment using Crossref.
    Returns (doi, url) if a confident match is found, otherwise None.
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

        # Slightly boost if first author matches.
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
