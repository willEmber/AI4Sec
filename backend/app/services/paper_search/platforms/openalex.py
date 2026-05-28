from __future__ import annotations

from ..config import Settings
from ..http_client import HTTPClient
from ..models import Paper
from ..utils import normalize_doi, normalize_whitespace, openalex_abstract_from_inverted_index


async def search_openalex(
    client: HTTPClient, *, query: str, limit: int, settings: Settings
) -> list[Paper]:
    params = {"search": query, "per_page": str(limit), "page": "1"}
    mailto = settings.pick_openalex_mailto()
    if mailto:
        params["mailto"] = mailto

    data = await client.get_json("https://api.openalex.org/works", params=params)
    results = data.get("results") or []

    papers: list[Paper] = []
    for item in results[:limit]:
        title = normalize_whitespace(item.get("title") or "")
        abstract = openalex_abstract_from_inverted_index(item.get("abstract_inverted_index"))

        doi = normalize_doi(item.get("doi") or "")
        url = ""
        primary = item.get("primary_location") or {}
        url = (primary.get("landing_page_url") or "").strip() or (item.get("id") or "").strip()
        if not url and doi:
            url = f"https://doi.org/{doi}"

        authorships = item.get("authorships") or []
        author_names: list[str] = []
        for a in authorships:
            author = (a or {}).get("author") or {}
            name = (author.get("display_name") or "").strip()
            if name:
                author_names.append(name)
        authors = "; ".join(author_names)

        year = 0
        raw_year = item.get("publication_year")
        if isinstance(raw_year, int):
            year = raw_year
        elif isinstance(raw_year, str) and raw_year.isdigit():
            year = int(raw_year)

        venue = ""
        source = (primary.get("source") or {})
        venue = normalize_whitespace(source.get("display_name") or "")

        papers.append(
            Paper(
                title=title,
                abstract=abstract,
                url=url,
                doi=doi,
                authors=authors,
                source_platform="OpenAlex",
                year=year,
                venue=venue,
            )
        )
    return papers
