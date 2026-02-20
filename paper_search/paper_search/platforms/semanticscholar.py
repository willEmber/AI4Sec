from __future__ import annotations

from ..config import Settings
from ..http_client import HTTPClient
from ..models import Paper
from ..utils import normalize_doi, normalize_whitespace


async def search_semanticscholar(
    client: HTTPClient, *, query: str, limit: int, settings: Settings
) -> list[Paper]:
    params = {
        "query": query,
        "limit": str(int(limit)),
        "fields": "title,abstract,authors,url,externalIds",
    }
    headers = {}
    if settings.semantic_api_key:
        headers["x-api-key"] = settings.semantic_api_key

    data = await client.get_json(
        "https://api.semanticscholar.org/graph/v1/paper/search", params=params, headers=headers
    )
    items = data.get("data") or []

    papers: list[Paper] = []
    for item in items[:limit]:
        title = normalize_whitespace(item.get("title") or "")
        abstract = normalize_whitespace(item.get("abstract") or "")
        url = normalize_whitespace(item.get("url") or "")

        external_ids = item.get("externalIds") or {}
        doi = normalize_doi(external_ids.get("DOI") or "")
        authors_raw = item.get("authors") or []
        authors = "; ".join(
            normalize_whitespace(a.get("name") or "")
            for a in authors_raw
            if normalize_whitespace(a.get("name") or "")
        )

        papers.append(
            Paper(
                title=title,
                abstract=abstract,
                url=url,
                doi=doi,
                authors=authors,
                source_platform="SemanticScholar",
            )
        )
    return papers
