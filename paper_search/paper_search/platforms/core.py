from __future__ import annotations

from ..http_client import HTTPClient
from ..utils import normalize_doi, normalize_whitespace


async def core_find_oa_url(
    client: HTTPClient, *, doi: str, title: str, api_key: str
) -> str | None:
    """
    Best-effort OA fulltext URL discovery via CORE API v3.
    Preference order within a result:
      - downloadUrl
      - sourceFulltextUrls (first item)
    """
    api_key = (api_key or "").strip()
    if not api_key:
        return None

    doi_norm = normalize_doi(doi)
    query = doi_norm or normalize_whitespace(title)
    if not query:
        return None

    url = "https://api.core.ac.uk/v3/search/works/"
    params = {"q": query, "limit": "1", "offset": "0", "sort": "relevance"}
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        data = await client.get_json(url, params=params, headers=headers)
    except Exception:
        return None

    results = (data or {}).get("results") or []
    if not results:
        return None

    first = results[0] or {}
    download_url = normalize_whitespace(first.get("downloadUrl") or "")
    if download_url:
        return download_url

    fulltext_urls = first.get("sourceFulltextUrls")
    if isinstance(fulltext_urls, str):
        candidate = normalize_whitespace(fulltext_urls)
        return candidate or None
    if isinstance(fulltext_urls, list):
        for u in fulltext_urls:
            candidate = normalize_whitespace(u or "")
            if candidate:
                return candidate

    return None

