"""Exa: embedding-based search, strongest for "find the page that is about X"
(project pages, personal homepages, technical write-ups), and a cached
contents endpoint that is cheap per page.

Search asks for query-guided highlights rather than full text, so a result
costs a snippet's worth of context, not a page's.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.services.web_search.credentials import EXA
from app.services.web_search.http import pooled_post
from app.services.web_search.keypool import KeyPool
from app.services.web_search.models import ProviderError, WebPage, WebResult, WebSearchRequest
from app.services.web_search.providers import MAX_PAGE_CHARS, clean_snippet, iso_date

SEARCH_URL = "https://api.exa.ai/search"
CONTENTS_URL = "https://api.exa.ai/contents"
QUOTA_STATUSES = frozenset({402})
# Exa bills a search by its first ten results and clamps above that.
MAX_RESULTS = 10
HIGHLIGHT_CHARS = 1500
LIVECRAWL_TIMEOUT_MS = 12_000


def _auth(key: str) -> dict[str, str]:
    return {"x-api-key": key}


async def search(
    client: httpx.AsyncClient, pool: KeyPool, req: WebSearchRequest
) -> tuple[list[WebResult], list[str]]:
    payload: dict[str, Any] = {
        "query": req.query,
        "type": "auto",
        "numResults": max(1, min(req.max_results, MAX_RESULTS)),
        "contents": {"highlights": {"query": req.query, "maxCharacters": HIGHLIGHT_CHARS}},
    }
    if req.topic == "news":
        payload["category"] = "news"
    remote: list[str] = []
    if req.start_date:
        payload["startPublishedDate"] = f"{req.start_date.isoformat()}T00:00:00.000Z"
    if req.end_date:
        payload["endPublishedDate"] = f"{req.end_date.isoformat()}T23:59:59.999Z"
    if req.has_date_filter:
        remote.append("date")
    if req.include_domains:
        payload["includeDomains"] = list(req.include_domains)
    if req.exclude_domains:
        payload["excludeDomains"] = list(req.exclude_domains)
    if req.include_domains or req.exclude_domains:
        remote.append("domains")

    data = await pooled_post(
        client, pool, SEARCH_URL, body=payload, auth=_auth, quota_statuses=QUOTA_STATUSES
    )
    results: list[WebResult] = []
    for item in data.get("results") or []:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        highlights = [h for h in item.get("highlights") or [] if isinstance(h, str) and h.strip()]
        snippet = " … ".join(highlights) or item.get("text") or item.get("summary") or ""
        results.append(
            WebResult(
                url=str(item["url"]),
                title=clean_snippet(item.get("title"), 300),
                snippet=clean_snippet(snippet),
                published_date=iso_date(item.get("publishedDate")),
                provider=EXA,
                score=item.get("score") if isinstance(item.get("score"), (int, float)) else None,
            )
        )
    return results, remote


async def contents(client: httpx.AsyncClient, pool: KeyPool, url: str) -> WebPage:
    """The page text, from Exa's cache when it has one and live-crawled
    otherwise. A per-URL failure comes back in `statuses`, not as an HTTP
    error, and is reported as `empty` so the next provider is tried."""
    data = await pooled_post(
        client,
        pool,
        CONTENTS_URL,
        body={
            "urls": [url],
            "text": {"maxCharacters": MAX_PAGE_CHARS},
            "livecrawlTimeout": LIVECRAWL_TIMEOUT_MS,
        },
        auth=_auth,
        quota_statuses=QUOTA_STATUSES,
    )
    for status in data.get("statuses") or []:
        if isinstance(status, dict) and status.get("status") == "error":
            error = status.get("error") or {}
            raise ProviderError(EXA, "empty", str(error.get("tag") or "crawl failed"))
    for item in data.get("results") or []:
        text = (item or {}).get("text") or ""
        if text.strip():
            return WebPage(
                url=url,
                final_url=str(item.get("url") or url),
                markdown=text[:MAX_PAGE_CHARS],
                provider=EXA,
                title=clean_snippet(item.get("title"), 300),
                published_date=iso_date(item.get("publishedDate")),
            )
    raise ProviderError(EXA, "empty", "no content returned")
