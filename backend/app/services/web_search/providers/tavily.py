"""Tavily: general and news search, and plain page extraction.

Search returns up to three ≤500-character chunks per source already picked for
the query, which is the right size for the agent to judge a result without
reading the page. `published_date` only comes back for `topic=news`; for
general search a date filter is applied by Tavily (on publish *or* last-update
date) but the result carries no date of its own.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.services.web_search.credentials import TAVILY
from app.services.web_search.http import pooled_post
from app.services.web_search.keypool import KeyPool
from app.services.web_search.models import ProviderError, WebPage, WebResult, WebSearchRequest
from app.services.web_search.providers import MAX_PAGE_CHARS, clean_snippet, iso_date

SEARCH_URL = "https://api.tavily.com/search"
EXTRACT_URL = "https://api.tavily.com/extract"
# 432: key or plan credit spent; 433: pay-as-you-go spending cap reached.
QUOTA_STATUSES = frozenset({432, 433})
MAX_RESULTS = 20


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


async def search_raw(client: httpx.AsyncClient, pool: KeyPool, payload: dict[str, Any]) -> dict[str, Any]:
    """The raw `/search` response, for callers that format it themselves."""
    return await pooled_post(
        client, pool, SEARCH_URL, body=payload, auth=_auth, quota_statuses=QUOTA_STATUSES
    )


async def search(
    client: httpx.AsyncClient, pool: KeyPool, req: WebSearchRequest
) -> tuple[list[WebResult], list[str]]:
    """Results and the filters Tavily applied itself."""
    payload: dict[str, Any] = {
        "query": req.query,
        "search_depth": "advanced" if req.depth == "deep" else "basic",
        "chunks_per_source": 3,
        "max_results": max(1, min(req.max_results, MAX_RESULTS)),
        "topic": "news" if req.topic == "news" else "general",
        "include_answer": False,
        "include_raw_content": False,
    }
    remote: list[str] = []
    if req.start_date:
        payload["start_date"] = req.start_date.isoformat()
    if req.end_date:
        payload["end_date"] = req.end_date.isoformat()
    if req.has_date_filter:
        remote.append("date")
    if req.include_domains:
        payload["include_domains"] = list(req.include_domains)
    if req.exclude_domains:
        payload["exclude_domains"] = list(req.exclude_domains)
    if req.include_domains or req.exclude_domains:
        remote.append("domains")

    data = await search_raw(client, pool, payload)
    results = [
        WebResult(
            url=str(item.get("url") or ""),
            title=clean_snippet(item.get("title"), 300),
            snippet=clean_snippet(item.get("content")),
            published_date=iso_date(item.get("published_date")),
            provider=TAVILY,
            score=item.get("score") if isinstance(item.get("score"), (int, float)) else None,
        )
        for item in data.get("results") or []
        if isinstance(item, dict) and item.get("url")
    ]
    return results, remote


async def extract(client: httpx.AsyncClient, pool: KeyPool, url: str) -> WebPage:
    """The whole page as Markdown. No `query`: with one, Tavily returns only
    the matching chunks, and the page is chunked and ranked here instead so
    that later questions about the same page can be answered from cache."""
    data = await pooled_post(
        client,
        pool,
        EXTRACT_URL,
        body={"urls": [url], "extract_depth": "basic", "format": "markdown"},
        auth=_auth,
        quota_statuses=QUOTA_STATUSES,
    )
    for item in data.get("results") or []:
        content = (item or {}).get("raw_content") or ""
        if content.strip():
            return WebPage(
                url=url,
                final_url=str(item.get("url") or url),
                markdown=content[:MAX_PAGE_CHARS],
                provider=TAVILY,
                title=clean_snippet(item.get("title"), 300),
            )
    failed = next(iter(data.get("failed_results") or []), None) or {}
    raise ProviderError(TAVILY, "empty", clean_snippet(failed.get("error") or "no content extracted", 200))
