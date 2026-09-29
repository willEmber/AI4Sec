"""Firecrawl: the heavy fallback. It renders JavaScript and parses PDFs, which
the other two cannot, but the free plan allows ten requests a minute and a
thousand credits a month per key — so it is tried last on both paths.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.services.web_search.credentials import FIRECRAWL
from app.services.web_search.http import pooled_post
from app.services.web_search.keypool import KeyPool
from app.services.web_search.models import ProviderError, WebPage, WebResult, WebSearchRequest
from app.services.web_search.providers import MAX_PAGE_CHARS, clean_snippet, iso_date

SEARCH_URL = "https://api.firecrawl.dev/v2/search"
SCRAPE_URL = "https://api.firecrawl.dev/v2/scrape"
QUOTA_STATUSES = frozenset({402})
MAX_RESULTS = 10
# Firecrawl's own cache is accepted up to two days old: a page read for a
# question is rarely one that changed this morning, and a cache hit is faster.
SCRAPE_MAX_AGE_MS = 2 * 24 * 3600 * 1000
SCRAPE_TIMEOUT_MS = 30_000
# Each PDF page costs a credit; a report longer than this is truncated.
PDF_MAX_PAGES = 20


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def _tbs(req: WebSearchRequest) -> str:
    """Google-style custom date range; `tbs` applies to web results only."""
    parts = ["cdr:1"]
    if req.start_date:
        d = req.start_date
        parts.append(f"cd_min:{d.month}/{d.day}/{d.year}")
    if req.end_date:
        d = req.end_date
        parts.append(f"cd_max:{d.month}/{d.day}/{d.year}")
    return ",".join(parts)


async def search(
    client: httpx.AsyncClient, pool: KeyPool, req: WebSearchRequest
) -> tuple[list[WebResult], list[str]]:
    query = req.query
    remote: list[str] = []
    # No domain parameters; the search operators do the same job.
    if req.include_domains:
        query += " (" + " OR ".join(f"site:{d}" for d in req.include_domains) + ")"
    if req.exclude_domains:
        query += "".join(f" -site:{d}" for d in req.exclude_domains)
    if req.include_domains or req.exclude_domains:
        remote.append("domains")

    source = "news" if req.topic == "news" else "web"
    payload: dict[str, Any] = {
        "query": query,
        "limit": max(1, min(req.max_results, MAX_RESULTS)),
        "sources": [source],
    }
    if req.has_date_filter and source == "web":
        payload["tbs"] = _tbs(req)
        remote.append("date")

    data = await pooled_post(
        client, pool, SEARCH_URL, body=payload, auth=_auth, quota_statuses=QUOTA_STATUSES
    )
    body = data.get("data")
    if isinstance(body, dict):
        items = body.get(source) or []
    elif isinstance(body, list):
        items = body
    else:
        items = []
    results = [
        WebResult(
            url=str(item.get("url") or ""),
            title=clean_snippet(item.get("title"), 300),
            snippet=clean_snippet(item.get("description") or item.get("snippet")),
            published_date=iso_date(item.get("date")),
            provider=FIRECRAWL,
        )
        for item in items
        if isinstance(item, dict) and item.get("url")
    ]
    return results, remote


async def scrape(client: httpx.AsyncClient, pool: KeyPool, url: str) -> WebPage:
    data = await pooled_post(
        client,
        pool,
        SCRAPE_URL,
        body={
            "url": url,
            "formats": ["markdown"],
            "onlyMainContent": True,
            "maxAge": SCRAPE_MAX_AGE_MS,
            "timeout": SCRAPE_TIMEOUT_MS,
            "parsers": [{"type": "pdf", "maxPages": PDF_MAX_PAGES}],
        },
        auth=_auth,
        quota_statuses=QUOTA_STATUSES,
        # Scraping waits on the target site as well as on Firecrawl.
        timeout=SCRAPE_TIMEOUT_MS / 1000 + 15,
    )
    doc = data.get("data") or {}
    meta = doc.get("metadata") or {}
    site_status = meta.get("statusCode")
    if isinstance(site_status, int) and site_status >= 400:
        raise ProviderError(FIRECRAWL, "empty", f"the site answered HTTP {site_status}")
    markdown = doc.get("markdown") or ""
    if not markdown.strip():
        raise ProviderError(FIRECRAWL, "empty", "no content returned")
    return WebPage(
        url=url,
        final_url=str(meta.get("url") or meta.get("sourceURL") or url),
        markdown=markdown[:MAX_PAGE_CHARS],
        provider=FIRECRAWL,
        title=clean_snippet(meta.get("title") or meta.get("ogTitle"), 300),
        published_date=iso_date(meta.get("publishedTime") or meta.get("article:published_time")),
        content_type=str(meta.get("contentType") or ""),
    )
