"""Reading one web page: which provider fetches it, and when to try another.

Exa contents first: it returns the page's main text (3k characters for the
vLLM docs front page, where Tavily's basic extract returned 290k of site
navigation), keeps its own cache, and costs a tenth of a cent. Tavily extract
second, Firecrawl last (renders JavaScript and parses PDFs, but has the
scarcest credit). A page that comes back nearly empty is usually
a script-rendered shell the first two cannot run, so it counts as not read and
the next provider is tried; if every provider returns a short page, the
longest one is kept — some pages really are short.
"""

from __future__ import annotations

import time
from typing import Awaitable, Callable, Mapping

import httpx

from app.services.paper_search.models import PlatformStatus
from app.services.web_search.credentials import EXA, FIRECRAWL, TAVILY, web_search_enabled
from app.services.web_search.http import new_client
from app.services.web_search.keypool import KeyPool, resolve_pool
from app.services.web_search.models import FetchOutcome, ProviderError, WebPage
from app.services.web_search.providers import exa, firecrawl, tavily
from app.services.web_search.urls import fetchable_reason

Fetcher = Callable[[httpx.AsyncClient, KeyPool, str], Awaitable[WebPage]]

FETCHERS: tuple[tuple[str, Fetcher], ...] = (
    (EXA, exa.contents),
    (TAVILY, tavily.extract),
    (FIRECRAWL, firecrawl.scrape),
)

# Below this, a page is most likely a JavaScript shell or a cookie wall.
MIN_PAGE_CHARS = 200


async def fetch_page(
    url: str,
    *,
    client: httpx.AsyncClient | None = None,
    pools: Mapping[str, KeyPool] | None = None,
) -> FetchOutcome:
    """Fetch `url` as Markdown. Raises `ValueError` for a URL that may not be
    fetched at all (see `urls.fetchable_reason`)."""
    reason = fetchable_reason(url)
    if reason:
        raise ValueError(reason)
    if not web_search_enabled():
        return FetchOutcome(
            page=None,
            attempts=[PlatformStatus(p, "unsupported", detail="web search is disabled") for p, _ in FETCHERS],
        )

    owns_client = client is None
    http = client if client is not None else new_client()
    attempts: list[PlatformStatus] = []
    best_short: WebPage | None = None
    try:
        for provider, fetcher in FETCHERS:
            pool = resolve_pool(provider, pools)
            t0 = time.perf_counter()
            try:
                page = await fetcher(http, pool, url)
            except ProviderError as exc:
                attempts.append(
                    PlatformStatus(provider, exc.status, detail=exc.detail, elapsed_s=time.perf_counter() - t0)
                )
                continue
            except Exception as exc:  # noqa: BLE001
                attempts.append(
                    PlatformStatus(provider, "failed", detail=type(exc).__name__, elapsed_s=time.perf_counter() - t0)
                )
                continue
            size = len(page.markdown.strip())
            elapsed = time.perf_counter() - t0
            if size < MIN_PAGE_CHARS:
                attempts.append(
                    PlatformStatus(
                        provider,
                        "empty",
                        count=size,
                        detail=f"only {size} characters; likely a script-rendered page",
                        elapsed_s=elapsed,
                    )
                )
                if best_short is None or size > len(best_short.markdown.strip()):
                    best_short = page
                continue
            attempts.append(PlatformStatus(provider, "ok", count=size, elapsed_s=elapsed))
            return FetchOutcome(page=page, attempts=attempts)
    finally:
        if owns_client:
            await http.aclose()
    return FetchOutcome(page=best_short, attempts=attempts)
