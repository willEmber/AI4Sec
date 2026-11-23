"""Keenable: web search with query-relevant page text in each result and
server-side date and domain filters.

Opt-in: it runs only when `KEENABLE_API_KEYS` is set, and it sits last in
every route, so it is asked only when the providers before it did not search.
Search only; page reading stays with the existing fetchers.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.services.web_search.credentials import KEENABLE
from app.services.web_search.http import pooled_post
from app.services.web_search.keypool import KeyPool
from app.services.web_search.models import WebResult, WebSearchRequest
from app.services.web_search.providers import clean_snippet, iso_date

SEARCH_URL = "https://api.keenable.ai/v1/search"
QUOTA_STATUSES = frozenset({402})
# Keenable accepts 1-50 results per request.
MAX_RESULTS = 50
SNIPPET_CHARS = 1500
# Identifies the app on Keenable's side; required on the keyless endpoint,
# sent here too so usage is labelled the same way.
APP_TITLE = "AI4Sec"


def _auth(key: str) -> dict[str, str]:
    return {"X-API-Key": key, "X-Keenable-Title": APP_TITLE}


async def search(
    client: httpx.AsyncClient, pool: KeyPool, req: WebSearchRequest
) -> tuple[list[WebResult], list[str]]:
    payload: dict[str, Any] = {
        "query": req.query,
        "max_results": max(1, min(req.max_results, MAX_RESULTS)),
        "snippet_max_length": SNIPPET_CHARS,
        # Instant-answer cards point at cards.keenable.ai URLs that expire
        # after a day, which makes them poor sources to cite.
        "disable_cards": True,
    }
    remote: list[str] = []
    if req.start_date:
        payload["published_after"] = req.start_date.isoformat()
    if req.end_date:
        payload["published_before"] = req.end_date.isoformat()
    if req.has_date_filter:
        remote.append("date")
    # `site` takes a single host; a longer include list is left to the local
    # domain filter.
    if len(req.include_domains) == 1:
        payload["site"] = req.include_domains[0]
    if req.exclude_domains:
        payload["exclude_domains"] = list(req.exclude_domains)
    if len(req.include_domains) == 1 or req.exclude_domains:
        remote.append("domains")

    data = await pooled_post(
        client, pool, SEARCH_URL, body=payload, auth=_auth, quota_statuses=QUOTA_STATUSES
    )
    results: list[WebResult] = []
    for item in data.get("results") or []:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        results.append(
            WebResult(
                url=str(item["url"]),
                title=clean_snippet(item.get("title"), 300),
                snippet=clean_snippet(item.get("snippet") or item.get("description")),
                published_date=iso_date(item.get("published_at")),
                provider=KEENABLE,
            )
        )
    return results, remote
