"""Web search orchestration: which provider to ask, when to ask the next one,
and what the caller is told about coverage.

A provider is asked only when the one before it *did not search* — refused,
rate-limited, out of credit, unreachable. One that searched and found nothing
is an answer, and the next provider is not asked to second-guess it: that
would spend credit on every narrow query, and the agent can broaden the query
itself, which is usually the better fix. `depth="deep"` is the exception,
asking two providers at once and merging them.
"""

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, Mapping

import httpx

from app.services.paper_search.models import PlatformStatus
from app.services.web_search.credentials import EXA, FIRECRAWL, TAVILY, web_search_enabled
from app.services.web_search.http import new_client
from app.services.web_search.keypool import KeyPool, resolve_pool
from app.services.web_search.models import (
    TOPICS,
    ProviderError,
    WebResult,
    WebSearchOutcome,
    WebSearchRequest,
)
from app.services.web_search.providers import exa, firecrawl, tavily
from app.services.web_search.urls import domain_matches

Searcher = Callable[
    [httpx.AsyncClient, KeyPool, WebSearchRequest], Awaitable[tuple[list[WebResult], list[str]]]
]

SEARCHERS: dict[str, Searcher] = {
    TAVILY: tavily.search,
    EXA: exa.search,
    FIRECRAWL: firecrawl.search,
}

# Tavily first for general and news (date filters, news index, cheap basic
# depth); Exa first for "find the page about X", where embedding search beats
# keyword search. Firecrawl is last everywhere: scarcest credit, slowest limit.
ROUTES: dict[str, tuple[str, ...]] = {
    "general": (TAVILY, EXA, FIRECRAWL),
    "news": (TAVILY, EXA, FIRECRAWL),
    "technical": (EXA, TAVILY, FIRECRAWL),
}

# Reciprocal-rank-fusion constant; 60 is the value from the original paper and
# what most hybrid-search systems use.
RRF_K = 60


async def search_web(
    req: WebSearchRequest,
    *,
    client: httpx.AsyncClient | None = None,
    pools: Mapping[str, KeyPool] | None = None,
) -> WebSearchOutcome:
    if not req.query.strip():
        raise ValueError("empty query")
    if req.topic not in TOPICS:
        raise ValueError(f"unknown topic {req.topic!r}")
    chain = ROUTES[req.topic]
    if not web_search_enabled():
        return WebSearchOutcome(
            results=[],
            providers=[PlatformStatus(p, "unsupported", detail="web search is disabled") for p in chain],
        )

    wanted = 2 if req.depth == "deep" else 1
    owns_client = client is None
    http = client if client is not None else new_client()
    statuses: list[PlatformStatus] = []
    answered: list[tuple[list[WebResult], list[str]]] = []
    try:
        idx = 0
        while len(answered) < wanted and idx < len(chain):
            batch = chain[idx : idx + (wanted - len(answered))]
            idx += len(batch)
            runs = await asyncio.gather(*(_run(p, http, pools, req) for p in batch))
            for status, results, remote in runs:
                statuses.append(status)
                if status.searched:
                    answered.append((results, remote))
    finally:
        if owns_client:
            await http.aclose()

    merged = _merge([results for results, _ in answered])
    # Which providers applied the date filter themselves: a dateless result
    # from one of them passed the provider's own date check.
    date_checked = {
        s.platform for s in statuses if s.searched and "date" in s.remote_filters
    }
    kept, excluded, unknown = _filter_locally(merged, req, date_checked)
    remote = sorted({f for _, r in answered for f in r})
    return WebSearchOutcome(
        results=kept[: req.max_results],
        providers=statuses,
        excluded=excluded,
        remote_filters=remote,
        unknown_date=unknown,
    )


async def _run(
    provider: str,
    client: httpx.AsyncClient,
    pools: Mapping[str, KeyPool] | None,
    req: WebSearchRequest,
) -> tuple[PlatformStatus, list[WebResult], list[str]]:
    pool = resolve_pool(provider, pools)
    t0 = time.perf_counter()
    try:
        results, remote = await SEARCHERS[provider](client, pool, req)
    except ProviderError as exc:
        return (
            PlatformStatus(provider, exc.status, detail=exc.detail, elapsed_s=time.perf_counter() - t0),
            [],
            [],
        )
    except Exception as exc:  # noqa: BLE001 — an adapter bug must not sink the search
        return (
            PlatformStatus(provider, "failed", detail=type(exc).__name__, elapsed_s=time.perf_counter() - t0),
            [],
            [],
        )
    results = _dedupe(results)
    return (
        PlatformStatus(
            provider,
            "ok" if results else "empty",
            count=len(results),
            remote_filters=list(remote),
            elapsed_s=time.perf_counter() - t0,
        ),
        results,
        remote,
    )


def _dedupe(results: list[WebResult]) -> list[WebResult]:
    seen: set[str] = set()
    out: list[WebResult] = []
    for r in results:
        if r.key and r.key not in seen:
            seen.add(r.key)
            r.providers = r.providers or [r.provider]
            out.append(r)
    return out


def _merge(lists: list[list[WebResult]]) -> list[WebResult]:
    """One list, fused by reciprocal rank when several providers answered.

    A page two providers both ranked highly beats one only a single provider
    liked, and the record keeps a date and a title from whichever provider
    had them.
    """
    if not lists:
        return []
    if len(lists) == 1:
        return list(lists[0])
    scores: dict[str, float] = {}
    first: dict[str, WebResult] = {}
    for results in lists:
        for rank, r in enumerate(results):
            scores[r.key] = scores.get(r.key, 0.0) + 1.0 / (RRF_K + rank + 1)
            held = first.get(r.key)
            if held is None:
                first[r.key] = r
                continue
            for p in r.providers:
                if p not in held.providers:
                    held.providers.append(p)
            held.published_date = held.published_date or r.published_date
            held.title = held.title or r.title
            if not held.snippet:
                held.snippet = r.snippet
    order = sorted(first, key=lambda k: scores[k], reverse=True)
    return [first[k] for k in order]


def _filter_locally(
    results: list[WebResult], req: WebSearchRequest, date_checked: set[str]
) -> tuple[list[WebResult], dict[str, int], list[WebResult]]:
    """Enforce what the providers were asked for, whatever they returned.

    A known date outside the range is dropped. An unknown date is kept when a
    provider that returned the page applied the range itself; otherwise it
    cannot be said to satisfy the range, and is set aside rather than passed
    off as matching.
    """
    kept: list[WebResult] = []
    unknown: list[WebResult] = []
    excluded: dict[str, int] = {}

    def drop(reason: str) -> None:
        excluded[reason] = excluded.get(reason, 0) + 1

    start = req.start_date.isoformat() if req.start_date else ""
    end = req.end_date.isoformat() if req.end_date else ""
    for r in results:
        if req.include_domains and not domain_matches(r.url, req.include_domains):
            drop("include_domains")
            continue
        if req.exclude_domains and domain_matches(r.url, req.exclude_domains):
            drop("exclude_domains")
            continue
        if req.has_date_filter:
            if r.published_date is None:
                if not any(p in date_checked for p in r.providers):
                    unknown.append(r)
                    continue
            elif (start and r.published_date < start) or (end and r.published_date > end):
                drop("date")
                continue
        kept.append(r)
    return kept, excluded, unknown
