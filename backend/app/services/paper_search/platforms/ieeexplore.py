from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from threading import Lock
import time
from typing import Any

from ..config import Settings
from ..debug import debug
from ..http_client import HTTPClient
from ..models import FILTER_PUBLICATION_TYPES, FILTER_YEAR, Paper, SearchFilters
from ..ratelimit import MissingCredentialError
from ..utils import normalize_doi, normalize_whitespace, strip_html


IEEE_API_URL = "https://ieeexploreapi.ieee.org/api/v1/search/articles"

SUPPORTED_FILTERS = frozenset({FILTER_YEAR, FILTER_PUBLICATION_TYPES})
SUPPORTED_SORTS = frozenset({"relevance"})

_TYPE_TO_IEEE = {"conference": "Conferences", "journal": "Journals"}


class _IeeeRateLimiter:
    def __init__(self) -> None:
        self._lock = Lock()
        self._second_window_start = 0.0
        self._second_window_calls = 0
        self._day_ordinal = 0
        self._day_calls = 0

    async def acquire(self, *, per_second: int, per_day: int) -> None:
        per_second = max(int(per_second), 1)
        per_day = max(int(per_day), 1)

        while True:
            sleep_s = 0.0
            with self._lock:
                now_monotonic = time.monotonic()
                now_day = datetime.now(timezone.utc).toordinal()

                if self._day_ordinal != now_day:
                    self._day_ordinal = now_day
                    self._day_calls = 0

                if self._day_calls >= per_day:
                    raise RuntimeError(
                        f"IEEE Xplore daily rate limit reached: {self._day_calls}/{per_day} (UTC day)"
                    )

                if (now_monotonic - self._second_window_start) >= 1.0:
                    self._second_window_start = now_monotonic
                    self._second_window_calls = 0

                if self._second_window_calls < per_second:
                    self._second_window_calls += 1
                    self._day_calls += 1
                    return

                sleep_s = max(0.001, 1.0 - (now_monotonic - self._second_window_start))

            await asyncio.sleep(sleep_s)


_IEEE_RATE_LIMITER = _IeeeRateLimiter()


def _extract_authors(item: dict[str, Any]) -> str:
    raw = item.get("authors")
    if isinstance(raw, dict):
        seq = raw.get("authors") or []
    elif isinstance(raw, list):
        seq = raw
    else:
        seq = []

    names: list[str] = []
    for a in seq:
        if not isinstance(a, dict):
            continue
        name = normalize_whitespace(
            a.get("full_name")
            or a.get("name")
            or a.get("authorName")
            or a.get("preferred_name")
            or ""
        )
        if name:
            names.append(name)
    return "; ".join(names)


def _extract_url(item: dict[str, Any], doi: str) -> str:
    for key in ("html_url", "article_url", "document_url", "pdf_url", "doi_url"):
        value = normalize_whitespace(item.get(key) or "")
        if value:
            return value
    if doi:
        return f"https://doi.org/{doi}"
    return ""


async def search_ieeexplore(
    client: HTTPClient,
    *,
    query: str,
    limit: int,
    settings: Settings,
    filters: SearchFilters | None = None,
    offset: int = 0,
) -> list[Paper]:
    api_key = normalize_whitespace(settings.ieee_api_key or "")
    if not api_key:
        debug("platform=IEEE Xplore skipped: PAPERSEARCH_IEEE_API_KEY not set")
        raise MissingCredentialError("PAPERSEARCH_IEEE_API_KEY is not set")

    await _IEEE_RATE_LIMITER.acquire(
        per_second=settings.ieee_per_second_limit,
        per_day=settings.ieee_daily_limit,
    )

    params = {
        "querytext": query,
        "apikey": api_key,
        "format": "json",
        "max_records": str(max(int(limit), 1)),
        "start_record": str(max(int(offset), 0) + 1),
    }
    if filters is not None:
        if filters.year_from:
            params["start_year"] = str(filters.year_from)
        if filters.year_to:
            params["end_year"] = str(filters.year_to)
        # The API takes one content type per request; with several, filter locally.
        types = [_TYPE_TO_IEEE[t] for t in filters.publication_types if t in _TYPE_TO_IEEE]
        if len(types) == 1:
            params["content_type"] = types[0]
    data = await client.get_json(IEEE_API_URL, params=params)
    items = data.get("articles") or []

    papers: list[Paper] = []
    for item in items[:limit]:
        if not isinstance(item, dict):
            continue

        title = normalize_whitespace(
            item.get("article_title")
            or item.get("title")
            or item.get("document_title")
            or item.get("publication_title")
            or ""
        )
        abstract = strip_html(item.get("abstract") or "")
        doi = normalize_doi(item.get("doi") or "")
        url = _extract_url(item, doi)
        authors = _extract_authors(item)

        year = 0
        raw_year = item.get("publication_year")
        if isinstance(raw_year, int):
            year = raw_year
        elif isinstance(raw_year, str) and raw_year.isdigit():
            year = int(raw_year)

        venue = normalize_whitespace(item.get("publication_title") or "")
        content_type = (item.get("content_type") or "").lower()
        venue_type = "conference" if "conference" in content_type else (
            "journal" if "journal" in content_type else ""
        )
        citation_counts: dict[str, int] = {}
        cites = item.get("citing_paper_count")
        if isinstance(cites, int) or (isinstance(cites, str) and cites.isdigit()):
            citation_counts["IEEE Xplore"] = int(cites)

        papers.append(
            Paper(
                title=title,
                abstract=abstract,
                url=url,
                doi=doi,
                authors=authors,
                source_platform="IEEE Xplore",
                year=year,
                venue=venue,
                venue_type=venue_type,
                citation_counts=citation_counts,
            )
        )
    return papers
