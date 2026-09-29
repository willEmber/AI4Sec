"""Shapes every provider adapter normalises into.

Provider status reuses `paper_search.PlatformStatus`: "a provider that could
not search is not a provider that found nothing" is the same rule for web
search as for paper search, and the agent should read one vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from app.services.paper_search.models import PlatformStatus
from app.services.web_search.urls import domain_of, normalize_url

TOPICS = ("general", "news", "technical")
DEPTHS = ("fast", "deep")


class ProviderError(RuntimeError):
    """A provider did not answer the request.

    `status` is a `PlatformStatus` status other than ok/empty. The message is
    built by the adapter and never contains a key.
    """

    def __init__(self, provider: str, status: str, detail: str = "") -> None:
        super().__init__(f"{provider}: {status}" + (f" — {detail}" if detail else ""))
        self.provider = provider
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class WebSearchRequest:
    query: str
    max_results: int = 8
    topic: str = "general"
    depth: str = "fast"
    start_date: date | None = None
    end_date: date | None = None
    include_domains: tuple[str, ...] = ()
    exclude_domains: tuple[str, ...] = ()

    @property
    def has_date_filter(self) -> bool:
        return self.start_date is not None or self.end_date is not None


@dataclass
class WebResult:
    url: str
    title: str = ""
    snippet: str = ""
    # ISO date (YYYY-MM-DD) when the provider reported one, else None.
    # Never a guess: "no date" and "old" must stay distinguishable.
    published_date: str | None = None
    provider: str = ""
    score: float | None = None
    # Every provider that returned this URL, after a deep search merged them.
    providers: list[str] = field(default_factory=list)

    @property
    def domain(self) -> str:
        return domain_of(self.url)

    @property
    def key(self) -> str:
        return normalize_url(self.url)

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "domain": self.domain,
            "title": self.title,
            "snippet": self.snippet,
            "published_date": self.published_date,
            "date_known": self.published_date is not None,
            "provider": self.provider,
            "providers": list(self.providers or [self.provider]),
        }


@dataclass
class WebSearchOutcome:
    results: list[WebResult]
    providers: list[PlatformStatus]
    # Results dropped by a local filter, by filter name.
    excluded: dict[str, int] = field(default_factory=dict)
    # Filters the answering provider applied itself; a result with no date
    # that survives a pushed-down date filter passed the provider's check.
    remote_filters: list[str] = field(default_factory=list)
    # Results with no date that no provider date-checked, under a date filter:
    # neither inside nor outside the range, so kept apart from `results`.
    unknown_date: list[WebResult] = field(default_factory=list)

    @property
    def searched(self) -> bool:
        return any(p.searched for p in self.providers)

    @property
    def complete(self) -> bool:
        """Every provider that was tried actually searched."""
        return all(p.searched for p in self.providers)


@dataclass
class WebPage:
    url: str
    markdown: str
    provider: str
    title: str = ""
    final_url: str = ""
    published_date: str | None = None
    content_type: str = ""

    @property
    def domain(self) -> str:
        return domain_of(self.final_url or self.url)


@dataclass
class FetchOutcome:
    page: WebPage | None
    attempts: list[PlatformStatus]

    @property
    def fetched(self) -> bool:
        return self.page is not None
