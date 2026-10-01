from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


PUBLIC_FIELDS: tuple[str, ...] = (
    "title",
    "abstract",
    "url",
    "doi",
    "authors",
    "year",
    "venue",
    "source_platform",
    "arxiv_id",
    "s2_paper_id",
    "openalex_id",
    "dblp_key",
    "openreview_id",
    "oa_pdf_url",
    "publication_date",
    "venue_type",
    "citation_count",
    "citation_counts",
    "influential_citation_count",
    "is_open_access",
    "is_retracted",
    "fields_of_study",
    "tldr",
    "acceptance",
    "sources",
)

SORTS = ("relevance", "citations", "recent")

# Filter names as the orchestrator and the adapters' capability sets use them.
FILTER_YEAR = "year"
FILTER_MIN_CITATIONS = "min_citations"
FILTER_VENUES = "venues"
FILTER_FIELDS_OF_STUDY = "fields_of_study"
FILTER_OPEN_ACCESS = "open_access"
FILTER_PUBLICATION_TYPES = "publication_types"

# Normalised publication types. Each adapter maps its provider's vocabulary
# onto these, so a filter means the same thing on every platform.
PUBLICATION_TYPES = ("conference", "journal", "preprint", "review")


@dataclass(frozen=True)
class SearchFilters:
    """Constraints pushed down to each platform that can apply them.

    A platform that cannot apply one leaves it to the orchestrator, which
    filters the merged results locally — and says so, because "the platform
    filtered" and "we filtered what the platform happened to return" are not
    the same coverage.
    """

    year_from: int = 0
    year_to: int = 0
    min_citations: int = 0
    venues: tuple[str, ...] = ()
    fields_of_study: tuple[str, ...] = ()
    open_access_only: bool = False
    publication_types: tuple[str, ...] = ()
    sort: str = "relevance"

    def active(self) -> frozenset[str]:
        names: set[str] = set()
        if self.year_from or self.year_to:
            names.add(FILTER_YEAR)
        if self.min_citations > 0:
            names.add(FILTER_MIN_CITATIONS)
        if self.venues:
            names.add(FILTER_VENUES)
        if self.fields_of_study:
            names.add(FILTER_FIELDS_OF_STUDY)
        if self.open_access_only:
            names.add(FILTER_OPEN_ACCESS)
        if self.publication_types:
            names.add(FILTER_PUBLICATION_TYPES)
        return frozenset(names)

    def year_range(self, sep: str = "-") -> str:
        """`2020-2024`, `2020-` or `-2024` — the form S2 and OpenAlex take."""
        if not (self.year_from or self.year_to):
            return ""
        lo = str(self.year_from) if self.year_from else ""
        hi = str(self.year_to) if self.year_to else ""
        if lo and hi and lo == hi:
            return lo
        return f"{lo}{sep}{hi}"


@dataclass
class Paper:
    title: str
    abstract: str
    url: str
    doi: str
    authors: str
    source_platform: str
    year: int = 0
    venue: str = ""

    # Identifiers. These are what let a search result be downloaded or
    # resolved later without another lookup: an arXiv id is one request, an OA
    # PDF link is one more.
    arxiv_id: str = ""
    s2_paper_id: str = ""
    openalex_id: str = ""
    dblp_key: str = ""
    openreview_id: str = ""
    oa_pdf_url: str = ""

    publication_date: str = ""
    venue_type: str = ""  # one of PUBLICATION_TYPES, or "" when unknown
    publication_types: list[str] = field(default_factory=list)
    venue_aliases: list[str] = field(default_factory=list)
    fields_of_study: list[str] = field(default_factory=list)

    # Citation counts per source. Sources disagree — OpenAlex often splits a
    # preprint and its published version into two works — so they are kept
    # apart rather than averaged, and `citation_count` is the largest.
    citation_counts: dict[str, int] = field(default_factory=dict)
    influential_citation_count: int | None = None
    is_open_access: bool | None = None
    is_retracted: bool | None = None
    tldr: str = ""
    # OpenReview's venue line: "ICLR 2025 Poster", "Submitted to ICLR 2025"…
    acceptance: str = ""
    sources: list[str] = field(default_factory=list)

    # Internal fields (not part of the public output schema)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def citation_count(self) -> int | None:
        if not self.citation_counts:
            return None
        return max(self.citation_counts.values())

    def to_dict(self, *, fields: Iterable[str] | None = None) -> dict[str, Any]:
        data: dict[str, Any] = {
            "title": self.title,
            "abstract": self.abstract,
            "url": self.url,
            "doi": self.doi,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "source_platform": self.source_platform,
            "arxiv_id": self.arxiv_id,
            "s2_paper_id": self.s2_paper_id,
            "openalex_id": self.openalex_id,
            "dblp_key": self.dblp_key,
            "openreview_id": self.openreview_id,
            "oa_pdf_url": self.oa_pdf_url,
            "publication_date": self.publication_date,
            "venue_type": self.venue_type,
            "citation_count": self.citation_count,
            "citation_counts": dict(self.citation_counts),
            "influential_citation_count": self.influential_citation_count,
            "is_open_access": self.is_open_access,
            "is_retracted": self.is_retracted,
            "fields_of_study": list(self.fields_of_study),
            "tldr": self.tldr,
            "acceptance": self.acceptance,
            "sources": list(self.sources or [self.source_platform]),
        }
        if fields is None:
            return data

        out: dict[str, Any] = {}
        for key in fields:
            if key in data:
                out[key] = data[key]
        return out


@dataclass
class PlatformStatus:
    """What one platform did for one search, reported to the caller.

    `status` is one of: ok, empty, failed, auth_failed, rate_limited,
    quota_exhausted, skipped_no_key, unsupported. Only `ok` and `empty` mean
    the platform was actually searched.
    """

    platform: str
    status: str
    count: int = 0
    detail: str = ""
    remote_filters: list[str] = field(default_factory=list)
    local_filters: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def searched(self) -> bool:
        return self.status in {"ok", "empty"}

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"platform": self.platform, "status": self.status, "count": self.count}
        if self.detail:
            out["detail"] = self.detail
        if self.local_filters:
            out["filtered_locally"] = list(self.local_filters)
        return out


@dataclass
class SearchOutcome:
    papers: list[Paper]
    platforms: list[PlatformStatus]
    rerank_used: bool = False
    # Candidates dropped by a local filter, by filter name. A candidate whose
    # value is unknown (no citation count, no venue) cannot pass a filter on
    # that value and is counted under `<filter>_unknown`.
    excluded: dict[str, int] = field(default_factory=dict)
    candidates: int = 0
