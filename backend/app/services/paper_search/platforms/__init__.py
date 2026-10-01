from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import arxiv, crossref, dblp, ieeexplore, openalex, openreview, semanticscholar
from .arxiv import search_arxiv
from .crossref import search_crossref
from .dblp import search_dblp
from .ieeexplore import search_ieeexplore
from .openalex import search_openalex
from .openreview import search_openreview
from .semanticscholar import search_semanticscholar


def _norm(name: str) -> str:
    return "".join(ch for ch in (name or "").casefold() if ch.isalnum())


@dataclass(frozen=True)
class PlatformSpec:
    """One searchable platform and what it can do server-side.

    `filters` and `sorts` are what the platform applies itself; anything else
    the orchestrator applies to the merged results and reports as local.
    """

    name: str
    search: Callable
    filters: frozenset[str]
    sorts: frozenset[str]


_SPECS = (
    PlatformSpec("SemanticScholar", search_semanticscholar, semanticscholar.SUPPORTED_FILTERS, semanticscholar.SUPPORTED_SORTS),
    PlatformSpec("OpenAlex", search_openalex, openalex.SUPPORTED_FILTERS, openalex.SUPPORTED_SORTS),
    PlatformSpec("arXiv", search_arxiv, arxiv.SUPPORTED_FILTERS, arxiv.SUPPORTED_SORTS),
    PlatformSpec("Crossref", search_crossref, crossref.SUPPORTED_FILTERS, crossref.SUPPORTED_SORTS),
    PlatformSpec("IEEE Xplore", search_ieeexplore, ieeexplore.SUPPORTED_FILTERS, ieeexplore.SUPPORTED_SORTS),
    PlatformSpec("DBLP", search_dblp, dblp.SUPPORTED_FILTERS, dblp.SUPPORTED_SORTS),
    PlatformSpec("OpenReview", search_openreview, openreview.SUPPORTED_FILTERS, openreview.SUPPORTED_SORTS),
)

_ALIASES = {
    "semanticscholar": "SemanticScholar",
    "s2": "SemanticScholar",
    "openalex": "OpenAlex",
    "arxiv": "arXiv",
    "crossref": "Crossref",
    "ieeexplore": "IEEE Xplore",
    "ieee": "IEEE Xplore",
    "dblp": "DBLP",
    "openreview": "OpenReview",
}

PLATFORMS: dict[str, PlatformSpec] = {spec.name: spec for spec in _SPECS}

# Kept for callers that only need the adapter.
SEARCHERS: dict[str, Callable] = {key: PLATFORMS[name].search for key, name in _ALIASES.items()}


def canonical_platform(name: str) -> str:
    """The registry name for any accepted spelling, or "" if unknown."""
    return _ALIASES.get(_norm(name), "")


def resolve_spec(name: str) -> PlatformSpec | None:
    canonical = canonical_platform(name)
    return PLATFORMS.get(canonical) if canonical else None


def resolve_platform(name: str) -> Callable | None:
    spec = resolve_spec(name)
    return spec.search if spec else None
