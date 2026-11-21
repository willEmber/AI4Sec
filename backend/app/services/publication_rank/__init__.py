"""Embedded publication_rank package (P17).

Four-tier venue ranking & credibility resolution:
  L1: In-process authoritative catalog (`venue_catalog.json.gz` — CCF 2026,
      ICORE/CORE, TH-CPL, CAAI, CCF-CN T1–T3, JCR 2025, CAS 2025, CAS Warning)
  L2: PostgreSQL `publication_rank_cache` (with SQLite fallback & priority guard)
  L3: Async EasyScholar (ISSN-first, singleflight) + OpenAlex `/sources` metrics
  L4: Tavily web-search + Utility LLM multi-system extraction (`unverified`)
"""

from .catalog import VenueCatalog, get_venue_catalog, load_catalog_from_path, reset_venue_catalog_cache
from .llm_rank import (
    LLMRankClient,
    UnifiedRankClient,
    query_publication_rank,
    query_publication_ranks,
    resolve_venue_profile,
    resolve_venue_profiles_batch,
)
from .models import RankingEntry, VenueLookupInput, VenueProfile
from .normalizer import (
    classify_subtrack,
    clean_venue_text,
    extract_dblp_prefix,
    extract_issns,
    is_preprint_venue,
    is_umbrella_series,
    normalize_issn,
    normalize_name_key,
    normalize_venue_display,
    parse_openreview_status,
    strip_venue_decorations,
)
from .publication_rank import (
    EasyScholarClient,
    PublicationRankResult,
    RateLimiter,
)
from .rank_cache import RankCache, canonical_cache_key
from .tavily_search import TavilySearchClient

__all__ = [
    "EasyScholarClient",
    "LLMRankClient",
    "PublicationRankResult",
    "RankCache",
    "RankingEntry",
    "RateLimiter",
    "TavilySearchClient",
    "UnifiedRankClient",
    "VenueCatalog",
    "VenueLookupInput",
    "VenueProfile",
    "canonical_cache_key",
    "classify_subtrack",
    "clean_venue_text",
    "extract_dblp_prefix",
    "extract_issns",
    "get_venue_catalog",
    "is_preprint_venue",
    "is_umbrella_series",
    "load_catalog_from_path",
    "normalize_issn",
    "normalize_name_key",
    "normalize_venue_display",
    "parse_openreview_status",
    "query_publication_rank",
    "query_publication_ranks",
    "reset_venue_catalog_cache",
    "resolve_venue_profile",
    "resolve_venue_profiles_batch",
    "strip_venue_decorations",
]
