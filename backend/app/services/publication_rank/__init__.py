"""Embedded publication_rank package.

Provides journal/conference SCI/CCF ranking lookup with a unified cascade:
SQLite cache → EasyScholar API → LLM web-search fallback.

Previously lived at the project-root ``PublicationRank`` package; now bundled
directly into the backend to remove the cross-package ``sys.path`` hack.
"""

from .llm_rank import LLMRankClient, UnifiedRankClient, query_publication_rank, query_publication_ranks
from .publication_rank import (
    EasyScholarClient,
    PublicationRankResult,
    RateLimiter,
)
from .rank_cache import RankCache

__all__ = [
    "EasyScholarClient",
    "LLMRankClient",
    "PublicationRankResult",
    "RankCache",
    "RateLimiter",
    "UnifiedRankClient",
    "query_publication_rank",
    "query_publication_ranks",
]
