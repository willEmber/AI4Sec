"""General web search and page reading over Tavily, Exa and Firecrawl.

The web counterpart of `paper_search`: each provider is an adapter that
normalises into `WebResult` / `WebPage`, every provider reports whether it
actually searched, and keys come from rotating pools so that one refused or
spent key does not take a provider down. See DEV_DOC/P7_通用网页检索.md.
"""

from app.services.web_search.chunking import PageChunk, chunks_from_offset, rank_chunks, split_markdown
from app.services.web_search.fetch import fetch_page
from app.services.web_search.keypool import KeyPool, pool_for, reset_pools
from app.services.web_search.models import (
    DEPTHS,
    TOPICS,
    FetchOutcome,
    ProviderError,
    WebPage,
    WebResult,
    WebSearchOutcome,
    WebSearchRequest,
)
from app.services.web_search.search import search_web
from app.services.web_search.urls import (
    domain_of,
    fetchable_reason,
    normalize_url,
    scholarly_identifiers,
)

__all__ = [
    "DEPTHS",
    "TOPICS",
    "FetchOutcome",
    "KeyPool",
    "PageChunk",
    "ProviderError",
    "WebPage",
    "WebResult",
    "WebSearchOutcome",
    "WebSearchRequest",
    "chunks_from_offset",
    "domain_of",
    "fetch_page",
    "fetchable_reason",
    "normalize_url",
    "pool_for",
    "rank_chunks",
    "reset_pools",
    "scholarly_identifiers",
    "search_web",
    "split_markdown",
]
