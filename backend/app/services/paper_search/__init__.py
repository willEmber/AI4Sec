"""Embedded paper_search package.

Provides multi-platform academic search aggregation with optional LLM rerank.
Previously lived at the project-root ``paper_search`` package; now bundled
directly into the backend to remove the cross-package ``sys.path`` hack.
"""

from .config import Settings, load_env_file
from .models import PUBLIC_FIELDS, Paper
from .search import search_papers
from .utils import (
    jaccard_similarity,
    normalize_doi,
    normalize_whitespace,
    openalex_abstract_from_inverted_index,
    title_fingerprint,
)

__all__ = [
    "PUBLIC_FIELDS",
    "Paper",
    "Settings",
    "jaccard_similarity",
    "load_env_file",
    "normalize_doi",
    "normalize_whitespace",
    "openalex_abstract_from_inverted_index",
    "search_papers",
    "title_fingerprint",
]
