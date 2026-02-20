from __future__ import annotations

from typing import Callable

from .arxiv import search_arxiv
from .crossref import search_crossref
from .ieeexplore import search_ieeexplore
from .openalex import search_openalex
from .semanticscholar import search_semanticscholar


def _norm(name: str) -> str:
    return "".join(ch for ch in (name or "").casefold() if ch.isalnum())


SEARCHERS: dict[str, Callable] = {
    _norm("OpenAlex"): search_openalex,
    _norm("arXiv"): search_arxiv,
    _norm("Crossref"): search_crossref,
    _norm("CrossRef"): search_crossref,
    _norm("SemanticScholar"): search_semanticscholar,
    _norm("Semantic Scholar"): search_semanticscholar,
    _norm("IEEE Xplore"): search_ieeexplore,
    _norm("IEEEXplore"): search_ieeexplore,
    _norm("IEEE"): search_ieeexplore,
}


def resolve_platform(name: str) -> Callable | None:
    return SEARCHERS.get(_norm(name))
