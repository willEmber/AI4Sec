from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger("scholar.citation_graph")

# Ensure paper_search utils are importable
_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from paper_search.paper_search.utils import (
    jaccard_similarity,
    normalize_whitespace,
    openalex_abstract_from_inverted_index,
)

# ---------------------------------------------------------------------------
# Shared data class
# ---------------------------------------------------------------------------

@dataclass
class PaperMetadata:
    title: str = ""
    doi: str = ""
    arxiv_id: str = ""
    openalex_id: str = ""
    s2_paper_id: str = ""
    year: int = 0
    venue: str = ""
    authors: str = ""
    abstract_text: str = ""
    cited_by_count: int = 0


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

_OPENALEX_SEM = asyncio.Semaphore(10)
_S2_SEM = asyncio.Semaphore(1)

_RETRY_STATUSES = {429, 500, 502, 503, 504}

# Time-based rate limiter for Semantic Scholar API (max ~1 req/1.1s)
_s2_last_request: float = 0.0
_s2_throttle_lock = asyncio.Lock()


async def _s2_throttle() -> None:
    """Enforce minimum 1.1s between S2 API requests to avoid 429s."""
    global _s2_last_request
    async with _s2_throttle_lock:
        now = asyncio.get_event_loop().time()
        wait = 1.1 - (now - _s2_last_request)
        if wait > 0:
            await asyncio.sleep(wait)
        _s2_last_request = asyncio.get_event_loop().time()


async def _get_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    semaphore: asyncio.Semaphore | None = None,
    max_retries: int = 3,
) -> dict[str, Any] | None:
    """GET request with retry/backoff. Returns None on failure."""
    sem = semaphore or asyncio.Semaphore(999)
    for attempt in range(1, max_retries + 1):
        try:
            async with sem:
                resp = await client.get(url, params=params, headers=headers)
            if resp.status_code == 404:
                return None
            if resp.status_code in _RETRY_STATUSES:
                delay = min(2 ** attempt, 10)
                logger.debug(f"HTTP {resp.status_code} from {url}, retry {attempt}/{max_retries} in {delay}s")
                await asyncio.sleep(delay)
                continue
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, httpx.ReadTimeout, Exception) as e:
            if attempt >= max_retries:
                logger.debug(f"Failed GET {url}: {e}")
                return None
            await asyncio.sleep(min(2 ** attempt, 10))
    return None


async def _post_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    json_body: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    semaphore: asyncio.Semaphore | None = None,
    max_retries: int = 3,
) -> dict[str, Any] | None:
    """POST request with retry/backoff. Returns None on failure."""
    sem = semaphore or asyncio.Semaphore(999)
    for attempt in range(1, max_retries + 1):
        try:
            async with sem:
                resp = await client.post(url, json=json_body, params=params, headers=headers)
            if resp.status_code in _RETRY_STATUSES:
                delay = min(2 ** attempt, 10)
                await asyncio.sleep(delay)
                continue
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, httpx.ReadTimeout, Exception) as e:
            if attempt >= max_retries:
                logger.debug(f"Failed POST {url}: {e}")
                return None
            await asyncio.sleep(min(2 ** attempt, 10))
    return None


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------

_OA_BASE = "https://api.openalex.org"
_OA_SELECT = "id,title,publication_year,primary_location,authorships,cited_by_count,abstract_inverted_index,ids,referenced_works,related_works"


def _oa_mailto_param() -> dict[str, str]:
    email = os.getenv("PAPERSEARCH_CONTACT_EMAIL", "").split(",")[0].strip()
    if email:
        return {"mailto": email}
    return {}


def _oa_extract_doi(ids: dict[str, Any] | None) -> str:
    if not ids:
        return ""
    doi = ids.get("doi", "") or ""
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]
    return doi.strip().lower()


def _oa_extract_arxiv(ids: dict[str, Any] | None) -> str:
    if not ids:
        return ""
    # OpenAlex stores arXiv as "https://arxiv.org/abs/XXXX.XXXXX"
    arxiv = ids.get("openalex", "") or ""  # not here
    # Check the ids dict for arxiv
    for key in ("arxiv", "pmid", "pmcid"):
        pass
    # Actually OpenAlex puts arXiv IDs in ids.openalex or we extract from DOI
    return ""


def _oa_extract_venue(work: dict[str, Any]) -> str:
    loc = work.get("primary_location") or {}
    source = loc.get("source") or {}
    return normalize_whitespace(source.get("display_name", ""))


def _oa_extract_authors(work: dict[str, Any], limit: int = 5) -> str:
    authorships = work.get("authorships") or []
    names = []
    for a in authorships[:limit]:
        author = a.get("author") or {}
        name = author.get("display_name", "")
        if name:
            names.append(name)
    return ", ".join(names)


def _oa_work_to_metadata(work: dict[str, Any]) -> PaperMetadata:
    ids = work.get("ids") or {}
    doi = _oa_extract_doi(ids)
    openalex_id = (work.get("id") or "").replace("https://openalex.org/", "")
    abstract = openalex_abstract_from_inverted_index(work.get("abstract_inverted_index"))

    return PaperMetadata(
        title=normalize_whitespace(work.get("title", "") or ""),
        doi=doi,
        openalex_id=openalex_id,
        year=work.get("publication_year") or 0,
        venue=_oa_extract_venue(work),
        authors=_oa_extract_authors(work),
        abstract_text=abstract,
        cited_by_count=work.get("cited_by_count") or 0,
    )


def _oa_works_to_metadata(items: list[dict[str, Any]]) -> list[PaperMetadata]:
    results = []
    for item in items:
        meta = _oa_work_to_metadata(item)
        if meta.title:
            results.append(meta)
    return results


async def openalex_resolve_id(
    client: httpx.AsyncClient,
    doi: str = "",
    title: str = "",
    arxiv_id: str = "",
) -> str | None:
    """Resolve to an OpenAlex work ID (e.g., 'W1234567890')."""
    params = _oa_mailto_param()

    # Try DOI first
    if doi:
        data = await _get_json(
            client,
            f"{_OA_BASE}/works/doi:{doi}",
            params=params,
            semaphore=_OPENALEX_SEM,
        )
        if data and data.get("id"):
            return data["id"].replace("https://openalex.org/", "")

    # Try title search
    if title:
        params_search = {**params, "search": title, "per_page": "1", "select": "id,title"}
        data = await _get_json(
            client,
            f"{_OA_BASE}/works",
            params=params_search,
            semaphore=_OPENALEX_SEM,
        )
        if data:
            results = data.get("results") or []
            if results:
                oa_title = normalize_whitespace(results[0].get("title", ""))
                if jaccard_similarity(title, oa_title) >= 0.6:
                    return results[0]["id"].replace("https://openalex.org/", "")

    return None


async def openalex_get_work(
    client: httpx.AsyncClient,
    openalex_id: str,
) -> dict[str, Any] | None:
    """Fetch a full work record."""
    params = {**_oa_mailto_param(), "select": _OA_SELECT}
    return await _get_json(
        client,
        f"{_OA_BASE}/works/{openalex_id}",
        params=params,
        semaphore=_OPENALEX_SEM,
    )


async def openalex_get_referenced_works(
    client: httpx.AsyncClient,
    openalex_id: str,
) -> list[PaperMetadata]:
    """Get works referenced by the given paper."""
    work = await openalex_get_work(client, openalex_id)
    if not work:
        return []

    ref_ids = work.get("referenced_works") or []
    if not ref_ids:
        return []

    # Clean IDs
    clean_ids = [rid.replace("https://openalex.org/", "") for rid in ref_ids]
    return await _oa_batch_fetch(client, clean_ids)


async def openalex_get_cited_by(
    client: httpx.AsyncClient,
    openalex_id: str,
    limit: int = 80,
) -> list[PaperMetadata]:
    """Get works that cite the given paper, sorted by citation count."""
    params = {
        **_oa_mailto_param(),
        "filter": f"cites:{openalex_id}",
        "per_page": str(min(limit, 200)),
        "sort": "cited_by_count:desc",
        "select": _OA_SELECT,
    }
    data = await _get_json(
        client,
        f"{_OA_BASE}/works",
        params=params,
        semaphore=_OPENALEX_SEM,
    )
    if not data:
        return []
    return _oa_works_to_metadata(data.get("results") or [])


async def openalex_get_related_works(
    client: httpx.AsyncClient,
    openalex_id: str,
) -> list[PaperMetadata]:
    """Get related works from the work record."""
    work = await openalex_get_work(client, openalex_id)
    if not work:
        return []

    related_ids = work.get("related_works") or []
    if not related_ids:
        return []

    clean_ids = [rid.replace("https://openalex.org/", "") for rid in related_ids[:50]]
    return await _oa_batch_fetch(client, clean_ids)


async def _oa_batch_fetch(
    client: httpx.AsyncClient,
    openalex_ids: list[str],
    batch_size: int = 50,
) -> list[PaperMetadata]:
    """Batch fetch OpenAlex works by ID using filter pipe."""
    all_results: list[PaperMetadata] = []

    for i in range(0, len(openalex_ids), batch_size):
        batch = openalex_ids[i : i + batch_size]
        filter_str = "|".join(batch)
        params = {
            **_oa_mailto_param(),
            "filter": f"openalex:{filter_str}",
            "per_page": str(len(batch)),
            "select": _OA_SELECT,
        }
        data = await _get_json(
            client,
            f"{_OA_BASE}/works",
            params=params,
            semaphore=_OPENALEX_SEM,
        )
        if data:
            all_results.extend(_oa_works_to_metadata(data.get("results") or []))

    return all_results


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------

_S2_BASE = "https://api.semanticscholar.org"
_S2_FIELDS = "title,abstract,year,venue,authors,externalIds,citationCount"


def _s2_headers() -> dict[str, str]:
    key = os.getenv("PAPERSEARCH_SEMANTICSCHOLAR_API_KEY", "").strip()
    if key:
        return {"x-api-key": key}
    return {}


def _s2_extract_ids(external_ids: dict[str, Any] | None) -> tuple[str, str]:
    """Extract DOI and arXiv ID from S2 externalIds."""
    if not external_ids:
        return "", ""
    doi = (external_ids.get("DOI") or "").strip().lower()
    arxiv = (external_ids.get("ArXiv") or "").strip()
    return doi, arxiv


def _s2_paper_to_metadata(paper: dict[str, Any]) -> PaperMetadata:
    ext_ids = paper.get("externalIds") or {}
    doi, arxiv = _s2_extract_ids(ext_ids)
    s2_id = paper.get("paperId") or ""

    authors_list = paper.get("authors") or []
    authors_str = ", ".join(
        a.get("name", "") for a in authors_list[:5] if a.get("name")
    )

    return PaperMetadata(
        title=normalize_whitespace(paper.get("title", "") or ""),
        doi=doi,
        arxiv_id=arxiv,
        s2_paper_id=s2_id,
        year=paper.get("year") or 0,
        venue=normalize_whitespace(paper.get("venue", "") or ""),
        authors=authors_str,
        abstract_text=normalize_whitespace(paper.get("abstract", "") or ""),
        cited_by_count=paper.get("citationCount") or 0,
    )


def _s2_papers_to_metadata(items: list[dict[str, Any]], key: str = "") -> list[PaperMetadata]:
    results = []
    for item in items:
        paper = item.get(key, item) if key else item
        if not paper or not isinstance(paper, dict):
            continue
        meta = _s2_paper_to_metadata(paper)
        if meta.title:
            results.append(meta)
    return results


async def s2_resolve_id(
    client: httpx.AsyncClient,
    doi: str = "",
    arxiv_id: str = "",
    title: str = "",
) -> str | None:
    """Resolve to a Semantic Scholar paper ID."""
    headers = _s2_headers()

    # Try DOI
    if doi:
        await _s2_throttle()
        data = await _get_json(
            client,
            f"{_S2_BASE}/graph/v1/paper/DOI:{doi}",
            params={"fields": "paperId"},
            headers=headers,
            semaphore=_S2_SEM,
        )
        if data and data.get("paperId"):
            return data["paperId"]

    # Try arXiv
    if arxiv_id:
        await _s2_throttle()
        data = await _get_json(
            client,
            f"{_S2_BASE}/graph/v1/paper/ARXIV:{arxiv_id}",
            params={"fields": "paperId"},
            headers=headers,
            semaphore=_S2_SEM,
        )
        if data and data.get("paperId"):
            return data["paperId"]

    # Try title search
    if title:
        await _s2_throttle()
        data = await _get_json(
            client,
            f"{_S2_BASE}/graph/v1/paper/search",
            params={"query": title[:200], "limit": "1", "fields": "paperId,title"},
            headers=headers,
            semaphore=_S2_SEM,
        )
        if data:
            papers = data.get("data") or []
            if papers:
                s2_title = normalize_whitespace(papers[0].get("title", ""))
                if jaccard_similarity(title, s2_title) >= 0.6:
                    return papers[0].get("paperId")

    return None


async def s2_get_references(
    client: httpx.AsyncClient,
    s2_id: str,
    limit: int = 100,
) -> list[PaperMetadata]:
    """Get references of a paper."""
    await _s2_throttle()
    headers = _s2_headers()
    data = await _get_json(
        client,
        f"{_S2_BASE}/graph/v1/paper/{s2_id}/references",
        params={"fields": _S2_FIELDS, "limit": str(min(limit, 1000))},
        headers=headers,
        semaphore=_S2_SEM,
    )
    if not data:
        return []
    return _s2_papers_to_metadata(data.get("data") or [], key="citedPaper")


async def s2_get_citations(
    client: httpx.AsyncClient,
    s2_id: str,
    limit: int = 80,
) -> list[PaperMetadata]:
    """Get papers that cite the given paper."""
    await _s2_throttle()
    headers = _s2_headers()
    data = await _get_json(
        client,
        f"{_S2_BASE}/graph/v1/paper/{s2_id}/citations",
        params={"fields": _S2_FIELDS, "limit": str(min(limit, 1000))},
        headers=headers,
        semaphore=_S2_SEM,
    )
    if not data:
        return []
    return _s2_papers_to_metadata(data.get("data") or [], key="citingPaper")


async def s2_get_recommendations(
    client: httpx.AsyncClient,
    s2_id: str,
    limit: int = 50,
) -> list[PaperMetadata]:
    """Get recommended papers based on a seed paper."""
    await _s2_throttle()
    headers = _s2_headers()
    data = await _post_json(
        client,
        f"{_S2_BASE}/recommendations/v1/papers/",
        json_body={"positivePaperIds": [s2_id]},
        params={"fields": _S2_FIELDS, "limit": str(min(limit, 500))},
        headers=headers,
        semaphore=_S2_SEM,
    )
    if not data:
        return []
    papers = data.get("recommendedPapers") or []
    return [_s2_paper_to_metadata(p) for p in papers if p and p.get("title")]


# ---------------------------------------------------------------------------
# Crossref DOI resolution
# ---------------------------------------------------------------------------

_CR_BASE = "https://api.crossref.org"


async def crossref_resolve_doi(
    client: httpx.AsyncClient,
    title: str,
    authors: str = "",
) -> str | None:
    """Try to find a DOI via Crossref bibliographic search. Returns DOI or None."""
    if not title or len(title.strip()) < 10:
        return None

    query = title
    params: dict[str, str] = {
        "query.bibliographic": query,
        "rows": "3",
        "select": "DOI,title",
    }
    email = os.getenv("PAPERSEARCH_CONTACT_EMAIL", "").split(",")[0].strip()
    if email:
        params["mailto"] = email

    data = await _get_json(
        client,
        f"{_CR_BASE}/works",
        params=params,
    )
    if not data:
        return None

    items = (data.get("message") or {}).get("items") or []
    for item in items:
        cr_titles = item.get("title") or []
        cr_title = normalize_whitespace(cr_titles[0]) if cr_titles else ""
        if jaccard_similarity(title, cr_title) >= 0.8:
            return (item.get("DOI") or "").strip().lower()

    return None
