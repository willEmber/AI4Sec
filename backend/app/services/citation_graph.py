from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from app.services.paper_search import (
    Settings as PSSettings,
    jaccard_similarity,
    load_env_file,
    normalize_whitespace,
    openalex_abstract_from_inverted_index,
)

logger = logging.getLogger("scholar.citation_graph")

# Ensure .env is loaded so PAPERSEARCH_* vars are available.
load_env_file(str(Path(__file__).resolve().parents[3] / ".env"))

# Shared paper_search settings for email round-robin
_ps_settings = PSSettings.from_env()

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
    # OpenAlex work IDs this paper references (bibliographic coupling input)
    referenced_works: list[str] = field(default_factory=list)
    # S2 citation-edge signals (only set on citations/references responses)
    influential: bool = False
    intents: list[str] = field(default_factory=list)


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
_OA_SELECT = "id,title,publication_year,primary_location,locations,authorships,cited_by_count,abstract_inverted_index,ids,referenced_works,related_works"

# Preprint servers: never the venue we want when a published version exists
_OA_PREPRINT_RE = re.compile(
    r"arxiv|preprint|biorxiv|medrxiv|ssrn|research square|repec", re.IGNORECASE
)


def _oa_mailto_param() -> dict[str, str]:
    email = _ps_settings.pick_openalex_mailto()
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
    """Venue of the published version, not a preprint server or repository.

    For arXiv-first papers OpenAlex's primary_location is arXiv, which used
    to make every top-conference paper score as a preprint. But "any
    non-preprint location" is not enough either: OpenAlex lists institutional
    repositories (HAL, LA Referencia, Apollo, UvA-DARE, …) as locations, and
    those beat the real journal/conference by list order. OpenAlex marks all
    of them — arXiv included — with source.type == "repository", so filter by
    type first and keep the name regex as a backstop for untyped sources.
    """
    def _loc_source(loc: dict[str, Any] | None) -> tuple[str, str]:
        source = (loc or {}).get("source") or {}
        name = normalize_whitespace(source.get("display_name", "") or "")
        stype = (source.get("type") or "").strip().lower()
        return name, stype

    def _is_published_outlet(name: str, stype: str) -> bool:
        if not name or stype == "repository":
            return False
        return not _OA_PREPRINT_RE.search(name)

    primary_name, primary_type = _loc_source(work.get("primary_location"))
    if _is_published_outlet(primary_name, primary_type):
        return primary_name

    for loc in work.get("locations") or []:
        name, stype = _loc_source(loc)
        if _is_published_outlet(name, stype):
            return name

    return primary_name


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
    referenced = [
        rid.replace("https://openalex.org/", "")
        for rid in (work.get("referenced_works") or [])
    ]

    return PaperMetadata(
        title=normalize_whitespace(work.get("title", "") or ""),
        doi=doi,
        openalex_id=openalex_id,
        year=work.get("publication_year") or 0,
        venue=_oa_extract_venue(work),
        authors=_oa_extract_authors(work),
        abstract_text=abstract,
        cited_by_count=work.get("cited_by_count") or 0,
        referenced_works=referenced,
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


async def openalex_get_recent_cited_by(
    client: httpx.AsyncClient,
    openalex_id: str,
    since_year: int,
    limit: int = 40,
) -> list[PaperMetadata]:
    """Top-cited *recent* citers — the dedicated frontier channel.

    The all-time citers channel is dominated by decade-old landmarks, so for
    an older center paper the newest high-impact follow-ups never crack its
    top-N and the frontier quota ends up fed by whatever low-impact recent
    citers drifted in through other channels. Restricting to works published
    since `since_year` (still sorted by citations) fixes the supply side.
    """
    params = {
        **_oa_mailto_param(),
        "filter": f"cites:{openalex_id},from_publication_date:{since_year}-01-01",
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


async def openalex_batch_fetch_by_doi(
    client: httpx.AsyncClient,
    dois: list[str],
    batch_size: int = 40,
) -> dict[str, PaperMetadata]:
    """Batch fetch OpenAlex works by DOI. Returns {lowercase doi: metadata}."""
    result: dict[str, PaperMetadata] = {}
    clean = [d.strip().lower() for d in dois if d and d.strip()]

    for i in range(0, len(clean), batch_size):
        batch = clean[i : i + batch_size]
        filter_str = "|".join(batch)
        params = {
            **_oa_mailto_param(),
            "filter": f"doi:{filter_str}",
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
            for meta in _oa_works_to_metadata(data.get("results") or []):
                if meta.doi:
                    result[meta.doi] = meta

    return result


async def openalex_search_one(
    client: httpx.AsyncClient,
    title: str,
    min_similarity: float = 0.6,
) -> PaperMetadata | None:
    """Resolve a single paper by title search; returns full metadata or None."""
    if not title or len(title.strip()) < 10:
        return None
    params = {
        **_oa_mailto_param(),
        "search": title[:300],
        "per_page": "1",
        "select": _OA_SELECT,
    }
    data = await _get_json(
        client,
        f"{_OA_BASE}/works",
        params=params,
        semaphore=_OPENALEX_SEM,
    )
    if not data:
        return None
    results = data.get("results") or []
    if not results:
        return None
    meta = _oa_work_to_metadata(results[0])
    if jaccard_similarity(title, meta.title) < min_similarity:
        return None
    return meta


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
            # Edge-level citation signals live on the wrapper item, not the paper
            if key:
                meta.influential = bool(item.get("isInfluential"))
                meta.intents = [i for i in (item.get("intents") or []) if isinstance(i, str)]
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
    """Get papers that cite the given paper, highest-cited first.

    The S2 citations endpoint returns entries in no useful order (roughly
    recency), so a naive first-N slice hands back mostly low-impact recent
    citers. Over-fetch a larger page and keep the top `limit` by citation
    count instead.
    """
    await _s2_throttle()
    headers = _s2_headers()
    fetch = min(max(limit * 3, 300), 1000)
    data = await _get_json(
        client,
        f"{_S2_BASE}/graph/v1/paper/{s2_id}/citations",
        params={
            "fields": _S2_FIELDS + ",intents,isInfluential",
            "limit": str(fetch),
        },
        headers=headers,
        semaphore=_S2_SEM,
    )
    if not data:
        return []
    metas = _s2_papers_to_metadata(data.get("data") or [], key="citingPaper")
    metas.sort(key=lambda m: m.cited_by_count, reverse=True)
    return metas[:limit]


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
    email = _ps_settings.pick_crossref_mailto()
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
