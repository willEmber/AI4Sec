from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import math
import re
import time
from typing import Iterable

import requests

from .config import DEFAULT_USER_AGENT, Settings, load_env_file
from .debug import debug
from .http_client import HTTPClient, HTTPStatusError
from .llm import LLMConfig, embeddings, rerank
from .models import (
    FILTER_FIELDS_OF_STUDY,
    FILTER_MIN_CITATIONS,
    FILTER_OPEN_ACCESS,
    FILTER_PUBLICATION_TYPES,
    FILTER_VENUES,
    FILTER_YEAR,
    PUBLIC_FIELDS,
    SORTS,
    Paper,
    PlatformStatus,
    SearchFilters,
    SearchOutcome,
)
from .platforms import PlatformSpec, resolve_spec
from .platforms.crossref import guess_doi_from_crossref
from .platforms.openreview import OpenReviewLoginRequired
from .logger import logger
from .ratelimit import OPENALEX_BUDGET, MissingCredentialError, QuotaExhaustedError
from .utils import (
    normalize_doi,
    normalize_whitespace,
    title_fingerprint,
)

# A merged pool larger than this is handed to the rerank model for a first
# cut when the caller asked for "auto": below it, lexical order over what the
# platforms already ranked is as good and costs nothing.
RERANK_AUTO_THRESHOLD = 40

_PREPRINT_VENUE_RE = re.compile(r"^(arxiv|corr)\b|preprint", re.IGNORECASE)


def _norm_platform_name(name: str) -> str:
    return "".join(ch for ch in (name or "").casefold() if ch.isalnum())


def _looks_like_snippet(text: str) -> bool:
    t = normalize_whitespace(text)
    if not t:
        return False
    if "…" in t or "..." in t:
        return True
    return t.endswith("…") or t.endswith("...")


_DOI_IN_TEXT_RE = re.compile(r"10\.\d{4,9}/[^\s\"<>]+", re.IGNORECASE)


def _extract_doi_from_text(text: str) -> str:
    if not text:
        return ""
    m = _DOI_IN_TEXT_RE.search(text)
    if not m:
        return ""
    raw = (m.group(0) or "").strip()
    raw = raw.rstrip(".,);]}>'\"")
    return normalize_doi(raw)


def _merge_into(primary: Paper, incoming: Paper) -> None:
    """Fold a duplicate from another platform into the record already kept.

    Field by field rather than first-wins: arXiv knows the arXiv id, S2 the
    citation count, dblp the venue the community uses. Nothing already known is
    overwritten, except a preprint-server venue by a real one.
    """
    if incoming.abstract:
        if (
            not primary.abstract
            or (
                len(incoming.abstract) > len(primary.abstract)
                and _looks_like_snippet(primary.abstract)
            )
        ):
            primary.abstract = incoming.abstract
    for attr in (
        "url", "doi", "authors", "arxiv_id", "s2_paper_id", "openalex_id", "dblp_key",
        "openreview_id", "oa_pdf_url", "publication_date", "tldr", "acceptance",
    ):
        if not getattr(primary, attr) and getattr(incoming, attr):
            setattr(primary, attr, getattr(incoming, attr))
    if not primary.year and incoming.year:
        primary.year = incoming.year
    if incoming.venue and (not primary.venue or _PREPRINT_VENUE_RE.search(primary.venue)):
        if not _PREPRINT_VENUE_RE.search(incoming.venue) or not primary.venue:
            primary.venue = incoming.venue
            if incoming.venue_type:
                primary.venue_type = incoming.venue_type
    if not primary.venue_type or primary.venue_type == "preprint":
        if incoming.venue_type and incoming.venue_type != "preprint":
            primary.venue_type = incoming.venue_type
        elif not primary.venue_type:
            primary.venue_type = incoming.venue_type
    for source, count in incoming.citation_counts.items():
        primary.citation_counts.setdefault(source, count)
    if primary.influential_citation_count is None:
        primary.influential_citation_count = incoming.influential_citation_count
    # "Yes" from any source wins: one source knowing of an OA copy or a
    # retraction notice is enough, another not knowing proves nothing.
    if incoming.is_open_access is not None:
        primary.is_open_access = bool(primary.is_open_access) or incoming.is_open_access
    if incoming.is_retracted is not None:
        primary.is_retracted = bool(primary.is_retracted) or incoming.is_retracted
    for attr in ("fields_of_study", "publication_types", "venue_aliases"):
        merged = list(getattr(primary, attr))
        for value in getattr(incoming, attr):
            if value not in merged:
                merged.append(value)
        setattr(primary, attr, merged)
    if incoming.source_platform not in primary.sources:
        primary.sources.append(incoming.source_platform)


def _normalize_output_fields(fields: list[str] | None) -> list[str] | None:
    if fields is None:
        return None

    allowed_map = {k.casefold(): k for k in PUBLIC_FIELDS}
    picked: list[str] = []
    seen: set[str] = set()
    unknown: list[str] = []

    for raw in fields:
        key = (raw or "").strip()
        if not key:
            continue
        actual = allowed_map.get(key.casefold())
        if not actual:
            unknown.append(key)
            continue
        if actual in seen:
            continue
        picked.append(actual)
        seen.add(actual)

    if unknown:
        raise ValueError(f"unknown fields: {unknown}. allowed: {list(PUBLIC_FIELDS)}")
    if not picked:
        raise ValueError(f"fields is empty. allowed: {list(PUBLIC_FIELDS)}")
    return picked


def _dedupe_keep_first(papers: Iterable[Paper]) -> list[Paper]:
    by_doi: dict[str, Paper] = {}
    by_arxiv: dict[str, Paper] = {}
    by_title: dict[str, Paper] = {}
    by_compact: dict[str, Paper] = {}
    ordered: list[Paper] = []

    for paper in papers:
        paper.title = normalize_whitespace(paper.title)
        paper.abstract = normalize_whitespace(paper.abstract)
        paper.url = normalize_whitespace(paper.url)
        paper.doi = normalize_doi(paper.doi)
        paper.authors = normalize_whitespace(paper.authors)
        if not paper.sources:
            paper.sources = [paper.source_platform]

        doi_key = paper.doi
        arxiv_key = paper.arxiv_id.lower()
        title_key = title_fingerprint(paper.title)
        # "Fine-tuning" and "Finetuning" are the same title on two platforms.
        compact_key = re.sub(r"[^a-z0-9]+", "", paper.title.lower())

        existing = (
            (by_doi.get(doi_key) if doi_key else None)
            or (by_arxiv.get(arxiv_key) if arxiv_key else None)
            or by_title.get(title_key)
            or (by_compact.get(compact_key) if len(compact_key) >= 20 else None)
        )
        if existing is not None:
            _merge_into(existing, paper)
            if existing.doi:
                by_doi.setdefault(existing.doi, existing)
            if existing.arxiv_id:
                by_arxiv.setdefault(existing.arxiv_id.lower(), existing)
            continue

        ordered.append(paper)
        by_title[title_key] = paper
        if len(compact_key) >= 20:
            by_compact[compact_key] = paper
        if doi_key:
            by_doi[doi_key] = paper
        if arxiv_key:
            by_arxiv[arxiv_key] = paper

    return ordered


def _classify_failure(platform: str, exc: BaseException) -> tuple[str, str]:
    """Map an adapter exception onto a status the caller can act on."""
    if isinstance(exc, QuotaExhaustedError):
        return "quota_exhausted", str(exc)
    if isinstance(exc, MissingCredentialError):
        return "skipped_no_key", str(exc)
    if isinstance(exc, OpenReviewLoginRequired):
        return "auth_failed", str(exc)
    if isinstance(exc, HTTPStatusError):
        if exc.status_code in (401, 403):
            return "auth_failed", f"HTTP {exc.status_code}: the API key was rejected or access is gated"
        if exc.status_code == 429:
            return "rate_limited", "HTTP 429: too many requests"
        if exc.status_code == 406 and platform == "arXiv":
            return "rate_limited", "HTTP 406: arXiv is throttling this host"
        return "failed", f"HTTP {exc.status_code}: {normalize_whitespace(exc.body)[:160]}"
    if isinstance(exc, (requests.Timeout, asyncio.TimeoutError)):
        return "failed", "timeout"
    return "failed", f"{type(exc).__name__}: {str(exc)[:200]}"


async def _run_searchers(
    client: HTTPClient,
    *,
    query: str,
    specs: list[PlatformSpec],
    settings: Settings,
    filters: SearchFilters | None = None,
    page: int = 1,
    per_platform: dict[str, int] | None = None,
) -> tuple[list[Paper], list[PlatformStatus]]:
    active = filters.active() if filters else frozenset()
    sort = filters.sort if filters else "relevance"

    async def _timed(spec: PlatformSpec) -> tuple[PlatformSpec, list[Paper] | BaseException, float, int]:
        limit = (per_platform or {}).get(spec.name) or settings.limit_for_platform(spec.name)
        t0 = time.perf_counter()
        try:
            out = await spec.search(
                client,
                query=query,
                limit=limit,
                settings=settings,
                filters=filters,
                offset=(max(int(page), 1) - 1) * limit,
            )
            return spec, out, time.perf_counter() - t0, limit
        except Exception as e:  # noqa: BLE001 — every failure becomes a status
            return spec, e, time.perf_counter() - t0, limit

    timed_results = await asyncio.gather(*(_timed(spec) for spec in specs))

    merged: list[Paper] = []
    statuses: list[PlatformStatus] = []
    for spec, result, elapsed_s, _limit in timed_results:
        status = PlatformStatus(
            platform=spec.name,
            status="ok",
            remote_filters=sorted(active & spec.filters),
            local_filters=sorted(active - spec.filters)
            + (["sort"] if sort not in spec.sorts else []),
            elapsed_s=round(elapsed_s, 3),
        )
        if logger is not None:
            logger.info(f"[paper_search] platform={spec.name} search_time_s={elapsed_s:.3f}")
        if isinstance(result, BaseException):
            status.status, status.detail = _classify_failure(spec.name, result)
            if logger is not None:
                logger.warning(f"[paper_search] platform={spec.name} search failed: {status.detail}")
            debug(f"platform={spec.name} search failed: {result}")
        else:
            status.count = len(result)
            status.status = "ok" if result else "empty"
            debug(f"platform={spec.name} results={len(result)}")
            merged.extend(result)
        statuses.append(status)
    return merged, statuses


def _venue_matches(paper: Paper, venues: tuple[str, ...]) -> bool:
    names = [paper.venue, *paper.venue_aliases, paper.acceptance if paper.venue else ""]
    haystack = [normalize_whitespace(n).casefold() for n in names if n]
    for wanted in venues:
        w = normalize_whitespace(wanted).casefold()
        if not w:
            continue
        for name in haystack:
            if name == w or re.search(rf"(^|[^a-z0-9]){re.escape(w)}([^a-z0-9]|$)", name):
                return True
    return False


def _apply_local_filters(
    papers: list[Paper], filters: SearchFilters | None
) -> tuple[list[Paper], dict[str, int]]:
    """Check every merged candidate against every active filter.

    Applied to all candidates, not only those from platforms that could not
    filter: a platform's own filter and ours agree on what they both know, so
    re-checking costs nothing and catches providers that filter loosely.

    Year is the exception that is *not* enforced on unknown values here — the
    caller separates undated candidates and reports them, because "no year on
    record" is a fact the reader should see, not a silent drop. For the other
    filters an unknown value cannot be shown to pass, so it is dropped and
    counted under `<filter>_unknown`.
    """
    if filters is None or not filters.active():
        return papers, {}
    excluded: dict[str, int] = {}

    def drop(reason: str) -> None:
        excluded[reason] = excluded.get(reason, 0) + 1

    kept: list[Paper] = []
    for p in papers:
        if p.year:
            if filters.year_from and p.year < filters.year_from:
                drop(FILTER_YEAR)
                continue
            if filters.year_to and p.year > filters.year_to:
                drop(FILTER_YEAR)
                continue
        if filters.min_citations > 0:
            count = p.citation_count
            if count is None:
                drop(f"{FILTER_MIN_CITATIONS}_unknown")
                continue
            if count < filters.min_citations:
                drop(FILTER_MIN_CITATIONS)
                continue
        if filters.venues and not _venue_matches(p, filters.venues):
            # A rejected OpenReview submission has a known outcome, not an unknown venue.
            drop(FILTER_VENUES if (p.venue or p.acceptance) else f"{FILTER_VENUES}_unknown")
            continue
        if filters.open_access_only and not (p.is_open_access or p.oa_pdf_url):
            drop(FILTER_OPEN_ACCESS)
            continue
        if filters.publication_types:
            kinds = {p.venue_type, *p.publication_types} - {""}
            if kinds and not (kinds & set(filters.publication_types)):
                drop(FILTER_PUBLICATION_TYPES)
                continue
        if filters.fields_of_study and p.fields_of_study:
            wanted = {f.casefold() for f in filters.fields_of_study}
            if not wanted & {f.casefold() for f in p.fields_of_study}:
                drop(FILTER_FIELDS_OF_STUDY)
                continue
        kept.append(p)
    return kept, excluded


def _sort_papers(papers: list[Paper], sort: str) -> list[Paper]:
    """Stable sort by the requested key; unknown values go last, in input order."""
    if sort == "citations":
        return sorted(
            papers, key=lambda p: (p.citation_count is None, -(p.citation_count or 0))
        )
    if sort == "recent":
        # ISO dates and bare years compare correctly as strings; "" sorts last.
        return sorted(
            papers,
            key=lambda p: p.publication_date or (f"{p.year:04d}" if p.year else ""),
            reverse=True,
        )
    return papers


def _simple_rank(query: str, papers: list[Paper], limit: int) -> list[Paper]:
    """Simple lexical ranking (no LLM).

    Score = 3 * (#query tokens matched in title) + 1 * (#query tokens matched in abstract).
    Ties keep the original order (stable).
    """
    q_tokens = set(re.findall(r"[a-z0-9]+", (query or "").lower()))
    if not q_tokens:
        return papers[:limit]

    def score(p: Paper) -> tuple[int, int]:
        title_tokens = set(re.findall(r"[a-z0-9]+", (p.title or "").lower()))
        abstract_tokens = set(re.findall(r"[a-z0-9]+", (p.abstract or "").lower()))
        title_hits = len(q_tokens & title_tokens)
        abstract_hits = len(q_tokens & abstract_tokens)
        return (3 * title_hits + abstract_hits, title_hits)

    indexed = list(enumerate(papers))
    indexed.sort(key=lambda x: (-score(x[1])[0], -score(x[1])[1], x[0]))
    return [p for _, p in indexed[:limit]]


def _paper_text_for_embedding(p: Paper, *, max_abstract_chars: int = 4000) -> str:
    title = normalize_whitespace(p.title)
    abstract = normalize_whitespace(p.abstract)[: max(int(max_abstract_chars), 0)]

    parts: list[str] = []
    if title:
        parts.append(f"Title: {title}")
    if abstract:
        parts.append(f"Abstract: {abstract}")
    return "\n".join(parts) if parts else "(empty paper)"


def _is_meaningful_abstract(text: str) -> bool:
    t = normalize_whitespace(text)
    if not t:
        return False
    low = t.casefold()
    if "no abstract available" in low:
        return False
    if "abstract not available" in low:
        return False
    if low in {"n/a", "na", "none"}:
        return False
    if "暂无摘要" in t:
        return False
    return True


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


async def _embedding_rank(
    client: HTTPClient,
    *,
    query: str,
    papers: list[Paper],
    limit: int,
    settings: Settings,
) -> tuple[list[Paper], bool]:
    if not papers:
        return [], False

    q = normalize_whitespace(query)
    k = min(max(int(limit), 1), len(papers))
    if not q:
        return papers[:k], False
    if not (settings.llm_base_url and settings.embed_model):
        return _simple_rank(q, papers, k), False

    cfg = LLMConfig(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        max_retries=settings.llm_max_retries,
        retry_base_delay=settings.llm_retry_base_delay,
        retry_max_delay=settings.llm_retry_max_delay,
    )

    titles = [normalize_whitespace(p.title) or "(untitled paper)" for p in papers]
    abstracts: list[str] = []
    abstract_pos: list[int | None] = []
    for p in papers:
        a = normalize_whitespace(p.abstract)
        if _is_meaningful_abstract(a):
            abstract_pos.append(len(abstracts))
            abstracts.append(a[:4000])
        else:
            abstract_pos.append(None)
    try:
        vecs = await embeddings(
            client,
            cfg=cfg,
            model=settings.embed_model,
            texts=[q] + titles + abstracts,
            batch_size=32,
        )
        expected = 1 + len(titles) + len(abstracts)
        if len(vecs) != expected:
            raise RuntimeError(f"unexpected embeddings size: {len(vecs)} vs {expected}")
        q_vec = vecs[0]
        title_vecs = vecs[1 : 1 + len(titles)]
        abstract_vecs = vecs[1 + len(titles) :]

        scored: list[tuple[int, float]] = []
        for i in range(len(papers)):
            title_sim = _cosine_similarity(q_vec, title_vecs[i])
            ap = abstract_pos[i]
            abstract_sim = _cosine_similarity(q_vec, abstract_vecs[ap]) if ap is not None else 0.0

            abstract_sim = max(abstract_sim, 0.0)
            score = 0.7 * title_sim + 0.3 * abstract_sim
            if ap is None:
                score -= 0.03
            scored.append((i, score))

        scored.sort(key=lambda x: (-x[1], x[0]))
        ranked = [papers[i] for i, _ in scored[:k]]
        debug(f"embedding_rank_applied model={settings.embed_model} picked={len(ranked)}/{k}")
        return ranked, True
    except Exception as e:
        debug(f"embedding_rank_failed; fallback to simple rank. error={e}")
        return _simple_rank(q, papers, k), False


def _paper_text_for_rerank(p: Paper, *, max_chars: int) -> str:
    title = normalize_whitespace(p.title)
    abstract = normalize_whitespace(p.abstract)

    parts: list[str] = []
    if title:
        parts.append(f"Title: {title}")
    if _is_meaningful_abstract(abstract):
        parts.append(f"Abstract: {abstract}")

    text = "\n".join(parts) if parts else "(empty paper)"
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars]
    return text


async def _rerank_rank(
    client: HTTPClient,
    *,
    query: str,
    papers: list[Paper],
    limit: int,
    settings: Settings,
) -> tuple[list[Paper], bool, dict[str, int]]:
    if not papers:
        return [], False, {"docs": 0, "docs_with_abstract": 0, "total_chars": 0, "max_doc_chars": 0}

    q = normalize_whitespace(query)
    k = min(max(int(limit), 1), len(papers))
    if not q:
        return papers[:k], False, {"docs": 0, "docs_with_abstract": 0, "total_chars": 0, "max_doc_chars": 0}
    if not (settings.llm_base_url and settings.rerank_model):
        return _simple_rank(q, papers, k), False, {"docs": 0, "docs_with_abstract": 0, "total_chars": 0, "max_doc_chars": 0}

    cfg = LLMConfig(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        max_retries=int(settings.rerank_max_retries),
        retry_base_delay=settings.llm_retry_base_delay,
        retry_max_delay=settings.llm_retry_max_delay,
        rerank_url=settings.rerank_url,
    )

    docs_with_abstract = 0
    docs: list[str] = []
    max_chars = max(int(settings.rerank_max_doc_chars), 200)
    total_chars = 0
    max_doc_chars = 0
    for p in papers:
        abstract = normalize_whitespace(p.abstract)
        if _is_meaningful_abstract(abstract):
            docs_with_abstract += 1
        doc = _paper_text_for_rerank(p, max_chars=max_chars)
        docs.append(doc)
        n = len(doc)
        total_chars += n
        max_doc_chars = max(max_doc_chars, n)

    stats = {
        "docs": len(docs),
        "docs_with_abstract": docs_with_abstract,
        "total_chars": total_chars,
        "max_doc_chars": max_doc_chars,
    }
    try:
        ranked = await rerank(
            client,
            cfg=cfg,
            model=settings.rerank_model,
            query=q,
            documents=docs,
            top_n=k,
        )
        if not ranked:
            debug("rerank returned empty results; fallback to simple rank.")
            return _simple_rank(q, papers, k), False, stats

        picked: list[Paper] = []
        seen: set[int] = set()
        for idx, _score in ranked:
            if not isinstance(idx, int):
                continue
            if idx < 0 or idx >= len(papers) or idx in seen:
                continue
            picked.append(papers[idx])
            seen.add(idx)
            if len(picked) >= k:
                break

        if not picked:
            debug("rerank did not return any valid indices; fallback to simple rank.")
            return _simple_rank(q, papers, k), False, stats

        for i, p in enumerate(papers):
            if len(picked) >= k:
                break
            if i in seen:
                continue
            picked.append(p)
        debug(f"rerank_applied model={settings.rerank_model} picked={len(picked)}/{k}")
        return picked[:k], True, stats
    except Exception as e:
        debug(f"rerank_failed; fallback to simple rank. error={e}")
        return _simple_rank(q, papers, k), False, stats




def _resolve_specs(platforms: list[str]) -> tuple[list[PlatformSpec], list[PlatformStatus]]:
    specs: list[PlatformSpec] = []
    unknown: list[PlatformStatus] = []
    seen: set[str] = set()
    for name in platforms:
        name = (name or "").strip()
        if not name:
            continue
        spec = resolve_spec(name)
        if spec is None:
            unknown.append(PlatformStatus(platform=name, status="unsupported", detail="unknown platform"))
            continue
        if spec.name in seen:
            continue
        seen.add(spec.name)
        specs.append(spec)
    return specs, unknown


async def _enrich_missing_dois(papers: list[Paper], *, settings: Settings, headers: dict[str, str]) -> None:
    missing_doi_before = sum(1 for p in papers if not p.doi)
    doi_from_url = 0
    to_enrich: list[Paper] = []
    for p in papers:
        if p.doi:
            continue
        doi = _extract_doi_from_text(p.url)
        if doi:
            p.doi = doi
            if not p.url:
                p.url = f"https://doi.org/{doi}"
            doi_from_url += 1
            continue
        # A paper with an arXiv or OpenReview id is already downloadable and
        # resolvable; a Crossref title lookup would only add latency.
        if p.arxiv_id or p.openreview_id:
            continue
        to_enrich.append(p)

    enriched_doi = 0
    t0 = time.perf_counter()
    if settings.doi_enrich_enabled and to_enrich:
        doi_client = HTTPClient(timeout=settings.doi_enrich_timeout_s, headers=headers)
        sem = asyncio.Semaphore(max(int(settings.doi_enrich_max_concurrency), 1))

        async def enrich_one(p: Paper) -> bool:
            async with sem:
                enriched = await guess_doi_from_crossref(
                    doi_client, title=p.title, authors=p.authors, settings=settings
                )
                if not enriched:
                    return False
                doi, url = enriched
                p.doi = doi
                if not p.url:
                    p.url = url
                return True

        results = await asyncio.gather(*(enrich_one(p) for p in to_enrich))
        enriched_doi = sum(1 for ok in results if ok)

    if logger is not None and missing_doi_before:
        logger.info(
            "[paper_search] doi_enrich_time_s={} enabled={} missing_doi_before={} doi_from_url={} enriched_doi={}",
            f"{time.perf_counter() - t0:.3f}",
            settings.doi_enrich_enabled,
            missing_doi_before,
            doi_from_url,
            enriched_doi,
        )


async def search_papers_detailed(
    query: str,
    platforms: list[str],
    *,
    filters: SearchFilters | None = None,
    final_limit: int | None = None,
    page: int = 1,
    per_platform: int | None = None,
    rerank_mode: str = "auto",
    settings: Settings | None = None,
) -> SearchOutcome:
    """Search several platforms and return the papers *and* what each platform did.

    `rerank_mode`: "on" always hands the merged pool to the rerank model when
    one is configured, "off" never does, "auto" only when the pool exceeds
    RERANK_AUTO_THRESHOLD — a large sweep is where a model's first cut pays
    for itself; for a small one lexical order over the platforms' own ranking
    is as good. Rerank only applies to relevance order.

    When OpenAlex's daily budget is known to be spent it is not called at all;
    Semantic Scholar takes its place with a doubled share of the pool.
    """
    if settings is None:
        load_env_file(".env")
        settings = Settings.from_env()
    if final_limit is not None:
        max_limit = max(int(settings.final_limit_max), 1)
        settings = replace(settings, final_limit=min(max(int(final_limit), 1), max_limit))
    limit = settings.final_limit
    sort = (filters.sort if filters else "relevance") or "relevance"
    if sort not in SORTS:
        raise ValueError(f"sort must be one of {SORTS}")

    q_norm = normalize_whitespace(query)
    if settings.query_max_chars > 0 and len(q_norm) > int(settings.query_max_chars):
        raise ValueError(f"q too long: {len(q_norm)} > max {int(settings.query_max_chars)} chars")

    specs, statuses = _resolve_specs(platforms)
    if settings.platforms_max > 0 and len(specs) > int(settings.platforms_max):
        raise ValueError(f"too many platforms: {len(specs)} > max {int(settings.platforms_max)}")

    shares: dict[str, int] = {}
    if per_platform:
        shares = {spec.name: int(per_platform) for spec in specs}
    else:
        shares = {spec.name: max(settings.limit_for_platform(spec.name), limit) for spec in specs}

    names = [s.name for s in specs]
    if "OpenAlex" in names and OPENALEX_BUDGET.exhausted():
        skipped = PlatformStatus(
            platform="OpenAlex",
            status="quota_exhausted",
            detail=str(QuotaExhaustedError("OpenAlex", OPENALEX_BUDGET.reset_in())),
        )
        statuses.append(skipped)
        oa_share = shares.pop("OpenAlex", limit)
        specs = [s for s in specs if s.name != "OpenAlex"]
        s2 = resolve_spec("SemanticScholar")
        if s2 is not None and s2.name not in [s.name for s in specs]:
            specs.insert(0, s2)
            shares[s2.name] = oa_share
        elif s2 is not None:
            shares[s2.name] = shares.get(s2.name, limit) + oa_share

    headers = {"User-Agent": DEFAULT_USER_AGENT}
    client = HTTPClient(timeout=30.0, headers=headers)
    rerank_client = HTTPClient(timeout=settings.rerank_timeout_s, headers=headers)

    t_total0 = time.perf_counter()
    if logger is not None:
        logger.info(
            "[paper_search] start q={} platforms={} final_limit={} sort={} filters={}",
            q_norm,
            [s.name for s in specs],
            limit,
            sort,
            sorted(filters.active()) if filters else [],
        )

    raw, run_statuses = await _run_searchers(
        client, query=query, specs=specs, settings=settings, filters=filters, page=page, per_platform=shares
    )
    statuses = run_statuses + statuses

    merged = _dedupe_keep_first(raw)
    candidates = len(merged)
    kept, excluded = _apply_local_filters(merged, filters)

    rerank_used = False
    if sort == "relevance":
        use_rerank = rerank_mode == "on" or (rerank_mode == "auto" and len(kept) > RERANK_AUTO_THRESHOLD)
        if use_rerank:
            ranked, rerank_used, stats = await _rerank_rank(
                rerank_client, query=query, papers=kept, limit=limit, settings=settings
            )
            if logger is not None:
                logger.info(
                    "[paper_search] rerank used={} docs={} ranked={}",
                    rerank_used,
                    int(stats.get("docs", 0)),
                    len(ranked),
                )
        else:
            ranked = _simple_rank(q_norm, kept, min(limit, len(kept)) or 1) if kept else []
    else:
        ranked = _sort_papers(kept, sort)
    final = ranked[:limit]

    await _enrich_missing_dois(final, settings=settings, headers=headers)

    if logger is not None:
        logger.info(
            f"[paper_search] total_time_s={time.perf_counter() - t_total0:.3f} "
            f"candidates={candidates} kept={len(kept)} final={len(final)}"
        )
    return SearchOutcome(
        papers=final,
        platforms=statuses,
        rerank_used=rerank_used,
        excluded=excluded,
        candidates=candidates,
    )


async def search_papers(
    query: str,
    platforms: list[str],
    *,
    final_limit: int | None = None,
    summary_enabled: bool | None = None,
    fields: list[str] | None = None,
    settings: Settings | None = None,
    filters: SearchFilters | None = None,
    rerank_mode: str = "on",
) -> str:
    """Search papers across multiple platforms and return a JSON string.

    The list-of-papers view of `search_papers_detailed`, for callers that do
    not act on per-platform status. Rerank defaults to "on" here, which is how
    this function always behaved when a rerank model was configured.
    """
    _ = summary_enabled  # deprecated and no-op
    fields = _normalize_output_fields(fields)
    outcome = await search_papers_detailed(
        query,
        platforms,
        filters=filters,
        final_limit=final_limit,
        rerank_mode=rerank_mode,
        settings=settings,
    )
    return json.dumps([p.to_dict(fields=fields) for p in outcome.papers], ensure_ascii=False, indent=2)
