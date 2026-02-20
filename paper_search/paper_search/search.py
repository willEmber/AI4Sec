from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import math
import re
import time
from typing import Iterable

from .config import DEFAULT_USER_AGENT, Settings, load_env_file
from .debug import debug
from .http_client import HTTPClient
from .llm import LLMConfig, embeddings, rerank
from .models import PUBLIC_FIELDS, Paper
from .platforms import resolve_platform
from .platforms.crossref import guess_doi_from_crossref
from .logger import logger
from .utils import (
    normalize_doi,
    normalize_whitespace,
    title_fingerprint,
)


def _norm_platform_name(name: str) -> str:
    return "".join(ch for ch in (name or "").casefold() if ch.isalnum())


def _looks_like_snippet(text: str) -> bool:
    t = normalize_whitespace(text)
    if not t:
        return False
    # Scholar/serp snippets typically contain ellipses.
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
    if incoming.abstract:
        if (
            not primary.abstract
            or (
                len(incoming.abstract) > len(primary.abstract)
                and _looks_like_snippet(primary.abstract)
            )
        ):
            primary.abstract = incoming.abstract
    if not primary.url and incoming.url:
        primary.url = incoming.url
    if not primary.doi and incoming.doi:
        primary.doi = incoming.doi
    if not primary.authors and incoming.authors:
        primary.authors = incoming.authors


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
    by_title: dict[str, Paper] = {}
    ordered: list[Paper] = []

    for paper in papers:
        paper.title = normalize_whitespace(paper.title)
        paper.abstract = normalize_whitespace(paper.abstract)
        paper.url = normalize_whitespace(paper.url)
        paper.doi = normalize_doi(paper.doi)
        paper.authors = normalize_whitespace(paper.authors)

        doi_key = paper.doi
        title_key = title_fingerprint(paper.title)

        if doi_key and doi_key in by_doi:
            _merge_into(by_doi[doi_key], paper)
            continue

        if title_key in by_title:
            existing = by_title[title_key]
            # If the new one has a DOI and the existing doesn't, upgrade the existing.
            if doi_key and not existing.doi:
                existing.doi = doi_key
                by_doi[doi_key] = existing
            _merge_into(existing, paper)
            continue

        ordered.append(paper)
        by_title[title_key] = paper
        if doi_key:
            by_doi[doi_key] = paper

    return ordered


async def _run_searchers(
    client: HTTPClient,
    *,
    query: str,
    platforms: list[str],
    settings: Settings,
) -> list[Paper]:
    selected: list[tuple[str, callable]] = []
    seen: set[str] = set()
    for name in platforms:
        key = _norm_platform_name(name)
        if not key or key in seen:
            continue
        fn = resolve_platform(name)
        if fn:
            selected.append((name, fn))
            seen.add(key)

    async def _timed(platform_name: str, fn) -> tuple[str, list[Paper] | Exception, float]:
        t0 = time.perf_counter()
        try:
            out = await fn(client, query=query, limit=settings.limit_for_platform(platform_name), settings=settings)
            return platform_name, out, time.perf_counter() - t0
        except Exception as e:
            return platform_name, e, time.perf_counter() - t0

    tasks = [asyncio.create_task(_timed(platform_name, fn)) for platform_name, fn in selected]
    timed_results = await asyncio.gather(*tasks)

    merged: list[Paper] = []
    for platform_name, result, elapsed_s in timed_results:
        if logger is not None:
            logger.info(f"[paper_search] platform={platform_name} search_time_s={elapsed_s:.3f}")
        if isinstance(result, Exception):
            if logger is not None:
                logger.warning(f"[paper_search] platform={platform_name} search failed: {result}")
            debug(f"platform={platform_name} search failed: {result}")
            continue
        debug(f"platform={platform_name} results={len(result)}")
        merged.extend(result)
    return merged


def _simple_rank(query: str, papers: list[Paper], limit: int) -> list[Paper]:
    """
    Simple lexical ranking (no LLM).
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
    # Common placeholder strings from upstream providers.
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

            # Avoid punishing longer abstracts; only let abstract help when it matches.
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


async def search_papers(
    query: str,
    platforms: list[str],
    *,
    final_limit: int | None = None,
    summary_enabled: bool | None = None,
    fields: list[str] | None = None,
    settings: Settings | None = None,
) -> str:
    """
    Search papers across multiple platforms and return a JSON string.
    Required keys per item:
      title, abstract, url, doi, authors, source_platform

    If *settings* is provided it is used directly; otherwise configuration
    is loaded from environment variables / ``.env`` as before.
    """
    if settings is None:
        load_env_file(".env")
        settings = Settings.from_env()
    fields = _normalize_output_fields(fields)
    if final_limit is not None:
        max_limit = max(int(settings.final_limit_max), 1)
        effective = max(int(final_limit), 1)
        effective = min(effective, max_limit)
        settings = replace(settings, final_limit=effective)
    _ = summary_enabled  # deprecated and no-op

    q_norm = normalize_whitespace(query)
    if settings.query_max_chars > 0 and len(q_norm) > int(settings.query_max_chars):
        raise ValueError(f"q too long: {len(q_norm)} > max {int(settings.query_max_chars)} chars")

    platforms = [p.strip() for p in (platforms or []) if (p or "").strip()]
    unique_platforms = {_norm_platform_name(p) for p in platforms if _norm_platform_name(p)}
    if settings.platforms_max > 0 and len(unique_platforms) > int(settings.platforms_max):
        raise ValueError(
            f"too many platforms: {len(unique_platforms)} > max {int(settings.platforms_max)}"
        )

    headers = {"User-Agent": DEFAULT_USER_AGENT}

    client = HTTPClient(timeout=30.0, headers=headers)
    rerank_client = HTTPClient(timeout=settings.rerank_timeout_s, headers=headers)

    t_total0 = time.perf_counter()
    if logger is not None:
        logger.info(
            "[paper_search] start q={} platforms={} final_limit={}",
            normalize_whitespace(query),
            platforms,
            settings.final_limit,
        )

    t0 = time.perf_counter()
    raw = await _run_searchers(client, query=query, platforms=platforms, settings=settings)
    if logger is not None:
        logger.info(f"[paper_search] searchers_time_s={time.perf_counter() - t0:.3f} raw={len(raw)}")

    t0 = time.perf_counter()
    merged = _dedupe_keep_first(raw)
    if logger is not None:
        logger.info(f"[paper_search] dedupe_time_s={time.perf_counter() - t0:.3f} merged={len(merged)}")
    debug(f"merged_after_dedupe={len(merged)}")

    t0 = time.perf_counter()
    ranked, rerank_used, rerank_stats = await _rerank_rank(
        rerank_client, query=query, papers=merged, limit=settings.final_limit, settings=settings
    )
    if logger is not None:
        logger.info(
            "[paper_search] rerank_time_s={} used={} ranked={} docs={} docs_with_abstract={} max_doc_chars={} total_chars={}",
            f"{time.perf_counter() - t0:.3f}",
            rerank_used,
            len(ranked),
            int(rerank_stats.get("docs", 0)),
            int(rerank_stats.get("docs_with_abstract", 0)),
            int(rerank_stats.get("max_doc_chars", 0)),
            int(rerank_stats.get("total_chars", 0)),
        )
    final = ranked[:settings.final_limit]

    # Best-effort DOI enrichment for items without DOI (improves metadata completeness).
    t0 = time.perf_counter()
    missing_doi_before = sum(1 for p in final if not p.doi)
    doi_from_url = 0
    to_enrich: list[Paper] = []
    for p in final:
        if p.doi:
            continue
        doi = _extract_doi_from_text(p.url)
        if doi:
            p.doi = doi
            if not p.url:
                p.url = f"https://doi.org/{doi}"
            doi_from_url += 1
            continue
        to_enrich.append(p)

    enriched_doi = 0
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
        elapsed = time.perf_counter() - t0
        logger.info(
            "[paper_search] doi_enrich_time_s={} enabled={} missing_doi_before={} doi_from_url={} enriched_doi={}",
            f"{elapsed:.3f}",
            settings.doi_enrich_enabled,
            missing_doi_before,
            doi_from_url,
            enriched_doi,
        )

    if logger is not None:
        logger.info(f"[paper_search] total_time_s={time.perf_counter() - t_total0:.3f} final={len(final)}")
    return json.dumps([p.to_dict(fields=fields) for p in final], ensure_ascii=False, indent=2)
