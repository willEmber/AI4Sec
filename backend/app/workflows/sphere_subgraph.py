from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Coroutine

import httpx

from app.config import get_settings
from app.db import database as db
from app.models.paper_ir import PaperIR
from app.models.sphere_models import (
    CandidateSource,
    ComparisonRow,
    EdgeType,
    GapIdea,
    KeyHub,
    LibraryMatch,
    ReadingPath,
    SphereConfig,
    SphereEdge,
    SphereNode,
    SphereOutput,
    SphereState,
    ThemeCluster,
    TimelineEntry,
    make_node_id,
)
from app.services.llm_service import get_llm_service
from app.services.paper_search import normalize_whitespace, title_fingerprint
from app.workflows.progress import emit_progress as _emit_progress
from app.workflows.state import MainGraphState

logger = logging.getLogger("scholar.graph")


def _is_center_paper(center: SphereNode, title: str, doi: str = "") -> bool:
    """Check if a candidate is actually the center paper (via DOI or title fingerprint)."""
    if doi and center.doi and doi.strip().lower() == center.doi.strip().lower():
        return True
    if title and center.title:
        return title_fingerprint(title) == title_fingerprint(center.title)
    return False


# ──────────────────────────────────────────────────────────────────────
# Reference extraction (kept from original)
# ──────────────────────────────────────────────────────────────────────

def _extract_references(paper_ir: PaperIR) -> list[dict[str, str]]:
    """Extract reference entries from blocks."""
    refs: list[dict[str, str]] = []
    in_ref_section = False

    for block in paper_ir.blocks:
        section_lower = block.section_path.lower()
        if "reference" in section_lower or "bibliography" in section_lower:
            in_ref_section = True
        elif block.type == "title" and in_ref_section:
            in_ref_section = False

        if not in_ref_section:
            continue

        if block.type in ("text", "ref_text", "list"):
            text = block.text.strip()
            if not text or len(text) < 10:
                continue

            ref: dict[str, str] = {"raw": text, "page": str(block.page_idx + 1)}

            # Try to extract DOI
            doi_match = re.search(r'10\.\d{4,}/[^\s,\]]+', text)
            if doi_match:
                ref["doi"] = doi_match.group(0).rstrip(".")

            # Try to extract arXiv ID
            arxiv_match = re.search(r'(?:arXiv:?)(\d{4}\.\d{4,5}(?:v\d+)?)', text, re.IGNORECASE)
            if arxiv_match:
                ref["arxiv_id"] = arxiv_match.group(1)

            # Try to extract year
            year_match = re.search(r'\b(19|20)\d{2}\b', text)
            if year_match:
                ref["year"] = year_match.group(0)

            refs.append(ref)

    return refs


def _extract_title_from_citation(raw: str) -> str:
    """Extract paper title from a raw citation string."""
    text = raw.strip()
    text = re.sub(r'^\[?\d+\]?[\.\)\s]*', '', text).strip()

    m = re.search(r'["\u201c]([^"\u201d]{10,})["\u201d]', text)
    if m:
        return m.group(1).strip()

    m = re.search(r':\s+(.+?)\.\s+(?:In[:\s]|arXiv)', text)
    if m and len(m.group(1).strip()) >= 15:
        return m.group(1).strip()

    m = re.search(
        r'\b(?:19|20)\d{2}[a-z]?\b[\.\,\;\:\s]+(.+?)(?:\.\s+(?:In |Proceedings|Proc\.|arXiv|IEEE|ACM|AAAI|NeurIPS|ICML|ICLR|CVPR|ICCV|ECCV|SIGMOD|VLDB|KDD|WWW|Advances in|Trans\.|Journal))',
        text,
        re.IGNORECASE,
    )
    if m and len(m.group(1).strip()) > 15:
        return m.group(1).strip().rstrip(".")

    m = re.search(r'\b(?:19|20)\d{2}[a-z]?\b[\.\,\s]+([^\.]{15,}?)\.', text)
    if m:
        title = m.group(1).strip()
        if len(title) > 15:
            return title

    parts = [p.strip() for p in text.split('. ') if p.strip()]
    if len(parts) >= 3:
        candidates = parts[1:-1]
        best = max(candidates, key=len) if candidates else ""
        if len(best) > 15:
            return best

    return text[:120]


# ──────────────────────────────────────────────────────────────────────
# Step 1: Initialize from PDF
# ──────────────────────────────────────────────────────────────────────

async def step_1_init_from_pdf(
    sphere: SphereState,
    state: MainGraphState,
    run_id: str,
) -> SphereState:
    """Parse PaperIR, extract references, create center + seed nodes."""
    await _emit_progress(run_id, "sphere_init_from_pdf", "running")

    paper_ir = PaperIR.model_validate_json(state["paper_ir_json"])
    paper_id = state["paper_id"]

    # Create center node
    center_id = make_node_id(title=paper_ir.title)
    center = SphereNode(
        node_id=center_id,
        title=paper_ir.title,
        layer=2,
        mineru_parsed=True,
        source=CandidateSource.SEED_REF,
    )
    sphere.center_node = center
    sphere.nodes[center_id] = center

    # Extract references from PDF
    refs = _extract_references(paper_ir)
    sphere.pdf_refs = refs
    logger.info(f"[{paper_id}] sphere step 1: {len(refs)} refs extracted from PDF")

    # Create seed nodes from PDF references
    for ref in refs:
        title = _extract_title_from_citation(ref.get("raw", ""))
        if len(title.strip()) < 10:
            continue

        doi = ref.get("doi", "")
        node_id = make_node_id(doi=doi, title=title)
        if node_id == center_id or node_id in sphere.nodes:
            continue
        # Skip refs that are actually the center paper (title/DOI match)
        if _is_center_paper(center, title, doi):
            continue

        year = 0
        if ref.get("year"):
            try:
                year = int(ref["year"])
            except ValueError:
                pass

        node = SphereNode(
            node_id=node_id,
            title=title,
            doi=doi,
            arxiv_id=ref.get("arxiv_id", ""),
            year=year,
            source=CandidateSource.SEED_REF,
            sources=[CandidateSource.SEED_REF],
        )
        sphere.nodes[node_id] = node
        sphere.edges.append(SphereEdge(
            source_node_id=center_id,
            target_node_id=node_id,
            edge_type=EdgeType.CITES,
        ))

    logger.info(f"[{paper_id}] sphere step 1: {len(sphere.nodes)} nodes, {len(sphere.edges)} edges after seed init")
    await _emit_progress(run_id, "sphere_init_from_pdf", "done")
    return sphere


# ──────────────────────────────────────────────────────────────────────
# Step 2: Extract core metadata
# ──────────────────────────────────────────────────────────────────────

async def step_2_extract_core_metadata(
    sphere: SphereState,
    state: MainGraphState,
    run_id: str,
) -> SphereState:
    """Extract DOI, year, authors from PaperIR and DB."""
    await _emit_progress(run_id, "extract_core_metadata", "running")

    paper_id = state["paper_id"]
    paper_ir = PaperIR.model_validate_json(state["paper_ir_json"])
    center = sphere.center_node
    if not center:
        await _emit_progress(run_id, "extract_core_metadata", "done")
        return sphere

    # Try to get DOI from DB
    paper_row = await db.fetch_one(
        "SELECT doi FROM papers WHERE paper_id = ?", (paper_id,)
    )
    if paper_row and paper_row.get("doi"):
        center.doi = paper_row["doi"]

    # Extract DOI from text if not in DB
    if not center.doi:
        for block in paper_ir.blocks[:20]:
            doi_match = re.search(r'10\.\d{4,}/[^\s,\]]+', block.text)
            if doi_match:
                center.doi = doi_match.group(0).rstrip(".").lower()
                break

    # Extract year from first-page blocks
    for block in paper_ir.blocks[:30]:
        if block.page_idx > 1:
            break
        year_match = re.search(r'\b(20[0-2]\d)\b', block.text)
        if year_match:
            center.year = int(year_match.group(0))
            break

    # Extract authors from first-page blocks
    for block in paper_ir.blocks[:10]:
        if block.page_idx > 0:
            break
        # Heuristic: block after title, before abstract, with commas
        if block.type == "text" and "," in block.text and len(block.text) < 500:
            text = block.text.strip()
            if not any(kw in text.lower() for kw in ("abstract", "introduction", "keywords")):
                center.authors = normalize_whitespace(text[:300])
                break

    logger.info(
        f"[{paper_id}] sphere step 2: doi={center.doi or '(none)'} "
        f"year={center.year} authors_len={len(center.authors)}"
    )
    await _emit_progress(run_id, "extract_core_metadata", "done")
    return sphere


# ──────────────────────────────────────────────────────────────────────
# Step 3: Resolve canonical IDs
# ──────────────────────────────────────────────────────────────────────

async def step_3_resolve_canonical_ids(
    sphere: SphereState,
    state: MainGraphState,
    run_id: str,
) -> SphereState:
    """Resolve OpenAlex and S2 IDs for the center paper."""
    await _emit_progress(run_id, "resolve_canonical_ids", "running")

    from app.services.citation_graph import (
        crossref_resolve_doi,
        openalex_resolve_id,
        s2_resolve_id,
    )

    paper_id = state["paper_id"]
    center = sphere.center_node
    if not center:
        await _emit_progress(run_id, "resolve_canonical_ids", "done")
        return sphere

    async with httpx.AsyncClient(timeout=30.0) as client:
        # If no DOI, try Crossref first
        if not center.doi and center.title:
            try:
                doi = await crossref_resolve_doi(client, center.title, center.authors)
                if doi:
                    center.doi = doi
                    logger.info(f"[{paper_id}] sphere step 3: resolved DOI via Crossref: {doi}")
            except Exception as e:
                logger.debug(f"[{paper_id}] sphere step 3: Crossref DOI lookup failed: {e}")

        # Resolve OpenAlex and S2 in parallel
        oa_task = openalex_resolve_id(client, doi=center.doi, title=center.title)
        s2_task = s2_resolve_id(client, doi=center.doi, arxiv_id=center.arxiv_id, title=center.title)

        oa_id, s2_id = await asyncio.gather(oa_task, s2_task, return_exceptions=True)

        if isinstance(oa_id, str):
            center.openalex_id = oa_id
        elif isinstance(oa_id, Exception):
            logger.debug(f"[{paper_id}] sphere step 3: OpenAlex resolve failed: {oa_id}")

        if isinstance(s2_id, str):
            center.s2_paper_id = s2_id
        elif isinstance(s2_id, Exception):
            logger.debug(f"[{paper_id}] sphere step 3: S2 resolve failed: {s2_id}")

    logger.info(
        f"[{paper_id}] sphere step 3: openalex={center.openalex_id or '(none)'} "
        f"s2={center.s2_paper_id or '(none)'} doi={center.doi or '(none)'}"
    )
    await _emit_progress(run_id, "resolve_canonical_ids", "done")
    return sphere


# ──────────────────────────────────────────────────────────────────────
# Search query extraction (LLM)
# ──────────────────────────────────────────────────────────────────────

_QUERY_EXTRACTION_SYSTEM = """You are a research paper keyword extractor. Given a paper's title, abstract, and introduction, generate 3-5 diverse search queries for finding related academic literature.

Requirements:
- Each query should be a concise phrase (3-8 words) focusing on a different research aspect
- Cover different dimensions: core methodology, application domain, key technique, related approaches or baselines
- Do NOT simply repeat or paraphrase the full title
- Queries should be effective for academic search engines (Semantic Scholar, arXiv, etc.)

Return a JSON array of strings, for example:
["diffusion model watermarking", "robust image steganography deep learning", "edge-guided content generation"]

Return ONLY a valid JSON array, no markdown fences or explanation."""


async def _extract_search_queries(
    paper_ir: PaperIR,
    center: SphereNode,
    model: str,
    run_id: str,
    paper_id: str,
) -> list[str]:
    """Use LLM to extract 3-5 diverse search queries from the paper content."""
    # Build context from PaperIR: title + abstract + keywords + introduction
    parts: list[str] = [f"Title: {paper_ir.title}"]

    abstract_text = ""
    keywords_text = ""
    intro_text = ""

    for block in paper_ir.blocks:
        section_lower = block.section_path.lower()
        if "abstract" in section_lower and block.type == "text":
            abstract_text += block.text.strip() + " "
        elif "keyword" in section_lower and block.type in ("text", "list"):
            keywords_text += block.text.strip() + " "
        elif "introduction" in section_lower and block.type == "text":
            intro_text += block.text.strip() + " "

    if abstract_text:
        parts.append(f"Abstract: {abstract_text[:1000]}")
    if keywords_text:
        parts.append(f"Keywords: {keywords_text[:300]}")
    if intro_text:
        parts.append(f"Introduction (excerpt): {intro_text[:1500]}")

    context = "\n\n".join(parts)

    try:
        llm = get_llm_service()
        messages = [
            {"role": "system", "content": _QUERY_EXTRACTION_SYSTEM},
            {"role": "user", "content": context},
        ]
        response = await llm.chat(messages, model=model, temperature=0.3, max_tokens=512)
        response = _strip_json_fences(response)
        queries = json.loads(response)
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            queries = [q.strip() for q in queries if q.strip()]
            if queries:
                logger.info(
                    f"[{paper_id}] sphere: extracted {len(queries)} search queries: {queries}"
                )
                return queries[:5]
    except Exception as e:
        logger.warning(f"[{paper_id}] sphere: query extraction failed, falling back to title: {e}")

    # Fallback: use title
    return [center.title]


# ──────────────────────────────────────────────────────────────────────
# Step 4: Expand graph candidates
# ──────────────────────────────────────────────────────────────────────

async def step_4_expand_graph_candidates(
    sphere: SphereState,
    state: MainGraphState,
    run_id: str,
) -> SphereState:
    """Expand the citation graph by querying OpenAlex, S2, and paper_search."""
    await _emit_progress(run_id, "expand_graph_candidates", "running")

    from app.services.citation_graph import (
        PaperMetadata,
        openalex_get_cited_by,
        openalex_get_referenced_works,
        openalex_get_related_works,
        s2_get_citations,
        s2_get_recommendations,
        s2_get_references,
    )

    paper_id = state["paper_id"]
    center = sphere.center_node
    if not center:
        await _emit_progress(run_id, "expand_graph_candidates", "done")
        return sphere

    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks: list[tuple[str, Coroutine]] = []

        # OpenAlex channels
        if center.openalex_id:
            tasks.append(("oa_refs", openalex_get_referenced_works(client, center.openalex_id)))
            tasks.append(("oa_cited_by", openalex_get_cited_by(client, center.openalex_id)))
            tasks.append(("oa_related", openalex_get_related_works(client, center.openalex_id)))

        # S2 channels
        if center.s2_paper_id:
            tasks.append(("s2_refs", s2_get_references(client, center.s2_paper_id)))
            tasks.append(("s2_cited_by", s2_get_citations(client, center.s2_paper_id)))
            tasks.append(("s2_reco", s2_get_recommendations(client, center.s2_paper_id)))

        # Run all API calls in parallel
        if tasks:
            coros = [t[1] for t in tasks]
            labels = [t[0] for t in tasks]
            results = await asyncio.gather(*coros, return_exceptions=True)

            source_map = {
                "oa_refs": CandidateSource.OPENALEX_REF,
                "oa_cited_by": CandidateSource.OPENALEX_CITED_BY,
                "oa_related": CandidateSource.OPENALEX_RELATED,
                "s2_refs": CandidateSource.S2_REF,
                "s2_cited_by": CandidateSource.S2_CITED_BY,
                "s2_reco": CandidateSource.S2_RECO,
            }

            edge_type_map = {
                "oa_refs": EdgeType.CITES,
                "oa_cited_by": EdgeType.CITED_BY,
                "oa_related": EdgeType.RELATED,
                "s2_refs": EdgeType.CITES,
                "s2_cited_by": EdgeType.CITED_BY,
                "s2_reco": EdgeType.RELATED,
            }

            for label, result in zip(labels, results):
                if isinstance(result, Exception):
                    logger.warning(f"[{paper_id}] sphere step 4: {label} failed: {result}")
                    continue
                if not isinstance(result, list):
                    continue

                source = source_map.get(label, CandidateSource.QUERY_SEARCH)
                edge_type = edge_type_map.get(label, EdgeType.RELATED)
                count = 0

                for meta in result:
                    if not isinstance(meta, PaperMetadata) or not meta.title:
                        continue
                    node_id = make_node_id(doi=meta.doi, title=meta.title)
                    if node_id == center.node_id:
                        continue
                    # Skip candidates that are actually the center paper
                    if _is_center_paper(center, meta.title, meta.doi):
                        continue

                    if node_id in sphere.nodes:
                        # Merge metadata into existing node
                        existing = sphere.nodes[node_id]
                        if not existing.doi and meta.doi:
                            existing.doi = meta.doi
                        if not existing.openalex_id and meta.openalex_id:
                            existing.openalex_id = meta.openalex_id
                        if not existing.s2_paper_id and meta.s2_paper_id:
                            existing.s2_paper_id = meta.s2_paper_id
                        if not existing.abstract_text and meta.abstract_text:
                            existing.abstract_text = meta.abstract_text
                        if meta.cited_by_count > existing.cited_by_count:
                            existing.cited_by_count = meta.cited_by_count
                        if not existing.venue and meta.venue:
                            existing.venue = meta.venue
                        if not existing.authors and meta.authors:
                            existing.authors = meta.authors
                        if meta.year and not existing.year:
                            existing.year = meta.year
                        # Track all sources that contributed to this node
                        if source not in existing.sources:
                            existing.sources.append(source)
                    else:
                        node = SphereNode(
                            node_id=node_id,
                            title=meta.title,
                            doi=meta.doi,
                            arxiv_id=meta.arxiv_id,
                            openalex_id=meta.openalex_id,
                            s2_paper_id=meta.s2_paper_id,
                            year=meta.year,
                            venue=meta.venue,
                            authors=meta.authors,
                            abstract_text=meta.abstract_text,
                            cited_by_count=meta.cited_by_count,
                            source=source,
                            sources=[source],
                        )
                        sphere.nodes[node_id] = node
                        count += 1

                    # Add edge
                    if edge_type == EdgeType.CITED_BY:
                        sphere.edges.append(SphereEdge(
                            source_node_id=node_id,
                            target_node_id=center.node_id,
                            edge_type=edge_type,
                        ))
                    else:
                        sphere.edges.append(SphereEdge(
                            source_node_id=center.node_id,
                            target_node_id=node_id,
                            edge_type=edge_type,
                        ))

                logger.info(f"[{paper_id}] sphere step 4: {label} -> {count} new nodes")

    # Extract diverse search queries via LLM while waiting for S2 rate limits
    paper_ir = PaperIR.model_validate_json(state["paper_ir_json"])
    query_task = asyncio.create_task(
        _extract_search_queries(
            paper_ir, center, state.get("llm_model", ""), run_id, paper_id,
        )
    )
    # Brief delay to let S2 rate limits recover; LLM extraction runs in parallel
    await asyncio.sleep(2.0)
    queries = await query_task

    # Paper search channel — multiple keyword-driven queries
    try:
        from app.services.paper_search import (
            Settings as PSSettings,
            load_env_file,
            search_papers,
        )

        # Ensure .env is loaded so PAPERSEARCH_* vars are available.
        load_env_file(str(Path(__file__).resolve().parents[3] / ".env"))

        # Load full settings from env (API keys, emails, etc.),
        # then override LLM fields from backend config.
        _cfg = get_settings()
        _ps_base = PSSettings.from_env()
        _ps_settings = PSSettings(
            **{
                **{f.name: getattr(_ps_base, f.name) for f in _ps_base.__dataclass_fields__.values()},
                "llm_base_url": _cfg.llm_base_url,
                "llm_api_key": _cfg.llm_api_key,
                "rerank_model": _cfg.rerank_model,
            }
        )

        per_query_limit = 15
        _search_sem = asyncio.Semaphore(2)  # max 2 concurrent searches

        async def _run_search(query: str) -> list[dict]:
            async with _search_sem:
                result_json = await search_papers(
                    query,
                    platforms=["OpenAlex", "SemanticScholar", "arXiv", "Crossref", "IEEE Xplore"],
                    final_limit=per_query_limit,
                    settings=_ps_settings,
                )
                return json.loads(result_json)

        results_lists = await asyncio.gather(
            *[_run_search(q) for q in queries],
            return_exceptions=True,
        )

        # Merge results from all queries into sphere nodes
        search_count = 0
        for qi, result_list in enumerate(results_lists):
            if isinstance(result_list, Exception):
                logger.warning(
                    f"[{paper_id}] sphere step 4: query_search[{qi}] "
                    f"q={queries[qi]!r} failed: {result_list}"
                )
                continue
            q_count = 0
            for paper in result_list:
                title = paper.get("title", "")
                doi = paper.get("doi", "")
                if not title:
                    continue
                node_id = make_node_id(doi=doi, title=title)
                if node_id == center.node_id or node_id in sphere.nodes:
                    continue
                # Skip candidates that are actually the center paper
                if _is_center_paper(center, title, doi):
                    continue

                authors_raw = paper.get("authors", "")
                if isinstance(authors_raw, list):
                    authors_raw = ", ".join(
                        a if isinstance(a, str) else a.get("name", "")
                        for a in authors_raw[:5]
                    )

                node = SphereNode(
                    node_id=node_id,
                    title=title,
                    doi=doi,
                    year=paper.get("year", 0) or 0,
                    venue=paper.get("venue", "") or "",
                    abstract_text=paper.get("abstract", ""),
                    authors=authors_raw,
                    source=CandidateSource.QUERY_SEARCH,
                    sources=[CandidateSource.QUERY_SEARCH],
                )
                sphere.nodes[node_id] = node
                sphere.edges.append(SphereEdge(
                    source_node_id=center.node_id,
                    target_node_id=node_id,
                    edge_type=EdgeType.RELATED,
                ))
                q_count += 1
                search_count += 1
            logger.info(
                f"[{paper_id}] sphere step 4: query_search[{qi}] "
                f"q={queries[qi]!r} -> {q_count} new nodes"
            )
        logger.info(
            f"[{paper_id}] sphere step 4: query_search total -> "
            f"{search_count} new nodes from {len(queries)} queries"
        )
    except Exception as e:
        logger.warning(f"[{paper_id}] sphere step 4: paper_search failed: {e}")

    # Library channel — semantically related papers from the user's own Dify
    # knowledge base. Captured into sphere.library_matches (rendered as its own
    # section) and, when a hit coincides with a graph node, that node is tagged
    # LIBRARY. Fault-tolerant and bounded so a slow/unavailable KB never breaks
    # the run (mirrors the paper_search guard above). Default search method is
    # fast (full_text) per the project's "fast-first" retrieval choice.
    _settings = get_settings()
    if _settings.dify_enabled and _settings.dify_sphere_top_k > 0 and center.title:
        try:
            from app.services import dify_client

            def _lib_title(name: str) -> str:
                base = name.rsplit("/", 1)[-1]
                base = re.sub(r"\.(md|markdown|pdf)$", "", base, flags=re.IGNORECASE)
                base = re.sub(r"(_paper)?_clean$", "", base, flags=re.IGNORECASE)
                return base.replace("_", " ").strip()

            records = await dify_client.search_records(
                center.title,
                top_k=_settings.dify_sphere_top_k,
                search_method=_settings.dify_search_method,
            )
            seen_docs: set[str] = set()
            for rec in records:
                doc_id = rec.get("document_id", "")
                if not doc_id or doc_id in seen_docs:
                    continue
                seen_docs.add(doc_id)
                title = _lib_title(rec.get("document_name", ""))
                if not title or _is_center_paper(center, title, ""):
                    continue
                try:
                    score = float(rec.get("score") or 0.0)
                except (TypeError, ValueError):
                    score = 0.0
                sphere.library_matches.append(LibraryMatch(
                    document_id=doc_id,
                    title=title,
                    score=score,
                    snippet=(rec.get("content", "") or "").strip()[:300],
                ))
                # If this paper already surfaced via another source, tag the node.
                node_id = make_node_id(title=title)
                existing = sphere.nodes.get(node_id)
                if existing is not None:
                    if CandidateSource.LIBRARY not in existing.sources:
                        existing.sources.append(CandidateSource.LIBRARY)
                    if not existing.library_document_id:
                        existing.library_document_id = doc_id
            logger.info(
                f"[{paper_id}] sphere step 4: library -> {len(sphere.library_matches)} "
                f"matches from KB"
            )
        except Exception as e:
            logger.warning(f"[{paper_id}] sphere step 4: library channel failed: {e}")

    # Enforce candidate_cap by trimming lowest-cited_by_count
    cap = sphere.config.candidate_cap
    if len(sphere.nodes) > cap + 1:  # +1 for center
        non_center = [n for nid, n in sphere.nodes.items() if nid != center.node_id]
        non_center.sort(key=lambda n: n.cited_by_count, reverse=True)
        keep_ids = {center.node_id} | {n.node_id for n in non_center[:cap]}
        sphere.nodes = {nid: n for nid, n in sphere.nodes.items() if nid in keep_ids}
        sphere.edges = [
            e for e in sphere.edges
            if e.source_node_id in keep_ids and e.target_node_id in keep_ids
        ]

    logger.info(
        f"[{paper_id}] sphere step 4: total {len(sphere.nodes)} nodes, "
        f"{len(sphere.edges)} edges after expansion"
    )
    await _emit_progress(run_id, "expand_graph_candidates", "done")
    return sphere


# ──────────────────────────────────────────────────────────────────────
# Step 5: Deduplicate and score
# ──────────────────────────────────────────────────────────────────────

async def step_5_dedup_and_score(
    sphere: SphereState,
    state: MainGraphState,
    run_id: str,
) -> SphereState:
    """Compute PageRank, score all nodes, select layer1 and layer2."""
    await _emit_progress(run_id, "dedup_and_score", "running")

    from app.services.sphere_scorer import (
        compute_pagerank,
        mmr_select,
        multi_objective_score,
    )

    paper_id = state["paper_id"]
    center = sphere.center_node
    if not center:
        await _emit_progress(run_id, "dedup_and_score", "done")
        return sphere

    # Compute PageRank
    pagerank = compute_pagerank(sphere.nodes, sphere.edges)

    # Score all non-center nodes
    candidates: list[SphereNode] = []
    for nid, node in sphere.nodes.items():
        if nid == center.node_id:
            continue
        multi_objective_score(
            node, pagerank, center.title, center.abstract_text, sphere.config
        )
        candidates.append(node)

    # MMR diversity selection for layer 1
    layer1_ids = mmr_select(candidates, sphere.config.layer1_cap)
    sphere.layer1_node_ids = layer1_ids
    for nid in layer1_ids:
        if nid in sphere.nodes:
            sphere.nodes[nid].layer = 1

    # Top from layer1 for layer 2 (full-text parsing candidates)
    layer1_nodes = [sphere.nodes[nid] for nid in layer1_ids if nid in sphere.nodes]
    layer1_nodes.sort(key=lambda n: n.score_total, reverse=True)
    layer2_ids = [n.node_id for n in layer1_nodes[:sphere.config.pdf_parse_cap]]
    sphere.layer2_node_ids = layer2_ids
    for nid in layer2_ids:
        if nid in sphere.nodes:
            sphere.nodes[nid].layer = 2

    # Persist to SQLite
    try:
        node_rows = [
            (
                n.node_id, run_id, n.doi, n.arxiv_id, n.openalex_id,
                n.s2_paper_id, n.title, n.year, n.venue, n.authors,
                n.abstract_text, n.cited_by_count, n.pdf_path,
                1 if n.mineru_parsed else 0,
                ",".join(s.value for s in n.sources) if n.sources else n.source.value,
                n.score_total, n.layer, n.cluster_id,
            )
            for n in sphere.nodes.values()
        ]
        await db.execute_many(
            """INSERT OR REPLACE INTO sphere_nodes
            (node_id, run_id, doi, arxiv_id, openalex_id, s2_paper_id,
             title, year, venue, authors, abstract_text, cited_by_count,
             pdf_path, mineru_parsed, source, score_total, layer, cluster_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            node_rows,
        )

        edge_rows = [
            (run_id, e.source_node_id, e.target_node_id, e.edge_type.value, e.weight)
            for e in sphere.edges
        ]
        await db.execute_many(
            """INSERT INTO sphere_edges (run_id, source_node_id, target_node_id, edge_type, weight)
            VALUES (?, ?, ?, ?, ?)""",
            edge_rows,
        )
    except Exception as e:
        logger.warning(f"[{paper_id}] sphere step 5: DB persist failed: {e}")

    logger.info(
        f"[{paper_id}] sphere step 5: layer1={len(layer1_ids)} layer2={len(layer2_ids)} "
        f"total_scored={len(candidates)}"
    )
    await _emit_progress(run_id, "dedup_and_score", "done")
    return sphere


# ──────────────────────────────────────────────────────────────────────
# Step 6: Layer 0 metadata summaries
# ──────────────────────────────────────────────────────────────────────

async def step_6_layer0_summarize_metadata(
    sphere: SphereState,
    state: MainGraphState,
    run_id: str,
) -> SphereState:
    """Generate metadata-only reason strings for all nodes (no LLM)."""
    await _emit_progress(run_id, "layer0_summarize_metadata", "running")

    zh = state.get("language", "en") == "zh"
    lbl = (
        {"source": "来源", "cited": "被引", "year": "年份", "venue": "刊物",
         "score": "评分", "fallback": "PDF 参考文献"}
        if zh
        else {"source": "Source", "cited": "Cited by", "year": "Year", "venue": "Venue",
              "score": "Score", "fallback": "PDF reference"}
    )

    for nid, node in sphere.nodes.items():
        if node == sphere.center_node:
            continue
        parts: list[str] = []
        # Show all sources that contributed to this node
        all_sources = node.sources if node.sources else [node.source]
        non_seed = [s for s in all_sources if s != CandidateSource.SEED_REF]
        if non_seed:
            labels = [s.value.replace("_", " ") for s in non_seed]
            parts.append(f"{lbl['source']}: {', '.join(labels)}")
        if node.cited_by_count > 0:
            parts.append(f"{lbl['cited']} {node.cited_by_count}")
        if node.year:
            parts.append(f"{lbl['year']}: {node.year}")
        if node.venue:
            parts.append(f"{lbl['venue']}: {node.venue}")
        if node.score_total > 0:
            parts.append(f"{lbl['score']}: {node.score_total:.3f}")
        node.reason = " | ".join(parts) if parts else lbl["fallback"]

    await _emit_progress(run_id, "layer0_summarize_metadata", "done")
    return sphere


# ──────────────────────────────────────────────────────────────────────
# Step 7: Layer 1 abstract analysis
# ──────────────────────────────────────────────────────────────────────

_ABSTRACT_SNAP_SYSTEM = """You are a research paper analysis assistant. Given a batch of paper titles and abstracts, extract structured information for each paper.

Return a JSON array where each element has:
{
  "idx": <0-based index>,
  "method_summary": "brief description of the method/approach",
  "contribution": "main contribution",
  "conclusion": "key conclusion",
  "task": "problem/task addressed",
  "dataset": "datasets used (if mentioned)",
  "metric_keywords": "metrics or evaluation criteria"
}

Be concise. If information is not available from the abstract, use empty string.
Return ONLY valid JSON array, no markdown fences."""


async def step_7_layer1_abstract_snap(
    sphere: SphereState,
    state: MainGraphState,
    run_id: str,
) -> SphereState:
    """Batch LLM analysis of abstracts for layer1 nodes."""
    await _emit_progress(run_id, "layer1_abstract_snap", "running")

    paper_id = state["paper_id"]
    model = state.get("llm_model", "")

    # Collect layer1 nodes with abstracts
    nodes_with_abstracts: list[tuple[int, SphereNode]] = []
    for nid in sphere.layer1_node_ids:
        node = sphere.nodes.get(nid)
        if node and node.abstract_text and len(node.abstract_text.strip()) > 30:
            nodes_with_abstracts.append((len(nodes_with_abstracts), node))

    if not nodes_with_abstracts:
        logger.info(f"[{paper_id}] sphere step 7: no abstracts to analyze")
        await _emit_progress(run_id, "layer1_abstract_snap", "done")
        return sphere

    llm = get_llm_service()
    batch_size = 20

    async def _analyze_batch(
        sub_batch: list[tuple[int, SphereNode]],
    ) -> None:
        """Analyze a batch of papers via LLM. On failure of large batches, split and retry."""
        papers_text = ""
        for idx, (_, node) in enumerate(sub_batch):
            papers_text += f"\n[{idx}] Title: {node.title}\nAbstract: {node.abstract_text[:800]}\n"

        messages = [
            {"role": "system", "content": _ABSTRACT_SNAP_SYSTEM},
            {"role": "user", "content": f"Analyze these {len(sub_batch)} papers:\n{papers_text}"},
        ]

        try:
            response = await llm.chat(messages, model=model, temperature=0.2, max_tokens=4096)
            response = response.strip()
            if response.startswith("```"):
                response = re.sub(r'^```\w*\n?', '', response)
                response = re.sub(r'\n?```$', '', response)

            extractions = json.loads(response)
            if isinstance(extractions, list):
                for ext in extractions:
                    idx = ext.get("idx", -1)
                    if 0 <= idx < len(sub_batch):
                        _, node = sub_batch[idx]
                        node.method_summary = ext.get("method_summary", "")
                        node.contribution = ext.get("contribution", "")
                        node.conclusion = ext.get("conclusion", "")
                        node.task = ext.get("task", "")
                        node.dataset = ext.get("dataset", "")
                        node.metric_keywords = ext.get("metric_keywords", "")
        except Exception as e:
            if len(sub_batch) > 5:
                mid = len(sub_batch) // 2
                logger.warning(
                    f"[{paper_id}] sphere step 7: batch of {len(sub_batch)} failed, "
                    f"splitting into {mid}+{len(sub_batch)-mid} and retrying — {e}"
                )
                await _analyze_batch(sub_batch[:mid])
                await _analyze_batch(sub_batch[mid:])
            else:
                logger.warning(f"[{paper_id}] sphere step 7: batch LLM failed ({len(sub_batch)} papers): {e}")

    for batch_start in range(0, len(nodes_with_abstracts), batch_size):
        batch = nodes_with_abstracts[batch_start : batch_start + batch_size]
        await _analyze_batch(batch)

    analyzed = sum(1 for nid in sphere.layer1_node_ids
                   if sphere.nodes.get(nid, SphereNode()).method_summary)
    logger.info(f"[{paper_id}] sphere step 7: analyzed {analyzed}/{len(nodes_with_abstracts)} abstracts")
    await _emit_progress(run_id, "layer1_abstract_snap", "done")
    return sphere


# ──────────────────────────────────────────────────────────────────────
# Step 8: Download and MinerU parse (MVP: skipped)
# ──────────────────────────────────────────────────────────────────────

async def step_8_download_and_parse(
    sphere: SphereState,
    state: MainGraphState,
    run_id: str,
) -> SphereState:
    """Download PDFs for layer2 nodes via OA Resolver / Elsevier TDM / Wiley TDM.

    MinerU parsing of the downloaded PDFs is left to a future step.
    """
    await _emit_progress(run_id, "download_and_mineru_parse", "running")

    paper_id = state["paper_id"]
    if sphere.config.pdf_parse_cap <= 0 or not sphere.layer2_node_ids:
        logger.info(
            f"[{paper_id}] sphere step 8: skipped (pdf_parse_cap={sphere.config.pdf_parse_cap})"
        )
        await _emit_progress(run_id, "download_and_mineru_parse", "done")
        return sphere

    from app.services.paper_downloader import (
        DownloaderCredentials,
        DEFAULT_USER_AGENT,
        download_paper,
        safe_filename,
    )

    settings = get_settings()
    credentials = DownloaderCredentials(
        unpaywall_email=settings.unpaywall_email,
        core_api_key=settings.core_api_key,
        elsevier_api_key=settings.elsevier_api_key,
        elsevier_inst_token=settings.elsevier_inst_token,
        wiley_tdm_token=settings.wiley_tdm_token,
    )

    has_any_cred = any(
        [
            credentials.unpaywall_email,
            credentials.core_api_key,
            credentials.elsevier_api_key,
            credentials.wiley_tdm_token,
        ]
    )
    if not has_any_cred:
        logger.warning(
            f"[{paper_id}] sphere step 8: no downloader credentials configured, skipping"
        )
        await _emit_progress(run_id, "download_and_mineru_parse", "done")
        return sphere

    refs_dir = settings.data_dir / "papers" / paper_id / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)

    targets: list[tuple[str, SphereNode, Path]] = []
    for nid in sphere.layer2_node_ids:
        node = sphere.nodes.get(nid)
        if not node or not node.doi:
            continue
        if node.pdf_path and Path(node.pdf_path).exists():
            continue
        dest = refs_dir / f"{safe_filename(node.doi)}.pdf"
        targets.append((nid, node, dest))

    if not targets:
        logger.info(f"[{paper_id}] sphere step 8: no DOI-bearing layer2 nodes to download")
        await _emit_progress(run_id, "download_and_mineru_parse", "done")
        return sphere

    semaphore = asyncio.Semaphore(4)

    async with httpx.AsyncClient(
        headers={"User-Agent": DEFAULT_USER_AGENT},
        timeout=60.0,
    ) as client:
        async def _do_one(node: SphereNode, dest: Path):
            async with semaphore:
                return await download_paper(
                    node.doi,
                    dest,
                    credentials=credentials,
                    title=node.title,
                    client=client,
                    timeout=60.0,
                    retries=3,
                )

        results = await asyncio.gather(
            *(_do_one(node, dest) for _, node, dest in targets),
            return_exceptions=True,
        )

    ok_count = 0
    fail_count = 0
    for (nid, node, dest), result in zip(targets, results):
        if isinstance(result, Exception):
            fail_count += 1
            logger.warning(f"[{paper_id}] sphere step 8: {node.doi} crashed: {result}")
            continue
        if result.ok and result.pdf_path:
            node.pdf_path = result.pdf_path
            ok_count += 1
        else:
            fail_count += 1
            logger.info(
                f"[{paper_id}] sphere step 8: {node.doi} failed via {result.source}: {result.detail}"
            )

    logger.info(
        f"[{paper_id}] sphere step 8: downloaded {ok_count}/{len(targets)} layer2 PDFs "
        f"(fail={fail_count}) → {refs_dir}"
    )
    await _emit_progress(run_id, "download_and_mineru_parse", "done")
    return sphere


# ──────────────────────────────────────────────────────────────────────
# Step 9: Synthesize landscape (3 LLM calls)
# ──────────────────────────────────────────────────────────────────────

_CLASSIFIER_SYSTEM = """You are a research theme classifier. Given a list of papers with their metadata and analysis, identify 3-8 thematic clusters.

Return a JSON object:
{
  "themes": [
    {
      "name": "theme name",
      "definition": "1-2 sentence definition",
      "paper_indices": [0, 3, 7]
    }
  ]
}

Group papers by shared methodology, problem domain, or approach. Every paper should belong to at least one theme. Return ONLY valid JSON, no markdown fences."""

_COMPARATOR_SYSTEM = """You are a research comparator. Given the center paper and top comparison papers, produce a detailed comparison.

Return a JSON object:
{
  "rows": [
    {
      "idx": 0,
      "problem": "problem addressed",
      "assumption": "key assumptions",
      "method": "methodology",
      "dataset": "datasets used",
      "metric": "evaluation metrics",
      "strength": "key strengths",
      "weakness": "key weaknesses"
    }
  ]
}

idx=0 is the center paper. Subsequent indices match the comparison papers in order.
Be specific and concise. Return ONLY valid JSON, no markdown fences."""

_ADVISOR_SYSTEM = """You are a research advisor. Given theme clusters, a comparison matrix, and key hub papers, identify research gaps and suggest reading paths.

Return a JSON object:
{
  "overview": "2-3 paragraph landscape overview describing the research field, key trends, and state of the art",
  "gaps": [
    {
      "title": "gap title",
      "description": "explanation of the gap and why it matters",
      "evidence_indices": [0, 2, 5]
    }
  ],
  "reading_paths": [
    {
      "name": "fast",
      "description": "Quick introduction path (3-5 papers)",
      "paper_indices": [0, 1, 3]
    },
    {
      "name": "deep",
      "description": "Thorough understanding path (8-12 papers)",
      "paper_indices": [0, 1, 2, 3, 5, 7, 8, 10]
    },
    {
      "name": "frontier",
      "description": "Cutting-edge research path (5-8 papers)",
      "paper_indices": [2, 4, 6, 9, 11]
    }
  ]
}

Return ONLY valid JSON, no markdown fences."""


# Appended to the three synthesis role prompts when the user requested Chinese.
# The JSON *keys* must stay English (the code parses them), so we only ask the
# model to write the natural-language *values* in Chinese — this avoids a full
# post-hoc translation pass over the assembled report.
_SPHERE_ZH_DIRECTIVE = """

输出语言要求:JSON 的所有键名(key)保持英文不变;但所有自然语言文本字段的值(如 name、definition、problem、assumption、method、dataset、metric、strength、weakness、title、description、overview 等)一律用简体中文撰写。论文标题、作者姓名、期刊/会议名称、以及专有技术术语可保留英文(术语首次出现可在括号内附中文解释)。仍然只返回合法 JSON,不要 markdown 代码围栏或任何解释。"""


async def step_9_synthesize_landscape(
    sphere: SphereState,
    state: MainGraphState,
    run_id: str,
) -> SphereState:
    """Run 3 LLM roles: Classifier, Comparator, Research Advisor."""
    await _emit_progress(run_id, "synthesize_landscape", "running")

    paper_id = state["paper_id"]
    model = state.get("llm_model", "")
    language = state.get("language", "en")
    center = sphere.center_node
    if not center:
        await _emit_progress(run_id, "synthesize_landscape", "done")
        return sphere

    def _localize(base: str) -> str:
        """Append the Chinese output directive to a role prompt when lang=zh."""
        return base + _SPHERE_ZH_DIRECTIVE if language == "zh" else base

    llm = get_llm_service()

    # Build paper list for LLM context (layer1 nodes with data)
    paper_list: list[SphereNode] = []
    for nid in sphere.layer1_node_ids:
        node = sphere.nodes.get(nid)
        if node:
            paper_list.append(node)

    # Create indexed paper descriptions
    def _paper_desc(idx: int, node: SphereNode) -> str:
        parts = [f"[{idx}] {node.title}"]
        if node.year:
            parts.append(f"Year: {node.year}")
        if node.venue:
            parts.append(f"Venue: {node.venue}")
        if node.authors:
            parts.append(f"Authors: {node.authors[:100]}")
        if node.abstract_text:
            parts.append(f"Abstract: {node.abstract_text[:300]}")
        if node.method_summary:
            parts.append(f"Method: {node.method_summary}")
        if node.contribution:
            parts.append(f"Contribution: {node.contribution}")
        if node.task:
            parts.append(f"Task: {node.task}")
        return "\n".join(parts)

    papers_context = "\n\n".join(_paper_desc(i, p) for i, p in enumerate(paper_list))

    # ── Prepare Comparator inputs (needed before launching gather) ──
    # Filter out any nodes that are duplicates of the center paper
    comparison_papers = [
        p for p in paper_list[:sphere.config.comparison_top_k]
        if not _is_center_paper(center, p.title, p.doi)
    ]
    paper_ir = PaperIR.model_validate_json(state["paper_ir_json"])
    center_text_parts = []
    for block in paper_ir.blocks:
        if block.type in ("text", "title", "list", "equation"):
            center_text_parts.append(f"{block.text.strip()} [p.{block.page_idx + 1}]")
    center_text = "\n".join(center_text_parts)
    if len(center_text) > 6000:
        center_text = center_text[:6000] + "\n[...truncated...]"

    comp_context = f"CENTER PAPER:\n[0] {center.title}\n{center_text}\n\nCOMPARISON PAPERS:\n"
    for i, node in enumerate(comparison_papers):
        comp_context += f"\n{_paper_desc(i + 1, node)}\n"

    # ── LLM Role 1: Classifier (async) ──
    async def _run_classifier() -> dict[str, Any] | None:
        try:
            classifier_messages = [
                {"role": "system", "content": _localize(_CLASSIFIER_SYSTEM)},
                {"role": "user", "content": f"Classify these {len(paper_list)} papers into themes:\n\n{papers_context}"},
            ]
            classifier_resp = await llm.chat(classifier_messages, model=model, temperature=0.2, max_tokens=4096)
            classifier_resp = _strip_json_fences(classifier_resp)
            return json.loads(classifier_resp)
        except Exception as e:
            logger.warning(f"[{paper_id}] sphere step 9: Classifier failed: {e}")
            return None

    # ── LLM Role 2: Comparator (async) ──
    async def _run_comparator() -> dict[str, Any] | None:
        try:
            comparator_messages = [
                {"role": "system", "content": _localize(_COMPARATOR_SYSTEM)},
                {"role": "user", "content": f"Compare these papers:\n\n{comp_context}"},
            ]
            comparator_resp = await llm.chat(comparator_messages, model=model, temperature=0.2, max_tokens=8192)
            comparator_resp = _strip_json_fences(comparator_resp)
            return json.loads(comparator_resp)
        except Exception as e:
            logger.warning(f"[{paper_id}] sphere step 9: Comparator failed: {e}")
            return None

    # Run Classifier + Comparator concurrently (they are independent)
    classifier_data, comparator_data = await asyncio.gather(
        _run_classifier(), _run_comparator(),
    )

    # Process Classifier results
    if classifier_data:
        for theme_data in classifier_data.get("themes", []):
            indices = theme_data.get("paper_indices", [])
            node_ids = [paper_list[i].node_id for i in indices if 0 <= i < len(paper_list)]
            sphere.output.themes.append(ThemeCluster(
                name=theme_data.get("name", ""),
                definition=theme_data.get("definition", ""),
                representative_node_ids=node_ids,
            ))
            # Assign cluster IDs
            for i, nid in enumerate(node_ids):
                if nid in sphere.nodes:
                    sphere.nodes[nid].cluster_id = len(sphere.output.themes) - 1

    # Process Comparator results (with dedup by node_id)
    if comparator_data:
        seen_comp_nids: set[str] = set()
        for row_data in comparator_data.get("rows", []):
            idx = row_data.get("idx", -1)
            if idx == 0:
                nid = center.node_id
                title = center.title
            elif 1 <= idx <= len(comparison_papers):
                nid = comparison_papers[idx - 1].node_id
                title = comparison_papers[idx - 1].title
            else:
                continue

            # Skip duplicate rows for the same paper
            if nid in seen_comp_nids:
                continue
            seen_comp_nids.add(nid)

            sphere.output.comparison_table.append(ComparisonRow(
                node_id=nid,
                title=title,
                problem=row_data.get("problem", ""),
                assumption=row_data.get("assumption", ""),
                method=row_data.get("method", ""),
                dataset=row_data.get("dataset", ""),
                metric=row_data.get("metric", ""),
                strength=row_data.get("strength", ""),
                weakness=row_data.get("weakness", ""),
            ))

    # ── Build timeline (no LLM) ──
    year_groups: dict[int, list[str]] = {}
    for nid, node in sphere.nodes.items():
        if node.year > 0:
            year_groups.setdefault(node.year, []).append(nid)
    for year in sorted(year_groups.keys()):
        sphere.output.timeline.append(TimelineEntry(
            year=year,
            node_ids=year_groups[year],
        ))

    # ── Build key hubs (no LLM) ──
    from app.services.sphere_scorer import compute_pagerank
    pagerank = compute_pagerank(sphere.nodes, sphere.edges)
    hub_candidates = [
        (nid, node, pagerank.get(nid, 0.0))
        for nid, node in sphere.nodes.items()
        if nid != center.node_id
    ]
    hub_candidates.sort(key=lambda x: x[2], reverse=True)
    for nid, node, pr in hub_candidates[:10]:
        sphere.output.key_hubs.append(KeyHub(
            node_id=nid,
            title=node.title,
            pagerank=pr,
            cited_by_count=node.cited_by_count,
            reason=node.reason,
        ))

    # ── LLM Role 3: Research Advisor ──
    try:
        themes_desc = json.dumps(
            [{"name": t.name, "definition": t.definition, "papers": len(t.representative_node_ids)}
             for t in sphere.output.themes],
            indent=2,
        )
        hubs_desc = json.dumps(
            [{"title": h.title, "citations": h.cited_by_count, "pagerank": round(h.pagerank, 3)}
             for h in sphere.output.key_hubs],
            indent=2,
        )
        comp_desc = json.dumps(
            [{"title": r.title, "method": r.method, "strength": r.strength, "weakness": r.weakness}
             for r in sphere.output.comparison_table],
            indent=2,
        )

        advisor_context = (
            f"THEMES:\n{themes_desc}\n\n"
            f"KEY HUBS:\n{hubs_desc}\n\n"
            f"COMPARISON SUMMARY:\n{comp_desc}\n\n"
            f"PAPER LIST:\n{papers_context}"
        )

        advisor_messages = [
            {"role": "system", "content": _localize(_ADVISOR_SYSTEM)},
            {"role": "user", "content": f"Analyze research landscape:\n\n{advisor_context}"},
        ]
        advisor_resp = await llm.chat(advisor_messages, model=model, temperature=0.3, max_tokens=4096)
        advisor_resp = _strip_json_fences(advisor_resp)
        advisor_data = json.loads(advisor_resp)

        sphere.output.sphere_overview = advisor_data.get("overview", "")

        for gap_data in advisor_data.get("gaps", []):
            indices = gap_data.get("evidence_indices", [])
            node_ids = [paper_list[i].node_id for i in indices if 0 <= i < len(paper_list)]
            sphere.output.gaps_and_ideas.append(GapIdea(
                title=gap_data.get("title", ""),
                description=gap_data.get("description", ""),
                evidence_node_ids=node_ids,
            ))

        for path_data in advisor_data.get("reading_paths", []):
            indices = path_data.get("paper_indices", [])
            node_ids = [paper_list[i].node_id for i in indices if 0 <= i < len(paper_list)]
            sphere.output.reading_paths.append(ReadingPath(
                name=path_data.get("name", ""),
                description=path_data.get("description", ""),
                node_ids=node_ids,
            ))
    except Exception as e:
        logger.warning(f"[{paper_id}] sphere step 9: Advisor failed: {e}")

    logger.info(
        f"[{paper_id}] sphere step 9: themes={len(sphere.output.themes)} "
        f"comparison={len(sphere.output.comparison_table)} "
        f"gaps={len(sphere.output.gaps_and_ideas)} "
        f"paths={len(sphere.output.reading_paths)}"
    )
    await _emit_progress(run_id, "synthesize_landscape", "done")
    return sphere


def _strip_json_fences(text: str) -> str:
    """Remove markdown code fences from JSON response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```\w*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
    return text.strip()


# ──────────────────────────────────────────────────────────────────────
# Step 10: Render output
# ──────────────────────────────────────────────────────────────────────

async def step_10_render_output(
    sphere: SphereState,
    state: MainGraphState,
    run_id: str,
) -> tuple[str, str]:
    """Convert SphereOutput to markdown and JSON."""
    await _emit_progress(run_id, "render_output", "running")

    center = sphere.center_node
    output = sphere.output
    paper_id = state["paper_id"]
    zh = state.get("language", "en") == "zh"

    # Localized scaffolding for the rendered report (the LLM fills the prose
    # fields; these are the fixed headings/labels the code emits).
    t = (
        {
            "overview_h": "## 研究全景概览\n", "themes_h": "\n### 主题聚类\n",
            "timeline_h": "## 脉络与时间线\n", "comparison_h": "## 对比矩阵\n\n",
            "comparison_cols": "| 论文 | 问题 | 方法 | 数据集 | 指标 | 优势 | 不足 |\n",
            "hubs_h": "## 关键枢纽论文\n", "gaps_h": "## 研究空白与想法\n",
            "paths_h": "## 推荐阅读路径\n", "cited_by": "被引", "more": "篇",
            "evidence_from": "证据来源", "path_suffix": "路径", "unknown": "[未知论文]",
            "library_h": "## 你的文献库中的相关工作\n",
            "library_intro": "以下论文来自你自己的知识库,与本文语义相关:\n",
        }
        if zh
        else {
            "overview_h": "## Research Landscape Overview\n", "themes_h": "\n### Thematic Clusters\n",
            "timeline_h": "## Lineage & Timeline\n", "comparison_h": "## Comparison Matrix\n\n",
            "comparison_cols": "| Paper | Problem | Method | Dataset | Metric | Strength | Weakness |\n",
            "hubs_h": "## Key Hub Papers\n", "gaps_h": "## Research Gaps & Ideas\n",
            "paths_h": "## Suggested Reading Paths\n", "cited_by": "cited by", "more": "more",
            "evidence_from": "Evidence from", "path_suffix": "Path", "unknown": "[unknown paper]",
            "library_h": "## Related Work in Your Library\n",
            "library_intro": "These papers from your own knowledge base are semantically related to this paper:\n",
        }
    )

    # Node lookup helper
    def _node_ref(node_id: str) -> str:
        node = sphere.nodes.get(node_id)
        if not node:
            return t["unknown"]
        year_str = f" ({node.year})" if node.year else ""
        return f"**{node.title}**{year_str}"

    md_parts: list[str] = []

    # ── Section 1: Landscape Map ──
    md_parts.append(t["overview_h"])
    if output.sphere_overview:
        md_parts.append(output.sphere_overview + "\n")

    if output.themes:
        md_parts.append(t["themes_h"])
        for i, theme in enumerate(output.themes, 1):
            md_parts.append(f"**{i}. {theme.name}**: {theme.definition}\n")
            for nid in theme.representative_node_ids[:8]:
                node = sphere.nodes.get(nid)
                if node:
                    year_str = f" ({node.year})" if node.year else ""
                    cite_str = f" — {t['cited_by']} {node.cited_by_count}" if node.cited_by_count else ""
                    md_parts.append(f"- {node.title}{year_str}{cite_str}\n")
            md_parts.append("\n")

    # ── Section 2: Timeline ──
    if output.timeline:
        md_parts.append(t["timeline_h"])
        for entry in output.timeline:
            nodes_in_year = [sphere.nodes.get(nid) for nid in entry.node_ids if sphere.nodes.get(nid)]
            if nodes_in_year:
                titles = ", ".join(n.title[:60] for n in nodes_in_year[:5])
                extra = f" (+{len(nodes_in_year) - 5} {t['more']})" if len(nodes_in_year) > 5 else ""
                md_parts.append(f"- **{entry.year}**: {titles}{extra}\n")
        md_parts.append("\n")

    # ── Section 3: Comparison Matrix ──
    if output.comparison_table:
        md_parts.append(t["comparison_h"])
        md_parts.append(t["comparison_cols"])
        md_parts.append("|-------|---------|--------|---------|--------|----------|----------|\n")
        for row in output.comparison_table:
            short_title = row.title[:40] + ("..." if len(row.title) > 40 else "")
            md_parts.append(
                f"| {short_title} | {row.problem} | {row.method} | "
                f"{row.dataset} | {row.metric} | {row.strength} | {row.weakness} |\n"
            )
        md_parts.append("\n")

    # ── Section 4: Key Hubs ──
    if output.key_hubs:
        md_parts.append(t["hubs_h"])
        for hub in output.key_hubs:
            md_parts.append(
                f"- **{hub.title}** — PageRank: {hub.pagerank:.3f}, "
                f"Citations: {hub.cited_by_count}"
            )
            if hub.reason:
                md_parts.append(f" ({hub.reason})")
            md_parts.append("\n")
        md_parts.append("\n")

    # ── Section 5: Gaps & Ideas ──
    if output.gaps_and_ideas:
        md_parts.append(t["gaps_h"])
        for i, gap in enumerate(output.gaps_and_ideas, 1):
            md_parts.append(f"### {i}. {gap.title}\n\n")
            md_parts.append(f"{gap.description}\n\n")
            if gap.evidence_node_ids:
                evidence = [_node_ref(nid) for nid in gap.evidence_node_ids[:5]]
                md_parts.append(f"*{t['evidence_from']}*: {', '.join(evidence)}\n\n")

    # ── Section 6: Reading Paths ──
    if output.reading_paths:
        md_parts.append(t["paths_h"])
        for path in output.reading_paths:
            md_parts.append(f"### {path.name.title()} {t['path_suffix']}\n\n")
            md_parts.append(f"{path.description}\n\n")
            for j, nid in enumerate(path.node_ids, 1):
                node = sphere.nodes.get(nid)
                if node:
                    year_str = f" ({node.year})" if node.year else ""
                    md_parts.append(f"{j}. {node.title}{year_str}\n")
            md_parts.append("\n")

    # ── Section 7: Related work in the user's own library (Dify KB) ──
    if sphere.library_matches:
        md_parts.append(t["library_h"])
        md_parts.append(t["library_intro"])
        for m in sphere.library_matches[:15]:
            label = f"[{m.title}](/library?doc={m.document_id})" if m.document_id else m.title
            score = f" ({m.score:.2f})" if m.score and m.score > 0 else ""
            md_parts.append(f"- {label}{score}\n")
        md_parts.append("\n")

    markdown = "".join(md_parts)

    # Build JSON output
    json_data = json.dumps({
        "mode": "sphere",
        "paper_id": paper_id,
        "title": center.title if center else "",
        "num_nodes": len(sphere.nodes),
        "num_edges": len(sphere.edges),
        "num_layer1": len(sphere.layer1_node_ids),
        "num_layer2": len(sphere.layer2_node_ids),
        "num_library_matches": len(sphere.library_matches),
        "library_matches": [m.model_dump() for m in sphere.library_matches],
        "sphere_output": output.model_dump(),
    }, ensure_ascii=False)

    logger.info(f"[{paper_id}] sphere step 10: markdown={len(markdown)} chars")
    await _emit_progress(run_id, "render_output", "done")
    return markdown, json_data


# ──────────────────────────────────────────────────────────────────────
# Main orchestrator
# ──────────────────────────────────────────────────────────────────────

async def run_research_sphere(state: MainGraphState) -> dict[str, Any]:
    """Run the 10-step Research Sphere pipeline."""
    paper_id = state["paper_id"]
    run_id = state.get("run_id", "")
    t0 = time.perf_counter()

    settings = get_settings()
    config = SphereConfig(
        radius=settings.sphere_radius,
        candidate_cap=settings.sphere_candidate_cap,
        layer1_cap=settings.sphere_layer1_cap,
        pdf_parse_cap=settings.sphere_pdf_parse_cap,
    )
    sphere = SphereState(config=config)

    logger.info(f"[{paper_id}] sphere: Starting 10-step pipeline (config={config.model_dump()})")

    try:
        sphere = await step_1_init_from_pdf(sphere, state, run_id)
        sphere = await step_2_extract_core_metadata(sphere, state, run_id)
        sphere = await step_3_resolve_canonical_ids(sphere, state, run_id)
        sphere = await step_4_expand_graph_candidates(sphere, state, run_id)
        sphere = await step_5_dedup_and_score(sphere, state, run_id)
        sphere = await step_6_layer0_summarize_metadata(sphere, state, run_id)
        sphere = await step_7_layer1_abstract_snap(sphere, state, run_id)
        sphere = await step_8_download_and_parse(sphere, state, run_id)
        sphere = await step_9_synthesize_landscape(sphere, state, run_id)
        markdown, json_data = await step_10_render_output(sphere, state, run_id)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.exception(f"[{paper_id}] sphere: Pipeline failed at {elapsed:.1f}s — {e}")
        return {
            "error": f"Research Sphere failed: {e}",
            "progress": state.get("progress", []) + [{"step": "run_sphere", "status": "failed"}],
        }

    elapsed = time.perf_counter() - t0
    logger.info(f"[{paper_id}] sphere: Pipeline completed in {elapsed:.1f}s — {len(markdown)} chars markdown")

    return {
        "final_markdown": markdown,
        "analysis_language": state.get("language", "en"),
        "final_json": json_data,
        "sphere_state_json": sphere.model_dump_json(),
        "progress": state.get("progress", []) + [{"step": "run_sphere", "status": "done"}],
    }
