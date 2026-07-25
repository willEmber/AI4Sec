"""Research Sphere pipeline.

Redesigned around three principles (inspired by Connected Papers / Elicit):

1. **Provenance tiers** — citation-graph evidence (references, citers,
   recommendations) is trusted; keyword-search hits are third-class citizens
   that must clear quality gates before they may enter the report.
2. **Quality × relevance gating** — every candidate gets a citation/venue
   quality score plus an LLM-judged relation to the center paper
   (foundation / competitor / follow-up / …); "unrelated" is discarded.
3. **Quota-based core set** — the final ~40-paper core set is assembled per
   relation partition (foundations, method neighbors & competitors,
   follow-ups, recent frontier, user library), then all synthesis (themes,
   comparison, gaps, reading paths) runs only on that core set.

Pipeline steps (step names are progress keys mirrored in frontend i18n):
    sphere_init_from_pdf → extract_core_metadata → resolve_canonical_ids
    → expand_graph_candidates → enrich_candidates → score_and_rank
    → relevance_gate → build_similarity_graph → select_core_set
    → layer1_abstract_snap → download_and_mineru_parse
    → synthesize_landscape → render_output
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Coroutine

import httpx

from app.config import get_settings
from app.db import database as db
from app.models.paper_ir import PaperIR
from app.models.sphere_models import (
    CandidateSource,
    ComparisonRow,
    CorePartition,
    EdgeType,
    GapIdea,
    KeyHub,
    LibraryMatch,
    ReadingPath,
    RelationType,
    SphereConfig,
    SphereEdge,
    SphereNode,
    SphereOutput,
    SphereState,
    ThemeCluster,
    TimelineEntry,
    make_node_id,
    tier_for_sources,
)
from app.services.llm_service import get_llm_service
from app.services.paper_search import normalize_whitespace, title_fingerprint
from app.services.sphere_scorer import is_preprint_venue, is_survey_title, normalize_venue
from app.workflows.progress import emit_progress as _emit_progress
from app.workflows.state import MainGraphState

logger = logging.getLogger("scholar.graph")


def _current_year() -> int:
    return _dt.date.today().year


def _is_center_paper(center: SphereNode, title: str, doi: str = "") -> bool:
    """Check if a candidate is actually the center paper (via DOI or title fingerprint)."""
    if doi and center.doi and doi.strip().lower() == center.doi.strip().lower():
        return True
    if title and center.title:
        return title_fingerprint(title) == title_fingerprint(center.title)
    return False


def _candidates(sphere: SphereState) -> list[SphereNode]:
    """All non-center nodes."""
    center_id = sphere.center_node.node_id if sphere.center_node else ""
    return [n for nid, n in sphere.nodes.items() if nid != center_id]


def _drop_nodes(sphere: SphereState, node_ids: set[str]) -> None:
    """Remove nodes and any edges touching them."""
    if not node_ids:
        return
    sphere.nodes = {nid: n for nid, n in sphere.nodes.items() if nid not in node_ids}
    sphere.edges = [
        e for e in sphere.edges
        if e.source_node_id not in node_ids and e.target_node_id not in node_ids
    ]


# ──────────────────────────────────────────────────────────────────────
# Funnel instrumentation
# ──────────────────────────────────────────────────────────────────────

def _funnel(sphere: SphereState, stage: str, **fields: Any) -> None:
    """Record one pipeline-funnel entry (surfaced in final_json.debug)."""
    entry: dict[str, Any] = {"stage": stage}
    entry.update(fields)
    sphere.funnel.append(entry)


def _node_briefs(nodes: list[SphereNode], limit: int = 10) -> list[str]:
    """Compact who-was-here previews, highest-cited first — every drop point
    records these so a missing landmark paper can be traced to the exact
    stage that lost it."""
    ranked = sorted(nodes, key=lambda n: n.cited_by_count, reverse=True)
    return [
        f"{n.title[:90]} ({n.year or '?'}, cited {n.cited_by_count}, T{n.tier})"
        for n in ranked[:limit]
    ]


# ──────────────────────────────────────────────────────────────────────
# Reference extraction (from the PDF)
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

    m = re.search(r'["“]([^"”]{10,})["”]', text)
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
# LLM reference parsing (regex fallback per entry)
# ──────────────────────────────────────────────────────────────────────

_REF_PARSE_SYSTEM = """You are a bibliography parser. You will receive numbered raw reference strings extracted from a paper's References section. For each entry, extract:
- "title": the cited paper's title exactly as written (original language, no trailing period)
- "year": the 4-digit publication year (0 if absent)
- "doi": the DOI if present, else ""

Return ONLY a JSON array with one entry per input reference:
[{"idx": 0, "title": "...", "year": 2017, "doi": ""}]
No markdown fences, no explanation."""


async def _parse_references_llm(
    raw_refs: list[str],
    model: str,
    paper_id: str,
) -> dict[int, dict[str, Any]]:
    """LLM-parse raw citation strings into {ref_index: {title, year, doi}}.

    Regex title extraction misparses enough entries that real references
    (LSTM, Bahdanau attention) used to vanish from the graph. Batches are
    independent; a failed batch just falls back to regex for its entries.
    """
    llm = get_llm_service()
    parsed: dict[int, dict[str, Any]] = {}
    batch_size = 40

    async def _parse_batch(start: int, chunk: list[str]) -> None:
        listing = "\n".join(f"[{i}] {r[:400]}" for i, r in enumerate(chunk))
        try:
            resp = await llm.chat(
                [
                    {"role": "system", "content": _REF_PARSE_SYSTEM},
                    {"role": "user", "content": f"Parse these {len(chunk)} references:\n\n{listing}"},
                ],
                model=model, temperature=0.0, max_tokens=4096,
            )
            data = json.loads(_strip_json_fences(resp))
            if not isinstance(data, list):
                raise ValueError("ref parse response is not a list")
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                idx = entry.get("idx", -1)
                if 0 <= idx < len(chunk):
                    parsed[start + idx] = entry
        except Exception as e:
            logger.warning(
                f"[{paper_id}] sphere init: LLM ref parse failed for batch "
                f"@{start} ({len(chunk)} refs), falling back to regex — {e}"
            )

    await asyncio.gather(*[
        _parse_batch(start, raw_refs[start : start + batch_size])
        for start in range(0, len(raw_refs), batch_size)
    ])
    return parsed


# ──────────────────────────────────────────────────────────────────────
# Step 1: Initialize from PDF
# ──────────────────────────────────────────────────────────────────────

async def step_init_from_pdf(
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
        tier=1,
        mineru_parsed=True,
        source=CandidateSource.SEED_REF,
    )
    sphere.center_node = center
    sphere.nodes[center_id] = center

    # Extract references from PDF
    refs = _extract_references(paper_ir)
    sphere.pdf_refs = refs
    logger.info(f"[{paper_id}] sphere init: {len(refs)} refs extracted from PDF")

    # Parse titles/years via LLM (much more accurate than the regex heuristics);
    # regex remains the per-entry fallback when a batch fails.
    model = state.get("llm_model", "")
    parsed: dict[int, dict[str, Any]] = {}
    if refs and model:
        parsed = await _parse_references_llm([r.get("raw", "") for r in refs], model, paper_id)
        logger.info(f"[{paper_id}] sphere init: LLM parsed {len(parsed)}/{len(refs)} refs")

    year_now = _current_year()

    # Create seed nodes from PDF references
    for ref_idx, ref in enumerate(refs):
        p = parsed.get(ref_idx) or {}
        title = str(p.get("title") or "").strip()
        if len(title) < 10:
            title = _extract_title_from_citation(ref.get("raw", ""))
        if len(title.strip()) < 10:
            continue

        # Regex-found DOI is verbatim from the PDF; prefer it over the LLM's
        doi = ref.get("doi", "") or str(p.get("doi") or "").strip().lower()
        node_id = make_node_id(doi=doi, title=title)
        if node_id == center_id or node_id in sphere.nodes:
            continue
        # Skip refs that are actually the center paper (title/DOI match)
        if _is_center_paper(center, title, doi):
            continue

        year = 0
        try:
            llm_year = int(p.get("year") or 0)
            if 1900 <= llm_year <= year_now + 1:
                year = llm_year
        except (TypeError, ValueError):
            pass
        if not year and ref.get("year"):
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

    _funnel(
        sphere, "init",
        pdf_refs=len(refs),
        llm_parsed=len(parsed),
        seed_nodes=len(sphere.nodes) - 1,
    )
    logger.info(f"[{paper_id}] sphere init: {len(sphere.nodes)} nodes, {len(sphere.edges)} edges after seed init")
    await _emit_progress(run_id, "sphere_init_from_pdf", "done")
    return sphere


# ──────────────────────────────────────────────────────────────────────
# Step 2: Extract core metadata
# ──────────────────────────────────────────────────────────────────────

async def step_extract_core_metadata(
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

    # Abstract for the relevance gate / scoring
    if not center.abstract_text:
        abstract_parts: list[str] = []
        for block in paper_ir.blocks:
            if "abstract" in block.section_path.lower() and block.type == "text":
                abstract_parts.append(block.text.strip())
        center.abstract_text = normalize_whitespace(" ".join(abstract_parts))[:2000]

    logger.info(
        f"[{paper_id}] sphere metadata: doi={center.doi or '(none)'} "
        f"year={center.year} authors_len={len(center.authors)} "
        f"abstract_len={len(center.abstract_text)}"
    )
    await _emit_progress(run_id, "extract_core_metadata", "done")
    return sphere


# ──────────────────────────────────────────────────────────────────────
# Step 3: Resolve canonical IDs
# ──────────────────────────────────────────────────────────────────────

async def step_resolve_canonical_ids(
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
                    logger.info(f"[{paper_id}] sphere: resolved DOI via Crossref: {doi}")
            except Exception as e:
                logger.debug(f"[{paper_id}] sphere: Crossref DOI lookup failed: {e}")

        # Resolve OpenAlex and S2 in parallel
        oa_task = openalex_resolve_id(client, doi=center.doi, title=center.title)
        s2_task = s2_resolve_id(client, doi=center.doi, arxiv_id=center.arxiv_id, title=center.title)

        oa_id, s2_id = await asyncio.gather(oa_task, s2_task, return_exceptions=True)

        if isinstance(oa_id, str):
            center.openalex_id = oa_id
        elif isinstance(oa_id, Exception):
            logger.debug(f"[{paper_id}] sphere: OpenAlex resolve failed: {oa_id}")

        if isinstance(s2_id, str):
            center.s2_paper_id = s2_id
        elif isinstance(s2_id, Exception):
            logger.debug(f"[{paper_id}] sphere: S2 resolve failed: {s2_id}")

    logger.info(
        f"[{paper_id}] sphere ids: openalex={center.openalex_id or '(none)'} "
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

def _merge_meta_into_node(node: SphereNode, meta: Any) -> None:
    """Fill missing node fields from a PaperMetadata-like object."""
    if not node.doi and meta.doi:
        node.doi = meta.doi
    if not node.arxiv_id and getattr(meta, "arxiv_id", ""):
        node.arxiv_id = meta.arxiv_id
    if not node.openalex_id and meta.openalex_id:
        node.openalex_id = meta.openalex_id
    if not node.s2_paper_id and getattr(meta, "s2_paper_id", ""):
        node.s2_paper_id = meta.s2_paper_id
    if not node.abstract_text and meta.abstract_text:
        node.abstract_text = meta.abstract_text
    if meta.cited_by_count > node.cited_by_count:
        node.cited_by_count = meta.cited_by_count
    if not node.venue and meta.venue:
        node.venue = meta.venue
    if not node.authors and meta.authors:
        node.authors = meta.authors
    if meta.year and not node.year:
        node.year = meta.year
    if not node.referenced_ids and getattr(meta, "referenced_works", None):
        node.referenced_ids = list(meta.referenced_works)
    if getattr(meta, "influential", False):
        node.influential = True


async def step_expand_graph_candidates(
    sphere: SphereState,
    state: MainGraphState,
    run_id: str,
) -> SphereState:
    """Expand the citation graph by querying OpenAlex, S2, and paper_search."""
    await _emit_progress(run_id, "expand_graph_candidates", "running")

    from app.services.citation_graph import (
        PaperMetadata,
        openalex_get_cited_by,
        openalex_get_recent_cited_by,
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

    channel_counts: dict[str, int] = {}

    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks: list[tuple[str, Coroutine]] = []

        # OpenAlex channels
        if center.openalex_id:
            tasks.append(("oa_refs", openalex_get_referenced_works(client, center.openalex_id)))
            tasks.append(("oa_cited_by", openalex_get_cited_by(client, center.openalex_id)))
            tasks.append(("oa_related", openalex_get_related_works(client, center.openalex_id)))
            # Frontier supply: top-cited citers from the last ~3 years only —
            # the all-time citers list is landmarks, never the new wave
            tasks.append((
                "oa_recent_cited_by",
                openalex_get_recent_cited_by(client, center.openalex_id, _current_year() - 3),
            ))

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
                "oa_recent_cited_by": CandidateSource.OPENALEX_CITED_BY,
                "s2_refs": CandidateSource.S2_REF,
                "s2_cited_by": CandidateSource.S2_CITED_BY,
                "s2_reco": CandidateSource.S2_RECO,
            }

            edge_type_map = {
                "oa_refs": EdgeType.CITES,
                "oa_cited_by": EdgeType.CITED_BY,
                "oa_related": EdgeType.RELATED,
                "oa_recent_cited_by": EdgeType.CITED_BY,
                "s2_refs": EdgeType.CITES,
                "s2_cited_by": EdgeType.CITED_BY,
                "s2_reco": EdgeType.RELATED,
            }

            for label, result in zip(labels, results):
                if isinstance(result, Exception):
                    logger.warning(f"[{paper_id}] sphere expand: {label} failed: {result}")
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
                        existing = sphere.nodes[node_id]
                        _merge_meta_into_node(existing, meta)
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
                            referenced_ids=list(meta.referenced_works),
                            influential=meta.influential,
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

                channel_counts[label] = count
                logger.info(f"[{paper_id}] sphere expand: {label} -> {count} new nodes")

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

    # Paper search channel — weakest evidence tier (T3). These hits share
    # keywords with the center paper but have no citation-graph link; they must
    # clear the quality + relevance gates later before they can be reported.
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
                    f"[{paper_id}] sphere expand: query_search[{qi}] "
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
                if node_id == center.node_id:
                    continue
                # Skip candidates that are actually the center paper
                if _is_center_paper(center, title, doi):
                    continue
                if node_id in sphere.nodes:
                    existing = sphere.nodes[node_id]
                    if CandidateSource.QUERY_SEARCH not in existing.sources:
                        existing.sources.append(CandidateSource.QUERY_SEARCH)
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
                f"[{paper_id}] sphere expand: query_search[{qi}] "
                f"q={queries[qi]!r} -> {q_count} new nodes"
            )
        channel_counts["query_search"] = search_count
        logger.info(
            f"[{paper_id}] sphere expand: query_search total -> "
            f"{search_count} new nodes from {len(queries)} queries"
        )
    except Exception as e:
        logger.warning(f"[{paper_id}] sphere expand: paper_search failed: {e}")

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
                f"[{paper_id}] sphere expand: library -> {len(sphere.library_matches)} "
                f"matches from KB"
            )
        except Exception as e:
            logger.warning(f"[{paper_id}] sphere expand: library channel failed: {e}")

    # Assign provenance tiers now that all channels have reported
    for node in _candidates(sphere):
        node.tier = tier_for_sources(node.sources or [node.source])

    # Enforce candidate_cap: keep strongest provenance first, then citations
    cap = sphere.config.candidate_cap
    cands = _candidates(sphere)
    cap_dropped: list[SphereNode] = []
    if len(cands) > cap:
        cands.sort(key=lambda n: (n.tier, -n.cited_by_count))
        cap_dropped = cands[cap:]
        keep_ids = {center.node_id} | {n.node_id for n in cands[:cap]}
        _drop_nodes(sphere, {nid for nid in sphere.nodes if nid not in keep_ids})

    _funnel(
        sphere, "expand",
        channels=channel_counts,
        library_matches=len(sphere.library_matches),
        candidates=len(sphere.nodes) - 1,
        cap_dropped=len(cap_dropped),
        cap_dropped_top=_node_briefs(cap_dropped),
    )
    logger.info(
        f"[{paper_id}] sphere expand: total {len(sphere.nodes)} nodes, "
        f"{len(sphere.edges)} edges after expansion"
    )
    await _emit_progress(run_id, "expand_graph_candidates", "done")
    return sphere


# ──────────────────────────────────────────────────────────────────────
# Step 5: Enrich candidates (OpenAlex batch metadata completion)
# ──────────────────────────────────────────────────────────────────────

async def step_enrich_candidates(
    sphere: SphereState,
    state: MainGraphState,
    run_id: str,
) -> SphereState:
    """Complete metadata for every candidate via OpenAlex.

    Seed refs extracted from the PDF arrive with nothing but a title guess;
    without this step they lose every ranking comparison to search-channel
    noise (the exact failure mode of the old pipeline). Three passes:
      1. nodes with an OpenAlex ID but no reference list → batch fetch by ID
      2. nodes with a DOI but no OpenAlex ID → batch fetch by DOI
      3. remaining title-only nodes → bounded per-title search
    Afterwards, nodes that turn out to be the same work are merged.
    """
    await _emit_progress(run_id, "enrich_candidates", "running")

    from app.services.citation_graph import (
        _oa_batch_fetch,
        openalex_batch_fetch_by_doi,
        openalex_get_work,
        openalex_search_one,
        _oa_work_to_metadata,
    )

    paper_id = state["paper_id"]
    center = sphere.center_node
    if not center:
        await _emit_progress(run_id, "enrich_candidates", "done")
        return sphere

    cands = _candidates(sphere)
    by_oa_id = {n.openalex_id: n for n in cands if n.openalex_id and not n.referenced_ids}
    with_doi = [n for n in cands if not n.openalex_id and n.doi]
    title_only = [n for n in cands if not n.openalex_id and not n.doi]

    enriched = 0
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Pass 0: center's own reference list (for coupling with candidates)
        if center.openalex_id and not center.referenced_ids:
            try:
                work = await openalex_get_work(client, center.openalex_id)
                if work:
                    _merge_meta_into_node(center, _oa_work_to_metadata(work))
            except Exception as e:
                logger.debug(f"[{paper_id}] sphere enrich: center fetch failed: {e}")

        # Pass 1: refresh nodes that have an OpenAlex ID but no reference list
        if by_oa_id:
            try:
                metas = await _oa_batch_fetch(client, list(by_oa_id.keys()))
                for meta in metas:
                    node = by_oa_id.get(meta.openalex_id)
                    if node:
                        _merge_meta_into_node(node, meta)
                        enriched += 1
            except Exception as e:
                logger.warning(f"[{paper_id}] sphere enrich: OA-id batch failed: {e}")

        # Pass 2: resolve DOI-bearing nodes
        if with_doi:
            try:
                doi_map = await openalex_batch_fetch_by_doi(
                    client, [n.doi for n in with_doi]
                )
                for node in with_doi:
                    meta = doi_map.get(node.doi.strip().lower())
                    if meta:
                        _merge_meta_into_node(node, meta)
                        enriched += 1
            except Exception as e:
                logger.warning(f"[{paper_id}] sphere enrich: DOI batch failed: {e}")

        # Pass 3: title-only nodes (mostly PDF seed refs) — bounded search
        title_only = title_only[:80]
        sem = asyncio.Semaphore(8)

        async def _resolve_title(node: SphereNode) -> None:
            async with sem:
                try:
                    meta = await openalex_search_one(client, node.title)
                    if meta:
                        _merge_meta_into_node(node, meta)
                except Exception as e:
                    logger.debug(
                        f"[{paper_id}] sphere enrich: title search failed for "
                        f"{node.title[:50]!r}: {e}"
                    )

        if title_only:
            await asyncio.gather(*[_resolve_title(n) for n in title_only])
            enriched += sum(1 for n in title_only if n.openalex_id)

    # Merge nodes that resolved to the same work (a PDF ref keyed by title and
    # an API result keyed by DOI can be the same paper).
    merged = _merge_duplicate_nodes(sphere)

    with_refs = sum(1 for n in _candidates(sphere) if n.referenced_ids)
    _funnel(
        sphere, "enrich",
        enriched=enriched,
        merged_dupes=merged,
        with_references=with_refs,
        candidates=len(sphere.nodes) - 1,
    )
    logger.info(
        f"[{paper_id}] sphere enrich: enriched={enriched} merged_dupes={merged} "
        f"nodes_with_refs={with_refs}/{len(sphere.nodes) - 1}"
    )
    await _emit_progress(run_id, "enrich_candidates", "done")
    return sphere


def _years_compatible(a: int, b: int, tolerance: int = 2) -> bool:
    """Same-title nodes are merged only when their years plausibly agree
    (preprint vs published version); distinct papers sharing a title but
    decades apart stay separate."""
    return a == 0 or b == 0 or abs(a - b) <= tolerance


def _merge_duplicate_nodes(sphere: SphereState) -> int:
    """Merge nodes that are the same work: shared DOI/OpenAlex ID, or same
    title fingerprint with compatible years (arXiv preprint vs published
    version carry *different* DOIs, so identity keys alone miss them)."""
    center = sphere.center_node
    center_id = center.node_id if center else ""

    def _keys(node: SphereNode) -> list[str]:
        keys = []
        if node.doi:
            keys.append(f"doi:{node.doi.strip().lower()}")
        if node.openalex_id:
            keys.append(f"oa:{node.openalex_id}")
        return keys

    canonical: dict[str, str] = {}         # identity key -> surviving node_id
    by_title: dict[str, list[str]] = {}    # title fingerprint -> surviving node_ids
    remap: dict[str, str] = {}             # dropped node_id -> surviving node_id

    def _merge_into(survivor: SphereNode, node: SphereNode, version_merge: bool) -> None:
        # Two versions of one work split the citation record between them;
        # exact-identity duplicates would double count, so only sum then.
        total_citations = (
            survivor.cited_by_count + node.cited_by_count
            if version_merge
            else max(survivor.cited_by_count, node.cited_by_count)
        )
        _merge_meta_into_node(survivor, node)
        survivor.cited_by_count = total_citations
        # Prefer the published outlet (and its year) over a preprint server
        if (
            (not survivor.venue or is_preprint_venue(survivor.venue))
            and node.venue and not is_preprint_venue(node.venue)
        ):
            survivor.venue = node.venue
            if node.year:
                survivor.year = node.year
            if node.sci_rank:
                survivor.sci_rank = node.sci_rank
            if node.ccf_rank:
                survivor.ccf_rank = node.ccf_rank
        elif (
            version_merge
            and node.year and survivor.year and node.year < survivor.year
            and node.venue and not is_preprint_venue(node.venue)
            and survivor.venue and not is_preprint_venue(survivor.venue)
        ):
            # Both versions are formal publications — a conference paper and
            # its later journal reprint (AlexNet: NIPS 2012 vs CACM 2017).
            # The earlier year is the real publication date.
            survivor.year = node.year
        if not survivor.referenced_ids and node.referenced_ids:
            survivor.referenced_ids = list(node.referenced_ids)
        for s in node.sources:
            if s not in survivor.sources:
                survivor.sources.append(s)
        survivor.tier = tier_for_sources(survivor.sources or [survivor.source])
        if node.influential:
            survivor.influential = True
        if node.library_document_id and not survivor.library_document_id:
            survivor.library_document_id = node.library_document_id
        # Keep the stronger relevance judgement (relevant if gate already ran)
        if node.relevance > survivor.relevance:
            survivor.relevance = node.relevance
            survivor.relation_type = node.relation_type
            survivor.relation_reason = node.relation_reason

    for nid, node in list(sphere.nodes.items()):
        if nid == center_id:
            continue
        target: str | None = None
        version_merge = False
        for key in _keys(node):
            if key in canonical and canonical[key] != nid:
                target = canonical[key]
                break
        if target is None:
            fp = title_fingerprint(node.title) if node.title else ""
            if fp:
                for other_nid in by_title.get(fp, []):
                    other = sphere.nodes.get(other_nid)
                    if other is not None and _years_compatible(node.year, other.year):
                        target = other_nid
                        version_merge = True
                        break

        if target is None:
            for key in _keys(node):
                canonical[key] = nid
            fp = title_fingerprint(node.title) if node.title else ""
            if fp:
                by_title.setdefault(fp, []).append(nid)
            continue

        survivor = sphere.nodes[target]
        _merge_into(survivor, node, version_merge)
        # The survivor now also answers for the merged node's identity keys
        for key in _keys(node):
            canonical.setdefault(key, target)
        remap[nid] = target
        del sphere.nodes[nid]

    if remap:
        seen: set[tuple[str, str, str]] = set()
        new_edges: list[SphereEdge] = []
        for e in sphere.edges:
            src = remap.get(e.source_node_id, e.source_node_id)
            tgt = remap.get(e.target_node_id, e.target_node_id)
            if src == tgt:
                continue
            sig = (src, tgt, e.edge_type.value)
            if sig in seen:
                continue
            seen.add(sig)
            new_edges.append(SphereEdge(
                source_node_id=src, target_node_id=tgt,
                edge_type=e.edge_type, weight=e.weight,
            ))
        sphere.edges = new_edges

    return len(remap)


# ──────────────────────────────────────────────────────────────────────
# Step 6: Quality scoring + hard gates
# ──────────────────────────────────────────────────────────────────────

async def step_score_and_rank(
    sphere: SphereState,
    state: MainGraphState,
    run_id: str,
) -> SphereState:
    """Venue ranks (EasyScholar), quality scores, T3 hard gate, gate cap."""
    await _emit_progress(run_id, "score_and_rank", "running")

    from app.services.sphere_scorer import compute_quality_score

    paper_id = state["paper_id"]
    config = sphere.config
    cands = _candidates(sphere)

    # ── Venue ranks via EasyScholar (cache-backed; LLM fallback disabled
    #    because a sphere run can carry ~100 distinct venues). Queries use
    #    normalized names: EasyScholar's CCF table keys on "CVPR", not
    #    "2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition" ──
    venue_rank_hits = 0
    if config.venue_rank_enabled and cands:
        venue_count: dict[str, int] = {}
        for n in cands:
            v = normalize_venue(n.venue)
            if v and not is_preprint_venue(v):
                venue_count[v] = venue_count.get(v, 0) + 1
        venues = sorted(venue_count, key=venue_count.get, reverse=True)[:100]
        if venues:
            try:
                from app.services.publication_rank import UnifiedRankClient

                async with UnifiedRankClient(use_llm=False) as rank_client:
                    results = await rank_client.query_batch(venues, concurrency=2)
                rank_map = {
                    v: r for v, r in zip(venues, results)
                    if r is not None and r.success
                }
                for n in cands:
                    r = rank_map.get(normalize_venue(n.venue))
                    if r:
                        n.sci_rank = (r.sci or "").strip()
                        n.ccf_rank = (r.ccf or "").strip()
                        if n.sci_rank or n.ccf_rank:
                            venue_rank_hits += 1
                logger.info(
                    f"[{paper_id}] sphere score: venue ranks for {venue_rank_hits} nodes "
                    f"({len(venues)} distinct venues queried)"
                )
            except Exception as e:
                logger.warning(f"[{paper_id}] sphere score: venue rank lookup failed: {e}")

    # ── Quality scores ──
    year_now = _current_year()
    for n in cands:
        compute_quality_score(n, config, year_now)

    # ── T3 hard gate: keyword-search-only hits need real-world evidence of
    #    quality before they may even reach the LLM relevance gate ──
    t3_dropped_nodes = [
        n for n in cands
        if n.tier >= 3
        and n.cited_by_count < config.t3_min_citations
        and not n.sci_rank
        and not n.ccf_rank
    ]
    _drop_nodes(sphere, {n.node_id for n in t3_dropped_nodes})

    # ── Gate cap: strongest provenance first, then quality ──
    cands = _candidates(sphere)
    gate_cap_dropped: list[SphereNode] = []
    if len(cands) > config.gate_cap:
        cands.sort(key=lambda n: (n.tier, -n.quality_score))
        gate_cap_dropped = cands[config.gate_cap:]
        keep = {n.node_id for n in cands[:config.gate_cap]}
        center_id = sphere.center_node.node_id if sphere.center_node else ""
        _drop_nodes(
            sphere,
            {nid for nid in sphere.nodes if nid != center_id and nid not in keep},
        )

    _funnel(
        sphere, "score",
        venue_rank_hits=venue_rank_hits,
        t3_dropped=len(t3_dropped_nodes),
        t3_dropped_top=_node_briefs(t3_dropped_nodes),
        gate_cap_dropped=len(gate_cap_dropped),
        gate_cap_dropped_top=_node_briefs(gate_cap_dropped),
        candidates=len(sphere.nodes) - 1,
    )
    logger.info(
        f"[{paper_id}] sphere score: t3_dropped={len(t3_dropped_nodes)} "
        f"remaining={len(sphere.nodes) - 1}"
    )
    await _emit_progress(run_id, "score_and_rank", "done")
    return sphere


# ──────────────────────────────────────────────────────────────────────
# Step 7: LLM relevance gate (Elicit-style per-paper judgement)
# ──────────────────────────────────────────────────────────────────────

_RELEVANCE_GATE_SYSTEM = """You are a strict research-literature triage assistant. You will be given a CENTER paper (title + abstract) and a numbered list of CANDIDATE papers. For each candidate, judge its true relationship to the center paper's research topic and lineage.

Relations:
- "foundation": prior work the center paper builds upon (inherited methods/ideas)
- "method_neighbor": same methodology family or adjacent technique
- "competitor": addresses the same problem/task, directly comparable
- "follow_up": extends, improves, analyzes, or scales the center paper's approach
- "application": mainly applies the center paper's technique inside a different domain-specific system
- "unrelated": only superficial keyword overlap — not meaningfully related

Relevance (how important for understanding the center paper's research landscape):
3 = essential, 2 = clearly relevant, 1 = marginal, 0 = irrelevant noise

Be harsh. A paper that merely reuses a common component (e.g. plugs an attention module into an unrelated domain pipeline) is "application" with relevance 1 at best; pure keyword coincidence is "unrelated" with relevance 0.

Return ONLY a JSON array, one entry per candidate:
[{"idx": 0, "relation": "foundation", "relevance": 3, "reason": "one short sentence"}]
No markdown fences, no explanation."""

_GATE_ZH_DIRECTIVE = """

输出语言要求:JSON 键名与 relation 取值保持英文;"reason" 字段用一句简体中文说明。仍然只返回合法 JSON 数组。"""

_VALID_RELATIONS = {r.value for r in RelationType} - {RelationType.UNKNOWN.value}


async def step_relevance_gate(
    sphere: SphereState,
    state: MainGraphState,
    run_id: str,
) -> SphereState:
    """LLM-judge every candidate's relation to the center paper; drop noise."""
    await _emit_progress(run_id, "relevance_gate", "running")

    paper_id = state["paper_id"]
    model = state.get("llm_model", "")
    language = state.get("language", "en")
    config = sphere.config
    center = sphere.center_node
    if not center:
        await _emit_progress(run_id, "relevance_gate", "done")
        return sphere

    cands = _candidates(sphere)
    if not cands:
        await _emit_progress(run_id, "relevance_gate", "done")
        return sphere

    system_prompt = _RELEVANCE_GATE_SYSTEM + (
        _GATE_ZH_DIRECTIVE if language == "zh" else ""
    )
    center_desc = f"CENTER PAPER:\nTitle: {center.title}"
    if center.year:
        center_desc += f"\nYear: {center.year}"
    if center.abstract_text:
        center_desc += f"\nAbstract: {center.abstract_text[:1200]}"

    llm = get_llm_service()
    batch_size = 20
    sem = asyncio.Semaphore(3)

    async def _judge_batch(batch: list[SphereNode]) -> None:
        lines: list[str] = []
        for i, n in enumerate(batch):
            line = f"[{i}] {n.title}"
            meta_bits = []
            if n.year:
                meta_bits.append(str(n.year))
            if n.venue:
                meta_bits.append(n.venue[:60])
            if n.cited_by_count:
                meta_bits.append(f"cited {n.cited_by_count}")
            if meta_bits:
                line += f" ({'; '.join(meta_bits)})"
            if n.abstract_text:
                line += f"\nAbstract: {n.abstract_text[:400]}"
            lines.append(line)

        user = (
            f"{center_desc}\n\nCANDIDATES ({len(batch)}):\n\n" + "\n\n".join(lines)
        )
        try:
            async with sem:
                resp = await llm.chat(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user},
                    ],
                    model=model, temperature=0.1, max_tokens=4096,
                )
            data = json.loads(_strip_json_fences(resp))
            if not isinstance(data, list):
                raise ValueError("gate response is not a list")
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                idx = entry.get("idx", -1)
                if not (0 <= idx < len(batch)):
                    continue
                node = batch[idx]
                relation = str(entry.get("relation", "")).strip().lower()
                if relation in _VALID_RELATIONS:
                    node.relation_type = RelationType(relation)
                try:
                    node.relevance = max(0, min(3, int(entry.get("relevance", 1))))
                except (TypeError, ValueError):
                    node.relevance = 1
                node.relation_reason = str(entry.get("reason", "")).strip()[:200]
        except Exception as e:
            if len(batch) > 5:
                mid = len(batch) // 2
                logger.warning(
                    f"[{paper_id}] sphere gate: batch of {len(batch)} failed, "
                    f"splitting and retrying — {e}"
                )
                await _judge_batch(batch[:mid])
                await _judge_batch(batch[mid:])
            else:
                logger.warning(f"[{paper_id}] sphere gate: batch failed ({len(batch)}): {e}")

    batches = [cands[i : i + batch_size] for i in range(0, len(cands), batch_size)]
    await asyncio.gather(*[_judge_batch(b) for b in batches])

    judged = [n for n in cands if n.relevance >= 0]
    dropped_nodes = [
        n for n in judged
        if n.relation_type == RelationType.UNRELATED or n.relevance < config.min_relevance
    ]
    to_drop = {n.node_id for n in dropped_nodes}
    survivors = len(cands) - len(to_drop)

    # Safety valve: if the gate nuked nearly everything (e.g. a misbehaving
    # model), keep the judged-but-low nodes rather than emptying the report.
    safety_valve = survivors < 10 and bool(to_drop)
    if safety_valve:
        logger.warning(
            f"[{paper_id}] sphere gate: only {survivors} survivors, "
            f"skipping drop of {len(to_drop)} to avoid an empty report"
        )
        to_drop = set()
        dropped_nodes = []

    _drop_nodes(sphere, to_drop)

    relation_hist: dict[str, int] = {}
    for n in judged:
        relation_hist[n.relation_type.value] = relation_hist.get(n.relation_type.value, 0) + 1
    _funnel(
        sphere, "gate",
        judged=len(judged),
        unjudged=len(cands) - len(judged),
        relations=relation_hist,
        dropped=len(dropped_nodes),
        dropped_top=_node_briefs(dropped_nodes),
        safety_valve=safety_valve,
        candidates=len(sphere.nodes) - 1,
    )
    logger.info(
        f"[{paper_id}] sphere gate: judged={len(judged)}/{len(cands)} "
        f"dropped={len(to_drop)} remaining={len(sphere.nodes) - 1}"
    )
    await _emit_progress(run_id, "relevance_gate", "done")
    return sphere


# ──────────────────────────────────────────────────────────────────────
# Step 8: Build similarity graph (coupling + candidate-level citations)
# ──────────────────────────────────────────────────────────────────────

async def step_build_similarity_graph(
    sphere: SphereState,
    state: MainGraphState,
    run_id: str,
) -> SphereState:
    """Connected Papers-style edges among candidates, PageRank, communities."""
    await _emit_progress(run_id, "build_similarity_graph", "running")

    from app.services.sphere_scorer import (
        build_similarity_edges,
        compute_pagerank,
        label_propagation,
    )

    paper_id = state["paper_id"]
    center = sphere.center_node

    sim_edges = build_similarity_edges(sphere.nodes)
    sphere.edges.extend(sim_edges)

    pagerank = compute_pagerank(sphere.nodes, sphere.edges)
    for nid, node in sphere.nodes.items():
        node.score_graph = pagerank.get(nid, 0.0)

    # Communities over candidates only (the center would glue everything
    # into one blob); used later as citation-structure-based theme seeds.
    center_id = center.node_id if center else ""
    cand_ids = [nid for nid in sphere.nodes if nid != center_id]
    communities = label_propagation(cand_ids, sim_edges)
    for nid, cid in communities.items():
        if nid in sphere.nodes:
            sphere.nodes[nid].cluster_id = cid

    n_coupling = sum(1 for e in sim_edges if e.edge_type == EdgeType.COUPLING)
    n_cites = len(sim_edges) - n_coupling
    logger.info(
        f"[{paper_id}] sphere graph: +{n_cites} candidate-cites, "
        f"+{n_coupling} coupling edges; "
        f"{len(set(communities.values()))} communities"
    )
    await _emit_progress(run_id, "build_similarity_graph", "done")
    return sphere


# ──────────────────────────────────────────────────────────────────────
# Step 9: Quota-based core set selection
# ──────────────────────────────────────────────────────────────────────

_PARTITION_ORDER = ["foundation", "method", "follow_up", "application", "frontier", "survey", "library"]


async def step_select_core_set(
    sphere: SphereState,
    state: MainGraphState,
    run_id: str,
) -> SphereState:
    """Assemble the core set per relation partition; build reason strings."""
    await _emit_progress(run_id, "select_core_set", "running")

    from app.services.sphere_scorer import combined_score

    paper_id = state["paper_id"]
    config = sphere.config
    center = sphere.center_node
    zh = state.get("language", "en") == "zh"

    cands = _candidates(sphere)
    for n in cands:
        combined_score(n)

    center_year = center.year if center else 0

    selected: list[str] = []
    selected_set: set[str] = set()
    partitions: dict[str, list[str]] = {k: [] for k in _PARTITION_ORDER}

    # Default ranking inside a bucket: combined score, then raw citations —
    # the tie-break matters because many landmark papers saturate the
    # citation component and would otherwise sort by dict insertion order.
    def _rank_key(n: SphereNode) -> tuple:
        return (n.score_total, n.cited_by_count)

    def _take(
        bucket: list[SphereNode], quota: int, partition: str, key=None,
    ) -> None:
        bucket.sort(key=key or _rank_key, reverse=True)
        for n in bucket:
            if len(partitions[partition]) >= quota or len(selected) >= config.core_cap:
                break
            if n.node_id in selected_set:
                continue
            selected.append(n.node_id)
            selected_set.add(n.node_id)
            partitions[partition].append(n.node_id)

    # Surveys are quarantined: they get their own small partition (bucket 6)
    #  and never compete for the relation-based seats — a survey judged
    # "method_neighbor" is still not a method paper.
    regular = [n for n in cands if not is_survey_title(n.title)]

    # 1. Foundations: what the center paper builds on
    _take(
        [n for n in regular if n.relation_type == RelationType.FOUNDATION],
        config.quota_foundation, "foundation",
    )
    # 2. Method neighbors & competitors
    _take(
        [n for n in regular if n.relation_type in (RelationType.METHOD_NEIGHBOR, RelationType.COMPETITOR)],
        config.quota_method, "method",
    )
    # 3. Follow-up work (S2 "influential" citers get a score nudge already)
    _take(
        [n for n in regular if n.relation_type == RelationType.FOLLOW_UP],
        config.quota_follow_up, "follow_up",
    )
    # 4. Cross-domain applications of the center paper's ideas — a real part
    #    of a foundational paper's story, but capped so they can't flood the
    #    report (nor masquerade as "method neighbors" via backfill)
    _take(
        [n for n in regular if n.relation_type == RelationType.APPLICATION and n.relevance >= 2],
        config.quota_application, "application",
    )
    # 5. Recent frontier: newest still-unselected clearly-relevant papers
    #    published after the center paper. Relative by design — an absolute
    #    year window stays empty for older center papers. The impact floor
    #    (citations or a ranked venue) keeps the "newest first" sort from
    #    being won by an arbitrary low-impact recent citer.
    _take(
        [
            n for n in regular
            if n.relevance >= 1
            and n.relation_type != RelationType.UNRELATED
            and (n.year > center_year if center_year else n.year > 0)
            and (
                n.cited_by_count >= config.frontier_min_citations
                or n.sci_rank or n.ccf_rank
            )
        ],
        config.quota_frontier, "frontier",
        key=lambda n: (n.year, n.score_total, n.cited_by_count),
    )
    # 6. Surveys: map-of-the-territory entry points, clearly relevant only
    _take(
        [n for n in cands if is_survey_title(n.title) and n.relevance >= 2],
        config.quota_survey, "survey",
    )
    # 7. User library hits
    _take(
        [n for n in cands if CandidateSource.LIBRARY in n.sources],
        config.quota_library, "library",
    )

    # Backfill remaining seats with the best of what's left (never unrelated
    # or surveys beyond their quota; application papers need relevance >= 2).
    backfilled = 0
    if len(selected) < config.core_cap:
        leftovers = [
            n for n in cands
            if n.node_id not in selected_set
            and n.relation_type != RelationType.UNRELATED
            and (n.relation_type != RelationType.APPLICATION or n.relevance >= 2)
            and not is_survey_title(n.title)
        ]
        leftovers.sort(key=_rank_key, reverse=True)
        for n in leftovers:
            if len(selected) >= config.core_cap:
                break
            partition = _partition_for(n, center_year, config)
            selected.append(n.node_id)
            selected_set.add(n.node_id)
            partitions[partition].append(n.node_id)
            backfilled += 1

    sphere.layer1_node_ids = selected
    for nid in selected:
        sphere.nodes[nid].layer = 1

    sphere.output.partitions = [
        CorePartition(key=k, node_ids=partitions[k])
        for k in _PARTITION_ORDER
        if partitions[k]
    ]

    # Layer 2: full-text parse candidates (DOI-bearing, best first)
    core_nodes = [sphere.nodes[nid] for nid in selected if nid in sphere.nodes]
    with_doi = [n for n in core_nodes if n.doi]
    with_doi.sort(key=lambda n: n.score_total, reverse=True)
    layer2_ids = [n.node_id for n in with_doi[:config.pdf_parse_cap]]
    sphere.layer2_node_ids = layer2_ids
    for nid in layer2_ids:
        sphere.nodes[nid].layer = 2

    # Reason strings (provenance + relation + quality), rendered in reports
    _build_reason_strings(sphere, zh)

    # Near misses: the strongest candidates that did NOT make the core set —
    # if a landmark paper is missing from the report, it shows up here.
    near_misses = [n for n in cands if n.node_id not in selected_set]
    near_misses.sort(key=_rank_key, reverse=True)
    _funnel(
        sphere, "select",
        core=len(selected),
        partitions={k: len(v) for k, v in partitions.items() if v},
        backfilled=backfilled,
        near_misses=len(near_misses),
        near_misses_top=[
            f"{n.title[:90]} ({n.year or '?'}, cited {n.cited_by_count}, "
            f"{n.relation_type.value}, rel {n.relevance}, score {n.score_total:.2f})"
            for n in near_misses[:10]
        ],
    )

    # Persist nodes + edges
    try:
        await _persist_sphere(sphere, run_id)
    except Exception as e:
        logger.warning(f"[{paper_id}] sphere select: DB persist failed: {e}")

    part_summary = {k: len(v) for k, v in partitions.items() if v}
    logger.info(
        f"[{paper_id}] sphere select: core={len(selected)} partitions={part_summary} "
        f"layer2={len(layer2_ids)}"
    )
    await _emit_progress(run_id, "select_core_set", "done")
    return sphere


def _partition_for(node: SphereNode, center_year: int, config: SphereConfig) -> str:
    """Best-fitting partition for a backfilled node."""
    if is_survey_title(node.title):
        return "survey"
    if node.relation_type == RelationType.FOUNDATION:
        return "foundation"
    if node.relation_type in (RelationType.METHOD_NEIGHBOR, RelationType.COMPETITOR):
        return "method"
    if node.relation_type == RelationType.FOLLOW_UP:
        return "follow_up"
    if node.relation_type == RelationType.APPLICATION:
        return "application"
    if CandidateSource.LIBRARY in node.sources:
        return "library"
    if (
        center_year and node.year > center_year
        and (
            node.cited_by_count >= config.frontier_min_citations
            or node.sci_rank or node.ccf_rank
        )
    ):
        # Same impact floor as the frontier quota — backfill must not smuggle
        # low-impact recent citers into the "frontier" section either
        return "frontier"
    return "method"


_RELATION_LABELS_ZH = {
    RelationType.FOUNDATION: "奠基",
    RelationType.METHOD_NEIGHBOR: "方法近邻",
    RelationType.COMPETITOR: "竞品",
    RelationType.FOLLOW_UP: "后续",
    RelationType.APPLICATION: "应用",
    RelationType.UNRELATED: "无关",
    RelationType.UNKNOWN: "未判定",
}


def _rank_badge(node: SphereNode) -> str:
    bits = []
    if node.sci_rank:
        bits.append(f"SCI-{node.sci_rank}")
    if node.ccf_rank:
        bits.append(f"CCF-{node.ccf_rank}")
    return "/".join(bits)


def _build_reason_strings(sphere: SphereState, zh: bool) -> None:
    lbl = (
        {"cited": "被引", "quality": "质量", "influential": "高影响引用",
         "fallback": "PDF 参考文献"}
        if zh
        else {"cited": "cited", "quality": "quality", "influential": "influential citation",
              "fallback": "PDF reference"}
    )
    center_id = sphere.center_node.node_id if sphere.center_node else ""
    for nid, node in sphere.nodes.items():
        if nid == center_id:
            continue
        parts: list[str] = [f"T{node.tier}"]
        if node.relation_type != RelationType.UNKNOWN:
            rel = (
                _RELATION_LABELS_ZH[node.relation_type] if zh
                else node.relation_type.value
            )
            rel_str = f"{rel}({node.relevance})" if node.relevance >= 0 else rel
            parts.append(rel_str)
        if node.cited_by_count > 0:
            parts.append(f"{lbl['cited']} {node.cited_by_count}")
        badge = _rank_badge(node)
        if badge:
            parts.append(badge)
        if node.influential:
            parts.append(lbl["influential"])
        if node.quality_score > 0:
            parts.append(f"{lbl['quality']} {node.quality_score:.2f}")
        node.reason = " | ".join(parts) if parts else lbl["fallback"]


async def _persist_sphere(sphere: SphereState, run_id: str) -> None:
    node_rows = [
        (
            n.node_id, run_id, n.doi, n.arxiv_id, n.openalex_id,
            n.s2_paper_id, n.title, n.year, n.venue, n.authors,
            n.abstract_text, n.cited_by_count, n.pdf_path,
            1 if n.mineru_parsed else 0,
            ",".join(s.value for s in n.sources) if n.sources else n.source.value,
            n.score_total, n.layer, n.cluster_id,
            n.tier, n.relation_type.value, n.relevance, n.relation_reason,
            n.quality_score, n.sci_rank, n.ccf_rank, 1 if n.influential else 0,
        )
        for n in sphere.nodes.values()
    ]
    await db.execute_many(
        """INSERT OR REPLACE INTO sphere_nodes
        (node_id, run_id, doi, arxiv_id, openalex_id, s2_paper_id,
         title, year, venue, authors, abstract_text, cited_by_count,
         pdf_path, mineru_parsed, source, score_total, layer, cluster_id,
         tier, relation_type, relevance, relation_reason,
         quality_score, sci_rank, ccf_rank, influential)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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


# ──────────────────────────────────────────────────────────────────────
# Step 10: Core-set abstract analysis
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


async def step_layer1_abstract_snap(
    sphere: SphereState,
    state: MainGraphState,
    run_id: str,
) -> SphereState:
    """Batch LLM analysis of abstracts for core-set nodes."""
    await _emit_progress(run_id, "layer1_abstract_snap", "running")

    paper_id = state["paper_id"]
    model = state.get("llm_model", "")

    # Collect core-set nodes with abstracts
    nodes_with_abstracts: list[tuple[int, SphereNode]] = []
    for nid in sphere.layer1_node_ids:
        node = sphere.nodes.get(nid)
        if node and node.abstract_text and len(node.abstract_text.strip()) > 30:
            nodes_with_abstracts.append((len(nodes_with_abstracts), node))

    if not nodes_with_abstracts:
        logger.info(f"[{paper_id}] sphere snap: no abstracts to analyze")
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
            extractions = json.loads(_strip_json_fences(response))
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
                    f"[{paper_id}] sphere snap: batch of {len(sub_batch)} failed, "
                    f"splitting into {mid}+{len(sub_batch)-mid} and retrying — {e}"
                )
                await _analyze_batch(sub_batch[:mid])
                await _analyze_batch(sub_batch[mid:])
            else:
                logger.warning(f"[{paper_id}] sphere snap: batch LLM failed ({len(sub_batch)} papers): {e}")

    for batch_start in range(0, len(nodes_with_abstracts), batch_size):
        batch = nodes_with_abstracts[batch_start : batch_start + batch_size]
        await _analyze_batch(batch)

    analyzed = sum(1 for nid in sphere.layer1_node_ids
                   if sphere.nodes.get(nid, SphereNode()).method_summary)
    logger.info(f"[{paper_id}] sphere snap: analyzed {analyzed}/{len(nodes_with_abstracts)} abstracts")
    await _emit_progress(run_id, "layer1_abstract_snap", "done")
    return sphere


# ──────────────────────────────────────────────────────────────────────
# Step 11: Download and MinerU parse
# ──────────────────────────────────────────────────────────────────────

async def step_download_and_parse(
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
            f"[{paper_id}] sphere download: skipped (pdf_parse_cap={sphere.config.pdf_parse_cap})"
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
            f"[{paper_id}] sphere download: no downloader credentials configured, skipping"
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
        logger.info(f"[{paper_id}] sphere download: no DOI-bearing layer2 nodes to download")
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
            logger.warning(f"[{paper_id}] sphere download: {node.doi} crashed: {result}")
            continue
        if result.ok and result.pdf_path:
            node.pdf_path = result.pdf_path
            ok_count += 1
        else:
            fail_count += 1
            logger.info(
                f"[{paper_id}] sphere download: {node.doi} failed via {result.source}: {result.detail}"
            )

    logger.info(
        f"[{paper_id}] sphere download: downloaded {ok_count}/{len(targets)} layer2 PDFs "
        f"(fail={fail_count}) → {refs_dir}"
    )
    await _emit_progress(run_id, "download_and_mineru_parse", "done")
    return sphere


# ──────────────────────────────────────────────────────────────────────
# Step 12: Synthesize landscape
# ──────────────────────────────────────────────────────────────────────

_CLUSTER_NAMER_SYSTEM = """You are a research theme namer. You will receive groups of papers that were clustered by citation-structure similarity (bibliographic coupling). For each group, produce a concise theme name and a 1-2 sentence definition based on the member papers.

Return ONLY a JSON array:
[{"group": 0, "name": "theme name", "definition": "1-2 sentence definition"}]
No markdown fences."""

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

_ADVISOR_SYSTEM = """You are a research advisor. Given relation partitions (foundations / method neighbors & competitors / follow-up work / recent frontier), theme clusters, a comparison matrix, and key hub papers — all centered on one paper — identify research gaps and suggest reading paths.

Return a JSON object:
{
  "overview": "2-3 paragraph landscape overview describing the research field around the center paper: its intellectual roots, main competing/adjacent approaches, how follow-up work evolved, and where the frontier is",
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


# Appended to the synthesis role prompts when the user requested Chinese.
# The JSON *keys* must stay English (the code parses them), so we only ask the
# model to write the natural-language *values* in Chinese — this avoids a full
# post-hoc translation pass over the assembled report.
_SPHERE_ZH_DIRECTIVE = """

输出语言要求:JSON 的所有键名(key)保持英文不变;但所有自然语言文本字段的值(如 name、definition、problem、assumption、method、dataset、metric、strength、weakness、title、description、overview 等)一律用简体中文撰写。论文标题、作者姓名、期刊/会议名称、以及专有技术术语可保留英文(术语首次出现可在括号内附中文解释)。仍然只返回合法 JSON,不要 markdown 代码围栏或任何解释。"""


async def step_synthesize_landscape(
    sphere: SphereState,
    state: MainGraphState,
    run_id: str,
) -> SphereState:
    """LLM synthesis over the core set: themes, comparison, gaps, paths."""
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

    # All synthesis runs ONLY on the core set
    paper_list: list[SphereNode] = [
        sphere.nodes[nid] for nid in sphere.layer1_node_ids if nid in sphere.nodes
    ]

    # Create indexed paper descriptions
    def _paper_desc(idx: int, node: SphereNode) -> str:
        parts = [f"[{idx}] {node.title}"]
        if node.year:
            parts.append(f"Year: {node.year}")
        if node.venue:
            venue_str = normalize_venue(node.venue)
            badge = _rank_badge(node)
            if badge:
                venue_str += f" ({badge})"
            parts.append(f"Venue: {venue_str}")
        if node.cited_by_count:
            parts.append(f"Citations: {node.cited_by_count}")
        if node.relation_type != RelationType.UNKNOWN:
            parts.append(f"Relation to center: {node.relation_type.value}")
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

    # ── Comparator inputs: only true comparables (competitors and method
    #    neighbors) — never applications of the center's technique, and never
    #    surveys (a survey has no method/dataset of its own to compare) ──
    comparables = [
        p for p in paper_list
        if p.relation_type in (RelationType.COMPETITOR, RelationType.METHOD_NEIGHBOR)
        and not _is_center_paper(center, p.title, p.doi)
        and not is_survey_title(p.title)
    ]
    comparables.sort(key=lambda n: (n.score_total, n.cited_by_count), reverse=True)
    if len(comparables) < 3:
        # Sparse gate results — fall back to the strongest core papers
        fallback = [
            p for p in paper_list
            if p not in comparables and not _is_center_paper(center, p.title, p.doi)
            and p.relation_type != RelationType.APPLICATION
            and not is_survey_title(p.title)
        ]
        fallback.sort(key=lambda n: (n.score_total, n.cited_by_count), reverse=True)
        comparables = (comparables + fallback)[:sphere.config.comparison_top_k]
    comparison_papers = comparables[:sphere.config.comparison_top_k]

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

    # ── Themes from citation-structure communities (Connected Papers idea);
    #    LLM only names them. Falls back to full LLM classification when the
    #    coupling graph is too sparse to form communities. ──
    async def _run_theme_builder() -> list[ThemeCluster]:
        groups: dict[int, list[SphereNode]] = {}
        for p in paper_list:
            if p.cluster_id >= 0:
                groups.setdefault(p.cluster_id, []).append(p)
        named_groups = [members for members in groups.values() if len(members) >= 3]

        if len(named_groups) >= 2:
            try:
                lines = []
                for gi, members in enumerate(named_groups):
                    titles = "; ".join(m.title[:80] for m in members[:8])
                    methods = "; ".join(
                        m.method_summary[:60] for m in members[:5] if m.method_summary
                    )
                    line = f"[group {gi}] papers: {titles}"
                    if methods:
                        line += f"\n  methods: {methods}"
                    lines.append(line)
                resp = await llm.chat(
                    [
                        {"role": "system", "content": _localize(_CLUSTER_NAMER_SYSTEM)},
                        {"role": "user", "content": "\n\n".join(lines)},
                    ],
                    model=model, temperature=0.2, max_tokens=2048,
                )
                data = json.loads(_strip_json_fences(resp))
                themes: list[ThemeCluster] = []
                if isinstance(data, list):
                    for entry in data:
                        gi = entry.get("group", -1)
                        if not (0 <= gi < len(named_groups)):
                            continue
                        members = named_groups[gi]
                        members.sort(key=lambda n: n.score_total, reverse=True)
                        themes.append(ThemeCluster(
                            name=entry.get("name", ""),
                            definition=entry.get("definition", ""),
                            representative_node_ids=[m.node_id for m in members],
                        ))
                if themes:
                    logger.info(
                        f"[{paper_id}] sphere synth: {len(themes)} themes from "
                        f"citation-structure communities"
                    )
                    return themes
            except Exception as e:
                logger.warning(f"[{paper_id}] sphere synth: cluster naming failed: {e}")

        # Fallback: LLM classification over the core set
        try:
            resp = await llm.chat(
                [
                    {"role": "system", "content": _localize(_CLASSIFIER_SYSTEM)},
                    {"role": "user", "content": f"Classify these {len(paper_list)} papers into themes:\n\n{papers_context}"},
                ],
                model=model, temperature=0.2, max_tokens=4096,
            )
            data = json.loads(_strip_json_fences(resp))
            themes = []
            for theme_data in data.get("themes", []):
                indices = theme_data.get("paper_indices", [])
                node_ids = [paper_list[i].node_id for i in indices if 0 <= i < len(paper_list)]
                themes.append(ThemeCluster(
                    name=theme_data.get("name", ""),
                    definition=theme_data.get("definition", ""),
                    representative_node_ids=node_ids,
                ))
            return themes
        except Exception as e:
            logger.warning(f"[{paper_id}] sphere synth: Classifier failed: {e}")
            return []

    # ── Comparator (async) ──
    async def _run_comparator() -> dict[str, Any] | None:
        try:
            comparator_messages = [
                {"role": "system", "content": _localize(_COMPARATOR_SYSTEM)},
                {"role": "user", "content": f"Compare these papers:\n\n{comp_context}"},
            ]
            comparator_resp = await llm.chat(comparator_messages, model=model, temperature=0.2, max_tokens=8192)
            return json.loads(_strip_json_fences(comparator_resp))
        except Exception as e:
            logger.warning(f"[{paper_id}] sphere synth: Comparator failed: {e}")
            return None

    themes, comparator_data = await asyncio.gather(
        _run_theme_builder(), _run_comparator(),
    )

    sphere.output.themes = themes

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

    # ── Timeline (core set + center only — no more garbage years) ──
    year_groups: dict[int, list[str]] = {}
    timeline_nodes = list(paper_list)
    if center.year > 0:
        timeline_nodes.append(center)
    for node in timeline_nodes:
        if node.year > 0:
            year_groups.setdefault(node.year, []).append(node.node_id)
    for year in sorted(year_groups.keys()):
        sphere.output.timeline.append(TimelineEntry(
            year=year,
            node_ids=year_groups[year],
        ))

    # ── Key hubs: PageRank over the *similarity graph* (candidate↔candidate
    #    citations + bibliographic coupling). Only meaningful when such edges
    #    exist; a pure star graph would make every node identical. ──
    center_id = center.node_id
    has_candidate_edges = any(
        e.edge_type == EdgeType.COUPLING
        or (e.edge_type == EdgeType.CITES and e.source_node_id != center_id and e.target_node_id != center_id)
        for e in sphere.edges
    )
    if has_candidate_edges:
        hub_candidates = [
            n for nid, n in sphere.nodes.items()
            if nid != center_id and n.score_graph > 0
        ]
        hub_candidates.sort(
            key=lambda n: (n.score_graph, n.cited_by_count), reverse=True
        )
        for node in hub_candidates[:10]:
            sphere.output.key_hubs.append(KeyHub(
                node_id=node.node_id,
                title=node.title,
                pagerank=node.score_graph,
                cited_by_count=node.cited_by_count,
                reason=node.reason,
            ))
    else:
        logger.info(
            f"[{paper_id}] sphere synth: no candidate-level edges — skipping key hubs"
        )

    # ── Research Advisor ──
    try:
        partition_desc = json.dumps(
            [
                {
                    "partition": p.key,
                    "papers": [
                        sphere.nodes[nid].title
                        for nid in p.node_ids[:12] if nid in sphere.nodes
                    ],
                }
                for p in sphere.output.partitions
            ],
            indent=2, ensure_ascii=False,
        )
        themes_desc = json.dumps(
            [{"name": t.name, "definition": t.definition, "papers": len(t.representative_node_ids)}
             for t in sphere.output.themes],
            indent=2, ensure_ascii=False,
        )
        hubs_desc = json.dumps(
            [{"title": h.title, "citations": h.cited_by_count, "pagerank": round(h.pagerank, 3)}
             for h in sphere.output.key_hubs],
            indent=2, ensure_ascii=False,
        )
        comp_desc = json.dumps(
            [{"title": r.title, "method": r.method, "strength": r.strength, "weakness": r.weakness}
             for r in sphere.output.comparison_table],
            indent=2, ensure_ascii=False,
        )

        advisor_context = (
            f"CENTER PAPER: {center.title} ({center.year})\n\n"
            f"RELATION PARTITIONS:\n{partition_desc}\n\n"
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
        advisor_data = json.loads(_strip_json_fences(advisor_resp))

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
        logger.warning(f"[{paper_id}] sphere synth: Advisor failed: {e}")

    logger.info(
        f"[{paper_id}] sphere synth: themes={len(sphere.output.themes)} "
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
# Step 13: Render output
# ──────────────────────────────────────────────────────────────────────

_PARTITION_TITLES = {
    "zh": {
        "foundation": "奠基与背景",
        "method": "方法近邻与竞品",
        "follow_up": "后续发展",
        "application": "跨域应用",
        "frontier": "最新前沿",
        "survey": "相关综述",
        "library": "你的文献库",
    },
    "en": {
        "foundation": "Foundations & Background",
        "method": "Method Neighbors & Competitors",
        "follow_up": "Follow-up Work",
        "application": "Cross-domain Applications",
        "frontier": "Recent Frontier",
        "survey": "Surveys & Reviews",
        "library": "From Your Library",
    },
}


async def step_render_output(
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
    lang = "zh" if zh else "en"

    # Localized scaffolding for the rendered report (the LLM fills the prose
    # fields; these are the fixed headings/labels the code emits).
    t = (
        {
            "overview_h": "## 研究全景概览\n",
            "core_h": "## 核心文献分区\n",
            "core_intro": "以下 {n} 篇论文经过来源分层、质量评估与相关性审查后入选核心集,按其与本文的关系分区:\n",
            "themes_h": "## 主题聚类\n",
            "themes_intro": "基于引文结构相似度(文献耦合)的社区划分:\n",
            "timeline_h": "## 脉络与时间线\n", "comparison_h": "## 对比矩阵\n\n",
            "comparison_cols": "| 论文 | 问题 | 方法 | 数据集 | 指标 | 优势 | 不足 |\n",
            "hubs_h": "## 关键枢纽论文\n", "gaps_h": "## 研究空白与想法\n",
            "paths_h": "## 推荐阅读路径\n", "cited_by": "被引", "more": "篇",
            "evidence_from": "证据来源", "path_suffix": "路径", "unknown": "[未知论文]",
            "library_h": "## 你的文献库中的相关工作\n",
            "library_intro": "以下论文来自你自己的知识库,与本文语义相关:\n",
            "center_marker": "(本文)",
        }
        if zh
        else {
            "overview_h": "## Research Landscape Overview\n",
            "core_h": "## Core Literature Partitions\n",
            "core_intro": "The following {n} papers passed provenance, quality, and relevance review, partitioned by their relation to this paper:\n",
            "themes_h": "## Thematic Clusters\n",
            "themes_intro": "Communities detected from citation-structure similarity (bibliographic coupling):\n",
            "timeline_h": "## Lineage & Timeline\n", "comparison_h": "## Comparison Matrix\n\n",
            "comparison_cols": "| Paper | Problem | Method | Dataset | Metric | Strength | Weakness |\n",
            "hubs_h": "## Key Hub Papers\n", "gaps_h": "## Research Gaps & Ideas\n",
            "paths_h": "## Suggested Reading Paths\n", "cited_by": "cited by", "more": "more",
            "evidence_from": "Evidence from", "path_suffix": "Path", "unknown": "[unknown paper]",
            "library_h": "## Related Work in Your Library\n",
            "library_intro": "These papers from your own knowledge base are semantically related to this paper:\n",
            "center_marker": "(this paper)",
        }
    )

    # Node lookup helper
    def _node_ref(node_id: str) -> str:
        node = sphere.nodes.get(node_id)
        if not node:
            return t["unknown"]
        year_str = f" ({node.year})" if node.year else ""
        return f"**{node.title}**{year_str}"

    def _node_line(node: SphereNode) -> str:
        """One bullet line: title (year) — venue [rank] — citations | reason."""
        bits: list[str] = []
        year_str = f" ({node.year})" if node.year else ""
        head = f"**{node.title}**{year_str}"
        if node.venue:
            venue = normalize_venue(node.venue)
            if len(venue) > 60:
                venue = venue[:57] + "…"
            badge = _rank_badge(node)
            if badge:
                venue += f" [{badge}]"
            bits.append(venue)
        if node.cited_by_count:
            bits.append(f"{t['cited_by']} {node.cited_by_count}")
        line = head
        if bits:
            line += " — " + " · ".join(bits)
        if node.relation_reason:
            line += f"\n  - {node.relation_reason}"
        return line

    md_parts: list[str] = []

    # ── Section 1: Overview ──
    md_parts.append(t["overview_h"])
    if output.sphere_overview:
        md_parts.append(output.sphere_overview + "\n")

    # ── Section 2: Core literature partitions ──
    if output.partitions:
        md_parts.append("\n" + t["core_h"])
        n_core = sum(len(p.node_ids) for p in output.partitions)
        md_parts.append(t["core_intro"].format(n=n_core) + "\n")
        titles = _PARTITION_TITLES[lang]
        for part in output.partitions:
            md_parts.append(f"### {titles.get(part.key, part.key)}\n")
            for nid in part.node_ids:
                node = sphere.nodes.get(nid)
                if node:
                    md_parts.append(f"- {_node_line(node)}\n")
            md_parts.append("\n")

    # ── Section 3: Themes ──
    if output.themes:
        md_parts.append(t["themes_h"])
        md_parts.append(t["themes_intro"])
        for i, theme in enumerate(output.themes, 1):
            md_parts.append(f"**{i}. {theme.name}**: {theme.definition}\n")
            for nid in theme.representative_node_ids[:8]:
                node = sphere.nodes.get(nid)
                if node:
                    year_str = f" ({node.year})" if node.year else ""
                    cite_str = f" — {t['cited_by']} {node.cited_by_count}" if node.cited_by_count else ""
                    md_parts.append(f"- {node.title}{year_str}{cite_str}\n")
            md_parts.append("\n")

    # ── Section 4: Timeline ──
    if output.timeline:
        md_parts.append(t["timeline_h"])
        center_id = center.node_id if center else ""
        for entry in output.timeline:
            nodes_in_year = [sphere.nodes.get(nid) for nid in entry.node_ids if sphere.nodes.get(nid)]
            if nodes_in_year:
                titles_strs = []
                for n in nodes_in_year[:5]:
                    label = n.title[:60]
                    if n.node_id == center_id:
                        label = f"**{label}** {t['center_marker']}"
                    titles_strs.append(label)
                extra = f" (+{len(nodes_in_year) - 5} {t['more']})" if len(nodes_in_year) > 5 else ""
                md_parts.append(f"- **{entry.year}**: {', '.join(titles_strs)}{extra}\n")
        md_parts.append("\n")

    # ── Section 5: Comparison Matrix ──
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

    # ── Section 6: Key Hubs ──
    if output.key_hubs:
        md_parts.append(t["hubs_h"])
        for hub in output.key_hubs:
            md_parts.append(
                f"- **{hub.title}** — PageRank: {hub.pagerank:.3f}, "
                f"{t['cited_by']}: {hub.cited_by_count}"
            )
            if hub.reason:
                md_parts.append(f" ({hub.reason})")
            md_parts.append("\n")
        md_parts.append("\n")

    # ── Section 7: Gaps & Ideas ──
    if output.gaps_and_ideas:
        md_parts.append(t["gaps_h"])
        for i, gap in enumerate(output.gaps_and_ideas, 1):
            md_parts.append(f"### {i}. {gap.title}\n\n")
            md_parts.append(f"{gap.description}\n\n")
            if gap.evidence_node_ids:
                evidence = [_node_ref(nid) for nid in gap.evidence_node_ids[:5]]
                md_parts.append(f"*{t['evidence_from']}*: {', '.join(evidence)}\n\n")

    # ── Section 8: Reading Paths ──
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

    # ── Section 9: Related work in the user's own library (Dify KB) ──
    if sphere.library_matches:
        md_parts.append(t["library_h"])
        md_parts.append(t["library_intro"])
        for m in sphere.library_matches[:15]:
            label = f"[{m.title}](/library?doc={m.document_id})" if m.document_id else m.title
            score = f" ({m.score:.2f})" if m.score and m.score > 0 else ""
            md_parts.append(f"- {label}{score}\n")
        md_parts.append("\n")

    markdown = "".join(md_parts)

    # ── Node dictionary for structured (non-markdown) renderers ──
    # The markdown above flattens every node into a prose bullet; the frontend
    # needs the same nodes as fields so it can render cards, badges and filters.
    # Only nodes actually referenced by the report are emitted, keeping the
    # payload proportional to the core set rather than to every candidate.
    referenced_ids: set[str] = set()
    for part in output.partitions:
        referenced_ids.update(part.node_ids)
    for theme in output.themes:
        referenced_ids.update(theme.representative_node_ids)
    for entry in output.timeline:
        referenced_ids.update(entry.node_ids)
    for hub in output.key_hubs:
        referenced_ids.add(hub.node_id)
    for row in output.comparison_table:
        referenced_ids.add(row.node_id)
    for gap in output.gaps_and_ideas:
        referenced_ids.update(gap.evidence_node_ids)
    for path in output.reading_paths:
        referenced_ids.update(path.node_ids)
    if center:
        referenced_ids.add(center.node_id)

    def _node_payload(node: SphereNode) -> dict[str, Any]:
        return {
            "node_id": node.node_id,
            "title": node.title,
            "year": node.year,
            "venue": normalize_venue(node.venue) if node.venue else "",
            "authors": node.authors,
            "doi": node.doi,
            "arxiv_id": node.arxiv_id,
            "openalex_id": node.openalex_id,
            "cited_by_count": node.cited_by_count,
            "sci_rank": node.sci_rank,
            "ccf_rank": node.ccf_rank,
            "tier": node.tier,
            "quality_score": round(node.quality_score, 3),
            "relevance": node.relevance,
            "relation_type": node.relation_type.value,
            "relation_reason": node.relation_reason,
            "reason": node.reason,
            "cluster_id": node.cluster_id,
            "influential": node.influential,
            "library_document_id": node.library_document_id,
            "method_summary": node.method_summary,
            "contribution": node.contribution,
            "task": node.task,
            "dataset": node.dataset,
        }

    nodes_payload = {
        nid: _node_payload(node)
        for nid in sorted(referenced_ids)
        if (node := sphere.nodes.get(nid)) is not None
    }

    # Build JSON output
    json_data = json.dumps({
        "mode": "sphere",
        "paper_id": paper_id,
        "title": center.title if center else "",
        "num_nodes": len(sphere.nodes),
        "num_edges": len(sphere.edges),
        "num_core": len(sphere.layer1_node_ids),
        "num_layer2": len(sphere.layer2_node_ids),
        "language": lang,
        "center_node_id": center.node_id if center else "",
        "nodes": nodes_payload,
        "partitions": [p.model_dump() for p in output.partitions],
        "num_library_matches": len(sphere.library_matches),
        "library_matches": [m.model_dump() for m in sphere.library_matches],
        "sphere_output": output.model_dump(),
        # Per-stage candidate funnel (counts + top dropped papers per stage);
        # answers "where did paper X disappear" without rerunning the pipeline.
        "debug": {"funnel": sphere.funnel},
    }, ensure_ascii=False)

    logger.info(f"[{paper_id}] sphere render: markdown={len(markdown)} chars")
    await _emit_progress(run_id, "render_output", "done")
    return markdown, json_data


# ──────────────────────────────────────────────────────────────────────
# Main orchestrator
# ──────────────────────────────────────────────────────────────────────

async def run_research_sphere(state: MainGraphState) -> dict[str, Any]:
    """Run the redesigned Research Sphere pipeline."""
    paper_id = state["paper_id"]
    run_id = state.get("run_id", "")
    t0 = time.perf_counter()

    settings = get_settings()
    config = SphereConfig(
        radius=settings.sphere_radius,
        candidate_cap=settings.sphere_candidate_cap,
        gate_cap=settings.sphere_gate_cap,
        core_cap=settings.sphere_core_cap,
        pdf_parse_cap=settings.sphere_pdf_parse_cap,
        t3_min_citations=settings.sphere_t3_min_citations,
        venue_rank_enabled=settings.sphere_venue_rank_enabled,
        frontier_min_citations=settings.sphere_frontier_min_citations,
    )
    sphere = SphereState(config=config)

    logger.info(f"[{paper_id}] sphere: Starting pipeline (config={config.model_dump()})")

    try:
        sphere = await step_init_from_pdf(sphere, state, run_id)
        sphere = await step_extract_core_metadata(sphere, state, run_id)
        sphere = await step_resolve_canonical_ids(sphere, state, run_id)
        sphere = await step_expand_graph_candidates(sphere, state, run_id)
        sphere = await step_enrich_candidates(sphere, state, run_id)
        sphere = await step_score_and_rank(sphere, state, run_id)
        sphere = await step_relevance_gate(sphere, state, run_id)
        sphere = await step_build_similarity_graph(sphere, state, run_id)
        sphere = await step_select_core_set(sphere, state, run_id)
        sphere = await step_layer1_abstract_snap(sphere, state, run_id)
        sphere = await step_download_and_parse(sphere, state, run_id)
        sphere = await step_synthesize_landscape(sphere, state, run_id)
        markdown, json_data = await step_render_output(sphere, state, run_id)
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
