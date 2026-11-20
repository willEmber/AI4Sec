"""Where a paper was published, and how that venue ranks.

The paper's DOI and arXiv id are read from its first pages (GROBID's header
parse when that finds nothing and `GROBID_URL` is set); the venue and year come
from Crossref by DOI, then Semantic Scholar by DOI, arXiv id or title; the
venue's SCI / CCF rank from `publication_rank`. What is found is stored on the
`papers` row, so the next report on the same file asks nobody.

Moved out of the upload pipeline's graph (P16), which was its only home; the
conversation's mode tools reach it through `paper_ir_loader.ensure_pub_rank`.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import httpx

from app.db import database as db

logger = logging.getLogger("scholar.paper_metadata")


_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s,;\"')\]>]+)", re.ASCII)
_ARXIV_RE = re.compile(r"arXiv[:\s]*(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)


async def _extract_doi_from_ir(paper_ir_json: str) -> str:
    """Extract DOI from paper text blocks (prefer first few pages)."""
    from app.models.paper_ir import PaperIR

    ir = PaperIR.model_validate_json(paper_ir_json)
    # Search text blocks from early pages first
    for block in sorted(ir.blocks, key=lambda b: (b.page_idx, b.order_idx)):
        if block.page_idx > 3:
            break
        m = _DOI_RE.search(block.text)
        if m:
            doi = m.group(1).rstrip(".")
            return doi
    # Fallback: scan all blocks
    for block in ir.blocks:
        m = _DOI_RE.search(block.text)
        if m:
            doi = m.group(1).rstrip(".")
            return doi
    return ""


async def _extract_arxiv_id_from_ir(paper_ir_json: str) -> str:
    """Extract arXiv ID (e.g. 2301.12345) from paper text blocks."""
    from app.models.paper_ir import PaperIR

    ir = PaperIR.model_validate_json(paper_ir_json)
    for block in sorted(ir.blocks, key=lambda b: (b.page_idx, b.order_idx)):
        if block.page_idx > 3:
            break
        m = _ARXIV_RE.search(block.text)
        if m:
            return m.group(1)
    for block in ir.blocks:
        m = _ARXIV_RE.search(block.text)
        if m:
            return m.group(1)
    return ""


async def _crossref_lookup(doi: str) -> dict[str, Any]:
    """Query Crossref API for venue and year. Returns {"venue": ..., "year": ...}."""
    if not doi:
        return {}
    url = f"https://api.crossref.org/works/{doi}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers={"User-Agent": "ScholarApp/1.0 (mailto:scholar@example.com)"})
            resp.raise_for_status()
            data = resp.json().get("message", {})

        venue = ""
        container = data.get("container-title", [])
        if container:
            venue = container[0]

        year = 0
        for date_field in ("published-print", "published-online", "issued"):
            date_parts = data.get(date_field, {}).get("date-parts", [])
            if date_parts and date_parts[0] and date_parts[0][0]:
                year = int(date_parts[0][0])
                break

        return {"venue": venue, "year": year}
    except Exception as e:
        logger.warning("Crossref lookup failed for DOI %s: %s", doi, e)
        return {}


def _parse_s2_response(data: dict) -> dict[str, Any]:
    """Extract venue + year from Semantic Scholar paper record."""
    venue = ""
    pub_venue = data.get("publicationVenue") or {}
    if pub_venue.get("name"):
        venue = pub_venue["name"]
    if not venue:
        venue = data.get("venue", "")
    return {"venue": venue, "year": data.get("year", 0)}


async def _s2_lookup(doi: str = "", arxiv_id: str = "", title: str = "") -> dict[str, Any]:
    """Query Semantic Scholar for venue and year. Tries DOI -> arXiv -> title match."""
    fields = "venue,year,externalIds,publicationVenue"

    from app.services.paper_search.credentials import s2_headers
    from app.services.paper_search.ratelimit import S2_LIMITER

    async with httpx.AsyncClient(timeout=15.0, headers=s2_headers()) as client:
        if doi:
            try:
                await S2_LIMITER.wait()
                resp = await client.get(
                    f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
                    params={"fields": fields},
                )
                if resp.status_code == 200:
                    return _parse_s2_response(resp.json())
            except Exception as e:
                logger.warning("S2 DOI lookup failed: %s", e)

        if arxiv_id:
            try:
                await S2_LIMITER.wait()
                resp = await client.get(
                    f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_id}",
                    params={"fields": fields},
                )
                if resp.status_code == 200:
                    return _parse_s2_response(resp.json())
            except Exception as e:
                logger.warning("S2 arXiv lookup failed: %s", e)

        if title:
            try:
                await S2_LIMITER.wait()
                resp = await client.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search/match",
                    params={"query": title[:200], "fields": fields},
                )
                if resp.status_code == 200:
                    matches = resp.json().get("data", [])
                    if matches:
                        return _parse_s2_response(matches[0])
            except Exception as e:
                logger.warning("S2 title match failed: %s", e)

    return {}


async def enrich_metadata(
    paper_id: str, *, paper_ir_json: str = "", pdf_path: str = ""
) -> dict[str, Any]:
    """The paper's venue record: `venue`, `year`, `sci`, `ccf`, `doi` (and
    `arxiv_id` when it was looked up). Extracts DOI/arXiv from the paper text,
    queries Crossref → Semantic Scholar for the venue, then publication_rank for
    SCI/CCF.

    Multi-source fallback chain:
      Level 1: DOI → Crossref
      Level 2: DOI/arXiv → Semantic Scholar
      Level 3: Title → Semantic Scholar /paper/search/match

    Fault-tolerant: any failure logs a warning and returns `{}`; a venue is a
    nicety for a report, never a reason to fail one.
    """
    t0 = time.perf_counter()

    try:
        # 1. Check if paper already has full rank data (venue + at least one rank)
        existing = await db.fetch_one(
            "SELECT doi, venue, year, sci_rank, ccf_rank FROM papers WHERE paper_id = ?",
            (paper_id,),
        )
        if existing and existing.get("venue") and (existing.get("sci_rank") or existing.get("ccf_rank")):
            pub_rank = {
                "venue": existing["venue"],
                "year": existing.get("year", 0) or 0,
                "sci": existing.get("sci_rank", ""),
                "ccf": existing.get("ccf_rank", ""),
                # Carried so downstream nodes (Snap's signal lookup) can resolve
                # the paper in OpenAlex/S2 by DOI instead of by title.
                "doi": existing.get("doi", "") or "",
            }
            logger.info("[%s] enrich_metadata: CACHED from DB (venue=%s sci=%s ccf=%s) in %.2fs",
                         paper_id, existing["venue"], existing.get("sci_rank", "-"), existing.get("ccf_rank", "-"),
                         time.perf_counter() - t0)
            return pub_rank

        # 2. Extract DOI + arXiv ID from paper text (or reuse DOI from DB)
        doi = ""
        arxiv_id = ""
        if existing and existing.get("doi"):
            doi = existing["doi"]
        if not doi and paper_ir_json:
            doi = await _extract_doi_from_ir(paper_ir_json)
        if paper_ir_json:
            arxiv_id = await _extract_arxiv_id_from_ir(paper_ir_json)

        # 2b. GROBID fallback (optional, off unless GROBID_URL is set). The regex
        # scan above misses DOIs in running footers, scanned pages and layouts
        # whose reading order splits the header; GROBID parses the header as a
        # structure instead. Only consulted when the cheap path came up empty.
        grobid_venue = ""
        grobid_year = 0
        if not doi and not arxiv_id and pdf_path:
            from app.services.grobid_client import extract_header

            header = await extract_header(pdf_path)
            if header.available:
                doi = doi or header.doi
                arxiv_id = arxiv_id or header.arxiv_id
                grobid_venue = header.venue
                grobid_year = header.year
                if doi or arxiv_id:
                    logger.info(
                        "[%s] enrich_metadata: GROBID recovered doi=%s arxiv=%s",
                        paper_id, doi or "-", arxiv_id or "-",
                    )

        logger.info("[%s] enrich_metadata: DOI=%s arXiv=%s", paper_id, doi or "-", arxiv_id or "-")

        # 3. Multi-source venue + year lookup
        venue = ""
        year = 0

        # Reuse venue from DB if already stored (skip external queries for venue)
        if existing and existing.get("venue"):
            venue = existing["venue"]
            year = existing.get("year", 0) or 0
            logger.info("[%s] enrich_metadata: venue=%s (cached from DB)", paper_id, venue)
        else:
            # Level 1: Crossref (by DOI) — most authoritative for journals
            if doi:
                crossref = await _crossref_lookup(doi)
                venue = crossref.get("venue", "")
                year = crossref.get("year", 0)
                if venue:
                    logger.info("[%s] enrich_metadata: Crossref -> venue=%s year=%d", paper_id, venue, year)

            # Level 2+3: Semantic Scholar (by DOI/arXiv/title) — stronger for conferences
            if not venue:
                title = ""
                if paper_ir_json:
                    from app.models.paper_ir import PaperIR
                    ir = PaperIR.model_validate_json(paper_ir_json)
                    title = ir.title
                s2 = await _s2_lookup(doi=doi, arxiv_id=arxiv_id, title=title)
                if s2.get("venue"):
                    venue = s2["venue"]
                    year = s2.get("year", 0) or year
                    logger.info("[%s] enrich_metadata: S2 -> venue=%s year=%d", paper_id, venue, year)
                elif s2.get("year"):
                    year = s2["year"] or year

            # Level 4: GROBID's own reading of the header, if it ran.
            if not venue and grobid_venue:
                venue = grobid_venue
                year = year or grobid_year
                logger.info("[%s] enrich_metadata: GROBID -> venue=%s year=%d", paper_id, venue, year)

            if not venue:
                logger.warning("[%s] enrich_metadata: No venue found from any source", paper_id)

        # 4. Query publication_rank for SCI/CCF
        sci_rank = ""
        ccf_rank = ""
        if venue:
            try:
                logger.info("[%s] enrich_metadata: Querying publication_rank for '%s'...", paper_id, venue)
                from app.services.publication_rank import UnifiedRankClient
                async with UnifiedRankClient() as rank_client:
                    result = await rank_client.query(venue)
                    logger.info("[%s] enrich_metadata: publication_rank result: success=%s sci=%s ccf=%s error=%s",
                                 paper_id, result.success, result.sci, result.ccf, result.error)
                    if result.success:
                        sci_rank = result.sci or ""
                        ccf_rank = result.ccf or ""
                    else:
                        logger.warning("[%s] enrich_metadata: publication_rank query failed for '%s': %s", paper_id, venue, result.error)
            except Exception as e:
                logger.warning("[%s] enrich_metadata: publication_rank import/call error: %s", paper_id, e, exc_info=True)
        else:
            logger.info("[%s] enrich_metadata: No venue found, skipping publication_rank query", paper_id)

        # 5. Persist to DB
        if doi or venue or arxiv_id:
            await db.execute(
                "UPDATE papers SET doi = ?, venue = ?, year = ?, sci_rank = ?, ccf_rank = ? WHERE paper_id = ?",
                (doi, venue, year, sci_rank, ccf_rank, paper_id),
            )

        pub_rank = {
            "venue": venue,
            "year": year,
            "sci": sci_rank,
            "ccf": ccf_rank,
            "doi": doi,
            "arxiv_id": arxiv_id,
        }
        elapsed = time.perf_counter() - t0
        logger.info("[%s] enrich_metadata: DONE in %.1fs — venue=%s year=%d sci=%s ccf=%s",
                     paper_id, elapsed, venue or "(none)", year, sci_rank or "-", ccf_rank or "-")
        return pub_rank

    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.warning("[%s] enrich_metadata: FAILED in %.1fs — %s (non-blocking)", paper_id, elapsed, e)
        return {}
