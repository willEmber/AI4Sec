"""Where a paper was published, and how that venue ranks.

The paper's DOI and arXiv id are read from its first pages (GROBID's header
parse when that finds nothing and `GROBID_URL` is set); the venue and year come
from Crossref by DOI, then Semantic Scholar by DOI, arXiv id or title; the
venue's SCI / CCF rank from `publication_rank`. What is found is stored on the
`papers` row, so the next report on the same file asks nobody.

Semantic Scholar's `year` is when a paper first appeared, while its venue is
where it was published: a conference paper posted to arXiv first gets the
preprint's year (SinSR: 2023, at CVPR 2024). When S2 names the venue and lists
the published version's DOI, that DOI is kept and the year is Crossref's.
Rows stored before this are looked up again by `refresh_metadata`.

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
_REFERENCES_RE = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*|[A-Z])\.?\s+)?(?:references?|bibliography|参考文献)$", re.IGNORECASE
)


def _is_reference(block: Any) -> bool:
    """A block of the reference list, whose identifiers are other papers'.

    The CVF copy of Gaussian Shading has no arXiv stamp, and the scan used to
    take the arXiv id of its second reference: the paper was stored as ICLR
    2018, with that paper's rank.
    """
    if block.type == "ref_text":
        return True
    last = (block.section_path or "").rsplit("/", 1)[-1].strip()
    return bool(_REFERENCES_RE.match(last))


def _first_match(paper_ir_json: str, pattern: re.Pattern[str]) -> str:
    """The first match outside the references, the first four pages first."""
    from app.models.paper_ir import PaperIR

    ir = PaperIR.model_validate_json(paper_ir_json)
    blocks = [b for b in ir.blocks if not _is_reference(b)]
    early = sorted((b for b in blocks if b.page_idx <= 3), key=lambda b: (b.page_idx, b.order_idx))
    for block in [*early, *(b for b in blocks if b.page_idx > 3)]:
        m = pattern.search(block.text)
        if m:
            return m.group(1)
    return ""


async def _extract_doi_from_ir(paper_ir_json: str) -> str:
    """The paper's own DOI from its text, if printed (e.g. a journal's header)."""
    return _first_match(paper_ir_json, _DOI_RE).rstrip(".")


async def _extract_arxiv_id_from_ir(paper_ir_json: str) -> str:
    """The paper's own arXiv id (e.g. 2301.12345), usually its first-page stamp."""
    return _first_match(paper_ir_json, _ARXIV_RE)


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


_ARXIV_DOI_PREFIX = "10.48550/"


def _parse_s2_response(data: dict) -> dict[str, Any]:
    """Venue, year and the published version's DOI from an S2 paper record.

    arXiv's own DataCite DOI (`10.48550/arXiv.…`) is not a published version,
    and Crossref does not know it, so it is left out.
    """
    venue = ""
    pub_venue = data.get("publicationVenue") or {}
    if pub_venue.get("name"):
        venue = pub_venue["name"]
    if not venue:
        venue = data.get("venue", "")
    doi = str((data.get("externalIds") or {}).get("DOI") or "").strip()
    if doi.casefold().startswith(_ARXIV_DOI_PREFIX):
        doi = ""
    return {"venue": venue, "year": data.get("year", 0) or 0, "doi": doi}


async def _published_year(doi: str, fallback: int) -> int:
    """The year of the version `doi` names, by Crossref; `fallback` without one."""
    return (await _crossref_lookup(doi)).get("year", 0) or fallback


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
    paper_id: str,
    *,
    paper_ir_json: str = "",
    pdf_path: str = "",
    reuse_stored: bool = True,
    persist: bool = True,
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

    `reuse_stored=False` ignores what the row holds and `persist=False` writes
    nothing: together they say what a lookup finds today (`refresh_metadata`).
    """
    t0 = time.perf_counter()

    try:
        # 1. Check if paper already has full rank data (venue + at least one rank)
        existing = (
            await db.fetch_one(
                "SELECT doi, venue, year, sci_rank, ccf_rank FROM papers WHERE paper_id = ?",
                (paper_id,),
            )
            if reuse_stored
            else None
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
                    if not doi and s2.get("doi"):
                        # S2's year is the preprint's; the venue's is on its DOI.
                        doi = s2["doi"]
                        year = await _published_year(doi, year)
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
        if persist and (doi or venue or arxiv_id):
            await db.execute(
                "UPDATE papers SET doi = ?, venue = ?, year = ?, sci_rank = ?, ccf_rank = ? WHERE paper_id = ?",
                (doi, venue, year, sci_rank, ccf_rank, paper_id),
            )
            if venue:
                from app.services.paper_catalog import record_published_year

                await record_published_year(paper_id, doi=doi, year=year)

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


_FIELDS = ("doi", "venue", "year", "sci_rank", "ccf_rank")


async def refresh_metadata(paper_id: str, *, apply: bool = False) -> dict[str, Any] | None:
    """Look a stored venue record up again; the fields that changed.

    Rows whose venue came without a DOI are the ones that can be wrong and are
    never looked up again (a stored row is reused): a conference paper posted
    to arXiv first kept the preprint's year, and a paper without an arXiv
    stamp could carry a reference's venue. Returns `{field: (stored, found)}`
    for what differs (`{}` when nothing does), `None` when the row is not one
    of those or the lookup found no venue (a provider down): that changes
    nothing rather than blanking a row. With `apply` the new record is written
    and the paper's cached signals dropped, being computed from the old one.
    """
    row = await db.fetch_one(
        "SELECT doi, venue, year, sci_rank, ccf_rank FROM papers WHERE paper_id = ?", (paper_id,)
    )
    if not row or not row.get("venue") or row.get("doi"):
        return None
    from app.services.paper_ir_loader import load_paper_ir

    paper_ir = await load_paper_ir(paper_id)
    found = await enrich_metadata(
        paper_id, paper_ir_json=paper_ir.model_dump_json(), reuse_stored=False, persist=False
    )
    if not found.get("venue"):
        return None
    now = {
        "doi": found.get("doi") or "",
        "venue": found["venue"],
        "year": int(found.get("year") or 0),
        "sci_rank": found.get("sci") or "",
        "ccf_rank": found.get("ccf") or "",
    }
    was = {key: row.get(key) or ("" if key != "year" else 0) for key in _FIELDS}
    changed = {key: (was[key], now[key]) for key in _FIELDS if was[key] != now[key]}
    if changed and apply:
        async with db.transaction() as tx:
            await tx.execute(
                "UPDATE papers SET doi = ?, venue = ?, year = ?, sci_rank = ?, ccf_rank = ? "
                "WHERE paper_id = ?",
                (*(now[key] for key in _FIELDS), paper_id),
            )
            await tx.execute("DELETE FROM paper_signals WHERE paper_id = ?", (paper_id,))
        from app.services.paper_catalog import record_published_year

        await record_published_year(paper_id, doi=now["doi"], year=now["year"])
    return changed
