"""Where a paper was published, and how that venue ranks.

The paper's DOI and arXiv id are read from its first pages (GROBID's header
parse when that finds nothing and `GROBID_URL` is set); the venue and year come
from Crossref by DOI, then Semantic Scholar by DOI, arXiv id or title (continuing
past Crossref when Crossref returns a generic proceedings book series such as
Springer LNCS or PMLR); the venue's multi-system rank from `publication_rank`.
What is found is stored on the `papers` row, so the next report or turn on the
same file asks nobody.

Semantic Scholar's `year` is when a paper first appeared, while its venue is
where it was published: a conference paper posted to arXiv first gets the
preprint's year (SinSR: 2023, at CVPR 2024). When S2 names the venue and lists
the published version's DOI, that DOI is kept and the year is Crossref's.
Rows stored before this are looked up again by `refresh_metadata`.

Moved out of the upload pipeline's graph (P16), which was its only home; the
conversation's mode tools reach it through `paper_ir_loader.ensure_pub_rank`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

import httpx

from app.db import database as db
from app.services.publication_rank.models import VenueLookupInput
from app.services.publication_rank.normalizer import (
    clean_venue_text,
    extract_issns,
    is_preprint_venue,
    is_umbrella_series,
)

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
    """Query Crossref API for venue, year, ISSNs and conference event."""
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
            venue = clean_venue_text(container[0] or "")

        event_obj = data.get("event") or {}
        event_name = ""
        if isinstance(event_obj, dict):
            event_name = clean_venue_text(event_obj.get("acronym") or event_obj.get("name") or "")

        issns = extract_issns(data.get("ISSN") if isinstance(data.get("ISSN"), (list, str)) else None)

        year = 0
        for date_field in ("published-print", "published-online", "issued"):
            date_parts = data.get(date_field, {}).get("date-parts", [])
            if date_parts and date_parts[0] and date_parts[0][0]:
                year = int(date_parts[0][0])
                break

        out: dict[str, Any] = {"venue": venue, "year": year}
        if issns:
            out["issns"] = issns
        if event_name:
            out["event"] = event_name
        return out
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


def _parse_s2_venue_extras(data: dict) -> dict[str, Any]:
    pub_venue = data.get("publicationVenue") or {}
    ext = data.get("externalIds") or {}
    raw_issn = pub_venue.get("issn")
    return {
        "dblp_key": clean_venue_text(ext.get("DBLP") or ""),
        "issns": extract_issns(raw_issn if isinstance(raw_issn, (list, str)) else None),
        "aliases": [clean_venue_text(x) for x in (pub_venue.get("alternate_names") or []) if x],
    }


async def _published_year(doi: str, fallback: int) -> int:
    """The year of the version `doi` names, by Crossref; `fallback` without one."""
    return (await _crossref_lookup(doi)).get("year", 0) or fallback


async def _s2_lookup(doi: str = "", arxiv_id: str = "", title: str = "") -> dict[str, Any]:
    """Query Semantic Scholar for venue and year. Tries DOI -> arXiv -> title match."""
    fields = "title,venue,year,externalIds,publicationVenue"

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
                    payload = resp.json()
                    return {**_parse_s2_response(payload), **_parse_s2_venue_extras(payload)}
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
                    payload = resp.json()
                    return {**_parse_s2_response(payload), **_parse_s2_venue_extras(payload)}
            except Exception as e:
                logger.warning("S2 arXiv lookup failed: %s", e)

        if title:
            from app.services.paper_search import jaccard_similarity, normalize_whitespace

            clean_query = re.sub(r"([A-Za-z])-\s*([A-Za-z])", r"\1\2", title)
            try:
                await S2_LIMITER.wait()
                resp = await client.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search/match",
                    params={"query": clean_query[:200], "fields": fields},
                )
                if resp.status_code == 200:
                    matches = resp.json().get("data", [])
                    if matches:
                        candidate = matches[0]
                        cand_title = normalize_whitespace(candidate.get("title") or "")
                        if (
                            not cand_title
                            or jaccard_similarity(normalize_whitespace(clean_query), cand_title) >= 0.6
                            or jaccard_similarity(normalize_whitespace(title), cand_title) >= 0.6
                        ):
                            return {**_parse_s2_response(candidate), **_parse_s2_venue_extras(candidate)}
            except Exception as e:
                logger.warning("S2 title match failed: %s", e)

    return {}


_enrich_tasks: dict[str, asyncio.Task[dict[str, Any]]] = {}


def cancel_enrich_tasks() -> None:
    """Cancel in-flight singleflight tasks when shutting down the DB pool."""
    for task in list(_enrich_tasks.values()):
        if not task.done():
            task.cancel()
    _enrich_tasks.clear()


async def enrich_metadata(
    paper_id: str,
    *,
    paper_ir_json: str = "",
    pdf_path: str = "",
    reuse_stored: bool = True,
    persist: bool = True,
) -> dict[str, Any]:
    """The paper's venue record: `venue`, `year`, `sci`, `ccf`, `doi` (and
    `arxiv_id` and multi-system `ranks` when resolved).

    Concurrent default calls (`reuse_stored=True, persist=True`) for the same
    `paper_id` join one in-flight lookup so `parse_service`'s background task
    and `get_triage_signals` never double-query Crossref and Semantic Scholar.
    """
    if not (reuse_stored and persist):
        return await _enrich_metadata_once(
            paper_id,
            paper_ir_json=paper_ir_json,
            pdf_path=pdf_path,
            reuse_stored=reuse_stored,
            persist=persist,
        )

    loop = asyncio.get_running_loop()
    task = _enrich_tasks.get(paper_id)
    if task is None or task.done() or task.get_loop() is not loop:
        async def _run() -> dict[str, Any]:
            try:
                return await _enrich_metadata_once(
                    paper_id,
                    paper_ir_json=paper_ir_json,
                    pdf_path=pdf_path,
                    reuse_stored=True,
                    persist=True,
                )
            finally:
                if _enrich_tasks.get(paper_id) is task:
                    _enrich_tasks.pop(paper_id, None)

        task = asyncio.ensure_future(_run())
        _enrich_tasks[paper_id] = task
    return dict(await asyncio.shield(task))


async def _enrich_metadata_once(
    paper_id: str,
    *,
    paper_ir_json: str = "",
    pdf_path: str = "",
    reuse_stored: bool = True,
    persist: bool = True,
) -> dict[str, Any]:
    from app.services.publication_rank.paper_venue import (
        StoredVenue,
        _loads_extra,
        paper_venue_row,
        pub_rank_from_profile,
        venue_profile,
    )

    t0 = time.perf_counter()

    try:
        existing = None
        if reuse_stored:
            existing = await paper_venue_row(paper_id)
        if (
            existing
            and existing.get("venue")
            and not is_umbrella_series(existing["venue"])
        ):
            eff_doi = existing.get("effective_doi") or existing.get("doi") or ""
            eff_arxiv = existing.get("arxiv_id") or ""
            cached_prof = venue_profile(
                StoredVenue(
                    venue=existing["venue"],
                    sci_rank=existing.get("sci_rank") or "",
                    ccf_rank=existing.get("ccf_rank") or "",
                    extra=_loads_extra(existing.get("venue_ranks_json")),
                    doi=eff_doi,
                    arxiv_id=eff_arxiv,
                )
            )
            if (
                cached_prof.has_any_rank
                or cached_prof.subtrack in {"workshop", "companion", "findings"}
                or existing.get("sci_rank")
                or existing.get("ccf_rank")
            ):
                pub_rank = pub_rank_from_profile(
                    cached_prof,
                    raw_venue=existing["venue"],
                    year=int(existing.get("year") or 0),
                    doi=eff_doi,
                    arxiv_id=eff_arxiv,
                )
                logger.info(
                    "[%s] enrich_metadata: CACHED from DB (venue=%s sci=%s ccf=%s) in %.2fs",
                    paper_id, existing["venue"], pub_rank.get("sci") or "-", pub_rank.get("ccf") or "-",
                    time.perf_counter() - t0,
                )
                return pub_rank

        doi = ""
        arxiv_id = ""
        if existing and existing.get("doi"):
            doi = existing["doi"]
        if not doi and paper_ir_json:
            doi = await _extract_doi_from_ir(paper_ir_json)
        if paper_ir_json:
            arxiv_id = await _extract_arxiv_id_from_ir(paper_ir_json)

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

        venue = ""
        year = 0
        issns: list[str] = []
        crossref_event = ""
        dblp_key = ""
        venue_aliases: list[str] = []

        if existing and existing.get("venue") and not is_umbrella_series(existing["venue"]):
            venue = existing["venue"]
            year = existing.get("year", 0) or 0
            logger.info("[%s] enrich_metadata: venue=%s (cached from DB)", paper_id, venue)
        else:
            if doi:
                crossref = await _crossref_lookup(doi)
                venue = crossref.get("venue", "")
                year = crossref.get("year", 0)
                issns.extend(crossref.get("issns") or [])
                crossref_event = crossref.get("event", "")
                if venue:
                    logger.info("[%s] enrich_metadata: Crossref -> venue=%s year=%d", paper_id, venue, year)

            # Continue to S2 when Crossref returned nothing OR an umbrella proceedings series (LNCS / PMLR)
            if not venue or is_umbrella_series(venue) or is_preprint_venue(venue):
                title = ""
                if paper_ir_json:
                    from app.models.paper_ir import PaperIR

                    ir = PaperIR.model_validate_json(paper_ir_json)
                    title = ir.title
                s2 = await _s2_lookup(doi=doi, arxiv_id=arxiv_id, title=title)
                for i in s2.get("issns") or []:
                    if i not in issns:
                        issns.append(i)
                dblp_key = s2.get("dblp_key") or ""
                venue_aliases.extend(s2.get("aliases") or [])
                if s2.get("venue") and not is_umbrella_series(s2["venue"]):
                    venue = s2["venue"]
                    year = s2.get("year", 0) or year
                    if not doi and s2.get("doi"):
                        doi = s2["doi"]
                        year = await _published_year(doi, year)
                    logger.info("[%s] enrich_metadata: S2 -> venue=%s year=%d", paper_id, venue, year)
                elif s2.get("year"):
                    year = s2["year"] or year

            if not venue and grobid_venue:
                venue = grobid_venue
                year = year or grobid_year
                logger.info("[%s] enrich_metadata: GROBID -> venue=%s year=%d", paper_id, venue, year)

            if not venue:
                logger.warning("[%s] enrich_metadata: No venue found from any source", paper_id)

        sci_rank = ""
        ccf_rank = ""
        venue_summary: dict[str, Any] = {}
        if venue or dblp_key or crossref_event or issns:
            try:
                logger.info("[%s] enrich_metadata: Querying publication_rank for '%s'...", paper_id, venue)
                from app.services.publication_rank import UnifiedRankClient

                async with UnifiedRankClient() as rank_client:
                    if hasattr(rank_client, "resolve_profile"):
                        lookup = VenueLookupInput(
                            venue=venue,
                            venue_aliases=venue_aliases,
                            issns=issns,
                            dblp_key=dblp_key,
                            crossref_event=crossref_event,
                            is_preprint_source=bool(arxiv_id and is_preprint_venue(venue)),
                        )
                        profile = await rank_client.resolve_profile(lookup, allow_network=True)
                        if profile.has_any_rank and profile.display_venue and (
                            not venue or is_umbrella_series(venue) or is_preprint_venue(venue)
                        ):
                            venue = profile.display_venue
                        sci_rank = profile.sci_rank
                        ccf_rank = profile.ccf_rank
                        venue_summary = profile.agent_summary()
                    else:
                        result = await rank_client.query(venue)
                        if result.success:
                            sci_rank = result.sci or ""
                            ccf_rank = result.ccf or ""
            except Exception as e:
                logger.warning("[%s] enrich_metadata: publication_rank import/call error: %s", paper_id, e, exc_info=True)
        else:
            logger.info("[%s] enrich_metadata: No venue found, skipping publication_rank query", paper_id)

        if persist and (doi or venue or arxiv_id):
            await db.execute(
                "UPDATE papers SET doi = ?, venue = ?, year = ?, sci_rank = ?, ccf_rank = ? WHERE paper_id = ?",
                (doi, venue, year, sci_rank, ccf_rank, paper_id),
            )
            if venue_summary:
                try:
                    ranks_json = json.dumps(venue_summary, ensure_ascii=False)
                    await db.execute(
                        "UPDATE papers SET venue_ranks_json = ? WHERE paper_id = ?",
                        (ranks_json, paper_id),
                    )
                except Exception:
                    pass
            if venue:
                from app.services.paper_catalog import (
                    record_published_venue_ranks,
                    record_published_year,
                )

                await record_published_year(paper_id, doi=doi, year=year)
                try:
                    await record_published_venue_ranks(
                        paper_id, doi=doi, venue=venue, venue_ranks=venue_summary or None
                    )
                except Exception:
                    pass

        resolved_prof = venue_profile(
            StoredVenue(
                venue=venue,
                sci_rank=sci_rank,
                ccf_rank=ccf_rank,
                extra=venue_summary,
                doi=doi,
                arxiv_id=arxiv_id,
            )
        )
        pub_rank = pub_rank_from_profile(
            resolved_prof,
            raw_venue=venue,
            year=year,
            doi=doi,
            arxiv_id=arxiv_id,
        )
        elapsed = time.perf_counter() - t0
        logger.info(
            "[%s] enrich_metadata: DONE in %.1fs — venue=%s year=%d sci=%s ccf=%s",
            paper_id, elapsed, venue or "(none)", year, sci_rank or "-", ccf_rank or "-",
        )
        return pub_rank

    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.warning("[%s] enrich_metadata: FAILED in %.1fs — %s (non-blocking)", paper_id, elapsed, e)
        return {}


_FIELDS = ("doi", "venue", "year", "sci_rank", "ccf_rank")


async def refresh_metadata(paper_id: str, *, apply: bool = False) -> dict[str, Any] | None:
    """Look a stored venue record up again; returns `{field: (stored, found)}`."""
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
