"""Single deterministic read-time venue & tier resolver (P18 §2.1).

Every surface that reads a stored paper or work — session prompt badges,
`load_pub_rank`, Insight Snap triage signals, Research Sphere literature maps,
and frozen report citations — goes through :func:`venue_profile` so one paper
never reports two different tiers across the same turn.

The in-process catalog (`get_venue_catalog`, <0.1 ms) is always applied on
read; cached signals and cached literature maps therefore pick up catalog
updates immediately without cache invalidation, while saved reports freeze the
resolved summary at snapshot time.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .catalog import get_venue_catalog
from .models import RankingEntry, VenueLookupInput, VenueProfile
from .normalizer import is_preprint_venue, is_umbrella_series

logger = logging.getLogger("scholar.publication_rank.paper_venue")

_PAREN_RE = re.compile(r"\(([^()]+)\)\s*$")
_IF_RE = re.compile(r"^IF\s+(.+)$", re.IGNORECASE)
_YEAR_RE = re.compile(r"^\d{4}$")


@dataclass(frozen=True)
class StoredVenue:
    """Stored bibliographic columns of a `papers` or `literature_items` row."""

    venue: str = ""
    sci_rank: str = ""
    ccf_rank: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)
    doi: str = ""
    arxiv_id: str = ""


def parse_compact_rank(
    sys_name: str,
    raw_label: str,
    *,
    source: str,
    verified: bool,
) -> RankingEntry | None:
    """Parse a `compact_ranks()` value (e.g. `'A (2026)'`, `'Q1 (IF 10.2)'`, `'1区 Top (2025)'`)."""
    text = (raw_label or "").strip()
    if not text:
        return None
    edition_year = ""
    impact_factor = ""
    while m := _PAREN_RE.search(text):
        inner = m.group(1).strip()
        if m_if := _IF_RE.match(inner):
            impact_factor = m_if.group(1).strip()
        elif _YEAR_RE.match(inner):
            edition_year = inner
        else:
            break
        text = text[: m.start()].strip()

    top = False
    if sys_name == "CAS" and re.search(r"\btop\b", text, re.IGNORECASE):
        top = True
        text = re.sub(r"\s*\btop\b\s*", " ", text, flags=re.IGNORECASE).strip()
    if not text:
        return None
    return RankingEntry(
        ranking_system=sys_name,
        rank=text,
        edition_year=edition_year,
        top=top,
        impact_factor=impact_factor,
        source=source,
        verified=verified,
    )


def _merge_warnings(cat_warn: str, stored_warn: str) -> str:
    cw = (cat_warn or "").strip()
    sw = (stored_warn or "").strip()
    if not cw:
        return sw
    if not sw or sw in cw:
        return cw
    if cw in sw:
        return sw
    return f"{cw}; {sw}"


def venue_profile(stored: StoredVenue) -> VenueProfile:
    """Resolve one canonical `VenueProfile` by P18 §2.1's 6-rule priority."""
    extra = dict(stored.extra) if isinstance(stored.extra, Mapping) else {}
    raw_venue = (stored.venue or "").strip()
    canonical_stored = str(extra.get("canonical_venue") or "").strip()
    acronym_stored = str(extra.get("acronym") or "").strip()

    # Rule 6: preprint PDF flag
    is_preprint_pdf = bool(
        extra.get("preprint_file_only") or (stored.arxiv_id and not stored.doi)
    )

    aliases: list[str] = []
    if canonical_stored and canonical_stored != raw_venue:
        aliases.append(canonical_stored)
    if acronym_stored and acronym_stored not in (raw_venue, canonical_stored):
        aliases.append(acronym_stored)

    lookup_venue = raw_venue or canonical_stored
    cat_prof = get_venue_catalog().resolve(
        VenueLookupInput(
            venue=lookup_venue,
            venue_aliases=aliases,
            is_preprint_source=bool(
                is_preprint_pdf and (not lookup_venue or is_preprint_venue(lookup_venue))
            ),
        )
    )

    stored_sub = str(extra.get("subtrack") or "").strip().lower()
    stored_warn = str(extra.get("cas_warning") or "").strip()
    pres_tier = str(extra.get("presentation_tier") or "").strip()
    stored_note = str(extra.get("note") or "").strip()
    warning = _merge_warnings(cat_prof.warning, stored_warn)

    # Rule 1: Subtrack guard comes first. Workshop / Companion clears ALL tiers,
    # including stale main-venue `sci_rank` / `ccf_rank` stored on older rows.
    if cat_prof.subtrack in {"workshop", "companion"} or stored_sub in {"workshop", "companion"}:
        eff_sub = (
            cat_prof.subtrack
            if cat_prof.subtrack in {"workshop", "companion"}
            else stored_sub
        )
        return VenueProfile(
            query_venue=raw_venue or lookup_venue,
            canonical_name=cat_prof.canonical_name or canonical_stored or raw_venue,
            acronym=cat_prof.acronym or acronym_stored,
            venue_type=cat_prof.venue_type or str(extra.get("venue_type") or "conference"),
            subtrack=eff_sub,
            peer_reviewed=None,
            is_preprint_file=is_preprint_pdf,
            rankings=[],
            warning=warning,
            source="local_catalog",
            verification_status="verified",
        )

    if cat_prof.subtrack == "findings" or stored_sub == "findings":
        base_name = cat_prof.canonical_name or canonical_stored or raw_venue
        if base_name and "findings" not in base_name.lower():
            base_name = f"Findings of {base_name}"
        return VenueProfile(
            query_venue=raw_venue or lookup_venue,
            canonical_name=base_name,
            acronym=cat_prof.acronym or acronym_stored,
            venue_type=cat_prof.venue_type or "conference",
            subtrack="findings",
            peer_reviewed=True,
            is_preprint_file=is_preprint_pdf,
            openreview_tier=cat_prof.openreview_tier or pres_tier,
            rankings=[],
            warning=warning,
            note=cat_prof.note
            or stored_note
            or "Peer-reviewed Findings track (distinct from main conference)",
            source="local_catalog",
            verification_status="verified",
        )

    # Rule 2: Systems covered by the authoritative in-process catalog win.
    merged: list[RankingEntry] = list(cat_prof.rankings)
    covered = {r.ranking_system for r in merged}

    if stored.ccf_rank and cat_prof.ccf_rank and stored.ccf_rank.strip().upper() != cat_prof.ccf_rank:
        logger.debug(
            "venue_profile(%r): catalog CCF %s overrides stored %s",
            lookup_venue, cat_prof.ccf_rank, stored.ccf_rank,
        )
    if stored.sci_rank and cat_prof.sci_rank and stored.sci_rank.strip().upper() != cat_prof.sci_rank:
        logger.debug(
            "venue_profile(%r): catalog JCR %s overrides stored %s",
            lookup_venue, cat_prof.sci_rank, stored.sci_rank,
        )

    # Rule 3: Systems NOT in the catalog fall back to `venue_ranks_json.ranks`.
    stored_src = str(extra.get("source") or "").strip().lower()
    stored_ver_status = str(extra.get("verification_status") or "").strip().lower()
    per_sys_src = extra.get("rank_sources") if isinstance(extra.get("rank_sources"), Mapping) else {}
    per_sys_ver = extra.get("rank_verified") if isinstance(extra.get("rank_verified"), Mapping) else {}

    stored_ranks = extra.get("ranks")
    if isinstance(stored_ranks, Mapping):
        for sys_name, label in stored_ranks.items():
            sys_key = str(sys_name).strip().upper()
            if not sys_key or sys_key in covered:
                continue
            src = str(per_sys_src.get(sys_key) or stored_src or "legacy").strip().lower()
            if sys_key in per_sys_ver:
                ver = bool(per_sys_ver[sys_key])
            else:
                ver = src in {"local_catalog", "easyscholar"} and stored_ver_status != "unverified"
            parsed = parse_compact_rank(sys_key, str(label), source=src, verified=ver)
            if parsed is not None:
                merged.append(parsed)
                covered.add(sys_key)

    # Rule 4: Pre-P17 rows with bare `sci_rank` / `ccf_rank` and no `ranks` entry.
    if stored.ccf_rank and "CCF" not in covered:
        src = stored_src if stored_src in {"local_catalog", "easyscholar", "openalex", "llm_websearch"} else "legacy"
        ver = src in {"local_catalog", "easyscholar"} and stored_ver_status != "unverified"
        merged.append(
            RankingEntry(
                ranking_system="CCF",
                rank=stored.ccf_rank.strip().upper(),
                source=src,
                verified=ver,
            )
        )
        covered.add("CCF")

    if stored.sci_rank and "JCR" not in covered:
        src = stored_src if stored_src in {"local_catalog", "easyscholar", "openalex", "llm_websearch"} else "legacy"
        ver = src in {"local_catalog", "easyscholar"} and stored_ver_status != "unverified"
        merged.append(
            RankingEntry(
                ranking_system="JCR",
                rank=stored.sci_rank.strip().upper(),
                source=src,
                verified=ver,
            )
        )
        covered.add("JCR")

    # Rule 5: Merge warnings, notes, presentation tier, metrics.
    openreview_tier = cat_prof.openreview_tier or pres_tier
    note = cat_prof.note or stored_note
    metrics = dict(cat_prof.metrics or (extra.get("metrics") if isinstance(extra.get("metrics"), Mapping) else {}) or {})

    any_verified = any(r.verified for r in merged)
    if cat_prof.has_any_rank or cat_prof.subtrack == "preprint":
        top_source = "local_catalog"
        top_ver = "verified"
    elif any_verified:
        top_source = next((r.source for r in merged if r.verified and r.source), "easyscholar")
        top_ver = "verified"
    elif warning:
        top_source = "local_catalog" if cat_prof.warning else (stored_src or "local_catalog")
        top_ver = "verified"
    elif merged:
        top_source = merged[0].source or stored_src or "legacy"
        top_ver = "unverified"
    else:
        top_source = ""
        top_ver = "unknown"

    peer_reviewed = cat_prof.peer_reviewed
    if peer_reviewed is None:
        if isinstance(extra.get("peer_reviewed"), bool):
            peer_reviewed = extra["peer_reviewed"]
        elif any_verified:
            peer_reviewed = True

    return VenueProfile(
        query_venue=raw_venue or lookup_venue,
        canonical_name=cat_prof.canonical_name or canonical_stored or raw_venue,
        acronym=cat_prof.acronym or acronym_stored,
        full_name=cat_prof.full_name,
        venue_type=cat_prof.venue_type or str(extra.get("venue_type") or ""),
        subtrack=cat_prof.subtrack if cat_prof.subtrack != "main" else "main",
        peer_reviewed=peer_reviewed,
        is_preprint_file=is_preprint_pdf or cat_prof.is_preprint_file,
        openreview_tier=openreview_tier,
        issn_l=cat_prof.issn_l,
        issns=list(cat_prof.issns),
        dblp_prefix=cat_prof.dblp_prefix,
        openalex_source_id=cat_prof.openalex_source_id,
        rankings=merged,
        warning=warning,
        note=note,
        metrics=metrics,
        source=top_source,
        verification_status=top_ver,
    )


def pub_rank_from_profile(
    profile: VenueProfile,
    *,
    raw_venue: str = "",
    year: int = 0,
    doi: str = "",
    arxiv_id: str = "",
) -> dict[str, Any]:
    """Project a `VenueProfile` into the `pub_rank` dict contract."""
    venue = raw_venue
    if profile.has_any_rank and profile.display_venue and (
        not venue or is_umbrella_series(venue) or is_preprint_venue(venue)
    ):
        venue = profile.display_venue

    out: dict[str, Any] = {
        "venue": venue,
        "year": year or 0,
        "sci": profile.sci_rank,
        "ccf": profile.ccf_rank,
        "doi": doi or "",
    }
    if arxiv_id:
        out["arxiv_id"] = arxiv_id
    ranks = profile.compact_ranks()
    if ranks:
        out["ranks"] = ranks
        out["rank_sources"] = profile.rank_sources_map()
        out["rank_verified"] = profile.rank_verified_map()
    if profile.subtrack and profile.subtrack != "main":
        out["subtrack"] = profile.subtrack
    if profile.warning:
        out["cas_warning"] = profile.warning
    if profile.source:
        out["source"] = profile.source
    if profile.verification_status and profile.verification_status != "unknown":
        out["verification_status"] = profile.verification_status
    if profile.openreview_tier:
        out["presentation_tier"] = profile.openreview_tier
    if profile.is_preprint_file:
        out["preprint_file_only"] = True
    if profile.note:
        out["note"] = profile.note
    return out


def _loads_extra(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


async def paper_venue_row(paper_id: str) -> dict[str, Any] | None:
    """Fetch `papers` joined with its primary `literature_items` work (for `arxiv_id`)."""
    from app.db import database as db

    try:
        return await db.fetch_one(
            """SELECT p.doi, p.venue, p.year, p.sci_rank, p.ccf_rank, p.venue_ranks_json,
                      COALESCE(li.arxiv_id, '') AS arxiv_id,
                      COALESCE(NULLIF(p.doi, ''), li.doi, '') AS effective_doi,
                      COALESCE(NULLIF(p.venue, ''), li.venue, '') AS effective_venue
                 FROM papers p
                 LEFT JOIN LATERAL (
                     SELECT literature_id
                       FROM literature_files
                      WHERE paper_id = p.paper_id
                      ORDER BY is_primary DESC, created_at ASC
                      LIMIT 1
                 ) lf ON TRUE
                 LEFT JOIN literature_items li ON li.literature_id = lf.literature_id
                WHERE p.paper_id = ?""",
            (paper_id,),
        )
    except Exception:
        return await db.fetch_one(
            "SELECT doi, venue, year, sci_rank, ccf_rank FROM papers WHERE paper_id = ?",
            (paper_id,),
        )


async def paper_venue_profile(paper_id: str) -> VenueProfile:
    """Resolve `VenueProfile` for a local `papers` row."""
    row = await paper_venue_row(paper_id)
    if row is None:
        return VenueProfile()
    return venue_profile(
        StoredVenue(
            venue=row.get("effective_venue") or row.get("venue") or "",
            sci_rank=row.get("sci_rank") or "",
            ccf_rank=row.get("ccf_rank") or "",
            extra=_loads_extra(row.get("venue_ranks_json")),
            doi=row.get("effective_doi") or row.get("doi") or "",
            arxiv_id=row.get("arxiv_id") or "",
        )
    )


async def work_venue_records(
    literature_ids: list[str],
) -> dict[str, tuple[VenueProfile, int | None]]:
    """Batch-resolve `(VenueProfile, published_year)` for `literature_id`s in one query."""
    from app.db import database as db

    ids = [i for i in dict.fromkeys(literature_ids) if i]
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = await db.fetch_all(
        f"""SELECT li.literature_id,
                   li.year AS work_year,
                   li.year_known,
                   COALESCE(NULLIF(p.year, 0), CASE WHEN li.year_known = 1 THEN li.year ELSE 0 END) AS eff_year,
                   COALESCE(NULLIF(p.venue, ''), li.venue, '') AS venue,
                   COALESCE(p.sci_rank, '') AS sci_rank,
                   COALESCE(p.ccf_rank, '') AS ccf_rank,
                   COALESCE(NULLIF(p.venue_ranks_json, '{{}}'), li.venue_ranks_json, '{{}}') AS venue_ranks_json,
                   COALESCE(NULLIF(p.doi, ''), li.doi, '') AS doi,
                   COALESCE(li.arxiv_id, '') AS arxiv_id
              FROM literature_items li
              LEFT JOIN LATERAL (
                  SELECT paper_id
                    FROM literature_files
                   WHERE literature_id = li.literature_id
                   ORDER BY is_primary DESC, created_at ASC
                   LIMIT 1
              ) lf ON TRUE
              LEFT JOIN papers p ON p.paper_id = lf.paper_id
             WHERE li.literature_id IN ({placeholders})""",
        tuple(ids),
    )
    out: dict[str, tuple[VenueProfile, int | None]] = {}
    for r in rows:
        prof = venue_profile(
            StoredVenue(
                venue=r.get("venue") or "",
                sci_rank=r.get("sci_rank") or "",
                ccf_rank=r.get("ccf_rank") or "",
                extra=_loads_extra(r.get("venue_ranks_json")),
                doi=r.get("doi") or "",
                arxiv_id=r.get("arxiv_id") or "",
            )
        )
        eff_year = int(r.get("eff_year") or 0)
        out[r["literature_id"]] = (prof, eff_year if eff_year > 0 else None)
    return out


async def work_venue_profiles(literature_ids: list[str]) -> dict[str, VenueProfile]:
    """Batch-resolve `literature_id -> VenueProfile` in one database query."""
    records = await work_venue_records(literature_ids)
    return {lid: prof for lid, (prof, _year) in records.items()}
