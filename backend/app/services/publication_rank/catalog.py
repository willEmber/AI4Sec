"""In-process read-only authoritative venue catalog (`venue_catalog.json.gz`).

Loaded once per worker via ``get_venue_catalog()`` (~30 ms, ~20 MB RAM).
Provides <0.1 ms deterministic resolution for 100+ candidates at a time without
touching PostgreSQL or external APIs, keeping unit tests fast and offline-ready.
"""
from __future__ import annotations

import gzip
import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .models import RankingEntry, VenueLookupInput, VenueProfile
from .normalizer import (
    classify_subtrack,
    clean_venue_text,
    extract_dblp_prefix,
    extract_issns,
    is_preprint_venue,
    is_strict_satellite_workshop,
    is_umbrella_series,
    normalize_issn,
    normalize_name_key,
    parse_openreview_status,
    strip_venue_decorations,
)

logger = logging.getLogger("scholar.publication_rank.catalog")

DEFAULT_CATALOG_PATH = Path(__file__).resolve().parent / "data" / "venue_catalog.json.gz"

_TYPE_CODE_MAP = {"c": "conference", "j": "journal", "b": "book_series", "p": "preprint"}


def _entry_priority(raw_entry: dict[str, Any], venue_type_hint: str = "") -> tuple[int, int, int]:
    """Disambiguation score when an acronym maps to more than one venue."""
    vtype = _TYPE_CODE_MAP.get(raw_entry.get("t", ""), raw_entry.get("t", ""))
    type_match = 1 if (venue_type_hint and vtype == venue_type_hint) else 0
    ranks = raw_entry.get("r") or []
    best_tier = 0
    has_cs_dir = 0
    for r in ranks:
        sys_name, rank_val = r[0], str(r[1]).upper()
        if sys_name in {"CCF", "CORE", "TH_CPL", "CAAI", "CCF_CN"}:
            has_cs_dir = 1
        if sys_name == "CCF" and rank_val == "A":
            best_tier = max(best_tier, 100)
        elif sys_name == "CORE" and rank_val == "A*":
            best_tier = max(best_tier, 95)
        elif sys_name == "CCF" and rank_val == "B":
            best_tier = max(best_tier, 85)
        elif sys_name in {"CORE", "TH_CPL", "CAAI"} and rank_val == "A":
            best_tier = max(best_tier, 80)
        elif sys_name == "CCF_CN" and rank_val == "T1":
            best_tier = max(best_tier, 78)
        elif sys_name == "JCR" and rank_val == "Q1":
            best_tier = max(best_tier, 75)
        elif sys_name == "CCF" and rank_val == "C":
            best_tier = max(best_tier, 65)
        else:
            best_tier = max(best_tier, 40)
    return (type_match, has_cs_dir, best_tier)


class VenueCatalog:
    """Read-only hash-indexed academic venue catalog."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.version: str = str(payload.get("version") or "unknown")
        self.editions: dict[str, str] = dict(payload.get("editions") or {})
        self._venues: list[dict[str, Any]] = list(payload.get("venues") or [])

        self._by_id: dict[str, dict[str, Any]] = {}
        self._by_dblp: dict[str, list[dict[str, Any]]] = {}
        self._by_issn: dict[str, dict[str, Any]] = {}
        self._by_oa: dict[str, dict[str, Any]] = {}
        self._by_name: dict[str, list[dict[str, Any]]] = {}
        self._by_acronym: dict[str, list[dict[str, Any]]] = {}

        for entry in self._venues:
            vid = entry.get("id") or ""
            if vid:
                self._by_id[vid] = entry

            for d_pref in entry.get("d") or []:
                dp = d_pref.strip().lower()
                if dp:
                    self._by_dblp.setdefault(dp, []).append(entry)

            for raw_issn in entry.get("i") or []:
                issn = normalize_issn(raw_issn)
                if issn and issn not in self._by_issn:
                    self._by_issn[issn] = entry

            oa_id = (entry.get("oa") or "").strip().upper()
            if oa_id:
                self._by_oa[oa_id] = entry

            full_name = entry.get("n") or ""
            nk = normalize_name_key(full_name)
            if nk:
                self._by_name.setdefault(nk, []).append(entry)

            # Only run decoration-stripping regexes when the title actually carries
            # proceedings/year/parenthetical decorations (cuts cold load by ~8x).
            if "(" in full_name or full_name[:4].isdigit() or full_name.lower().startswith("proceedings"):
                stripped_n, _ = strip_venue_decorations(full_name)
                snk = normalize_name_key(stripped_n)
                if snk and snk != nk:
                    self._by_name.setdefault(snk, []).append(entry)

            for alias in entry.get("al") or []:
                ak = normalize_name_key(alias)
                if ak:
                    # Short uppercase aliases go to acronym index as well; full phrases go to name index
                    if len(alias.strip()) <= 16 and " " not in alias.strip():
                        self._by_acronym.setdefault(ak, []).append(entry)
                    else:
                        self._by_name.setdefault(ak, []).append(entry)

            acr = (entry.get("a") or "").strip()
            if acr:
                ack = normalize_name_key(acr)
                if ack:
                    self._by_acronym.setdefault(ack, []).append(entry)

    @property
    def size(self) -> int:
        return len(self._venues)

    def _pick_best(self, candidates: list[dict[str, Any]], venue_type_hint: str = "") -> dict[str, Any]:
        if len(candidates) == 1:
            return candidates[0]
        return max(candidates, key=lambda c: _entry_priority(c, venue_type_hint))

    def _to_profile(
        self,
        entry: dict[str, Any],
        *,
        query_venue: str = "",
        is_preprint_file: bool = False,
        openreview_tier: str = "",
    ) -> VenueProfile:
        vtype = _TYPE_CODE_MAP.get(entry.get("t", ""), entry.get("t", ""))
        rankings: list[RankingEntry] = []
        for r in entry.get("r") or []:
            # Tuple layout: [system, rank, category, edition_year, top_flag, impact_factor]
            sys_name = str(r[0])
            rank_val = str(r[1])
            cat = str(r[2]) if len(r) > 2 and r[2] else ""
            ed = str(r[3]) if len(r) > 3 and r[3] else self.editions.get(sys_name, "")
            top = bool(r[4]) if len(r) > 4 else False
            if_val = str(r[5]) if len(r) > 5 and r[5] else ""
            rankings.append(
                RankingEntry(
                    ranking_system=sys_name,
                    rank=rank_val,
                    category=cat,
                    edition_year=ed,
                    top=top,
                    impact_factor=if_val,
                    source="local_catalog",
                    verified=True,
                )
            )
        issns = [normalize_issn(x) for x in (entry.get("i") or []) if normalize_issn(x)]
        dblp_list = entry.get("d") or []
        return VenueProfile(
            query_venue=query_venue,
            canonical_name=entry.get("n") or entry.get("a") or query_venue,
            acronym=entry.get("a") or "",
            full_name=entry.get("n") or "",
            venue_type=vtype,
            subtrack="main",
            peer_reviewed=True,
            is_preprint_file=is_preprint_file,
            openreview_tier=openreview_tier,
            issn_l=issns[0] if issns else "",
            issns=issns,
            dblp_prefix=dblp_list[0] if dblp_list else "",
            openalex_source_id=entry.get("oa") or "",
            rankings=rankings,
            warning=str(entry.get("w") or ""),
            note=str(entry.get("note") or ""),
            metrics=dict(entry.get("m") or {}),
            source="local_catalog",
            verification_status="verified",
        )

    def _match_text(self, text: str, *, venue_type_hint: str = "") -> dict[str, Any] | None:
        cleaned = clean_venue_text(text)
        if not cleaned or is_preprint_venue(cleaned) or is_umbrella_series(cleaned):
            return None

        # 1) Direct ISSN embedded in string
        embedded_issns = extract_issns(cleaned)
        for issn in embedded_issns:
            if issn in self._by_issn:
                return self._by_issn[issn]

        # 1b) PACM conference-journal issues specifying the exact conference in parentheses,
        #     e.g. "Proceedings of the ACM on Programming Languages (POPL)" -> POPL (CCF A)
        if re.search(r"\b(?:proceedings\s+of\s+the\s+acm\s+on|proc\.?\s+acm)\b", cleaned, re.IGNORECASE):
            for m_sub in re.finditer(r"\(\s*(popl|oopsla|pldi|icfp|sigmod|fse|cscw|chi|sigmetrics)\s*\)", cleaned, re.IGNORECASE):
                sub_acr = normalize_name_key(m_sub.group(1))
                if sub_acr in self._by_acronym:
                    return self._pick_best(self._by_acronym[sub_acr], "conference")

        # 2) Exact normalized name match (for short single-word queries like "AI", "IOT", "SPIN",
        #    pool both `_by_name` and `_by_acronym` so minor homograph journals never shadow CS flagship venues!)
        raw_key = normalize_name_key(cleaned)
        is_short_token = len(cleaned) <= 16 and " " not in cleaned
        if is_short_token and (raw_key in self._by_name or raw_key in self._by_acronym):
            pooled = list(self._by_name.get(raw_key, ()))
            for c in self._by_acronym.get(raw_key, ()):
                if c not in pooled:
                    pooled.append(c)
            return self._pick_best(pooled, venue_type_hint)
        if raw_key in self._by_name:
            return self._pick_best(self._by_name[raw_key], venue_type_hint)

        # 3) Strip proceedings/year/ordinal wrappers and check full stripped name
        stripped, acronyms = strip_venue_decorations(cleaned)
        stripped_key = normalize_name_key(stripped)
        is_short_stripped = len(stripped) <= 16 and " " not in stripped
        if is_short_stripped and stripped_key and (stripped_key in self._by_name or stripped_key in self._by_acronym):
            pooled = list(self._by_name.get(stripped_key, ()))
            for c in self._by_acronym.get(stripped_key, ()):
                if c not in pooled:
                    pooled.append(c)
            return self._pick_best(pooled, venue_type_hint)
        if stripped_key and stripped_key in self._by_name:
            return self._pick_best(self._by_name[stripped_key], venue_type_hint)

        # 4) Multi-token short acronyms (e.g. "ACM MM", "IEEE S&P", "USENIX Security")
        if len(stripped) <= 22 and stripped_key in self._by_acronym:
            return self._pick_best(self._by_acronym[stripped_key], venue_type_hint)
        if len(cleaned) <= 22 and raw_key in self._by_acronym:
            return self._pick_best(self._by_acronym[raw_key], venue_type_hint)

        # 5) Check extracted parenthetical / canonical conference acronyms
        #    Guard: only use parenthetical acronym if the main text isn't a different journal
        #    containing an acronym in parentheses (e.g. conferences or exact known acronyms).
        for acr in acronyms:
            ack = normalize_name_key(acr)
            if ack in self._by_acronym:
                cand = self._pick_best(self._by_acronym[ack], venue_type_hint or "conference")
                # If stripped text looks like a journal ("letters", "bulletin", "advances in ..."),
                # verify we don't accidentally map "Pattern Recognition Letters (PRL)" onto "PR".
                cand_type = _TYPE_CODE_MAP.get(cand.get("t", ""), "")
                if cand_type == "conference" or stripped_key == ack:
                    return cand

        return None

    def resolve(self, inp: VenueLookupInput | str) -> VenueProfile:
        """Deterministically resolve a venue + identifiers against the local catalog."""
        if isinstance(inp, str):
            inp = VenueLookupInput(venue=inp)

        raw_venue = clean_venue_text(inp.venue)
        is_preprint_file = bool(inp.is_preprint_source or (raw_venue and is_preprint_venue(raw_venue)))

        # Parse OpenReview acceptance status if available
        or_venue, or_subtrack, or_tier = parse_openreview_status(inp.openreview_status)

        # Step 1: Collect candidate venue strings in priority order.
        # If primary `raw_venue` is empty, a preprint server, or an umbrella book
        # series (LNCS / PMLR / CCIS / CEUR-WS), probe published metadata fields!
        primary_is_shell = (
            not raw_venue
            or is_preprint_venue(raw_venue)
            or is_umbrella_series(raw_venue)
        )

        text_candidates: list[str] = []
        if not primary_is_shell and raw_venue:
            text_candidates.append(raw_venue)
        if or_venue and or_subtrack != "preprint":
            text_candidates.append(or_venue)
        if inp.crossref_event:
            ev = clean_venue_text(inp.crossref_event)
            if ev and not is_umbrella_series(ev):
                text_candidates.append(ev)
        for alias in inp.venue_aliases:
            al = clean_venue_text(alias)
            if al and not is_preprint_venue(al) and not is_umbrella_series(al):
                if al not in text_candidates:
                    text_candidates.append(al)

        effective_venue = text_candidates[0] if text_candidates else raw_venue
        # Never pass an umbrella proceedings series (e.g. "JMLR Workshop and Conference Proceedings")
        # into `classify_subtrack`, otherwise "Workshop" in the series title blocks DBLP unmasking!
        subtrack_probe_venue = text_candidates[0] if text_candidates else ("" if is_umbrella_series(raw_venue) else raw_venue)

        # Step 2: Subtrack guard (Workshop / Companion / Findings / Preprint)
        subtrack, host = classify_subtrack(subtrack_probe_venue, dblp_key=inp.dblp_key)
        if or_subtrack in {"workshop", "companion", "findings"} and subtrack == "main":
            subtrack = or_subtrack

        dblp_prefix = extract_dblp_prefix(inp.dblp_key)

        # Guard for standalone peer-reviewed CCF/CORE conferences whose official title
        # legitimately includes singular "Workshop" (e.g. IWQoS, HotOS, PKC, CSFW, CHES, ICER):
        # if it is NOT a workshop acronym (CVPRW / conf/cvprw) or plural/satellite track
        # ("Workshops", "Tiny Papers", "Doctoral Consortium"), check if the DBLP prefix or
        # exact stripped title is a standalone cataloged venue!
        if subtrack == "workshop" and not host and not is_strict_satellite_workshop(subtrack_probe_venue):
            if dblp_prefix and dblp_prefix in self._by_dblp:
                subtrack = "main"
            elif subtrack_probe_venue:
                probe_key = normalize_name_key(subtrack_probe_venue)
                stripped_probe, _ = strip_venue_decorations(subtrack_probe_venue)
                stripped_probe_key = normalize_name_key(stripped_probe)
                if probe_key in self._by_name or (stripped_probe_key and stripped_probe_key in self._by_name):
                    subtrack = "main"

        if subtrack in {"workshop", "companion"}:
            return VenueProfile(
                query_venue=raw_venue,
                canonical_name=effective_venue or raw_venue,
                acronym=f"{host}W" if host and not effective_venue else "",
                venue_type="conference",
                subtrack=subtrack,
                peer_reviewed=None,
                is_preprint_file=is_preprint_file,
                rankings=[],
                source="local_catalog",
                verification_status="verified",
            )

        if subtrack == "findings":
            label = f"Findings of {host}" if host else (effective_venue or raw_venue)
            return VenueProfile(
                query_venue=raw_venue,
                canonical_name=label,
                acronym=f"{host} Findings" if host else "",
                venue_type="conference",
                subtrack="findings",
                peer_reviewed=True,
                is_preprint_file=is_preprint_file,
                openreview_tier=or_tier,
                rankings=[],
                note=f"Peer-reviewed Findings track of {host or 'conference'} (distinct from main conference)",
                source="local_catalog",
                verification_status="verified",
            )

        # Step 3: Identifier-first exact lookup (DBLP prefix -> ISSN -> OpenAlex Source ID)
        if dblp_prefix and dblp_prefix in self._by_dblp:
            cands = self._by_dblp[dblp_prefix]
            # If PACMPL / PACMMOD maps to multiple conferences, try text match among them first
            if len(cands) > 1 and text_candidates:
                for tc in text_candidates:
                    tck = normalize_name_key(tc)
                    for c in cands:
                        if tck == normalize_name_key(c.get("a", "")) or tck == normalize_name_key(c.get("n", "")):
                            return self._to_profile(
                                c, query_venue=raw_venue, is_preprint_file=is_preprint_file, openreview_tier=or_tier
                            )
            chosen = self._pick_best(cands, inp.venue_type_hint)
            return self._to_profile(
                chosen, query_venue=raw_venue, is_preprint_file=is_preprint_file, openreview_tier=or_tier
            )

        for issn in extract_issns(inp.issns):
            if issn in self._by_issn:
                return self._to_profile(
                    self._by_issn[issn],
                    query_venue=raw_venue,
                    is_preprint_file=is_preprint_file,
                    openreview_tier=or_tier,
                )

        oa_id = (inp.openalex_source_id or "").replace("https://openalex.org/", "").strip().upper()
        if oa_id and oa_id in self._by_oa:
            return self._to_profile(
                self._by_oa[oa_id],
                query_venue=raw_venue,
                is_preprint_file=is_preprint_file,
                openreview_tier=or_tier,
            )

        # Step 4: Deterministic name / acronym lookup across candidate strings
        for tc in text_candidates:
            tc_sub, tc_host = classify_subtrack(tc)
            if tc_sub in {"companion", "findings", "preprint"}:
                continue
            if tc_sub == "workshop":
                if tc_host or is_strict_satellite_workshop(tc):
                    continue
                tck = normalize_name_key(tc)
                st_tc, _ = strip_venue_decorations(tc)
                st_tck = normalize_name_key(st_tc)
                if tck not in self._by_name and (not st_tck or st_tck not in self._by_name):
                    continue
            matched = self._match_text(tc, venue_type_hint=inp.venue_type_hint)
            if matched is not None:
                return self._to_profile(
                    matched,
                    query_venue=raw_venue,
                    is_preprint_file=is_preprint_file,
                    openreview_tier=or_tier,
                )

        # Step 5: Unmatched — classify pure preprint or umbrella shell cleanly
        if subtrack == "preprint" or (is_preprint_file and not text_candidates):
            return VenueProfile(
                query_venue=raw_venue,
                canonical_name=raw_venue or "arXiv",
                venue_type="preprint",
                subtrack="preprint",
                peer_reviewed=False,
                is_preprint_file=True,
                rankings=[],
                source="local_catalog",
                verification_status="verified",
            )

        if raw_venue and is_umbrella_series(raw_venue) and not text_candidates:
            return VenueProfile(
                query_venue=raw_venue,
                canonical_name=raw_venue,
                venue_type="book_series",
                subtrack="main",
                peer_reviewed=None,
                is_preprint_file=is_preprint_file,
                rankings=[],
                note="Proceedings book series container; specific conference venue not identified",
                source="",
                verification_status="unknown",
            )

        return VenueProfile(
            query_venue=raw_venue,
            canonical_name=effective_venue or raw_venue,
            venue_type=inp.venue_type_hint,
            subtrack="main",
            is_preprint_file=is_preprint_file,
            openreview_tier=or_tier,
            issns=extract_issns(inp.issns),
            dblp_prefix=dblp_prefix,
            openalex_source_id=oa_id,
            rankings=[],
            source="",
            verification_status="unknown",
        )

    def list_by_tier(
        self,
        *,
        system: str,
        ranks: set[str],
        venue_type: str = "",
    ) -> list[dict[str, Any]]:
        """Return catalog entries matching ``system`` and ``ranks`` (for Phase 4 filter pushdown)."""
        sys_up = system.strip().upper()
        want = {r.strip().upper() for r in ranks}
        out: list[dict[str, Any]] = []
        for entry in self._venues:
            if venue_type and _TYPE_CODE_MAP.get(entry.get("t", ""), "") != venue_type:
                continue
            for r in entry.get("r") or []:
                if str(r[0]).upper() == sys_up and str(r[1]).upper() in want:
                    out.append(entry)
                    break
        return out


def load_catalog_from_path(path: Path | str | None = None) -> VenueCatalog:
    target = Path(path) if path is not None else DEFAULT_CATALOG_PATH
    if not target.exists():
        logger.warning("Venue catalog file missing at %s; initializing empty catalog", target)
        return VenueCatalog({"version": "empty", "venues": []})
    with gzip.open(target, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    cat = VenueCatalog(payload)
    logger.info("Loaded venue catalog v=%s (%d venues) from %s", cat.version, cat.size, target.name)
    return cat


@lru_cache(maxsize=1)
def get_venue_catalog() -> VenueCatalog:
    return load_catalog_from_path(DEFAULT_CATALOG_PATH)


def reset_venue_catalog_cache() -> None:
    get_venue_catalog.cache_clear()
