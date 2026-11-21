"""Async OpenAlex `/sources` lookup for ISSN alignment and quantitative venue metrics (P17).

Used in L3 when a journal/conference is not in the static catalog:
1. Resolves canonical ``issn_l`` and ``issns`` so EasyScholar can be queried by
   exact ISSN even when the caller only had an abbreviated journal title.
2. Supplies quantitative citation impact metrics (``h_index``, ``two_yr_citedness``)
   for emerging open-review venues (e.g. TMLR, COLM) or cross-disciplinary outlets.

Shares ``OPENALEX_BUDGET`` with ``paper_search`` and enforces exact normalized
title/alias matching on search results so loose substrings never misattribute a venue.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from app.services.paper_search.credentials import openalex_auth_params
from app.services.paper_search.ratelimit import OPENALEX_BUDGET, is_quota_response

from .normalizer import clean_venue_text, extract_issns, normalize_issn, normalize_name_key

logger = logging.getLogger("scholar.publication_rank.openalex_source")

OPENALEX_SOURCES_URL = "https://api.openalex.org/sources"
_SELECT_FIELDS = (
    "id,display_name,abbreviated_title,alternate_titles,issn_l,issn,type,"
    "is_oa,is_in_doaj,host_organization_name,summary_stats"
)


def _parse_source_record(item: dict[str, Any]) -> dict[str, Any]:
    oa_id = (item.get("id") or "").replace("https://openalex.org/", "").strip()
    issn_l = normalize_issn(item.get("issn_l") or "")
    issns = extract_issns(item.get("issn") or [])
    if issn_l and issn_l not in issns:
        issns.insert(0, issn_l)

    stype = (item.get("type") or "").strip().lower()
    venue_type = stype if stype in {"journal", "conference"} else ""

    stats = item.get("summary_stats") or {}
    metrics: dict[str, Any] = {}
    if isinstance(stats.get("h_index"), int) and stats["h_index"] > 0:
        metrics["h_index"] = stats["h_index"]
    citedness = stats.get("2yr_mean_citedness")
    if isinstance(citedness, (int, float)) and citedness > 0:
        metrics["two_yr_citedness"] = round(float(citedness), 2)

    aliases: list[str] = []
    if item.get("abbreviated_title"):
        aliases.append(clean_venue_text(item["abbreviated_title"]))
    for alt in item.get("alternate_titles") or []:
        if isinstance(alt, str) and clean_venue_text(alt):
            aliases.append(clean_venue_text(alt))

    return {
        "openalex_source_id": oa_id,
        "display_name": clean_venue_text(item.get("display_name") or ""),
        "venue_type": venue_type,
        "issn_l": issn_l or (issns[0] if issns else ""),
        "issns": issns,
        "aliases": aliases,
        "is_oa": item.get("is_oa"),
        "is_in_doaj": item.get("is_in_doaj"),
        "publisher": clean_venue_text(item.get("host_organization_name") or ""),
        "metrics": metrics,
    }


def _exact_title_match(query_name: str, candidate: dict[str, Any]) -> bool:
    want = normalize_name_key(query_name)
    if not want:
        return False
    names = [candidate.get("display_name") or "", candidate.get("abbreviated_title") or ""]
    names.extend(x for x in (candidate.get("alternate_titles") or []) if isinstance(x, str))
    return any(normalize_name_key(n) == want for n in names if n)


async def _oa_get(client: httpx.AsyncClient, url: str, params: dict[str, str]) -> dict[str, Any] | None:
    if OPENALEX_BUDGET.exhausted():
        return None
    full_params = {**openalex_auth_params(), **params}
    try:
        resp = await client.get(
            url,
            params=full_params,
            headers={"User-Agent": "scholar-agent/1.0"},
        )
        if is_quota_response(resp.status_code, dict(resp.headers)):
            OPENALEX_BUDGET.mark_exhausted(dict(resp.headers))
            return None
        OPENALEX_BUDGET.observe(dict(resp.headers))
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.debug("OpenAlex /sources lookup failed (%s): %s", url, exc)
        return None


async def fetch_openalex_source(
    *,
    venue_name: str = "",
    issns: list[str] | None = None,
    source_id: str = "",
    client: httpx.AsyncClient | None = None,
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    """Resolve one venue against OpenAlex `/sources` (ID -> ISSN -> exact name)."""
    if OPENALEX_BUDGET.exhausted():
        return None

    clean_id = (source_id or "").replace("https://openalex.org/", "").strip().upper()
    clean_issns = extract_issns(issns)
    clean_name = clean_venue_text(venue_name)
    if not (clean_id or clean_issns or clean_name):
        return None

    own_client = client is None
    http = client or httpx.AsyncClient(timeout=timeout)
    try:
        if clean_id.startswith("S"):
            item = await _oa_get(http, f"{OPENALEX_SOURCES_URL}/{clean_id}", {"select": _SELECT_FIELDS})
            if item and item.get("id"):
                return _parse_source_record(item)

        if clean_issns:
            flt = "issn:" + "|".join(clean_issns[:10])
            data = await _oa_get(
                http,
                OPENALEX_SOURCES_URL,
                {"filter": flt, "per_page": "5", "select": _SELECT_FIELDS},
            )
            results = (data or {}).get("results") or []
            for cand in results:
                if isinstance(cand, dict) and (cand.get("type") or "") != "repository":
                    return _parse_source_record(cand)

        if clean_name:
            data = await _oa_get(
                http,
                OPENALEX_SOURCES_URL,
                {"search": clean_name[:200], "per_page": "5", "select": _SELECT_FIELDS},
            )
            results = (data or {}).get("results") or []
            for cand in results:
                if (
                    isinstance(cand, dict)
                    and (cand.get("type") or "") != "repository"
                    and _exact_title_match(clean_name, cand)
                ):
                    return _parse_source_record(cand)
        return None
    finally:
        if own_client:
            await http.aclose()
