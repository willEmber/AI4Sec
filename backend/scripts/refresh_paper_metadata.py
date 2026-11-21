"""Look up again the venue records stored without a DOI, or backfill multi-system ranks.

Those are the rows that can be wrong, and a stored row is never looked up
again: a conference paper posted to arXiv first kept the preprint's year
(Semantic Scholar's `year`), and a paper without an arXiv stamp could carry
the venue of an arXiv id read from its references. Each paper is looked up as
a new one would be, one at a time (S2 allows one request a second); a lookup
that finds no venue changes nothing.

With `--ranks`, runs strictly offline (<0.1ms per paper, no network): applies
`paper_venue.venue_profile` to every `papers` row with a venue, backfills
`venue_ranks_json` (and updates `sci_rank` / `ccf_rank` when the local catalog
or subtrack guard disagrees), and syncs linked `literature_items`.

    uv run python -m scripts.refresh_paper_metadata                  # what would change (network)
    uv run python -m scripts.refresh_paper_metadata --apply          # write it
    uv run python -m scripts.refresh_paper_metadata --ranks          # offline rank backfill dry-run
    uv run python -m scripts.refresh_paper_metadata --ranks --apply  # write offline rank backfill
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from app.config import get_settings
from app.db import database
from app.services.paper_ir_loader import PaperNotParsed
from app.services.paper_metadata import refresh_metadata

# S2 paces callers at one request a second and still answers bursts with 429.
PAUSE_SECONDS = 3.0


async def _run(apply: bool) -> int:
    settings = get_settings()
    database.configure(settings.database_url, min_size=1, max_size=2)
    await database.init_db()
    changed = 0
    try:
        rows = await database.fetch_all(
            "SELECT paper_id, title FROM papers "
            "WHERE venue <> '' AND COALESCE(doi, '') = '' ORDER BY paper_id"
        )
        for i, row in enumerate(rows):
            if i:
                await asyncio.sleep(PAUSE_SECONDS)
            label = f"{row['paper_id'][:8]} {(row.get('title') or '')[:60]}"
            try:
                found = await refresh_metadata(row["paper_id"], apply=apply)
            except PaperNotParsed:
                print(f"  {label}: not parsed, skipped")
                continue
            if found is None:
                print(f"  {label}: lookup found no venue (provider unavailable?), kept")
                continue
            if not found:
                print(f"  {label}: unchanged")
                continue
            changed += 1
            diff = "; ".join(f"{key} {was or '-'} -> {now or '-'}" for key, (was, now) in found.items())
            print(f"  {label}: {diff}")
        works = await _sync_work_years(apply)
    finally:
        await database.close_pool()
    print(f"{'Updated' if apply else 'Would update'} {changed} of {len(rows)} papers, {works} work years.")
    return 0


async def _sync_work_years(apply: bool) -> int:
    from app.services.paper_catalog import record_published_year

    stale = await database.fetch_all(
        """SELECT p.paper_id, p.doi, p.year, li.literature_id, li.year AS work_year, li.title
             FROM papers p
             JOIN literature_files lf ON lf.paper_id = p.paper_id
             JOIN literature_items li ON li.literature_id = lf.literature_id
            WHERE p.doi <> '' AND p.year > 0 AND p.venue <> ''
              AND (li.doi = '' OR LOWER(li.doi) = LOWER(p.doi))
              AND (li.year_known = 0 OR li.year <> p.year)"""
    )
    for row in stale:
        print(f"  work {row['literature_id']} {(row.get('title') or '')[:50]}: "
              f"year {row['work_year'] or '-'} -> {row['year']}")
        if apply:
            await record_published_year(row["paper_id"], doi=row["doi"], year=row["year"])
    return len(stale)


async def _backfill_ranks(apply: bool) -> int:
    """Offline backfill of `papers` and `literature_items` multi-system ranks."""
    from app.services.paper_catalog import record_published_venue_ranks
    from app.services.publication_rank.normalizer import is_preprint_venue
    from app.services.publication_rank.paper_venue import (
        StoredVenue,
        _loads_extra,
        venue_profile,
    )

    settings = get_settings()
    database.configure(settings.database_url, min_size=1, max_size=2)
    await database.init_db()
    paper_changes = 0
    work_changes = 0
    synced_lits: set[str] = set()
    try:
        rows = await database.fetch_all(
            """SELECT p.paper_id, p.title, p.doi, p.venue, p.year, p.sci_rank, p.ccf_rank,
                      p.venue_ranks_json,
                      COALESCE(li.arxiv_id, '') AS arxiv_id,
                      COALESCE(NULLIF(p.doi, ''), li.doi, '') AS effective_doi
                 FROM papers p
                 LEFT JOIN LATERAL (
                     SELECT literature_id
                       FROM literature_files
                      WHERE paper_id = p.paper_id
                      ORDER BY is_primary DESC, created_at ASC
                      LIMIT 1
                 ) lf ON TRUE
                 LEFT JOIN literature_items li ON li.literature_id = lf.literature_id
                WHERE COALESCE(p.venue, '') <> ''
                ORDER BY p.paper_id"""
        )
        for row in rows:
            pid = row["paper_id"]
            label = f"{pid[:8]} {(row.get('title') or '')[:55]}"
            was_extra = _loads_extra(row.get("venue_ranks_json"))
            prof = venue_profile(
                StoredVenue(
                    venue=row.get("venue") or "",
                    sci_rank=row.get("sci_rank") or "",
                    ccf_rank=row.get("ccf_rank") or "",
                    extra=was_extra,
                    doi=row.get("effective_doi") or "",
                    arxiv_id=row.get("arxiv_id") or "",
                )
            )
            now_sci = prof.sci_rank
            now_ccf = prof.ccf_rank
            now_summary = prof.agent_summary()
            now_json = json.dumps(now_summary, ensure_ascii=False)
            was_sci = row.get("sci_rank") or ""
            was_ccf = row.get("ccf_rank") or ""
            was_json = json.dumps(was_extra, ensure_ascii=False) if was_extra else "{}"

            diffs: list[str] = []
            if was_sci != now_sci:
                diffs.append(f"sci_rank {was_sci or '-'} -> {now_sci or '-'}")
            if was_ccf != now_ccf:
                diffs.append(f"ccf_rank {was_ccf or '-'} -> {now_ccf or '-'}")
            if was_json != now_json:
                diffs.append(f"venue_ranks {list((now_summary.get('ranks') or {}).keys()) or now_summary.get('subtrack') or 'updated'}")

            if diffs:
                paper_changes += 1
                print(f"  {label} [{row['venue']}]: {'; '.join(diffs)}")
                if apply:
                    await database.execute(
                        "UPDATE papers SET sci_rank = ?, ccf_rank = ?, venue_ranks_json = ? "
                        "WHERE paper_id = ?",
                        (now_sci, now_ccf, now_json, pid),
                    )
            linked = await database.fetch_all(
                """SELECT li.literature_id, li.venue, li.doi, li.venue_ranks_json
                     FROM literature_files lf
                     JOIN literature_items li ON li.literature_id = lf.literature_id
                    WHERE lf.paper_id = ?""",
                (pid,),
            )
            p_doi = (row.get("doi") or "").strip().lower()
            for lrow in linked:
                l_doi = (lrow.get("doi") or "").strip().lower()
                if p_doi and l_doi and p_doi != l_doi:
                    continue
                synced_lits.add(lrow["literature_id"])
                l_was = json.dumps(_loads_extra(lrow.get("venue_ranks_json")), ensure_ascii=False)
                l_venue = lrow.get("venue") or ""
                venue_needs_sync = (not l_venue or is_preprint_venue(l_venue)) and not is_preprint_venue(row.get("venue") or "")
                if l_was != now_json or venue_needs_sync:
                    work_changes += 1
            if apply:
                await record_published_venue_ranks(
                    pid,
                    doi=row.get("doi") or "",
                    venue=row.get("venue") or "",
                    venue_ranks=now_summary,
                )

        lit_rows = await database.fetch_all(
            """SELECT literature_id, title, venue, doi, arxiv_id, venue_ranks_json
                 FROM literature_items
                WHERE COALESCE(venue, '') <> ''
                ORDER BY literature_id"""
        )
        for lrow in lit_rows:
            lid = lrow["literature_id"]
            if lid in synced_lits:
                continue
            l_venue = lrow.get("venue") or ""
            was_extra = _loads_extra(lrow.get("venue_ranks_json"))
            if is_preprint_venue(l_venue) and not was_extra:
                continue
            prof = venue_profile(
                StoredVenue(
                    venue=l_venue,
                    extra=was_extra,
                    doi=lrow.get("doi") or "",
                    arxiv_id=lrow.get("arxiv_id") or "",
                )
            )
            now_summary = prof.agent_summary()
            if not was_extra and prof.subtrack == "preprint" and not prof.has_any_rank and not prof.warning:
                continue
            now_json = json.dumps(now_summary, ensure_ascii=False)
            was_json = json.dumps(was_extra, ensure_ascii=False) if was_extra else "{}"
            if was_json != now_json:
                work_changes += 1
                if apply:
                    await database.execute(
                        "UPDATE literature_items SET venue_ranks_json = ? WHERE literature_id = ?",
                        (now_json, lid),
                    )
        works_years = await _sync_work_years(apply)
    finally:
        await database.close_pool()
    print(
        f"{'Updated' if apply else 'Would update'} {paper_changes} of {len(rows)} papers, "
        f"{work_changes} work venue records, {works_years} work years."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="write the new records")
    parser.add_argument("--ranks", action="store_true", help="offline multi-system rank backfill (no network)")
    args = parser.parse_args()
    if args.ranks:
        return asyncio.run(_backfill_ranks(args.apply))
    return asyncio.run(_run(args.apply))


if __name__ == "__main__":
    sys.exit(main())
