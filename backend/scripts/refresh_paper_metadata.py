"""Look up again the venue records stored without a DOI.

Those are the rows that can be wrong, and a stored row is never looked up
again: a conference paper posted to arXiv first kept the preprint's year
(Semantic Scholar's `year`), and a paper without an arXiv stamp could carry
the venue of an arXiv id read from its references. Each paper is looked up as
a new one would be, one at a time (S2 allows one request a second); a lookup
that finds no venue changes nothing.

A second pass, local only, gives each file's work record the year of the
file's published version: work records keep the first year any source gave,
which for a paper on arXiv first was often the preprint's, and the prompt
lists the paper with it.

    uv run python -m scripts.refresh_paper_metadata            # what would change
    uv run python -m scripts.refresh_paper_metadata --apply    # write it
"""

from __future__ import annotations

import argparse
import asyncio
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="write the new records")
    return asyncio.run(_run(parser.parse_args().apply))


if __name__ == "__main__":
    sys.exit(main())
