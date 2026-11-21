"""Look up again the venue records stored without a DOI.

Those are the rows that can be wrong, and a stored row is never looked up
again: a conference paper posted to arXiv first kept the preprint's year
(Semantic Scholar's `year`), and a paper without an arXiv stamp could carry
the venue of an arXiv id read from its references. Each paper is looked up as
a new one would be, one at a time (S2 allows one request a second); a lookup
that finds no venue changes nothing.

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
    finally:
        await database.close_pool()
    print(f"{'Updated' if apply else 'Would update'} {changed} of {len(rows)} papers.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="write the new records")
    return asyncio.run(_run(parser.parse_args().apply))


if __name__ == "__main__":
    sys.exit(main())
