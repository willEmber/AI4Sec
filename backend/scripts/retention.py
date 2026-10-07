#!/usr/bin/env python3
"""Run the retention pass by hand, or see what it would remove.

Usage:
    uv run python -m scripts.retention --dry-run   # count, delete nothing
    uv run python -m scripts.retention             # the same pass the server runs daily

The rules and their thresholds are the server's (`RETENTION_*` in .env); a rule
set to 0 is skipped. A dry run counts idle visitors and orphan papers
separately, so the papers that would only become orphans once those visitors
are gone are not in its count.
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from app.config import get_settings
from app.db import database
from app.services import data_lifecycle


async def _run(dry_run: bool) -> int:
    settings = get_settings()
    database.configure(settings.database_url, min_size=1, max_size=2)
    await database.init_db()
    try:
        report = await data_lifecycle.run_retention(dry_run=dry_run)
    finally:
        await database.close_pool()
    print("Would remove:" if dry_run else "Removed:")
    for key, value in report.items():
        shown = f"{value / 1024 / 1024:.1f} MB" if key.endswith("_bytes") else str(value)
        print(f"  {key:<22} {shown}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="count what a pass would remove")
    return asyncio.run(_run(parser.parse_args().dry_run))


if __name__ == "__main__":
    sys.exit(main())
