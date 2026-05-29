#!/usr/bin/env python3
"""Clear publication_rank SQLite cache.

Usage:
    # Inside backend/ with uv:
    uv run python -m scripts.clear_rank_cache --stats
    uv run python -m scripts.clear_rank_cache --only-failures
    uv run python -m scripts.clear_rank_cache --all
    uv run python -m scripts.clear_rank_cache --delete "Neural Information Processing Systems"
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from app.services.publication_rank import RankCache


async def _run(args: argparse.Namespace) -> int:
    cache = RankCache(db_path=args.db) if args.db else RankCache()
    await cache.init()
    try:
        if args.stats:
            total = await cache.count()
            failures = await cache.count(only_failures=True)
            print(f"publication_rank cache: total={total} failures={failures}")
            return 0

        if args.delete:
            deleted = await cache.delete(args.delete)
            print(f"deleted {deleted} entry for name={args.delete!r}")
            return 0

        if args.only_failures:
            deleted = await cache.clear(only_failures=True)
            remaining = await cache.count()
            print(f"cleared {deleted} failure entries; remaining={remaining}")
            return 0

        if args.all:
            deleted = await cache.clear(only_failures=False)
            print(f"cleared all {deleted} entries")
            return 0

        # 默认行为：只清失败缓存（最常见的痛点）
        deleted = await cache.clear(only_failures=True)
        remaining = await cache.count()
        print(
            f"(default) cleared {deleted} failure entries; remaining={remaining}\n"
            "use --all to wipe everything, --stats to inspect, --delete NAME for a single row"
        )
        return 0
    finally:
        await cache.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect or clear the publication_rank SQLite cache.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stats", action="store_true", help="show cache row counts and exit")
    group.add_argument("--all", action="store_true", help="delete EVERY cache row (success + failure)")
    group.add_argument("--only-failures", action="store_true", help="delete only failure (success=0) rows")
    group.add_argument("--delete", metavar="NAME", help="delete a single cached publication by name")
    parser.add_argument("--db", help="custom cache DB path (defaults to settings.data_dir/rank_cache.sqlite3)")
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
