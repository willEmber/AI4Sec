from __future__ import annotations

import argparse
import asyncio
import sys

from .search import search_papers


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PaperSearch quick runner (smoke test / CLI).")
    parser.add_argument("-q", "--query", required=True, help="Search query text.")
    parser.add_argument(
        "--platforms",
        default="arXiv",
        help='Comma-separated platforms, e.g. "arXiv,OpenAlex,PubMed".',
    )
    parser.add_argument(
        "--fields",
        default="",
        help='Comma-separated output fields, e.g. "title,doi,url". Default: all fields.',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    platforms = [p.strip() for p in (args.platforms or "").split(",") if p.strip()]
    fields = [f.strip() for f in (args.fields or "").split(",") if f.strip()]

    result = asyncio.run(search_papers(query=args.query, platforms=platforms, fields=fields or None))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
