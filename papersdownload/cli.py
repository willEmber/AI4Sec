from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .downloader import DEFAULT_USER_AGENT, download_pdfs


def _read_dois_from_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    dois: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        dois.append(s)
    return dois


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="papersdownload", description="Download open-access PDFs by DOI.")
    parser.add_argument("doi", nargs="*", help="One or more DOIs (or https://doi.org/... URLs).")
    parser.add_argument("--input", type=str, help="Text file with one DOI per line.")
    parser.add_argument("--out", type=str, default="pdfs", help="Output directory for PDFs.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent download workers.")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per URL.")
    parser.add_argument("--report", type=str, default="download_report.jsonl", help="Write JSONL report.")
    parser.add_argument(
        "--prefer",
        choices=["europepmc", "resolver"],
        default="europepmc",
        help="Which strategy to try first (Europe PMC is the default).",
    )
    parser.add_argument(
        "--no-resolver",
        action="store_true",
        help="Disable OA URL fallback (Unpaywall/CORE), even if env vars are set.",
    )
    parser.add_argument(
        "--no-scihub",
        action="store_true",
        help="Disable Sci-Hub fallback as the last resort.",
    )
    parser.add_argument(
        "--no-elsevier",
        action="store_true",
        help="Disable Elsevier TDM fallback (ScienceDirect), even if ELSEVIER_API_KEY is set.",
    )
    parser.add_argument(
        "--no-wiley",
        action="store_true",
        help="Disable Wiley TDM fallback, even if WILEY_TDM_TOKEN is set.",
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default=DEFAULT_USER_AGENT,
        help="HTTP User-Agent header for requests.",
    )
    args = parser.parse_args(argv)

    dois: list[str] = []
    if args.input:
        dois.extend(_read_dois_from_file(Path(args.input)))
    dois.extend(args.doi or [])

    if not dois:
        logging.error("No DOI provided. Pass DOI(s) or use --input.")
        return 2

    if args.no_resolver:
        unpaywall_email: str | None = ""
        core_api_key: str | None = ""
    else:
        unpaywall_email = None
        core_api_key = None

    results = download_pdfs(
        dois,
        out_dir=args.out,
        workers=args.workers,
        timeout=args.timeout,
        retries=args.retries,
        prefer_europepmc=args.prefer == "europepmc",
        unpaywall_email=unpaywall_email,
        core_api_key=core_api_key,
        user_agent=args.user_agent,
        report_path=args.report,
        use_scihub=not args.no_scihub,
        use_elsevier=not args.no_elsevier,
        use_wiley=not args.no_wiley,
    )

    return 0 if all(r.ok for r in results) else 1
