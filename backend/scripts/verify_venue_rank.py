"""Standalone verification, side-by-side comparison, and benchmark CLI for P17 venue ranking.

Usage:
    cd backend && uv run python -m scripts.verify_venue_rank
    cd backend && uv run python -m scripts.verify_venue_rank \\
        --query "CVPR" "ACL (Findings)" "Computers & Electrical Engineering" "软件学报"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from app.services.publication_rank import (
    VenueLookupInput,
    get_venue_catalog,
    reset_venue_catalog_cache,
    resolve_venue_profile,
    resolve_venue_profiles_batch,
)

DEFAULT_LEGACY_CACHE = Path("/home/stdlibh/scholar/backend/data/rank_cache.sqlite3")

CURATED_STRESS_CASES: list[tuple[str, VenueLookupInput]] = [
    (
        "Springer LNCS -> ECCV unmasking (via DBLP key)",
        VenueLookupInput(
            venue="Lecture Notes in Computer Science",
            dblp_key="conf/eccv/He2024",
            issns=["0302-9743"],
        ),
    ),
    (
        "PMLR -> ICML unmasking (via DBLP key)",
        VenueLookupInput(
            venue="Proceedings of Machine Learning Research",
            dblp_key="conf/icml/2024",
            issns=["2640-3498"],
        ),
    ),
    (
        "Accepted arXiv preprint -> ICLR Oral promotion",
        VenueLookupInput(
            venue="arXiv",
            openreview_status="ICLR 2025 Oral",
            is_preprint_source=True,
        ),
    ),
    (
        "Unaccepted OpenReview submission -> stays preprint",
        VenueLookupInput(
            venue="arXiv",
            openreview_status="Submitted to ICLR 2025",
            is_preprint_source=True,
        ),
    ),
    ("ACL Findings track isolation", VenueLookupInput(venue="ACL (Findings)")),
    ("CVPR Workshop isolation", VenueLookupInput(venue="CVPRW 2024")),
    ("Journal exact boundary: Pattern Recognition", VenueLookupInput(venue="Pattern Recognition")),
    ("Journal exact boundary: Pattern Recognition Letters", VenueLookupInput(venue="Pattern Recognition Letters")),
    ("HTML entity decoding: Optics &amp; Laser Technology", VenueLookupInput(venue="Optics &amp; Laser Technology")),
    ("CAS Early Warning red flag", VenueLookupInput(venue="Computers & Electrical Engineering")),
    ("Chinese CCF T1 journal", VenueLookupInput(venue="软件学报")),
    ("Open-review ML journal: TMLR", VenueLookupInput(venue="Transactions on Machine Learning Research")),
]


async def run_verification(
    *,
    legacy_cache_path: Path,
    queries: list[str],
    live: bool,
) -> dict[str, Any]:
    reset_venue_catalog_cache()
    t0 = time.perf_counter()
    cat = get_venue_catalog()
    cold_load_ms = round((time.perf_counter() - t0) * 1000, 2)

    print(f"=== 1. Catalog Benchmark ===")
    print(f"  Version:        {cat.version}")
    print(f"  Total venues:   {cat.size:,}")
    print(f"  Cold load time: {cold_load_ms} ms")

    if queries:
        print(f"\n=== Ad-hoc Queries (live={live}) ===")
        for q in queries:
            prof = await resolve_venue_profile(q, allow_network=live, allow_llm=live)
            print(f"  [{q}]")
            print(f"    badge:   {prof.prompt_badge('zh')}")
            print(f"    summary: {json.dumps(prof.agent_summary(), ensure_ascii=False)}")
        return {"cold_load_ms": cold_load_ms, "catalog_size": cat.size}

    print(f"\n=== 2. Curated Edge-Case Stress Suite ({len(CURATED_STRESS_CASES)} cases) ===")
    for label, inp in CURATED_STRESS_CASES:
        prof = cat.resolve(inp)
        print(
            f"  • {label:50s} => badge={prof.prompt_badge('zh')!r:42s} "
            f"ranks={prof.compact_ranks()} warn={prof.warning!r}"
        )

    comparisons: list[dict[str, Any]] = []
    if legacy_cache_path.exists():
        conn = sqlite3.connect(f"file:{legacy_cache_path}?mode=ro", uri=True)
        rows = conn.execute(
            "SELECT name_display, sci, ccf, source, success FROM publication_rank_cache"
        ).fetchall()
        conn.close()

        names = [r[0] for r in rows]
        tb0 = time.perf_counter()
        new_profiles = await resolve_venue_profiles_batch(names, allow_network=False)
        batch_ms = round((time.perf_counter() - tb0) * 1000, 2)

        upgraded = 0
        print(f"\n=== 3. Replay Against Real Historical Cache ({len(rows)} entries in {batch_ms} ms) ===")
        for (name, old_sci, old_ccf, old_src, old_ok), prof in zip(rows, new_profiles):
            old_tag = f"{old_src}:{'OK' if old_ok else 'FAIL'}(SCI={old_sci or '-'},CCF={old_ccf or '-'})"
            new_tag = f"{prof.verification_status}:{prof.subtrack} {prof.compact_ranks() or prof.note[:28]}"
            improved = (
                (not old_ok and (prof.has_any_rank or prof.subtrack != "main" or prof.venue_type == "book_series"))
                or (old_src == "llm_websearch" and prof.verification_status == "verified")
            )
            if improved:
                upgraded += 1
                print(f"  [FIXED] {name[:48]:48s} | OLD: {old_tag:38s} -> NEW: {new_tag}")
            comparisons.append({"name": name, "old": old_tag, "new": new_tag, "improved": improved})
        print(f"  Summary: {upgraded}/{len(rows)} historical failures or unverified LLM entries upgraded offline!")

    return {
        "cold_load_ms": cold_load_ms,
        "catalog_size": cat.size,
        "replayed": len(comparisons),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and benchmark P17 venue ranking engine.")
    parser.add_argument("--legacy-cache", type=Path, default=DEFAULT_LEGACY_CACHE)
    parser.add_argument("--query", nargs="*", default=[], help="Specific venue names to look up")
    parser.add_argument("--live", action="store_true", help="Enable live EasyScholar/OpenAlex/LLM fallback")
    args = parser.parse_args()

    asyncio.run(
        run_verification(
            legacy_cache_path=args.legacy_cache,
            queries=list(args.query),
            live=bool(args.live),
        )
    )


if __name__ == "__main__":
    main()
