"""Rebuild a parsed paper's `PaperIR` and bibliographic record from what is stored.

The mode subgraphs (Snap / Lens / Sphere) were written for the upload pipeline,
where `paper_ir_json` and `pub_rank_json` arrived in graph state fresh from the
parse. Inside a conversation the paper was parsed earlier — by the agent's
`ensure_paper_parsed`, by the parse started when it was attached, or by a
recovery sweep — so the same two inputs are recovered from storage instead.

`PaperIR` is rebuilt from MinerU's `content_list.json` rather than from the
`blocks` table on purpose: that is what the pipeline built it from, so reports
made before and after the upload page went away rest on the same structure.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from app.db import database as db
from app.models.paper_ir import PaperIR
from app.services.paper_ir import parse_content_list

logger = logging.getLogger("scholar.services.paper_ir_loader")


class PaperNotParsed(RuntimeError):
    """No completed parse with an output directory exists for this paper."""


async def latest_parse_output_dir(paper_id: str) -> Path | None:
    """Output directory of the most recent completed MinerU parse, if it still exists."""
    rows = await db.fetch_all(
        """SELECT output_dir FROM mineru_parses
            WHERE paper_id = ? AND status = 'done' AND COALESCE(output_dir, '') <> ''
            ORDER BY created_at DESC""",
        (paper_id,),
    )
    for row in rows:
        path = Path(row["output_dir"])
        if path.exists():
            return path
    return None


async def load_paper_ir(paper_id: str) -> PaperIR:
    """Rebuild the `PaperIR` for a parsed paper. Raises `PaperNotParsed` otherwise."""
    output_dir = await latest_parse_output_dir(paper_id)
    if output_dir is None:
        raise PaperNotParsed(paper_id)
    # File IO plus JSON parsing of a whole paper: off the event loop.
    return await asyncio.to_thread(parse_content_list, output_dir, paper_id)


async def load_pub_rank(paper_id: str) -> dict[str, Any]:
    """The `pub_rank_json` payload, from what `enrich_metadata` last stored."""
    import json

    try:
        row = await db.fetch_one(
            "SELECT doi, venue, year, sci_rank, ccf_rank, venue_ranks_json FROM papers WHERE paper_id = ?",
            (paper_id,),
        )
    except Exception:
        row = await db.fetch_one(
            "SELECT doi, venue, year, sci_rank, ccf_rank FROM papers WHERE paper_id = ?",
            (paper_id,),
        )
    if row is None:
        return {}
    extra: dict[str, Any] = {}
    try:
        extra = json.loads(row.get("venue_ranks_json") or "{}")
    except Exception:
        extra = {}
    out: dict[str, Any] = {
        "venue": row.get("venue") or "",
        "year": row.get("year") or 0,
        "sci": row.get("sci_rank") or "",
        "ccf": row.get("ccf_rank") or "",
        "doi": row.get("doi") or "",
    }
    if extra.get("ranks"):
        out["ranks"] = extra["ranks"]
    if extra.get("subtrack"):
        out["subtrack"] = extra["subtrack"]
    if extra.get("cas_warning"):
        out["cas_warning"] = extra["cas_warning"]
    if extra.get("source"):
        out["source"] = extra["source"]
    if extra.get("verification_status"):
        out["verification_status"] = extra["verification_status"]
    return out


async def ensure_pub_rank(paper_id: str, paper_ir: PaperIR, pdf_path: Path | None) -> dict[str, Any]:
    """Return the paper's venue/rank record, looking it up once when missing.

    The lookup (`paper_metadata.enrich_metadata`) stores what it finds on the
    paper's row and is already non-blocking; here a failure simply leaves the
    record as stored.
    """
    stored = await load_pub_rank(paper_id)
    if stored.get("venue"):
        return stored
    try:
        from app.services.paper_metadata import enrich_metadata

        found = await enrich_metadata(
            paper_id,
            paper_ir_json=paper_ir.model_dump_json(),
            pdf_path=str(pdf_path) if pdf_path else "",
        )
        if found:
            return found
    except Exception:  # noqa: BLE001 — metadata is a nicety, never a blocker
        logger.warning("enrich_metadata failed for %s; continuing without venue", paper_id)
    return stored
