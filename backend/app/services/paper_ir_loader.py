"""Rebuild a parsed paper's `PaperIR` and bibliographic record from what is stored.

The mode subgraphs (Snap / Lens / Sphere) were written for the upload pipeline,
where `paper_ir_json` and `pub_rank_json` arrive in graph state fresh from the
parse. Inside a conversation the paper was parsed earlier — by an upload run,
by the agent's `ensure_paper_parsed`, or by a recovery sweep — so the same two
inputs have to be recovered from storage instead.

`PaperIR` is rebuilt from MinerU's `content_list.json` rather than from the
`blocks` table on purpose: that is exactly what the pipeline does, so a report
made in a conversation is built on the same structure as one made from the
upload page.
"""

from __future__ import annotations

import asyncio
import json
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
    row = await db.fetch_one(
        "SELECT doi, venue, year, sci_rank, ccf_rank FROM papers WHERE paper_id = ?",
        (paper_id,),
    )
    if row is None:
        return {}
    return {
        "venue": row.get("venue") or "",
        "year": row.get("year") or 0,
        "sci": row.get("sci_rank") or "",
        "ccf": row.get("ccf_rank") or "",
        "doi": row.get("doi") or "",
    }


async def ensure_pub_rank(paper_id: str, paper_ir: PaperIR, pdf_path: Path | None) -> dict[str, Any]:
    """Return the paper's venue/rank record, looking it up once when missing.

    Runs the pipeline's own `enrich_metadata` node so the conversation and the
    upload page agree on where a venue came from. Any failure there is already
    non-blocking; here it simply leaves the record as stored.
    """
    stored = await load_pub_rank(paper_id)
    if stored.get("venue"):
        return stored
    try:
        from app.workflows.main_graph import enrich_metadata

        result = await enrich_metadata(
            {
                "paper_id": paper_id,
                "paper_ir_json": paper_ir.model_dump_json(),
                "pdf_path": str(pdf_path) if pdf_path else "",
                "progress": [],
            }
        )
        payload = result.get("pub_rank_json") or ""
        if payload:
            return json.loads(payload)
    except Exception:  # noqa: BLE001 — metadata is a nicety, never a blocker
        logger.warning("enrich_metadata failed for %s; continuing without venue", paper_id)
    return stored
