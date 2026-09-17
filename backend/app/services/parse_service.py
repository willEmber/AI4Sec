"""Parsing a stored PDF, in a form that survives losing the process doing it.

A MinerU parse is the single most expensive thing this system does: minutes of
remote work, billed per document. The submission is what costs — once a batch
exists, MinerU is already working on it, and the result can be collected by
anyone who knows the batch id.

So the unit of work records that id the moment it exists, and asks two questions
before doing anything:

*Is it already parsed?* Then nothing runs.

*Did a previous attempt already submit a batch?* Then this attempt rejoins that
batch instead of paying for a second one (acceptance case A13). If that batch
has since failed or expired, the failure is recorded rather than papered over
with an immediate resubmission — the job's own retry policy decides whether a
fresh submission is worth it, and that decision belongs in one place.

The same function backs the agent tool and the recovery sweep, because "resume
this parse" must mean exactly one thing.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.db import database as db
from app.services import agent_jobs, paper_catalog
from app.services.paper_ir import build_and_store_paper_ir

logger = logging.getLogger("scholar.services.parse")


def parse_idempotency_key(paper_id: str) -> str:
    """The key this paper's parse is filed under.

    The parser configuration is part of it: re-parsing the same bytes with a
    different backend is different work and must not reuse the old result.
    """
    return f"parse:{paper_id}:{get_settings().mineru_model_version}"


async def resumable_batch(job_id: str) -> tuple[str, str]:
    """`(parse_id, batch_id)` for a submission this job can still rejoin.

    Empty strings when there is nothing to rejoin: no earlier attempt, no batch
    id recorded before the crash, or an attempt that already ended in failure —
    in which case rejoining would only wait for a result that will not arrive.
    """
    job = await agent_jobs.get_job_by_id(job_id)
    parse_id = ((job or {}).get("remote_id") or "").strip()
    if not parse_id:
        return "", ""
    row = await db.fetch_one(
        "SELECT remote_batch_id, status FROM mineru_parses WHERE parse_id = ?",
        (parse_id,),
    )
    if row is None:
        return "", ""
    batch_id = (row.get("remote_batch_id") or "").strip()
    if not batch_id or row.get("status") == "failed":
        return "", ""
    return parse_id, batch_id


async def run_parse_job(*, job_id: str, paper_id: str) -> dict[str, Any]:
    """Parse `paper_id`, rejoining an already-submitted batch when there is one.

    Runs as the body of an `agent_jobs` job, which is what makes "at most once"
    true across turns and restarts. Returns the identifiers a caller needs to
    report the parse, never the parsed content itself.
    """
    from app.services import mineru_adapter

    settings = get_settings()
    pdf_path = settings.data_dir / "papers" / paper_id / "original.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"No PDF stored for {paper_id}")

    parse_id, batch_id = await resumable_batch(job_id)
    if batch_id:
        logger.info(
            "Resuming parse of %s from MinerU batch %s (parse %s)", paper_id, batch_id, parse_id
        )
        output_dir = await mineru_adapter.resume_parse(paper_id, parse_id, batch_id)
        resumed = True
    else:
        parse_id = uuid.uuid4().hex[:16]
        await db.execute(
            "INSERT INTO mineru_parses (parse_id, paper_id, status) VALUES (?, ?, 'pending')",
            (parse_id, paper_id),
        )
        # Recorded before the submission returns: a crash between here and the
        # first poll must still leave something to rejoin.
        await agent_jobs.set_remote_id(job_id, parse_id)
        output_dir = await mineru_adapter.parse_pdf(paper_id, parse_id)
        resumed = False

    ir = await build_and_store_paper_ir(
        Path(output_dir),
        paper_id,
        parse_id=parse_id,
        parser_config={"backend": settings.mineru_model_version},
    )
    version = await paper_catalog.get_current_version(paper_id)
    return {
        "parse_id": parse_id,
        "version_id": (version or {}).get("version_id", ""),
        "sections": len(getattr(ir, "sections", []) or []),
        "resumed": resumed,
    }
