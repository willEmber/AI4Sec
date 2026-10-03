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

import asyncio
import contextlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.config import get_settings
from app.db import database as db
from app.models.agent_models import ErrorCode, EventType
from app.services import agent_jobs, paper_catalog
from app.services.paper_ir import build_and_store_paper_ir

if TYPE_CHECKING:
    from app.agents.context import AgentContext

logger = logging.getLogger("scholar.services.parse")


def parse_idempotency_key(paper_id: str) -> str:
    """The key this paper's parse is filed under.

    The parser configuration is part of it: re-parsing the same bytes with a
    different backend is different work and must not reuse the old result.
    """
    return f"parse:{paper_id}:{get_settings().mineru_model_version}"


async def run_agent_parse(
    ctx: AgentContext, paper_id: str, *, tool_name: str
) -> agent_jobs.JobHandle:
    """Await the keyed parse and publish its durable progress to this turn.

    Each waiter observes the same stored snapshot, including when the work was
    started by another turn in this process. Progress is not tied to the caller
    that happened to submit the batch, and replay uses the same persisted events.
    """
    from app.services import agent_runner

    key = parse_idempotency_key(paper_id)
    started = time.monotonic()

    async def publish(status: str = "running", **progress: Any) -> None:
        with contextlib.suppress(Exception):
            await agent_runner.emit(
                session_id=ctx.session_id,
                run_id=ctx.run_id,
                type=EventType.TOOL_PROGRESS,
                payload={
                    "tool": tool_name, "step": "mineru_parse", "status": status,
                    "paper_id": paper_id, "elapsed_s": round(time.monotonic() - started),
                    **progress,
                },
            )

    async def watch() -> None:
        while True:
            try:
                row = await db.fetch_one(
                    "SELECT p.status, p.progress_json, p.last_state_counts "
                    "FROM agent_jobs j JOIN mineru_parses p ON p.parse_id = j.remote_id "
                    "WHERE j.idempotency_key = ?", (key,),
                )
                if row:
                    snapshot = json.loads(row.get("progress_json") or "{}")
                    states = snapshot.get("state_counts") or json.loads(
                        row.get("last_state_counts") or "{}"
                    )
                    phase = snapshot.get("phase")
                    if not phase:
                        phase = next(
                            (s for s in ("running", "converting", "pending", "waiting-file") if states.get(s)),
                            "submitting",
                        )
                    if row["status"] == "done":
                        phase = "indexing"
                    await publish(
                        phase=phase,
                        extracted_pages=snapshot.get("extracted_pages", 0),
                        total_pages=snapshot.get("total_pages", 0),
                    )
            except Exception:
                # An unavailable diagnostic must not cancel expensive work.
                logger.debug("Parse progress unavailable for %s", paper_id, exc_info=True)
            await asyncio.sleep(max(1, get_settings().mineru_poll_interval_seconds))

    if paper_id in ctx.deferred_parses:
        await publish(status="waiting", phase="pending")
        raise agent_jobs.JobFailed(
            ErrorCode.TIMEOUT, "This paper is still being prepared; continue in a later turn."
        )
    await publish(phase="submitting")
    observer = asyncio.create_task(watch())

    async def work(job_id: str) -> dict[str, Any]:
        return await run_parse_job(job_id=job_id, paper_id=paper_id)

    try:
        handle = await agent_jobs.run_once(
            kind="parse", idempotency_key=key, work=work,
            session_id=ctx.session_id, run_id=ctx.run_id, request={"paper_id": paper_id},
        )
    except agent_jobs.JobFailed as exc:
        observer.cancel()
        if exc.code == ErrorCode.TIMEOUT:
            ctx.deferred_parses.add(paper_id)
        await publish(status="waiting" if exc.code == ErrorCode.TIMEOUT else "failed")
        raise
    except asyncio.CancelledError:
        observer.cancel()
        await publish(status="cancelled")
        raise
    finally:
        observer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await observer
    await publish(status="done" if handle.status == "done" else "waiting")
    if handle.status != "done":
        ctx.deferred_parses.add(paper_id)
    return handle


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
        output_dir = await mineru_adapter.resume_parse(
            paper_id, parse_id, batch_id, queue_timeout_s=settings.mineru_queue_timeout_seconds
        )
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
        output_dir = await mineru_adapter.parse_pdf(
            paper_id, parse_id, queue_timeout_s=settings.mineru_queue_timeout_seconds
        )
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
