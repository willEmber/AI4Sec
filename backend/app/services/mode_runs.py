"""The `runs` row of a mode report: stopping, deleting, heartbeat, recovery.

A report is made inside a conversation turn (`agents/tools/modes.py`); the
upload page that once started runs on their own, and the executor that ran
them, are gone (P16). What is left is the row's own life, which the report
page, the library and the exports read:

*The row is the truth.* The tool stamps its worker on the row and heartbeats
while it works (`heartbeating`); every terminal status it writes is
conditional on the run still being active, so a report that finishes anyway
cannot reopen a run somebody closed.

*Stopping is a change to the row.* `cancel` closes it; the heartbeat of
whichever replica is making the report notices and interrupts the tool, at its
next beat. The turn's own stop button does the same through the turn.

*An orphaned run is closed, not repeated.* Its turn died with the same worker
and is closed as `interrupted`; the turn that asks again finds the report job
released and makes it anew.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator

from app.db import database as db
from app.services import llm_usage
from app.services.agent_worker import HEARTBEAT_SECONDS, RUN_STALE_SECONDS, WORKER_ID

logger = logging.getLogger("scholar.runs")

ACTIVE_STATUSES = ("pending", "running")

INTERRUPTED = "Interrupted (the worker executing this run stopped)"


async def cancel(run_id: str) -> bool:
    """Close an active run. Returns whether it was still active.

    The tool making the report is interrupted by its own heartbeat, which
    stops beating on a closed row; the model call in flight is abandoned, not
    refunded.
    """
    closed = await db.execute(
        "UPDATE runs SET status = 'cancelled', error_msg = 'Cancelled by user', "
        "finished_at = now() WHERE run_id = ? AND status IN ('pending', 'running')",
        (run_id,),
    )
    if closed:
        logger.info(f"[run:{run_id}] Cancelled by user")
    return bool(closed)


async def delete(run_id: str) -> None:
    """Remove a run and everything stored for it. Stops it first if it is active."""
    await cancel(run_id)
    async with db.transaction() as tx:
        for table in ("sphere_edges", "sphere_nodes", "run_outputs", "runs"):
            await tx.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))


@contextlib.asynccontextmanager
async def heartbeating(run_id: str) -> AsyncIterator[None]:
    """Mark this process as the run's executor for as long as the block lasts.

    The beat is also how a cancel issued on another replica arrives: once the
    row is no longer active there is nothing to report on, and the task that
    entered the block is interrupted.
    """
    owner = asyncio.current_task()
    await db.execute(
        "UPDATE runs SET worker_id = ?, heartbeat_at = now() WHERE run_id = ?",
        (WORKER_ID, run_id),
    )

    async def beat() -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_SECONDS)
            try:
                alive = await db.execute(
                    "UPDATE runs SET heartbeat_at = now() "
                    "WHERE run_id = ? AND status IN ('pending', 'running')",
                    (run_id,),
                )
            except Exception:  # noqa: BLE001 — one missed beat is not a dead run
                logger.warning(f"[run:{run_id}] Heartbeat failed", exc_info=True)
                continue
            if not alive:
                if owner is not None and not owner.done():
                    owner.cancel()
                return

    beater = asyncio.create_task(beat(), name=f"run-heartbeat:{run_id}")
    try:
        yield
    finally:
        beater.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await beater


@contextlib.asynccontextmanager
async def recording_tokens(run_id: str) -> AsyncIterator[llm_usage.UsageMeter]:
    """Add what the block spends on the model to the run's `tokens`.

    Added, not set, and written however the block ends: a failed or cancelled
    report was billed up to the point it stopped.
    """
    meter = llm_usage.metering()
    try:
        with meter:
            yield meter
    finally:
        if meter.total:
            with contextlib.suppress(Exception):
                await asyncio.shield(
                    db.execute(
                        "UPDATE runs SET tokens = tokens + ? WHERE run_id = ?",
                        (meter.total, run_id),
                    )
                )


async def recover_stale(*, stale_after_seconds: int = RUN_STALE_SECONDS) -> dict[str, list[str]]:
    """Close runs whose executor stopped reporting.

    A report's turn is closed as `interrupted` by the same sweep, so there is
    nobody left to finish the report. A run still waiting from the upload
    page — `pending`, from before that page went away — has no executor
    either, and is closed the same way.
    """
    # The staleness test is in the write, so of two replicas sweeping at once
    # only one closes a run.
    async with db.transaction() as tx:
        rows = await tx.fetch_all(
            "UPDATE runs SET status = 'failed', error_msg = ?, finished_at = now(), "
            "worker_id = ?, heartbeat_at = now() "
            "WHERE status IN ('pending', 'running') "
            "AND COALESCE(heartbeat_at, started_at) < now() - make_interval(secs => ?) "
            "RETURNING run_id",
            (INTERRUPTED, WORKER_ID, stale_after_seconds),
        )
    closed = sorted(row["run_id"] for row in rows)
    for run_id in closed:
        logger.warning(f"[run:{run_id}] Worker gone; closed as interrupted")
    return {"mode_runs_closed": closed}
