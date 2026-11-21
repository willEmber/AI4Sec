"""The `runs` row of a mode report: stopping, deleting, heartbeat, recovery.

A report is written as a conversation turn's answer (P16,
`services/answer_reports.py`); the upload page that once started runs on their
own, the executor that ran them and the report tools are gone. What is left is
the row's own life, which the report page, the library and the exports read:

*The row is the truth.* The turn writing the report stamps its worker on the
row and heartbeats while it works (`heartbeating`); every terminal status it
writes is conditional on the run still being active, so a report that
finishes anyway cannot reopen a run somebody closed.

*Closing the row stops the report.* `cancel` closes it (deleting a report
does, first); the heartbeat of whichever replica is writing it notices and
interrupts the turn, at its next beat. The turn's own stop button closes the
row through the turn.

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
from app.services.agent_worker import HEARTBEAT_SECONDS, RUN_STALE_SECONDS, WORKER_ID

logger = logging.getLogger("scholar.runs")

INTERRUPTED = "Interrupted (the worker executing this run stopped)"


async def cancel(run_id: str) -> bool:
    """Close an active run. Returns whether it was still active.

    The turn writing the report is interrupted by its heartbeat, which stops
    beating on a closed row; the model call in flight is abandoned, not
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
