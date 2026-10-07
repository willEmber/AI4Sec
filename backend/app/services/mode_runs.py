"""Executing a mode run (Snap / Lens / Sphere / Q&A) started outside a conversation.

A run is a `runs` row and the row is the whole truth about it: the task that
executes one keeps nothing a second process would need. The task stamps its
worker on the row and heartbeats; progress is appended to the row; the event
stream follows the row. So any replica can show a run, a restart loses nothing
but the model call that was in flight, and stopping a run is a change to the
row that the executor notices wherever it is.

Unlike an agent turn, an orphaned run is run again rather than closed. A turn's
answer was streamed to the reader, so repeating it would print its first half
twice; a mode run shows nothing until it is finished, and what it had already
paid for — the parse, the metadata, the external signals — is cached where the
second attempt will find it.

Cancelling is the one terminal state the executor does not write itself. The
row is closed first and the task interrupted second, and every status the
executor writes is conditional on the run still being active, so a graph that
reaches its last node anyway cannot reopen a run somebody closed. A task that is
cancelled without the row being closed is a process shutting down, and is left
for recovery.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from app.config import get_settings
from app.db import database as db
from app.services.agent_worker import HEARTBEAT_SECONDS, RUN_STALE_SECONDS, WORKER_ID
from app.workflows.progress import emit_progress
from app.workflows.state import MainGraphState

logger = logging.getLogger("scholar.runs")

ACTIVE_STATUSES = ("pending", "running")

# A run whose worker died this many times stops being picked back up: whatever
# keeps killing it would keep being paid for.
MAX_ATTEMPTS = 3

# How long a run waits for a free slot before giving up.
SLOT_WAIT_SECONDS = 300.0

_tasks: dict[str, asyncio.Task[None]] = {}
_slots: asyncio.Semaphore | None = None
_compiled_graph: CompiledStateGraph | None = None


def _get_graph() -> CompiledStateGraph:
    global _compiled_graph
    if _compiled_graph is None:
        from app.workflows.main_graph import build_main_graph

        _compiled_graph = build_main_graph().compile()
    return _compiled_graph


def _get_slots() -> asyncio.Semaphore:
    global _slots
    if _slots is None:
        _slots = asyncio.Semaphore(max(1, get_settings().mode_run_concurrency))
    return _slots


def is_executing(run_id: str) -> bool:
    task = _tasks.get(run_id)
    return task is not None and not task.done()


def start(run_id: str) -> None:
    """Execute a `pending` run in this process."""
    if is_executing(run_id):
        return
    task = asyncio.create_task(_execute(run_id), name=f"run:{run_id}")
    _tasks[run_id] = task
    task.add_done_callback(lambda _t: _tasks.pop(run_id, None))


async def cancel(run_id: str) -> bool:
    """Close an active run and interrupt whoever is executing it.

    Returns whether the run was still active. The model call in flight is
    abandoned, not refunded; a request already inside a worker thread (the
    MinerU and search clients are synchronous) finishes there and is discarded.
    """
    closed = await db.execute(
        "UPDATE runs SET status = 'cancelled', error_msg = 'Cancelled by user', "
        "finished_at = now() WHERE run_id = ? AND status IN ('pending', 'running')",
        (run_id,),
    )
    task = _tasks.get(run_id)
    if task is not None and not task.done():
        task.cancel()
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


async def _fail(run_id: str, message: str) -> None:
    await db.execute(
        "UPDATE runs SET status = 'failed', error_msg = ?, finished_at = now() "
        "WHERE run_id = ? AND status IN ('pending', 'running')",
        (message[:2000], run_id),
    )


async def _execute(run_id: str) -> None:
    async with heartbeating(run_id):
        slots = _get_slots()
        try:
            await asyncio.wait_for(slots.acquire(), timeout=SLOT_WAIT_SECONDS)
        except asyncio.TimeoutError:
            logger.warning(f"[run:{run_id}] Timed out waiting for execution slot")
            await _fail(run_id, "Server busy, please retry later")
            return
        try:
            await _run_graph(run_id)
        finally:
            slots.release()


async def _run_graph(run_id: str) -> None:
    # Taking the run is conditional on it still being pending: it may have been
    # cancelled while it waited for a slot. A second attempt starts its step
    # list over, since it runs every step again.
    run = await db.execute_returning(
        "UPDATE runs SET status = 'running', attempts = attempts + 1, "
        "current_step = '', progress_json = '[]' "
        "WHERE run_id = ? AND status = 'pending' RETURNING *",
        (run_id,),
    )
    if run is None:
        return

    t0 = time.perf_counter()
    logger.info(f"[run:{run_id}] ▶ Graph execution started (attempt {run['attempts']})")
    initial_state: MainGraphState = {
        "paper_id": run["paper_id"],
        "run_id": run_id,
        "mode": run["mode"],
        "llm_model": run["llm_model"] or "",
        "language": run["language"],
        "user_question": run["user_question"] or "",
        "progress": [],
    }
    try:
        final_state: dict[str, Any] = {}
        async for event in _get_graph().astream(initial_state):
            # event is a dict mapping node_name -> output_dict
            for node_name, node_output in event.items():
                elapsed = time.perf_counter() - t0
                final_state.update(node_output)
                progress = node_output.get("progress", [])
                latest = progress[-1] if progress else {"step": node_name, "status": "done"}
                logger.info(f"[run:{run_id}] ✔ Node '{node_name}' done at {elapsed:.1f}s — {latest}")
                step = latest.get("step", node_name)
                status = latest.get("status", "done")
                extra = {k: v for k, v in latest.items() if k not in ("step", "status")}
                await emit_progress(run_id, step, status, **extra)

        elapsed = time.perf_counter() - t0
        error = final_state.get("error")
        if error:
            logger.error(f"[run:{run_id}] ✗ Graph failed at {elapsed:.1f}s — {error}")
            # `persist_output` records it; this covers a graph that never got there.
            await _fail(run_id, str(error))
        else:
            logger.info(f"[run:{run_id}] ✔ Graph completed at {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.exception(f"[run:{run_id}] ✗ Graph exception at {elapsed:.1f}s — {e}")
        await _fail(run_id, str(e))


async def recover_stale(*, stale_after_seconds: int = RUN_STALE_SECONDS) -> dict[str, list[str]]:
    """Take over runs whose executor stopped reporting.

    A run started from the upload page goes back to `pending` and is executed
    here, up to `MAX_ATTEMPTS`. A run made inside a conversation is closed: its
    turn is gone with the same worker, and the turn that asks again finds the
    report job released.
    """
    rows = await db.fetch_all(
        "SELECT run_id, COALESCE(agent_run_id, '') AS agent_run_id FROM runs "
        "WHERE status IN ('pending', 'running') "
        "AND COALESCE(heartbeat_at, started_at) < now() - make_interval(secs => ?)",
        (stale_after_seconds,),
    )
    resumed: list[str] = []
    closed: list[str] = []
    for row in rows:
        run_id = row["run_id"]
        if is_executing(run_id):
            # Ours, and merely late with a heartbeat.
            continue
        # The staleness test is repeated in the write, so of two replicas
        # sweeping at once only one takes the run.
        taken = await db.execute_returning(
            "UPDATE runs SET worker_id = ?, heartbeat_at = now(), "
            "status = CASE WHEN attempts >= ? OR ? <> '' THEN 'failed' ELSE 'pending' END, "
            "error_msg = CASE WHEN attempts >= ? OR ? <> '' "
            "THEN 'Interrupted (the worker executing this run stopped)' ELSE error_msg END, "
            "finished_at = CASE WHEN attempts >= ? OR ? <> '' THEN now() ELSE finished_at END "
            "WHERE run_id = ? AND status IN ('pending', 'running') "
            "AND COALESCE(heartbeat_at, started_at) < now() - make_interval(secs => ?) "
            "RETURNING status",
            (
                WORKER_ID,
                MAX_ATTEMPTS, row["agent_run_id"],
                MAX_ATTEMPTS, row["agent_run_id"],
                MAX_ATTEMPTS, row["agent_run_id"],
                run_id, stale_after_seconds,
            ),
        )
        if taken is None:
            continue
        if taken["status"] == "pending":
            start(run_id)
            resumed.append(run_id)
            logger.warning(f"[run:{run_id}] Worker gone; running it again")
        else:
            closed.append(run_id)
            logger.warning(f"[run:{run_id}] Worker gone; closed as interrupted")
    return {"mode_runs_resumed": resumed, "mode_runs_closed": closed}


async def stop() -> None:
    """Interrupt this process's runs at shutdown, leaving their rows for recovery."""
    tasks = list(_tasks.values())
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
