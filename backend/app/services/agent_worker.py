"""The worker identity, and what happens to work whose worker stopped existing.

Turns and jobs are executed by a process. Processes are restarted, deployed
over and killed, and when that happens mid-turn the database is left holding
rows that claim to be in progress with nobody in progress on them. Those rows
are not self-correcting: a run stuck in `running` keeps its session's
one-active-run slot forever, and a browser reconnecting to its event stream
waits for a terminal event that will never be written.

The fix is to make "is anyone executing this?" an observable fact rather than
an assumption. A worker stamps its identity on what it takes and says so every
few seconds; anything active whose last word is older than the stale window has
no executor, whoever is asking and whenever they ask. That test works with two
processes running, and — unlike "everything active at startup is dead" — it can
be exercised without killing one.

What recovery means differs by the kind of work, and the difference is the point:

*A turn is restarted, not resumed.* Its model output was streamed; the graph
checkpoint sits at the last completed step, so resuming would re-run the call
that was interrupted and emit the first half of the answer twice. So an orphaned
run is closed as `interrupted` — which costs the reader one "ask again" and
costs the operator nothing, because the expensive work that turn had already
done is keyed and will be reused rather than repeated.

*A parse is resumed.* MinerU has the file and a batch id was recorded before the
first poll, so the result is collectable. That is the half worth recovering, and
recovering it is the difference between one parse and two.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import socket
import uuid
from typing import Any

from app.db import agent_repository as repo
from app.models.agent_models import Availability, ErrorCode, EventType, RunStatus
from app.services import agent_jobs, paper_catalog, parse_service

# `agent_runner` is imported inside the functions that need it: it imports this
# module for the worker identity and heartbeat interval, so taking it at module
# level would be a cycle.

logger = logging.getLogger("scholar.agent.worker")

# Who this process is. Host and pid make it readable in a log; the random tail
# keeps two processes on one host from colliding after a fast restart reuses a
# pid.
WORKER_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"

# How often an executing run says it is still alive.
HEARTBEAT_SECONDS = 15

# How long silence is tolerated before a run counts as abandoned. Several
# missed heartbeats rather than one: a loaded event loop can be late without
# being dead, and declaring a live run orphaned would end a turn the reader is
# still watching.
RUN_STALE_SECONDS = 90

# How often the sweep looks. Recovery is not urgent — nothing is being lost
# while a row sits stale — so this is deliberately slower than the heartbeat.
SWEEP_INTERVAL_SECONDS = 30

# Retention removes what nobody is using; nothing depends on when. Once a day,
# and not in the first minutes of a process, which belong to recovery.
RETENTION_INTERVAL_SECONDS = 24 * 3600
RETENTION_FIRST_DELAY_SECONDS = 600

_sweeper: asyncio.Task[None] | None = None
_retention: asyncio.Task[None] | None = None
_resumes: set[asyncio.Task[None]] = set()


# ── Runs ────────────────────────────────────────────────────────────────────


async def recover_stale_runs(*, stale_after_seconds: int = RUN_STALE_SECONDS) -> list[str]:
    """Close runs whose worker stopped reporting. Returns the run ids closed."""
    from app.services import agent_runner

    recovered: list[str] = []
    for run in await repo.list_stale_runs(stale_after_seconds=stale_after_seconds):
        if agent_runner.is_executing(run.run_id):
            # This process is running it and merely late with a heartbeat.
            # Ending it here would kill a turn someone is watching.
            continue
        await repo.finish_run(
            run.run_id,
            status=RunStatus.FAILED,
            error_code=ErrorCode.INTERRUPTED.value,
            error_msg="The worker executing this turn stopped before it finished.",
            usage=run.usage,
        )
        with contextlib.suppress(Exception):
            await agent_runner.emit(
                session_id=run.session_id,
                run_id=run.run_id,
                type=EventType.RUN_FAILED,
                payload={
                    "code": ErrorCode.INTERRUPTED.value,
                    "error": "This turn was interrupted by a restart. Ask again — any "
                    "download or parse it had already done is reused, not repeated.",
                    "usage": run.usage,
                },
            )
        recovered.append(run.run_id)
        logger.warning(
            "Recovered interrupted run %s (worker %s, last heartbeat %s)",
            run.run_id, run.worker_id or "unknown", run.heartbeat_at or "never",
        )
    return recovered


# ── Jobs ────────────────────────────────────────────────────────────────────


async def recover_stale_jobs() -> dict[str, list[str]]:
    """Deal with jobs left `running` by a worker that is gone.

    A parse with a submitted batch is resumed in the background — that work is
    already paid for. Anything else is released, which puts it back within reach
    of the next caller that asks for it instead of leaving it looking busy.
    """
    resumed: list[str] = []
    released: list[str] = []
    for job in await agent_jobs.list_stale_jobs():
        job_id = job["job_id"]
        if job["attempts"] >= agent_jobs.MAX_ATTEMPTS:
            if await agent_jobs.fail_exhausted_job(job_id):
                logger.warning("Closed abandoned %s job %s at the retry limit", job["kind"], job_id)
            continue
        if job["kind"] == "parse" and await _parse_is_resumable(job):
            task = asyncio.create_task(_resume_parse(job))
            _resumes.add(task)
            task.add_done_callback(_resumes.discard)
            resumed.append(job_id)
            continue
        await agent_jobs.release_lease(job_id)
        released.append(job_id)
        logger.info("Released abandoned %s job %s", job["kind"], job_id)
    return {"resumed": resumed, "released": released}


async def _parse_is_resumable(job: dict[str, Any]) -> bool:
    _parse_id, batch_id = await parse_service.resumable_batch(job["job_id"])
    return bool(batch_id)


def _request_of(job: dict[str, Any]) -> dict[str, Any]:
    try:
        parsed = json.loads(job.get("request_json") or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def _resume_parse(job: dict[str, Any]) -> None:
    """Collect a parse whose batch outlived the worker that submitted it.

    Goes back through `agent_jobs.run_once`, so the resume takes the lease the
    same way any other attempt does — two workers sweeping at once cannot both
    collect the same batch, and the attempt counter still bounds how often a
    doomed parse is retried.
    """
    paper_id = str(_request_of(job).get("paper_id") or "")
    if not paper_id:
        await agent_jobs.release_lease(job["job_id"])
        return

    async def _work(job_id: str) -> dict[str, Any]:
        return await parse_service.run_parse_job(job_id=job_id, paper_id=paper_id)

    try:
        handle = await agent_jobs.run_once(
            kind="parse",
            idempotency_key=job["idempotency_key"],
            work=_work,
            session_id=job.get("session_id", ""),
            run_id="",
            request={"paper_id": paper_id},
        )
    except agent_jobs.JobFailed as exc:
        logger.warning("Could not resume parse of %s: %s", paper_id, exc.message)
        return

    if handle.status != "done":
        return
    logger.info(
        "Resumed parse of %s after a restart (version %s)",
        paper_id, handle.result.get("version_id", ""),
    )
    # The turn that asked for this is long gone, so nobody is waiting on an
    # event. What the reader will see is the paper listed as readable when they
    # next open the session, which is the only place this result can surface.
    session_id = job.get("session_id") or ""
    literature_id = await paper_catalog.literature_for_paper(paper_id)
    if session_id and literature_id:
        await repo.set_session_paper_availability(
            session_id=session_id,
            literature_id=literature_id,
            availability=Availability.PARSED,
        )


# ── The sweep ───────────────────────────────────────────────────────────────


async def sweep_once() -> dict[str, Any]:
    """One recovery pass. Separate from the loop so a test can run exactly one."""
    from app.services import mode_runs

    runs = await recover_stale_runs()
    jobs = await recover_stale_jobs()
    mode = await mode_runs.recover_stale()
    return {"runs_recovered": runs, **jobs, **mode}


async def _sweep_loop() -> None:
    while True:
        try:
            await sweep_once()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — a failed sweep must not end the sweeper
            logger.exception("Recovery sweep failed; will retry")
        await asyncio.sleep(SWEEP_INTERVAL_SECONDS)


async def _retention_loop() -> None:
    from app.services import data_lifecycle

    await asyncio.sleep(RETENTION_FIRST_DELAY_SECONDS)
    while True:
        try:
            report = await data_lifecycle.run_retention()
            if any(report.values()):
                logger.info("Retention pass: %s", report)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — a failed pass must not end the loop
            logger.exception("Retention pass failed; will retry tomorrow")
        await asyncio.sleep(RETENTION_INTERVAL_SECONDS)


async def start() -> None:
    """Begin recovering. Called from the app lifespan.

    The first sweep runs inline: whatever the previous process left behind
    should be cleaned up before this one starts serving, not one interval later
    while a browser waits on a run that ended with the old process.
    """
    global _sweeper, _retention
    if _sweeper is not None:
        return
    with contextlib.suppress(Exception):
        result = await sweep_once()
        if any(result.values()):
            logger.info("Startup recovery: %s", result)
    _sweeper = asyncio.create_task(_sweep_loop())
    _retention = asyncio.create_task(_retention_loop())


async def stop() -> None:
    """Stop sweeping and let any in-flight resume finish or be dropped."""
    global _sweeper, _retention
    for task in (_sweeper, _retention):
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
    _sweeper = _retention = None
    for task in list(_resumes):
        task.cancel()
    _resumes.clear()
    from app.services import mode_runs

    await mode_runs.stop()
