"""Idempotent records for the two slow things the agent does: download and parse.

A download or a MinerU parse costs minutes and real money, and the model may
ask for the same one twice — in one turn, in a retried request, or after a
process restart. So the unit of work is keyed, not repeated: an
`idempotency_key` derived from what is being fetched or parsed (plus the parser
configuration) decides whether this is new work or a result that already exists.

Three states matter to a caller:

*Done.* The stored result is returned. No network call happens at all, which is
what makes a repeated request cheap rather than merely correct (acceptance case
A12).

*Running.* If this process owns the work, the caller awaits the same task
instead of starting a second one. If another process owns it — or this one
restarted — the lease and `remote_id` are what a worker uses to pick it back up
(acceptance case A13); P4 adds that worker, and this module records what it will
need.

*Absent or retryable failure.* New work is started, under a lease.

The lease exists so a crashed worker's job becomes claimable again rather than
staying `running` forever.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from app.db import database as db
from app.models.agent_models import ErrorCode

logger = logging.getLogger("scholar.agent.jobs")

# How long a claim is good for. A job still `running` past this is assumed to
# belong to a worker that died, and may be claimed again.
LEASE_SECONDS = 1800

# A job that failed this many times stops being retried: repeatedly resubmitting
# an external parse that keeps failing costs money and gets no further.
MAX_ATTEMPTS = 3

# Jobs this process is currently executing, so a second caller in the same
# process awaits the same work rather than racing it.
_inflight: dict[str, asyncio.Task[dict[str, Any]]] = {}


class JobFailed(RuntimeError):
    """A job finished in `failed` and its error is what the caller must report."""

    def __init__(self, code: ErrorCode, message: str, *, attempts: int = 0) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.attempts = attempts


@dataclass
class JobHandle:
    """What a caller learns about a unit of work, whether or not it ran now."""

    job_id: str
    kind: str
    status: str
    result: dict[str, Any]
    reused: bool = False
    attempts: int = 0
    remote_id: str = ""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _lease_expiry() -> str:
    return (_now() + timedelta(seconds=LEASE_SECONDS)).isoformat()


def _lease_is_live(expires_at: str | None) -> bool:
    if not expires_at:
        return False
    try:
        expires = datetime.fromisoformat(expires_at)
    except ValueError:
        return False
    # The database hands timestamps back as naive UTC.
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    return expires > _now()


async def get_job(idempotency_key: str) -> dict[str, Any] | None:
    return await db.fetch_one(
        "SELECT * FROM agent_jobs WHERE idempotency_key = ?", (idempotency_key,)
    )


async def get_job_by_id(job_id: str) -> dict[str, Any] | None:
    return await db.fetch_one("SELECT * FROM agent_jobs WHERE job_id = ?", (job_id,))


async def list_stale_jobs(*, kind: str = "") -> list[dict[str, Any]]:
    """Jobs still marked `running` whose lease has lapsed.

    These are the ones a crash left behind: nobody is executing them, and
    without the sweep they would sit `running` until some caller happened to ask
    for the same work again. Ordered oldest first so the longest-abandoned work
    is recovered first.
    """
    sql = (
        "SELECT * FROM agent_jobs WHERE status = 'running' "
        "AND (lease_expires_at IS NULL OR lease_expires_at < ?)"
    )
    params: list[Any] = [_now().isoformat()]
    if kind:
        sql += " AND kind = ?"
        params.append(kind)
    sql += " ORDER BY updated_at"
    return await db.fetch_all(sql, tuple(params))


async def release_lease(job_id: str) -> None:
    """Hand an abandoned job back to whoever asks for it next.

    `_claim` would already take a job whose lease has lapsed, so this changes no
    permission. What it changes is legibility: a job sitting in `running` with
    nobody running it reads as work in progress, both to an operator and to the
    `status == "running"` branch a tool reports to the model.
    """
    await db.execute(
        """UPDATE agent_jobs
              SET status = 'pending', lease_owner = '', lease_expires_at = NULL,
                  updated_at = now()
            WHERE job_id = ? AND status = 'running'""",
        (job_id,),
    )


async def fail_exhausted_job(job_id: str) -> bool:
    """Close abandoned work that cannot be claimed again, without touching a live lease."""
    row = await db.execute_returning(
        """UPDATE agent_jobs
              SET status = 'failed', lease_owner = '', lease_expires_at = NULL,
                  error_code = CASE WHEN error_code = '' THEN 'upstream_error' ELSE error_code END,
                  error_msg = CASE WHEN error_msg = '' THEN 'Retry limit reached after an interrupted job' ELSE error_msg END,
                  updated_at = now()
            WHERE job_id = ? AND attempts >= ? AND status IN ('pending', 'running')
              AND (status = 'pending' OR lease_expires_at IS NULL OR lease_expires_at < ?)
           RETURNING job_id""",
        (job_id, MAX_ATTEMPTS, _now().isoformat()),
    )
    return row is not None


async def set_remote_id(job_id: str, remote_id: str) -> None:
    """Record the external task id, so a restart can query it instead of resubmitting."""
    await db.execute(
        "UPDATE agent_jobs SET remote_id = ?, updated_at = now() WHERE job_id = ?",
        (remote_id, job_id),
    )


async def forget(idempotency_key: str) -> bool:
    """Drop a finished job so the same work may be attempted again.

    Reuse is right for a result that cannot change — a parse of the same bytes
    with the same parser — and wrong for one that can. "No open-access copy
    exists" is true on the day it is recorded and may be false a month later, so
    the caller that knows its result has a shelf life expires it here rather
    than having the job layer guess a TTL for every kind of work.

    Refuses to touch a running job: a lease holder's work is not the caller's to
    discard.
    """
    row = await get_job(idempotency_key)
    if row is None or row["status"] == "running":
        return False
    await db.execute("DELETE FROM agent_jobs WHERE idempotency_key = ?", (idempotency_key,))
    return True


async def renew_lease(job_id: str) -> None:
    await db.execute(
        "UPDATE agent_jobs SET lease_expires_at = ?, updated_at = now() "
        "WHERE job_id = ?",
        (_lease_expiry(), job_id),
    )


async def _claim(
    *,
    kind: str,
    idempotency_key: str,
    session_id: str,
    run_id: str,
    request: dict[str, Any],
    owner: str,
) -> tuple[str, int] | None:
    """Take ownership of this unit of work, or return None if someone else holds it.

    The insert is the lock: `idx_agent_jobs_idempotent` is a unique index, so two
    concurrent callers cannot both create the row, and whoever loses falls
    through to the update, which only succeeds against a job that is finished,
    failed or whose lease has lapsed.
    """
    job_id = f"job_{uuid.uuid4().hex[:20]}"
    # RETURNING makes the outcome observable: `ON CONFLICT DO NOTHING` yields a row
    # only when it actually inserted, which is how the winner of the race is
    # decided without a separate read.
    inserted = await db.execute_returning(
        """INSERT INTO agent_jobs
               (job_id, kind, idempotency_key, session_id, run_id, status,
                attempts, lease_owner, lease_expires_at, request_json)
           VALUES (?, ?, ?, ?, ?, 'running', 1, ?, ?, ?)
           ON CONFLICT DO NOTHING
           RETURNING job_id""",
        (
            job_id,
            kind,
            idempotency_key,
            session_id,
            run_id,
            owner,
            _lease_expiry(),
            json.dumps(request, ensure_ascii=False),
        ),
    )
    if inserted is not None:
        return job_id, 1

    existing = await get_job(idempotency_key)
    if existing is None:  # deleted between the insert and the read
        return None
    if existing["status"] == "done":
        # Finished between this caller's first read and its claim attempt. The
        # re-claim below only guards against a *live* lease, so without this a
        # late caller would re-run work that had already succeeded — which is
        # exactly what the idempotency key exists to prevent.
        return None
    if existing["status"] == "running" and _lease_is_live(existing["lease_expires_at"]):
        return None
    if existing["attempts"] >= MAX_ATTEMPTS:
        return None

    claimed = await db.execute_returning(
        """UPDATE agent_jobs
              SET status = 'running', attempts = attempts + 1, lease_owner = ?,
                  lease_expires_at = ?, run_id = ?, updated_at = now()
            WHERE job_id = ? AND status <> 'done'
              AND attempts < ?
              AND (status <> 'running' OR lease_expires_at IS NULL OR lease_expires_at < ?)
           RETURNING job_id, attempts""",
        (
            owner,
            _lease_expiry(),
            run_id,
            existing["job_id"],
            MAX_ATTEMPTS,
            _now().isoformat(),
        ),
    )
    if claimed is None:
        return None
    return existing["job_id"], int(claimed["attempts"])


async def _finish(
    job_id: str, *, status: str, result: dict[str, Any] | None = None,
    error_code: str = "", error_msg: str = "",
    refund_attempt: bool = False,
) -> None:
    await db.execute(
        """UPDATE agent_jobs
              SET status = ?, result_json = ?, error_code = ?, error_msg = ?,
                  lease_owner = '', lease_expires_at = NULL,
                  attempts = GREATEST(0, attempts - ?),
                  updated_at = now()
            WHERE job_id = ?""",
        (
            status,
            json.dumps(result or {}, ensure_ascii=False),
            error_code,
            error_msg[:500],
            int(refund_attempt),
            job_id,
        ),
    )


def _stored_result(row: dict[str, Any]) -> dict[str, Any]:
    try:
        parsed = json.loads(row["result_json"] or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def _await_shared(task: asyncio.Task[dict[str, Any]], key: str) -> dict[str, Any]:
    """Give joined callers the same failure contract as the job's owner."""
    from app.services.mineru_adapter import MinerUPollTimeoutError

    try:
        return await asyncio.shield(task)
    except Exception as exc:
        row = await get_job(key)
        code = ErrorCode.TIMEOUT if isinstance(exc, MinerUPollTimeoutError) else ErrorCode.UPSTREAM_ERROR
        raise JobFailed(code, str(exc), attempts=row["attempts"] if row else 0) from exc


async def run_once(
    *,
    kind: str,
    idempotency_key: str,
    work: Callable[[str], Awaitable[dict[str, Any]]],
    session_id: str = "",
    run_id: str = "",
    request: dict[str, Any] | None = None,
) -> JobHandle:
    """Execute `work` at most once per `idempotency_key`, reusing any result.

    `work` receives the `job_id` so it can record an external task id through
    :func:`set_remote_id` — that id is the difference between resuming a
    submitted MinerU parse after a restart and paying for it twice.

    Raises :class:`JobFailed` when the job has failed and is out of attempts, so
    the caller reports a real error rather than an empty success.
    """
    existing = await get_job(idempotency_key)
    if existing is not None and existing["status"] == "done":
        return JobHandle(
            job_id=existing["job_id"],
            kind=existing["kind"],
            status="done",
            result=_stored_result(existing),
            reused=True,
            attempts=existing["attempts"],
            remote_id=existing["remote_id"],
        )

    # Same work already in flight here: wait on it rather than duplicating it.
    inflight = _inflight.get(idempotency_key)
    if inflight is not None:
        result = await _await_shared(inflight, idempotency_key)
        row = await get_job(idempotency_key)
        return JobHandle(
            job_id=row["job_id"] if row else "",
            kind=kind,
            status="done",
            result=result,
            reused=True,
            attempts=row["attempts"] if row else 0,
            remote_id=row["remote_id"] if row else "",
        )

    owner = f"proc-{uuid.uuid4().hex[:8]}"
    claim = await _claim(
        kind=kind,
        idempotency_key=idempotency_key,
        session_id=session_id,
        run_id=run_id,
        request=request or {},
        owner=owner,
    )
    if claim is None:
        row = await get_job(idempotency_key)
        if row is not None and row["status"] == "done":
            return JobHandle(
                job_id=row["job_id"], kind=row["kind"], status="done",
                result=_stored_result(row), reused=True,
                attempts=row["attempts"], remote_id=row["remote_id"],
            )
        if row is not None and row["attempts"] >= MAX_ATTEMPTS and (
            row["status"] != "running" or not _lease_is_live(row["lease_expires_at"])
        ):
            await fail_exhausted_job(row["job_id"])
            row = await get_job(idempotency_key)
            if row is not None and row["status"] != "done" and (
                row["status"] != "running" or not _lease_is_live(row["lease_expires_at"])
            ):
                raise JobFailed(
                    ErrorCode.UPSTREAM_ERROR,
                    row["error_msg"] or f"{kind} failed {row['attempts']} times; not retrying",
                    attempts=row["attempts"],
                )
        # Lost a race inside this process: the winner registered its task just
        # after our first check, so wait on that instead of reporting `running`.
        inflight = _inflight.get(idempotency_key)
        if inflight is not None:
            result = await _await_shared(inflight, idempotency_key)
            return JobHandle(
                job_id=row["job_id"] if row else "",
                kind=kind,
                status="done",
                result=result,
                reused=True,
                attempts=row["attempts"] if row else 0,
                remote_id=row["remote_id"] if row else "",
            )

        # One last look before reporting work as pending: the holder may have
        # finished in the moment between the read above and the in-flight check,
        # and telling the caller "still running" about work that is done would
        # cost it the result for no reason.
        await asyncio.sleep(0)
        row = await get_job(idempotency_key)
        if row is not None and row["status"] == "done":
            return JobHandle(
                job_id=row["job_id"], kind=row["kind"], status="done",
                result=_stored_result(row), reused=True,
                attempts=row["attempts"], remote_id=row["remote_id"],
            )

        # Someone else holds a live lease. Report it rather than starting a
        # duplicate: the caller turns this into a `partial` the model can act on.
        return JobHandle(
            job_id=row["job_id"] if row else "",
            kind=kind,
            status="running",
            result={},
            attempts=row["attempts"] if row else 0,
            remote_id=row["remote_id"] if row else "",
        )

    job_id, attempts = claim

    async def _run() -> dict[str, Any]:
        try:
            result = await work(job_id)
        except asyncio.CancelledError:
            # The turn was stopped, so this work has no waiter — but it did not
            # fail, and an external service may well have finished it. Hand the
            # lease back with `remote_id` intact: the next attempt rejoins that
            # submission instead of paying for it again. Marking it `failed`
            # here would spend an attempt on something that never went wrong.
            await _finish(job_id, status="pending", refund_attempt=True)
            raise
        except Exception as exc:  # noqa: BLE001 — recorded, then re-raised
            from app.services.mineru_adapter import MinerUPollTimeoutError

            waiting = isinstance(exc, MinerUPollTimeoutError)
            await _finish(
                job_id, status="pending" if waiting else "failed",
                error_code=ErrorCode.TIMEOUT.value if waiting else ErrorCode.UPSTREAM_ERROR.value,
                error_msg=str(exc), refund_attempt=waiting,
            )
            raise
        await _finish(job_id, status="done", result=result)
        return result

    task = asyncio.create_task(_run())
    _inflight[idempotency_key] = task
    try:
        result = await task
    except Exception as exc:  # noqa: BLE001
        from app.services.mineru_adapter import MinerUPollTimeoutError

        if isinstance(exc, MinerUPollTimeoutError):
            raise JobFailed(ErrorCode.TIMEOUT, str(exc), attempts=attempts) from exc
        logger.warning("Job %s (%s) failed on attempt %s: %s", job_id, kind, attempts, exc)
        raise JobFailed(ErrorCode.UPSTREAM_ERROR, str(exc), attempts=attempts) from exc
    finally:
        _inflight.pop(idempotency_key, None)

    row = await get_job(idempotency_key)
    return JobHandle(
        job_id=job_id,
        kind=kind,
        status="done",
        result=result,
        reused=False,
        attempts=attempts,
        remote_id=row["remote_id"] if row else "",
    )
