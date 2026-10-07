"""Removing data: what a reader deletes, and what nobody is using any more.

Until this module nothing was ever removed. Two different things are done here
and they share one set of deletions, so that "the reader deleted it" and "the
retention pass deleted it" cannot come to mean different amounts of data.

*What a reader deletes* is theirs: a conversation, a project, their account. A
conversation goes with everything made in it — messages, events, evidence, the
LangGraph thread, and the reports it produced.

*What nobody is using* is decided by the daily pass: streamed answer fragments
of finished turns (the answer itself is a message), expired login sessions and
old counters, anonymous visitors who have not come back, and paper files no run
and no conversation refers to.

Papers are the one thing a reader cannot delete. A file is stored once by its
content and belongs to nobody, so it is only ever collected by reference.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.db import agent_repository as repo
from app.db import database as db
from app.services import mode_runs

logger = logging.getLogger("scholar.lifecycle")

# How long a deletion waits for a turn it interrupted to finish writing.
_STOP_WAIT_SECONDS = 5.0

# Rows removed per statement, so a large backlog is worked off in short
# transactions rather than one long one.
_BATCH = 5000

# ── What a reader deletes ───────────────────────────────────────────────────


async def _stop_turns(session_id: str) -> None:
    from app.services import agent_runner

    active = await db.fetch_all(
        "SELECT run_id FROM agent_runs WHERE session_id = ? AND status IN ('pending', 'running')",
        (session_id,),
    )
    stopped = [row["run_id"] for row in active if await agent_runner.request_stop(row["run_id"])]
    deadline = asyncio.get_running_loop().time() + _STOP_WAIT_SECONDS
    while any(agent_runner.is_executing(run_id) for run_id in stopped):
        if asyncio.get_running_loop().time() > deadline:
            break
        await asyncio.sleep(0.1)


async def delete_session(session_id: str, *, owner_id: str) -> None:
    """Delete a conversation and everything made in it.

    Raises `SessionNotFound` for a session that is missing or not the caller's.
    The reports it produced go too: they were answers in this conversation.
    Downloads and parses stay — they are keyed, shared work, and the papers
    they produced are collected by reference like any other.
    """
    session = await repo.get_session(session_id, owner_id=owner_id)
    await _stop_turns(session_id)

    reports = await db.fetch_all(
        "SELECT run_id FROM runs WHERE agent_session_id = ?", (session_id,)
    )
    for row in reports:
        await mode_runs.delete(row["run_id"])

    async with db.transaction() as tx:
        for table in ("agent_events", "agent_messages", "session_papers", "evidence", "agent_runs"):
            await tx.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
        await tx.execute(
            "UPDATE agent_jobs SET session_id = '', run_id = '' WHERE session_id = ?", (session_id,)
        )
        await tx.execute(
            "UPDATE agent_memories SET source_session_id = '', source_run_id = '' "
            "WHERE source_session_id = ?",
            (session_id,),
        )
        await tx.execute("DELETE FROM agent_sessions WHERE session_id = ?", (session_id,))

    await _delete_thread(session.thread_id)
    logger.info("Deleted session %s (%d report(s))", session_id, len(reports))


async def _delete_thread(thread_id: str) -> None:
    """Drop the LangGraph thread. Best effort: the session is already gone, and
    a thread without one is unreachable."""
    from app.services import agent_runner

    saver = agent_runner._checkpointer()
    if saver is None or not thread_id:
        return
    try:
        await saver.adelete_thread(thread_id)
    except Exception:  # noqa: BLE001
        logger.warning("Could not delete checkpoint thread %s", thread_id, exc_info=True)


async def delete_project(project_id: str, *, owner_id: str, delete_sessions: bool) -> int:
    """Delete a project and its memories. Returns how many sessions were deleted.

    Its conversations are deleted with it, or — `delete_sessions=False` — moved
    out to stand on their own. Raises `ProjectNotFound`.
    """
    await repo.get_project(project_id, owner_id=owner_id)
    deleted = 0
    if delete_sessions:
        rows = await db.fetch_all(
            "SELECT session_id FROM agent_sessions WHERE project_id = ? AND owner_id = ?",
            (project_id, owner_id),
        )
        for row in rows:
            await delete_session(row["session_id"], owner_id=owner_id)
            deleted += 1
    async with db.transaction() as tx:
        await tx.execute(
            "UPDATE agent_sessions SET project_id = '' WHERE project_id = ?", (project_id,)
        )
        await tx.execute("DELETE FROM agent_memories WHERE project_id = ?", (project_id,))
        await tx.execute("DELETE FROM agent_projects WHERE project_id = ?", (project_id,))
    logger.info("Deleted project %s (%d session(s) with it)", project_id, deleted)
    return deleted


async def delete_principal(principal_id: str) -> dict[str, int]:
    """Delete everything a principal owns, then the principal.

    Its credential stops resolving with it. Returns what was removed.
    """
    sessions = await db.fetch_all(
        "SELECT session_id FROM agent_sessions WHERE owner_id = ?", (principal_id,)
    )
    for row in sessions:
        await delete_session(row["session_id"], owner_id=principal_id)
    runs = await db.fetch_all("SELECT run_id FROM runs WHERE owner_id = ?", (principal_id,))
    for row in runs:
        await mode_runs.delete(row["run_id"])

    async with db.transaction() as tx:
        for table in ("agent_memories", "agent_projects", "evidence", "quota_charges"):
            await tx.execute(f"DELETE FROM {table} WHERE owner_id = ?", (principal_id,))
        for table in ("auth_sessions", "user_identities", "users"):
            await tx.execute(f"DELETE FROM {table} WHERE principal_id = ?", (principal_id,))
        # Visitors merged into this account point at it and own nothing.
        await tx.execute("DELETE FROM agent_principals WHERE merged_into = ?", (principal_id,))
        await tx.execute("DELETE FROM agent_principals WHERE principal_id = ?", (principal_id,))
    removed = {"sessions": len(sessions), "runs": len(runs)}
    logger.info("Deleted principal %s: %s", principal_id, removed)
    return removed


async def owned_counts(principal_id: str) -> dict[str, int]:
    """What deleting this principal would remove, for the confirmation."""
    row = await db.fetch_one(
        "SELECT (SELECT COUNT(*) FROM agent_sessions WHERE owner_id = ?) AS sessions, "
        "(SELECT COUNT(*) FROM runs WHERE owner_id = ?) AS runs, "
        "(SELECT COUNT(*) FROM agent_projects WHERE owner_id = ?) AS projects",
        (principal_id, principal_id, principal_id),
    )
    return {key: int(row[key] or 0) for key in ("sessions", "runs", "projects")}


# ── What nobody is using ────────────────────────────────────────────────────


async def _delete_in_batches(count_sql: str, delete_sql: str, params: tuple[Any, ...], *, dry_run: bool) -> int:
    """Count what a rule matches, or delete it `_BATCH` rows at a time."""
    if dry_run:
        row = await db.fetch_one(count_sql, params)
        return int(row["n"] or 0)
    total = 0
    while True:
        removed = await db.execute(delete_sql, (*params, _BATCH))
        total += removed
        if removed < _BATCH:
            return total
        await asyncio.sleep(0)


async def prune_answer_fragments(days: int, *, dry_run: bool = False) -> int:
    """`message.delta` events of turns that ended more than `days` ago.

    They exist so a reconnecting browser can replay an answer in progress. Once
    the turn is over the answer is a message, and the activity replay already
    leaves them out.
    """
    where = (
        "e.type = 'message.delta' AND e.run_id = r.run_id "
        "AND r.status NOT IN ('pending', 'running') "
        "AND r.finished_at < now() - make_interval(days => ?)"
    )
    return await _delete_in_batches(
        f"SELECT COUNT(*) AS n FROM agent_events e, agent_runs r WHERE {where}",
        "DELETE FROM agent_events WHERE (session_id, seq) IN ("
        f"SELECT e.session_id, e.seq FROM agent_events e, agent_runs r WHERE {where} LIMIT ?)",
        (days,),
        dry_run=dry_run,
    )


async def prune_logs(days: int, *, dry_run: bool = False) -> int:
    """Login sessions past use, and counters older than anything reads them."""
    rules = (
        ("auth_sessions", "token_hash",
         "(expires_at < now() OR revoked_at IS NOT NULL) "
         "AND COALESCE(revoked_at, expires_at) < now() - make_interval(days => ?)"),
        ("quota_charges", "charge_id", "created_at < now() - make_interval(days => ?)"),
        ("traffic_visitors", "visitor_hash", "last_seen_at < now() - make_interval(days => ?)"),
    )
    total = 0
    for table, key, where in rules:
        total += await _delete_in_batches(
            f"SELECT COUNT(*) AS n FROM {table} WHERE {where}",
            f"DELETE FROM {table} WHERE {key} IN (SELECT {key} FROM {table} WHERE {where} LIMIT ?)",
            (days,),
            dry_run=dry_run,
        )
    return total


async def purge_idle_visitors(days: int, *, dry_run: bool = False) -> int:
    """Anonymous principals nobody has presented for `days`, with their data.

    A visitor who logged in was merged into the account and is not idle data;
    those rows go with the account.
    """
    rows = await db.fetch_all(
        "SELECT principal_id FROM agent_principals "
        "WHERE kind = 'anonymous' AND merged_into IS NULL "
        "AND last_seen_at < now() - make_interval(days => ?) "
        "ORDER BY last_seen_at LIMIT ?",
        (days, _BATCH),
    )
    if dry_run:
        return len(rows)
    for row in rows:
        try:
            await delete_principal(row["principal_id"])
        except Exception:  # noqa: BLE001 — one stubborn visitor must not stop the pass
            logger.exception("Could not purge idle visitor %s", row["principal_id"])
    return len(rows)


_ORPHAN_PAPERS = """
    SELECT p.paper_id FROM papers p
     WHERE p.created_at < now() - make_interval(days => ?)
       AND NOT EXISTS (SELECT 1 FROM runs r WHERE r.paper_id = p.paper_id)
       AND NOT EXISTS (SELECT 1 FROM session_papers sp WHERE sp.paper_id = p.paper_id)
       AND NOT EXISTS (SELECT 1 FROM literature_files lf
                         JOIN session_papers sp ON sp.literature_id = lf.literature_id
                        WHERE lf.paper_id = p.paper_id)
       AND NOT EXISTS (SELECT 1 FROM mineru_parses m
                        WHERE m.paper_id = p.paper_id
                          AND m.updated_at > now() - make_interval(days => 1))
     LIMIT ?
"""


def _paper_dir(paper_id: str) -> Path | None:
    root = (get_settings().data_dir / "papers").resolve()
    path = (root / paper_id).resolve()
    return path if path.parent == root else None


def _dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


async def delete_paper(paper_id: str) -> None:
    """Remove a paper's rows and files. For papers nothing refers to.

    Evidence quoting it stays: evidence is a snapshot of the passage, not a
    pointer into the file. The keyed download and parse records go, or a later
    request for the same paper would be handed a result that no longer exists.
    """
    async with db.transaction() as tx:
        for table in ("blocks", "paper_nodes", "paper_versions", "paper_signals",
                      "mineru_parses", "literature_files"):
            await tx.execute(f"DELETE FROM {table} WHERE paper_id = ?", (paper_id,))
        await tx.execute(
            "DELETE FROM agent_jobs WHERE idempotency_key LIKE ? OR result_json LIKE ?",
            (f"parse:{paper_id}:%", f"%{paper_id}%"),
        )
        await tx.execute("DELETE FROM papers WHERE paper_id = ?", (paper_id,))
    path = _paper_dir(paper_id)
    if path is not None:
        await asyncio.to_thread(shutil.rmtree, path, True)


async def collect_orphan_papers(days: int, *, dry_run: bool = False) -> dict[str, int]:
    """Papers older than `days` that no run and no conversation refers to."""
    rows = await db.fetch_all(_ORPHAN_PAPERS, (days, _BATCH))
    freed = 0
    for row in rows:
        path = _paper_dir(row["paper_id"])
        if path is not None:
            freed += await asyncio.to_thread(_dir_bytes, path)
        if dry_run:
            continue
        try:
            await delete_paper(row["paper_id"])
        except Exception:  # noqa: BLE001
            logger.exception("Could not collect orphan paper %s", row["paper_id"])
    return {"papers": len(rows), "bytes": freed}


def _sweep_redundant_parse_files(dry_run: bool) -> int:
    from app.services.mineru_adapter import drop_redundant_parse_files

    root = get_settings().data_dir / "papers"
    if not root.exists():
        return 0
    return sum(
        drop_redundant_parse_files(raw, dry_run=dry_run) for raw in root.glob("*/mineru/raw")
    )


async def run_retention(*, dry_run: bool = False) -> dict[str, int]:
    """One pass over every rule. A rule whose setting is 0 is skipped."""
    settings = get_settings()
    report: dict[str, int] = {}
    # A parse in flight has its archive on disk but not yet extracted, and
    # `drop_redundant_parse_files` leaves those alone.
    report["parse_archive_bytes"] = await asyncio.to_thread(_sweep_redundant_parse_files, dry_run)
    if settings.retention_event_delta_days > 0:
        report["answer_fragments"] = await prune_answer_fragments(
            settings.retention_event_delta_days, dry_run=dry_run
        )
    if settings.retention_log_days > 0:
        report["log_rows"] = await prune_logs(settings.retention_log_days, dry_run=dry_run)
    if settings.retention_anon_days > 0 and settings.auth_mode != "single_user":
        report["idle_visitors"] = await purge_idle_visitors(
            settings.retention_anon_days, dry_run=dry_run
        )
    # After the visitors: the papers only they referred to are orphans now.
    if settings.retention_orphan_paper_days > 0:
        orphans = await collect_orphan_papers(
            settings.retention_orphan_paper_days, dry_run=dry_run
        )
        report["orphan_papers"] = orphans["papers"]
        report["orphan_paper_bytes"] = orphans["bytes"]
    return report
