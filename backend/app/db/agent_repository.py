"""Persistence for agent sessions, literature identity, versions and evidence.

Everything the agent layer reads or writes goes through here, so ownership
checks and idempotency rules live in one place rather than being restated in
each API handler.

Two invariants are enforced by the schema rather than by code, because a
process restart or a second worker must not be able to violate them:

* one in-flight run per session (partial unique index on `agent_runs`), and
* one run per `(session_id, client_request_id)`.

`db.IntegrityError` from those indexes is translated into the exceptions
below.

Per-session sequence numbers (`agent_messages.seq`, `agent_events.seq`) are
allocated as `MAX(seq) + 1` under a transaction-scoped advisory lock on the
session. SQLite's single writer used to serialise this for free; under
PostgreSQL's concurrent writers two appends would otherwise read the same
maximum.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from app.db import database as db
from app.models.agent_models import (
    ACTIVE_RUN_STATUSES,
    AgentEvent,
    AgentMemory,
    AgentMessage,
    AgentProject,
    AgentRun,
    AgentSession,
    Availability,
    EventType,
    MemoryKind,
    MessageRole,
    RunStatus,
    SessionArtifact,
    SessionPaper,
)
from app.models.evidence_models import Evidence, Locator, SourceLevel

logger = logging.getLogger("scholar.agent.repo")


class AgentRepositoryError(Exception):
    """Base class for repository-level failures the API maps to HTTP codes."""


class SessionNotFound(AgentRepositoryError):
    """No such session, or it belongs to someone else.

    One exception for both on purpose: telling an unauthorized caller that a
    session exists is itself a leak.
    """


class ActiveRunConflict(AgentRepositoryError):
    """The session already has a pending or running run (HTTP 409)."""


class EvidenceNotFound(AgentRepositoryError):
    """No such evidence, or the caller may not read it."""


class ProjectNotFound(AgentRepositoryError):
    """No such project, or it belongs to someone else (same rule as sessions)."""


def _json_loads(value: Any, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


# ── Sessions ────────────────────────────────────────────────────────────────


def _row_to_session(row: dict[str, Any]) -> AgentSession:
    return AgentSession(
        session_id=row["session_id"],
        owner_id=row["owner_id"],
        thread_id=row["thread_id"],
        title=row["title"],
        language=row["language"],
        llm_model=row["llm_model"],
        config=_json_loads(row["config_json"], {}),
        status=row["status"],
        project_id=row.get("project_id") or "",
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


async def create_session(
    *,
    owner_id: str,
    title: str = "",
    language: str = "zh",
    llm_model: str = "",
    config: dict[str, Any] | None = None,
    project_id: str = "",
) -> AgentSession:
    """Create a session. Its LangGraph thread id is fixed at creation.

    A `project_id` must be one of the owner's projects (`ProjectNotFound`
    otherwise); the check is here so no caller can file a session under
    somebody else's project.
    """
    if project_id:
        await get_project(project_id, owner_id=owner_id)
    session_id = f"as_{uuid.uuid4().hex[:24]}"
    thread_id = f"th_{uuid.uuid4().hex}"
    await db.execute(
        """INSERT INTO agent_sessions
               (session_id, owner_id, thread_id, title, language, llm_model, config_json,
                project_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            session_id,
            owner_id,
            thread_id,
            title,
            language,
            llm_model,
            json.dumps(config or {}, ensure_ascii=False),
            project_id,
        ),
    )
    session = await get_session(session_id, owner_id=owner_id)
    logger.info("Created agent session %s for %s", session_id, owner_id)
    return session


async def get_session(session_id: str, *, owner_id: str) -> AgentSession:
    """Fetch a session, checking ownership. Raises `SessionNotFound` otherwise."""
    row = await db.fetch_one(
        "SELECT * FROM agent_sessions WHERE session_id = ?", (session_id,)
    )
    if row is None or row["owner_id"] != owner_id:
        raise SessionNotFound(session_id)
    return _row_to_session(row)


async def list_sessions(
    owner_id: str, *, limit: int = 50, project_id: str | None = None
) -> list[AgentSession]:
    """Recent sessions; `project_id` narrows to one project (`""` = sessions in none)."""
    sql = "SELECT * FROM agent_sessions WHERE owner_id = ? AND status = 'active'"
    params: list[Any] = [owner_id]
    if project_id is not None:
        sql += " AND project_id = ?"
        params.append(project_id)
    sql += " ORDER BY updated_at DESC LIMIT ?"
    params.append(limit)
    rows = await db.fetch_all(sql, tuple(params))
    return [_row_to_session(r) for r in rows]


async def touch_session(session_id: str) -> None:
    await db.execute(
        "UPDATE agent_sessions SET updated_at = now() WHERE session_id = ?",
        (session_id,),
    )


async def set_session_title(session_id: str, title: str) -> None:
    await db.execute(
        "UPDATE agent_sessions SET title = ?, updated_at = now() "
        "WHERE session_id = ? AND title = ''",
        (title, session_id),
    )


async def rename_session(session_id: str, title: str) -> None:
    """Set the title outright (`set_session_title` only fills an empty one)."""
    await db.execute(
        "UPDATE agent_sessions SET title = ?, updated_at = now() WHERE session_id = ?",
        (title, session_id),
    )


async def set_session_project(session_id: str, *, owner_id: str, project_id: str) -> AgentSession:
    """Move a session into a project, or out of one with `project_id=""`.

    Both ends are checked against `owner_id`: the session (`SessionNotFound`)
    and the target project (`ProjectNotFound`).
    """
    await get_session(session_id, owner_id=owner_id)
    if project_id:
        await get_project(project_id, owner_id=owner_id)
    async with db.transaction() as tx:
        await tx.execute(
            "UPDATE agent_sessions SET project_id = ?, updated_at = now() WHERE session_id = ?",
            (project_id, session_id),
        )
        if project_id:
            await tx.execute(
                "UPDATE agent_projects SET updated_at = now() WHERE project_id = ?",
                (project_id,),
            )
    return await get_session(session_id, owner_id=owner_id)


async def update_session_config(session_id: str, updates: dict[str, Any]) -> None:
    """Merge keys into `config_json`. Empty values are ignored, not written."""
    updates = {k: v for k, v in updates.items() if v not in ("", None)}
    if not updates:
        return
    row = await db.fetch_one(
        "SELECT config_json FROM agent_sessions WHERE session_id = ?", (session_id,)
    )
    if row is None:
        return
    config = _json_loads(row["config_json"], {})
    config.update(updates)
    await db.execute(
        "UPDATE agent_sessions SET config_json = ? WHERE session_id = ?",
        (json.dumps(config, ensure_ascii=False), session_id),
    )


# ── Projects ────────────────────────────────────────────────────────────────


_PROJECT_COLUMNS = """p.project_id, p.owner_id, p.title, p.description, p.status,
       p.created_at, p.updated_at,
       (SELECT COUNT(*) FROM agent_sessions s
         WHERE s.project_id = p.project_id AND s.status = 'active') AS session_count"""


def _row_to_project(row: dict[str, Any]) -> AgentProject:
    return AgentProject(
        project_id=row["project_id"],
        owner_id=row["owner_id"],
        title=row["title"],
        description=row["description"],
        status=row["status"],
        session_count=int(row.get("session_count") or 0),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


async def create_project(*, owner_id: str, title: str = "", description: str = "") -> AgentProject:
    project_id = f"pj_{uuid.uuid4().hex[:20]}"
    await db.execute(
        "INSERT INTO agent_projects (project_id, owner_id, title, description) VALUES (?, ?, ?, ?)",
        (project_id, owner_id, title.strip(), description.strip()),
    )
    logger.info("Created project %s for %s", project_id, owner_id)
    return await get_project(project_id, owner_id=owner_id)


async def get_project(project_id: str, *, owner_id: str) -> AgentProject:
    """Fetch a project, checking ownership. Raises `ProjectNotFound` otherwise."""
    row = await db.fetch_one(
        f"SELECT {_PROJECT_COLUMNS} FROM agent_projects p WHERE p.project_id = ?",
        (project_id,),
    )
    if row is None or row["owner_id"] != owner_id:
        raise ProjectNotFound(project_id)
    return _row_to_project(row)


async def list_projects(
    owner_id: str, *, include_archived: bool = False, limit: int = 100
) -> list[AgentProject]:
    status_clause = "" if include_archived else " AND p.status = 'active'"
    rows = await db.fetch_all(
        f"SELECT {_PROJECT_COLUMNS} FROM agent_projects p "
        f"WHERE p.owner_id = ?{status_clause} ORDER BY p.updated_at DESC LIMIT ?",
        (owner_id, limit),
    )
    return [_row_to_project(r) for r in rows]


async def update_project(
    project_id: str,
    *,
    owner_id: str,
    title: str | None = None,
    description: str | None = None,
    status: str | None = None,
) -> AgentProject:
    """Change the given fields; `None` leaves a field as it is."""
    await get_project(project_id, owner_id=owner_id)
    sets: list[str] = []
    params: list[Any] = []
    for column, value in (("title", title), ("description", description), ("status", status)):
        if value is not None:
            sets.append(f"{column} = ?")
            params.append(value.strip())
    if sets:
        await db.execute(
            f"UPDATE agent_projects SET {', '.join(sets)}, updated_at = now() "
            "WHERE project_id = ? AND owner_id = ?",
            (*params, project_id, owner_id),
        )
    return await get_project(project_id, owner_id=owner_id)


# Best first: when several sessions of a project hold the same work, the
# project shows it as readable if any of them can read it.
_AVAILABILITY_RANK = """CASE sp.availability
    WHEN 'parsed' THEN 0 WHEN 'pdf_ready' THEN 1 WHEN 'downloading' THEN 2
    WHEN 'candidate' THEN 3 ELSE 4 END"""


async def list_project_papers(
    project_id: str, *, exclude_session_id: str = "", limit: int = 200
) -> list[SessionPaper]:
    """The project's papers: its sessions' papers, one row per work, newest first.

    Derived rather than stored, so a session moving in or out of the project
    takes its papers with it. `exclude_session_id` drops the works that
    session already has, which is what "the project's *other* papers" means
    to a turn running in it. `session_id` on each row is the session the best
    copy was found in.
    """
    if not project_id:
        return []
    rows = await db.fetch_all(
        f"""SELECT DISTINCT ON (sp.literature_id)
                   sp.session_id, sp.literature_id, sp.paper_id, sp.availability,
                   sp.added_by, sp.note, sp.updated_at,
                   li.title, li.year, li.year_known, li.venue, li.doi, li.arxiv_id
              FROM session_papers sp
              JOIN agent_sessions s ON s.session_id = sp.session_id
              JOIN literature_items li ON li.literature_id = sp.literature_id
             WHERE s.project_id = ? AND s.status = 'active'
               AND sp.literature_id NOT IN (
                   SELECT literature_id FROM session_papers WHERE session_id = ?)
             ORDER BY sp.literature_id, {_AVAILABILITY_RANK}, (sp.paper_id = ''),
                      sp.updated_at DESC""",
        (project_id, exclude_session_id),
    )
    papers = [_row_to_session_paper(r) for r in rows]
    papers.sort(key=lambda p: p.updated_at, reverse=True)
    return papers[:limit]


async def find_project_paper(project_id: str, handle: str) -> SessionPaper | None:
    """A project paper by `paper_id` or `literature_id`; `None` if the project has none such."""
    handle = (handle or "").strip()
    if not project_id or not handle:
        return None
    for paper in await list_project_papers(project_id, limit=10_000):
        if handle in (paper.paper_id, paper.literature_id):
            return paper
    return None


# ── Runs ────────────────────────────────────────────────────────────────────


def _row_to_run(row: dict[str, Any]) -> AgentRun:
    return AgentRun(
        run_id=row["run_id"],
        session_id=row["session_id"],
        owner_id=row["owner_id"],
        client_request_id=row["client_request_id"],
        status=RunStatus(row["status"]),
        cancel_requested=bool(row["cancel_requested"]),
        error_code=row["error_code"],
        error_msg=row["error_msg"],
        budget=_json_loads(row["budget_json"], {}),
        usage=_json_loads(row["usage_json"], {}),
        llm_model=row["llm_model"],
        prompt_version=row["prompt_version"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        worker_id=row["worker_id"],
        heartbeat_at=row["heartbeat_at"],
    )


async def create_run(
    *,
    session_id: str,
    owner_id: str,
    client_request_id: str = "",
    budget: dict[str, Any] | None = None,
    llm_model: str = "",
    prompt_version: str = "",
) -> tuple[AgentRun, bool]:
    """Create a run for a session. Returns `(run, deduplicated)`.

    A repeat of a `client_request_id` returns the original run rather than
    starting another (acceptance case A12). A different request while one is
    still in flight raises `ActiveRunConflict`, which the API turns into a 409.
    """
    if client_request_id:
        existing = await db.fetch_one(
            "SELECT * FROM agent_runs WHERE session_id = ? AND client_request_id = ?",
            (session_id, client_request_id),
        )
        if existing is not None:
            return _row_to_run(existing), True

    run_id = f"ar_{uuid.uuid4().hex[:24]}"
    try:
        await db.execute(
            """INSERT INTO agent_runs
                   (run_id, session_id, owner_id, client_request_id, status,
                    budget_json, llm_model, prompt_version)
               VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)""",
            (
                run_id,
                session_id,
                owner_id,
                client_request_id,
                json.dumps(budget or {}, ensure_ascii=False),
                llm_model,
                prompt_version,
            ),
        )
    except db.IntegrityError as exc:
        # Either the active-run index or the idempotency index fired. Re-read to
        # tell which: a concurrent identical submit should still be a dedupe.
        if client_request_id:
            existing = await db.fetch_one(
                "SELECT * FROM agent_runs WHERE session_id = ? AND client_request_id = ?",
                (session_id, client_request_id),
            )
            if existing is not None:
                return _row_to_run(existing), True
        raise ActiveRunConflict(session_id) from exc

    row = await db.fetch_one("SELECT * FROM agent_runs WHERE run_id = ?", (run_id,))
    return _row_to_run(row), False


async def get_run(run_id: str, *, owner_id: str) -> AgentRun:
    row = await db.fetch_one("SELECT * FROM agent_runs WHERE run_id = ?", (run_id,))
    if row is None or row["owner_id"] != owner_id:
        raise SessionNotFound(run_id)
    return _row_to_run(row)


async def list_runs(session_id: str, *, limit: int = 50) -> list[AgentRun]:
    rows = await db.fetch_all(
        "SELECT * FROM agent_runs WHERE session_id = ? ORDER BY started_at DESC LIMIT ?",
        (session_id, limit),
    )
    return [_row_to_run(r) for r in rows]


async def claim_run(run_id: str, *, worker_id: str) -> bool:
    """Take a pending run as `worker_id`, starting its heartbeat.

    The status guard is the lock: two workers cannot both claim a pending run,
    and a run already `running` is not re-claimable here — recovering one that
    was abandoned goes through :func:`list_stale_runs` instead, which requires
    evidence that its worker stopped reporting.
    """
    claimed = await db.execute_returning(
        """UPDATE agent_runs
              SET status = 'running', worker_id = ?, heartbeat_at = now()
            WHERE run_id = ? AND status = 'pending'
           RETURNING run_id""",
        (worker_id, run_id),
    )
    return claimed is not None


async def heartbeat_run(run_id: str, *, worker_id: str) -> None:
    """Say this worker is still executing the run.

    Scoped to the owning worker so a process that lost its lease cannot keep a
    run looking alive after someone else has recovered it.
    """
    await db.execute(
        """UPDATE agent_runs SET heartbeat_at = now()
            WHERE run_id = ? AND worker_id = ? AND status = 'running'""",
        (run_id, worker_id),
    )


async def list_stale_runs(*, stale_after_seconds: int) -> list[AgentRun]:
    """Active runs whose worker has not checked in recently.

    A `pending` run counts once it has waited that long without being claimed:
    a crash between "persist the run" and "schedule it" leaves exactly that, and
    nothing else would ever pick it up.

    Times are compared on the database clock, which is what wrote them; mixing
    it with this process's clock would make staleness depend on clock skew
    between machines.
    """
    rows = await db.fetch_all(
        """SELECT * FROM agent_runs
            WHERE status IN ('pending', 'running')
              AND COALESCE(heartbeat_at, started_at) < now() - make_interval(secs => ?)
            ORDER BY started_at""",
        (int(stale_after_seconds),),
    )
    return [_row_to_run(r) for r in rows]


async def finish_run(
    run_id: str,
    *,
    status: RunStatus,
    error_code: str = "",
    error_msg: str = "",
    usage: dict[str, Any] | None = None,
) -> None:
    """Close out a run. Terminal statuses only."""
    if status in (RunStatus.PENDING, RunStatus.RUNNING):
        raise ValueError(f"finish_run needs a terminal status, got {status}")
    await db.execute(
        """UPDATE agent_runs
              SET status = ?, error_code = ?, error_msg = ?, usage_json = ?,
                  finished_at = now()
            WHERE run_id = ?""",
        (
            status.value,
            error_code,
            error_msg,
            json.dumps(usage or {}, ensure_ascii=False),
            run_id,
        ),
    )


async def request_cancel(run_id: str, *, owner_id: str) -> AgentRun:
    """Flag a run for cancellation. Idempotent — repeat calls are a no-op.

    Only the flag is set here. Stopping the work is the executor's job, which
    checks the flag before each scheduling step (development plan §7.2).
    """
    run = await get_run(run_id, owner_id=owner_id)
    if run.status in (RunStatus.PENDING, RunStatus.RUNNING):
        await db.execute(
            "UPDATE agent_runs SET cancel_requested = 1 WHERE run_id = ?", (run_id,)
        )
        run = await get_run(run_id, owner_id=owner_id)
    return run


async def is_cancel_requested(run_id: str) -> bool:
    row = await db.fetch_one(
        "SELECT cancel_requested FROM agent_runs WHERE run_id = ?", (run_id,)
    )
    return bool(row and row["cancel_requested"])


# ── Messages ────────────────────────────────────────────────────────────────


async def _lock_session_seq(tx: db.Transaction, session_id: str) -> None:
    """Serialise `seq` allocation for one session until the transaction ends.

    An advisory lock rather than `SELECT … FOR UPDATE` on the session row, so
    it holds even for an event whose session row is not visible to this
    transaction, and so it never blocks ordinary updates to the session.
    """
    await tx.execute("SELECT pg_advisory_xact_lock(hashtextextended(?, 0))", (session_id,))


async def append_message(
    *,
    session_id: str,
    role: MessageRole,
    content: str,
    run_id: str = "",
    citations: list[str] | None = None,
) -> AgentMessage:
    message_id = f"am_{uuid.uuid4().hex[:24]}"
    async with db.transaction() as tx:
        await _lock_session_seq(tx, session_id)
        row = await tx.fetch_one(
            """INSERT INTO agent_messages
                   (message_id, session_id, run_id, role, content, citations_json, seq)
               VALUES (?, ?, ?, ?, ?, ?,
                       (SELECT COALESCE(MAX(seq), 0) + 1 FROM agent_messages WHERE session_id = ?))
               RETURNING seq, created_at""",
            (
                message_id,
                session_id,
                run_id,
                role.value,
                content,
                json.dumps(citations or [], ensure_ascii=False),
                session_id,
            ),
        )
    assert row is not None
    return AgentMessage(
        message_id=message_id,
        session_id=session_id,
        run_id=run_id,
        role=role,
        content=content,
        citations=citations or [],
        seq=row["seq"],
        created_at=row["created_at"],
    )


# Named rather than `*`, so the search vector (P8) is not shipped with every read.
_MESSAGE_COLUMNS = ("message_id", "session_id", "run_id", "role", "content",
                    "citations_json", "seq", "created_at")


def _row_to_message(r: dict[str, Any]) -> AgentMessage:
    return AgentMessage(
        message_id=r["message_id"],
        session_id=r["session_id"],
        run_id=r["run_id"],
        role=MessageRole(r["role"]),
        content=r["content"],
        citations=_json_loads(r["citations_json"], []),
        seq=r["seq"],
        created_at=r["created_at"],
    )


async def list_messages(session_id: str, *, limit: int = 500) -> list[AgentMessage]:
    rows = await db.fetch_all(
        f"SELECT {', '.join(_MESSAGE_COLUMNS)} FROM agent_messages "
        "WHERE session_id = ? ORDER BY seq LIMIT ?",
        (session_id, limit),
    )
    return [_row_to_message(r) for r in rows]


async def search_terms(text: str) -> list[str]:
    """The terms the message index holds for `text` (`scholar_search_terms`)."""
    row = await db.fetch_one("SELECT scholar_search_terms(?) AS terms", (text or "",))
    return list((row or {}).get("terms") or [])


async def search_messages(
    owner_id: str,
    query: str,
    *,
    project_id: str | None = None,
    exclude_session_id: str = "",
    limit: int = 40,
) -> list[tuple[AgentMessage, dict[str, Any]]]:
    """`owner_id`'s messages matching any term of `query`, best first.

    Index and query are segmented by the same SQL function, so a Chinese query
    meets Chinese messages as bigrams on both sides. `project_id` narrows to
    one project (`""` = sessions in none); `exclude_session_id` leaves out the
    session that is asking. Each message comes with its session's `title`,
    `project_id` and the match `rank`.
    """
    columns = ", ".join(f"m.{c}" for c in _MESSAGE_COLUMNS)
    sql = f"""
        WITH q AS (
            SELECT string_agg(quote_literal(t), ' | ')::tsquery AS query
              FROM unnest(scholar_search_terms(?)) AS t
        )
        SELECT {columns}, s.title AS session_title, s.project_id,
               ts_rank(m.search_tsv, q.query) AS rank
          FROM agent_messages m
          JOIN agent_sessions s ON s.session_id = m.session_id
          CROSS JOIN q
         WHERE s.owner_id = ? AND s.status = 'active'
           AND m.role IN ('user', 'assistant')
           AND m.search_tsv @@ q.query"""
    params: list[Any] = [query or "", owner_id]
    if project_id is not None:
        sql += " AND s.project_id = ?"
        params.append(project_id)
    if exclude_session_id:
        sql += " AND m.session_id <> ?"
        params.append(exclude_session_id)
    sql += " ORDER BY rank DESC, m.created_at DESC LIMIT ?"
    params.append(limit)
    rows = await db.fetch_all(sql, tuple(params))
    return [
        (
            _row_to_message(r),
            {
                "session_title": r["session_title"],
                "project_id": r["project_id"],
                "rank": float(r["rank"] or 0.0),
            },
        )
        for r in rows
    ]


async def list_turn_messages(turns: list[tuple[str, str]]) -> list[AgentMessage]:
    """Every message of the given `(session_id, run_id)` turns, in order."""
    pairs = [(s, r) for s, r in dict.fromkeys(turns) if s and r]
    if not pairs:
        return []
    clause = " OR ".join("(session_id = ? AND run_id = ?)" for _ in pairs)
    rows = await db.fetch_all(
        f"SELECT {', '.join(_MESSAGE_COLUMNS)} FROM agent_messages "
        f"WHERE {clause} ORDER BY session_id, seq",
        tuple(v for pair in pairs for v in pair),
    )
    return [_row_to_message(r) for r in rows]


# ── Events ──────────────────────────────────────────────────────────────────


async def append_event(
    *,
    session_id: str,
    type: EventType,
    payload: dict[str, Any] | None = None,
    run_id: str = "",
) -> AgentEvent:
    """Persist an event and return it with its assigned `seq`.

    The sequence number is allocated inside the insert so two concurrent
    appends cannot collide, and the event is durable *before* anyone publishes
    it — a client that reconnects must be able to replay what it missed.
    """
    async with db.transaction() as tx:
        await _lock_session_seq(tx, session_id)
        row = await tx.fetch_one(
            """INSERT INTO agent_events (session_id, seq, run_id, type, payload_json)
               VALUES (?,
                       (SELECT COALESCE(MAX(seq), 0) + 1 FROM agent_events WHERE session_id = ?),
                       ?, ?, ?)
               RETURNING seq, schema_version, created_at""",
            (
                session_id,
                session_id,
                run_id,
                type.value,
                json.dumps(payload or {}, ensure_ascii=False),
            ),
        )
    assert row is not None
    return AgentEvent(
        schema_version=row["schema_version"],
        session_id=session_id,
        run_id=run_id,
        seq=row["seq"],
        type=type,
        timestamp=row["created_at"],
        payload=payload or {},
    )


async def list_events(
    session_id: str, *, after_seq: int = 0, limit: int = 500
) -> list[AgentEvent]:
    """Events after `after_seq`, oldest first — the SSE resume query."""
    rows = await db.fetch_all(
        "SELECT * FROM agent_events WHERE session_id = ? AND seq > ? ORDER BY seq LIMIT ?",
        (session_id, after_seq, limit),
    )
    return [
        AgentEvent(
            schema_version=r["schema_version"],
            session_id=r["session_id"],
            run_id=r["run_id"],
            seq=r["seq"],
            type=EventType(r["type"]),
            timestamp=r["created_at"],
            payload=_json_loads(r["payload_json"], {}),
        )
        for r in rows
    ]


async def list_run_events(
    run_id: str, *, limit: int = 500, exclude_types: tuple[str, ...] = ()
) -> list[AgentEvent]:
    """One run's events, oldest first — what a finished turn actually did.

    `exclude_types` exists for `message.delta`: a turn emits one per streamed
    fragment, and the answer they compose is already stored as a message. A
    client restoring history wants the tool activity, not a second copy of the
    text it is already rendering.
    """
    sql = "SELECT * FROM agent_events WHERE run_id = ?"
    params: list[Any] = [run_id]
    if exclude_types:
        sql += f" AND type NOT IN ({','.join('?' * len(exclude_types))})"
        params.extend(exclude_types)
    sql += " ORDER BY seq LIMIT ?"
    params.append(limit)
    rows = await db.fetch_all(sql, tuple(params))
    return [
        AgentEvent(
            schema_version=r["schema_version"],
            session_id=r["session_id"],
            run_id=r["run_id"],
            seq=r["seq"],
            type=EventType(r["type"]),
            timestamp=r["created_at"],
            payload=_json_loads(r["payload_json"], {}),
        )
        for r in rows
    ]


async def last_event_seq(session_id: str) -> int:
    row = await db.fetch_one(
        "SELECT COALESCE(MAX(seq), 0) AS seq FROM agent_events WHERE session_id = ?",
        (session_id,),
    )
    return int(row["seq"]) if row else 0


async def count_events(session_id: str, type: EventType) -> int:
    row = await db.fetch_one(
        "SELECT COUNT(*) AS n FROM agent_events WHERE session_id = ? AND type = ?",
        (session_id, type.value),
    )
    return int(row["n"]) if row else 0


# ── Artifacts ───────────────────────────────────────────────────────────────


async def list_session_artifacts(session_id: str, *, limit: int = 100) -> list[SessionArtifact]:
    """Mode reports announced in this conversation, oldest first.

    Sourced from `artifact.created` events rather than `runs.agent_session_id`
    alone. A reused report still belongs to the session that first produced it,
    but this conversation also showed it and must keep the card after a reload.
    `agent_run_id` is the turn that announced the card here, so it sits under
    the answer that produced (or reused) it.
    """
    rows = await db.fetch_all(
        """SELECT r.run_id, e.run_id AS agent_run_id, r.paper_id,
                  COALESCE(p.title, '') AS paper_title,
                  r.mode, r.language, r.status, e.created_at
             FROM agent_events e
             JOIN runs r ON r.run_id = (e.payload_json::jsonb ->> 'run_id')
             LEFT JOIN papers p ON p.paper_id = r.paper_id
            WHERE e.session_id = ? AND e.type = ?
            ORDER BY e.seq
            LIMIT ?""",
        (session_id, EventType.ARTIFACT_CREATED.value, limit),
    )
    return [
        SessionArtifact(
            run_id=r["run_id"],
            agent_run_id=r["agent_run_id"] or "",
            paper_id=r["paper_id"],
            paper_title=r["paper_title"] or "",
            mode=r["mode"],
            language=r["language"] or "en",
            status=r["status"],
            created_at=r["created_at"] or "",
        )
        for r in rows
    ]


# ── Long-term memory ────────────────────────────────────────────────────────


def _row_to_memory(row: dict[str, Any]) -> AgentMemory:
    try:
        kind = MemoryKind(row["kind"])
    except ValueError:
        kind = MemoryKind.FACT
    return AgentMemory(
        memory_id=row["memory_id"],
        owner_id=row["owner_id"],
        kind=kind,
        content=row["content"],
        source_session_id=row["source_session_id"],
        source_run_id=row["source_run_id"],
        project_id=row.get("project_id") or "",
        active=bool(row["active"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


async def add_memory(
    *,
    owner_id: str,
    content: str,
    kind: MemoryKind = MemoryKind.PREFERENCE,
    source_session_id: str = "",
    source_run_id: str = "",
    project_id: str = "",
) -> AgentMemory:
    """Store a memory for a principal, globally or for one of their projects.

    Saving the same text twice in the same scope updates the existing row
    instead of adding a duplicate — the model tends to re-state a preference
    it has already kept. The caller checks that `project_id` is the owner's.
    """
    content = content.strip()
    existing = await db.fetch_one(
        "SELECT * FROM agent_memories "
        "WHERE owner_id = ? AND project_id = ? AND content = ? AND active = 1",
        (owner_id, project_id, content),
    )
    if existing is not None:
        await db.execute(
            "UPDATE agent_memories SET kind = ?, updated_at = now() WHERE memory_id = ?",
            (kind.value, existing["memory_id"]),
        )
        row = await db.fetch_one(
            "SELECT * FROM agent_memories WHERE memory_id = ?", (existing["memory_id"],)
        )
        return _row_to_memory(row)

    memory_id = f"mem_{uuid.uuid4().hex[:20]}"
    await db.execute(
        """INSERT INTO agent_memories
               (memory_id, owner_id, kind, content, source_session_id, source_run_id,
                project_id)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (memory_id, owner_id, kind.value, content, source_session_id, source_run_id, project_id),
    )
    row = await db.fetch_one("SELECT * FROM agent_memories WHERE memory_id = ?", (memory_id,))
    return _row_to_memory(row)


async def list_memories(owner_id: str, *, limit: int = 100) -> list[AgentMemory]:
    rows = await db.fetch_all(
        """SELECT * FROM agent_memories
            WHERE owner_id = ? AND active = 1
            ORDER BY updated_at DESC, created_at DESC, memory_id DESC LIMIT ?""",
        (owner_id, limit),
    )
    return [_row_to_memory(r) for r in rows]


async def list_memories_for_prompt(
    owner_id: str, *, project_id: str = "", limit: int = 30
) -> list[AgentMemory]:
    """The memories a turn sees: its project's first, then global ones, within `limit`.

    Another project's memories never appear — "I am working on X" saved in one
    project would be wrong context in the next.
    """
    rows = await db.fetch_all(
        """SELECT * FROM agent_memories
            WHERE owner_id = ? AND active = 1 AND (project_id = '' OR project_id = ?)
            ORDER BY (project_id = ''), updated_at DESC, created_at DESC, memory_id DESC
            LIMIT ?""",
        (owner_id, project_id, limit),
    )
    return [_row_to_memory(r) for r in rows]


async def get_memory(memory_id: str, *, owner_id: str) -> AgentMemory | None:
    row = await db.fetch_one("SELECT * FROM agent_memories WHERE memory_id = ?", (memory_id,))
    if row is None or row["owner_id"] != owner_id:
        return None
    return _row_to_memory(row)


async def deactivate_memory(memory_id: str, *, owner_id: str) -> bool:
    """Forget a memory. Soft-deleted so an audit of a past answer still resolves it."""
    updated = await db.execute_returning(
        """UPDATE agent_memories SET active = 0, updated_at = now()
            WHERE memory_id = ? AND owner_id = ? AND active = 1
           RETURNING memory_id""",
        (memory_id, owner_id),
    )
    return updated is not None


# ── Session papers ──────────────────────────────────────────────────────────


async def attach_session_paper(
    *,
    session_id: str,
    literature_id: str,
    paper_id: str = "",
    availability: Availability = Availability.CANDIDATE,
    added_by: str = "agent",
    note: str = "",
) -> None:
    """Attach a paper to a session, or update what is known about it."""
    await db.execute(
        """INSERT INTO session_papers
               (session_id, literature_id, paper_id, availability, added_by, note)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(session_id, literature_id) DO UPDATE SET
               paper_id = CASE WHEN excluded.paper_id <> '' THEN excluded.paper_id
                               ELSE session_papers.paper_id END,
               availability = excluded.availability,
               note = excluded.note,
               updated_at = now()""",
        (session_id, literature_id, paper_id, availability.value, added_by, note),
    )


async def set_session_paper_availability(
    *, session_id: str, literature_id: str, availability: Availability, note: str = ""
) -> None:
    await db.execute(
        """UPDATE session_papers
              SET availability = ?, note = ?, updated_at = now()
            WHERE session_id = ? AND literature_id = ?""",
        (availability.value, note, session_id, literature_id),
    )


async def mark_paper_parsed(paper_id: str) -> None:
    """Record a finished parse on every session holding this file.

    Being parsed is a fact about the file, not about the session that happened
    to ask: a session left at `pdf_ready` would send its next turn to
    `ensure_paper_parsed` just to be told there is nothing to do.
    """
    await db.execute(
        """UPDATE session_papers
              SET availability = ?, note = '', updated_at = now()
            WHERE paper_id = ? AND availability = ?""",
        (Availability.PARSED.value, paper_id, Availability.PDF_READY.value),
    )


async def list_session_papers(session_id: str) -> list[SessionPaper]:
    """Session papers joined with their bibliographic record."""
    rows = await db.fetch_all(
        """SELECT sp.*, li.title, li.year, li.year_known, li.venue, li.doi, li.arxiv_id
             FROM session_papers sp
             JOIN literature_items li ON li.literature_id = sp.literature_id
            WHERE sp.session_id = ?
            ORDER BY sp.created_at""",
        (session_id,),
    )
    return [_row_to_session_paper(r) for r in rows]


def _row_to_session_paper(r: dict[str, Any]) -> SessionPaper:
    return SessionPaper(
        session_id=r["session_id"],
        literature_id=r["literature_id"],
        paper_id=r["paper_id"],
        availability=Availability(r["availability"]),
        added_by=r["added_by"],
        note=r["note"],
        title=r["title"],
        year=r["year"],
        year_known=bool(r["year_known"]),
        venue=r["venue"],
        doi=r["doi"],
        arxiv_id=r["arxiv_id"],
        updated_at=r["updated_at"],
    )


async def session_owns_paper(session_id: str, paper_id: str) -> bool:
    """Whether a local PDF is attached to this session — the evidence ACL."""
    row = await db.fetch_one(
        "SELECT 1 AS hit FROM session_papers WHERE session_id = ? AND paper_id = ? LIMIT 1",
        (session_id, paper_id),
    )
    return row is not None


# ── Evidence ────────────────────────────────────────────────────────────────


def _row_to_evidence(row: dict[str, Any]) -> Evidence:
    return Evidence(
        evidence_id=row["evidence_id"],
        owner_id=row["owner_id"],
        session_id=row["session_id"],
        literature_id=row["literature_id"],
        paper_id=row["paper_id"],
        parse_version=row["parse_version"],
        source_level=SourceLevel(row["source_level"]),
        locator=Locator(**_json_loads(row["locator_json"], {})),
        quote=row["quote"],
        content_hash=row["content_hash"],
        source_url=row["source_url"],
        provider=row["provider"],
        retrieved_at=row["retrieved_at"],
    )


async def insert_evidence(evidence: Evidence) -> Evidence:
    """Store evidence. Idempotent: re-reading a passage reuses the existing row.

    `DO NOTHING` rather than an upsert, because evidence is immutable — if the
    id is already present, the stored row is by definition the same content,
    and overwriting it would defeat the point.
    """
    await db.execute(
        """INSERT INTO evidence
               (evidence_id, owner_id, session_id, literature_id, paper_id, parse_version,
                source_level, locator_json, quote, content_hash, source_url, provider)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT (evidence_id) DO NOTHING""",
        (
            evidence.evidence_id,
            evidence.owner_id,
            evidence.session_id,
            evidence.literature_id,
            evidence.paper_id,
            evidence.parse_version,
            evidence.source_level.value,
            json.dumps(evidence.locator.model_dump(), ensure_ascii=False),
            evidence.quote,
            evidence.content_hash,
            evidence.source_url,
            evidence.provider,
        ),
    )
    row = await db.fetch_one(
        "SELECT * FROM evidence WHERE evidence_id = ?", (evidence.evidence_id,)
    )
    return _row_to_evidence(row)


async def get_evidence(evidence_id: str, *, owner_id: str = "") -> Evidence:
    """Fetch evidence, checking ownership when an owner is supplied."""
    row = await db.fetch_one("SELECT * FROM evidence WHERE evidence_id = ?", (evidence_id,))
    if row is None:
        raise EvidenceNotFound(evidence_id)
    if owner_id and row["owner_id"] and row["owner_id"] != owner_id:
        raise EvidenceNotFound(evidence_id)
    return _row_to_evidence(row)


async def get_evidence_many(evidence_ids: list[str], *, owner_id: str = "") -> list[Evidence]:
    if not evidence_ids:
        return []
    placeholders = ",".join("?" for _ in evidence_ids)
    rows = await db.fetch_all(
        f"SELECT * FROM evidence WHERE evidence_id IN ({placeholders})",
        tuple(evidence_ids),
    )
    out = []
    for r in rows:
        if owner_id and r["owner_id"] and r["owner_id"] != owner_id:
            continue
        out.append(_row_to_evidence(r))
    return out


async def literature_titles(literature_ids: list[str]) -> dict[str, str]:
    """`literature_id -> title` for the given works; unknown ids are absent."""
    ids = [i for i in dict.fromkeys(literature_ids) if i]
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = await db.fetch_all(
        f"SELECT literature_id, title FROM literature_items WHERE literature_id IN ({placeholders})",
        tuple(ids),
    )
    return {r["literature_id"]: r["title"] for r in rows}


async def list_session_evidence(session_id: str, *, limit: int = 200) -> list[Evidence]:
    rows = await db.fetch_all(
        "SELECT * FROM evidence WHERE session_id = ? ORDER BY retrieved_at DESC LIMIT ?",
        (session_id, limit),
    )
    return [_row_to_evidence(r) for r in rows]
