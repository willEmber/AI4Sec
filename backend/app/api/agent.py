"""Agent session API.

Ownership is checked on every route through the repository, which raises
`SessionNotFound` for both "no such session" and "not yours" — distinguishing
them would confirm a stranger's session exists.

Submitting a message persists the message and the run *before* returning, then
schedules the work: a client that never opens the event stream still has a run
it can poll, and a repeat submission is deduplicated by the database rather
than by whether the first one happened to be in flight.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from app.api.deps import (
    principal_or_new,
    quota_exceeded,
    require_caller,
    require_principal,
    resolve_caller,
)
from app.config import get_settings
from app.db import agent_repository as repo
from app.db import database as db
from app.models.agent_models import (
    AttachPapersRequest,
    Availability,
    CreateMemoryRequest,
    CreateProjectRequest,
    CreateSessionRequest,
    CreateSessionResponse,
    EventType,
    MessageRole,
    PostMessageRequest,
    PostMessageResponse,
    SessionContextStats,
    SessionDetailResponse,
    UpdateMemoryRequest,
    UpdateProjectRequest,
    UpdateSessionRequest,
)
from app.rate_limit import limiter
from app.services import (
    accounts,
    agent_runner,
    conversation_recall,
    data_lifecycle,
    evidence_service,
    identity,
    paper_catalog,
    parse_service,
)
from app.services.accounts import Caller

logger = logging.getLogger("scholar.api.agent")

router = APIRouter(prefix="/agent", tags=["agent"])

# How long an idle stream waits before sending a keepalive. Proxies routinely
# drop a silent connection at 60s, and a turn can think for longer than that.
_SSE_KEEPALIVE_SECONDS = 15.0
# How often a stream reads the event table for a turn this process is not the
# one executing.
_SSE_POLL_SECONDS = 1.0
# A stream ticket only has to outlive the gap between asking for it and
# opening the stream; reconnects ask again.
_STREAM_TICKET_SECONDS = 300


async def _attach_papers(session_id: str, paper_ids: list[str]) -> list[str]:
    """Attach local PDFs to a session, returning the ids that were accepted."""
    attached: list[str] = []
    for paper_id in dict.fromkeys(pid.strip() for pid in paper_ids if pid.strip()):
        row = await db.fetch_one(
            "SELECT paper_id FROM papers WHERE paper_id = ?", (paper_id,)
        )
        if row is None:
            logger.info("Ignoring unknown paper %s on session %s", paper_id, session_id)
            continue
        literature_id = await paper_catalog.ensure_literature_for_local_paper(paper_id)
        availability = await _availability_of(paper_id)
        await repo.attach_session_paper(
            session_id=session_id,
            literature_id=literature_id,
            paper_id=paper_id,
            availability=availability,
            added_by="user",
        )
        attached.append(paper_id)
        if availability is Availability.PDF_READY:
            # The reader is about to ask about this paper; parsing while they
            # type keeps those minutes out of their first turn.
            try:
                await parse_service.start_background_parse(paper_id, session_id=session_id)
            except Exception:
                logger.exception("Could not start parsing %s ahead of the first turn", paper_id)
    return attached


async def _availability_of(paper_id: str) -> Availability:
    """Whether a paper can actually be read yet.

    Checks `paper_nodes` rather than `paper_versions`: papers parsed before
    versioning existed have nodes but no version row, and they are perfectly
    readable.
    """
    row = await db.fetch_one(
        "SELECT 1 AS hit FROM paper_nodes WHERE paper_id = ? LIMIT 1", (paper_id,)
    )
    return Availability.PARSED if row else Availability.PDF_READY


@router.post("/sessions", response_model=CreateSessionResponse)
@limiter.limit("30/minute")
async def create_session(
    request: Request,
    body: CreateSessionRequest,
    principal_id: str = Depends(principal_or_new),
) -> CreateSessionResponse:
    """Create a session, optionally with papers already attached.

    The agent route that may mint a principal: a first-time visitor has no
    credential yet, and `principal_or_new` sets the new one as the auth cookie
    (and, for older clients, the `X-Agent-Token` response header).
    """
    try:
        session = await repo.create_session(
            owner_id=principal_id,
            title=body.title,
            language=body.language,
            llm_model=_allowed_model(body.llm_model),
            config={"owner_token": body.owner_token[:100]} if body.owner_token else None,
            project_id=body.project_id.strip(),
        )
    except repo.ProjectNotFound:
        raise HTTPException(status_code=404, detail="No such project.") from None
    if body.paper_ids:
        await _attach_papers(session.session_id, body.paper_ids)
    return CreateSessionResponse(
        session_id=session.session_id,
        thread_id=session.thread_id,
        created_at=session.created_at,
    )


async def _check_quota(
    caller: Caller, session_id: str, client_request_id: str
) -> accounts.QuotaUsage:
    """429 once today's turns or tokens are used up; otherwise today's usage.

    A resubmission of a turn that already exists is let through: it starts
    nothing, and refusing it would make a network retry look like a new turn.
    """
    usage = await accounts.daily_usage(caller)
    if not usage.exceeded:
        return usage
    if client_request_id and await db.fetch_one(
        "SELECT 1 AS hit FROM agent_runs WHERE session_id = ? AND client_request_id = ?",
        (session_id, client_request_id),
    ):
        return usage
    raise quota_exceeded(caller, usage)


async def _active_run_limit(caller: Caller) -> int:
    """Turns this caller may have in flight at once; 0 for no limit.

    The single local user and the admin are the operator, who is not being
    protected from themselves.
    """
    settings = get_settings()
    if settings.auth_mode == "single_user" or await accounts.is_admin(caller):
        return 0
    return max(0, settings.agent_max_active_runs_per_principal)


def _allowed_model(requested: str) -> str:
    """Only a model the operator listed may be chosen; anything else is the default.

    Same rule as the classic run endpoint: a caller must not be able to point a
    conversation at an arbitrary — possibly far more expensive — model name.
    The list is narrower than the report modes': a model is offered for
    conversations only once it has been verified against the agent's tools.
    """
    from app.services.llm_gateway import get_registry

    name = (requested or "").strip()
    allowed = get_registry().agent_models()
    if name and allowed and name not in allowed:
        logger.warning("Rejected unknown llm_model=%r for agent session; using default", name)
        return ""
    return name


@router.post("/sessions/{session_id}/papers")
@limiter.limit("30/minute")
async def attach_papers(
    request: Request,
    session_id: str,
    body: AttachPapersRequest,
    principal_id: str = Depends(require_principal),
) -> dict[str, Any]:
    """Attach uploaded PDFs to a session without sending a message.

    This is the upload path inside the conversation: the file goes through the
    ordinary `/papers/upload`, then is attached here so the sidebar and the
    next turn's prompt know about it.
    """
    try:
        await repo.get_session(session_id, owner_id=principal_id)
    except repo.SessionNotFound:
        raise HTTPException(status_code=404, detail="No such session.") from None
    attached = await _attach_papers(session_id, body.paper_ids)
    await repo.touch_session(session_id)
    return {
        "session_id": session_id,
        "attached": attached,
        "papers": [p.model_dump(mode="json") for p in await repo.list_session_papers(session_id)],
    }


@router.get("/sessions")
async def list_sessions(
    request: Request,
    project_id: str | None = Query(default=None, max_length=64),
    principal_id: str = Depends(require_principal),
) -> dict[str, Any]:
    """Recent sessions for the calling principal; `project_id` narrows to one project
    (an empty value lists the sessions in no project)."""
    sessions = await repo.list_sessions(principal_id, project_id=project_id)
    return {"sessions": [s.model_dump() for s in sessions]}


@router.patch("/sessions/{session_id}")
@limiter.limit("60/minute")
async def update_session(
    request: Request,
    session_id: str,
    body: UpdateSessionRequest,
    principal_id: str = Depends(require_principal),
) -> dict[str, Any]:
    """Rename a session, or move it into a project (`project_id=""` takes it out)."""
    try:
        session = await repo.get_session(session_id, owner_id=principal_id)
        if body.project_id is not None and body.project_id.strip() != session.project_id:
            session = await repo.set_session_project(
                session_id, owner_id=principal_id, project_id=body.project_id.strip()
            )
        if body.title is not None and body.title.strip():
            await repo.rename_session(session_id, body.title.strip())
            session = await repo.get_session(session_id, owner_id=principal_id)
    except repo.SessionNotFound:
        raise HTTPException(status_code=404, detail="No such session.") from None
    except repo.ProjectNotFound:
        raise HTTPException(status_code=404, detail="No such project.") from None
    return {"session": session.model_dump()}


@router.delete("/sessions/{session_id}")
@limiter.limit("30/minute")
async def delete_session(
    request: Request,
    session_id: str,
    principal_id: str = Depends(require_principal),
) -> dict[str, Any]:
    """Delete a conversation with its messages, evidence and the reports made in it."""
    try:
        await data_lifecycle.delete_session(session_id, owner_id=principal_id)
    except repo.SessionNotFound:
        raise HTTPException(status_code=404, detail="No such session.") from None
    return {"session_id": session_id, "deleted": True}


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(
    session_id: str,
    principal_id: str = Depends(require_principal),
) -> SessionDetailResponse:
    """Messages, papers and run history for one session."""
    try:
        session = await repo.get_session(session_id, owner_id=principal_id)
    except repo.SessionNotFound:
        raise HTTPException(status_code=404, detail="No such session.") from None
    runs = await repo.list_runs(session_id)
    project = None
    project_papers = []
    if session.project_id:
        try:
            project = await repo.get_project(session.project_id, owner_id=principal_id)
            project_papers = await repo.list_project_papers(
                project.project_id, exclude_session_id=session_id
            )
        except repo.ProjectNotFound:
            project = None
    last_tokens: int | None = None
    for run in runs:  # newest first
        tokens = (run.usage or {}).get("tokens")
        if isinstance(tokens, int):
            last_tokens = tokens
            break
    return SessionDetailResponse(
        session=session,
        messages=await repo.list_messages(session_id),
        papers=await repo.list_session_papers(session_id),
        runs=runs,
        artifacts=await repo.list_session_artifacts(session_id),
        context=SessionContextStats(
            compactions=await repo.count_events(session_id, EventType.CONTEXT_COMPACTED),
            last_turn_tokens=last_tokens,
        ),
        last_event_seq=await repo.last_event_seq(session_id),
        project=project,
        project_papers=project_papers,
    )


@router.post("/sessions/{session_id}/messages", response_model=PostMessageResponse)
@limiter.limit("30/minute")
async def post_message(
    request: Request,
    session_id: str,
    body: PostMessageRequest,
    caller: Caller = Depends(require_caller),
) -> PostMessageResponse:
    """Submit a turn. Persists first, schedules second, returns a run id."""
    principal_id = caller.principal_id
    try:
        session = await repo.get_session(session_id, owner_id=principal_id)
    except repo.SessionNotFound:
        raise HTTPException(status_code=404, detail="No such session.") from None

    question = (body.content or "").strip()
    if not question:
        raise HTTPException(status_code=422, detail="content must not be empty.")
    settings = get_settings()
    limit = settings.agent_max_message_chars
    if limit > 0 and len(question) > limit:
        # One message could otherwise fill the context and the turn's token
        # budget on its own. Long material belongs in an attached paper.
        raise HTTPException(
            status_code=422,
            detail={
                "code": "message_too_long",
                "message": f"A message may be at most {limit} characters.",
                "limit": limit,
                "length": len(question),
            },
        )

    quota = await _check_quota(caller, session_id, body.client_request_id)

    if body.paper_ids:
        await _attach_papers(session_id, body.paper_ids)
    if body.owner_token and not (session.config or {}).get("owner_token"):
        # Remembered on the session so mode reports made this turn and later
        # show up in the browser's compare matrix.
        await repo.update_session_config(session_id, {"owner_token": body.owner_token[:100]})
        session = await repo.get_session(session_id, owner_id=principal_id)

    try:
        run, deduplicated = await repo.create_run(
            session_id=session_id,
            owner_id=principal_id,
            client_request_id=body.client_request_id,
            llm_model=session.llm_model,
            prompt_version=_prompt_version(),
            # Stored on the run, so an answer can later be audited against the
            # limits it actually ran under rather than today's settings.
            budget=_run_budget(quota),
            max_active_for_owner=await _active_run_limit(caller),
        )
    except repo.TooManyActiveRuns:
        raise HTTPException(
            status_code=429,
            detail={
                "code": "too_many_active_runs",
                "message": "Too many conversations are answering at once. "
                "Wait for one to finish or stop it.",
                "limit": settings.agent_max_active_runs_per_principal,
            },
            headers={"Retry-After": "15"},
        ) from None
    except repo.ActiveRunConflict:
        raise HTTPException(
            status_code=409,
            detail="This session already has a run in progress.",
        ) from None

    if deduplicated:
        # The original run already exists and is either running or finished.
        # Re-scheduling it would double the external work the plan's A12 case
        # is specifically about.
        return PostMessageResponse(
            run_id=run.run_id,
            session_id=session_id,
            status=run.status,
            deduplicated=True,
        )

    await repo.append_message(
        session_id=session_id,
        role=MessageRole.USER,
        content=question,
        run_id=run.run_id,
    )
    await agent_runner.start_turn(
        session=session, run=run, question=question, mode=body.mode
    )
    return PostMessageResponse(
        run_id=run.run_id, session_id=session_id, status=run.status
    )


# ── Long-term memory ────────────────────────────────────────────────────────


async def _owned_project(project_id: str | None, principal_id: str) -> str | None:
    """`project_id` stripped, once it is known to be the caller's (or empty)."""
    if project_id is None:
        return None
    project_id = project_id.strip()
    if project_id:
        try:
            await repo.get_project(project_id, owner_id=principal_id)
        except repo.ProjectNotFound:
            raise HTTPException(status_code=404, detail="No such project.") from None
    return project_id


async def _memory_text(content: str, principal_id: str) -> str:
    """`content` as it may be stored: the rules `save_memory` applies."""
    from app.agents.tools.memory import looks_like_secret, memory_text_problem

    content = (content or "").strip()
    problem = memory_text_problem(content)
    if problem:
        raise HTTPException(status_code=422, detail=problem)
    if looks_like_secret(content) and looks_like_secret(
        content, await repo.list_owner_paper_ids(principal_id)
    ):
        raise HTTPException(status_code=422, detail="Credentials are never stored.")
    return content


@router.get("/memories")
async def list_memories(
    project_id: str | None = Query(default=None, max_length=64),
    principal_id: str = Depends(require_principal),
) -> dict[str, Any]:
    """What the agent remembers about the calling principal.

    Everything by default. With `project_id` it is what a conversation in that
    project works with — its memories and the global ones — and with an empty
    `project_id`, what one outside any project does.
    """
    memories = await repo.list_memories(principal_id, project_id=project_id)
    return {
        "memories": [m.model_dump(mode="json") for m in memories],
        "limit": repo.MAX_MEMORIES_PER_OWNER,
    }


@router.post("/memories")
@limiter.limit("30/minute")
async def create_memory(
    request: Request,
    body: CreateMemoryRequest,
    principal_id: str = Depends(require_principal),
) -> dict[str, Any]:
    """Let the reader add a memory by hand. Same rules as the tool, except
    that what the reader types needs no confirming."""
    content = await _memory_text(body.content, principal_id)
    project_id = await _owned_project(body.project_id, principal_id) or ""
    try:
        memory = await repo.add_memory(
            owner_id=principal_id, content=content, kind=body.kind, project_id=project_id
        )
    except repo.MemoryLimitReached:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "memory_limit",
                "message": f"At most {repo.MAX_MEMORIES_PER_OWNER} memories are kept.",
            },
        ) from None
    return memory.model_dump(mode="json")


@router.patch("/memories/{memory_id}")
@limiter.limit("60/minute")
async def update_memory(
    request: Request,
    memory_id: str,
    body: UpdateMemoryRequest,
    principal_id: str = Depends(require_principal),
) -> dict[str, Any]:
    """Edit a memory's text, kind or scope, or confirm one the agent saved."""
    content = None if body.content is None else await _memory_text(body.content, principal_id)
    try:
        memory = await repo.update_memory(
            memory_id,
            owner_id=principal_id,
            content=content,
            kind=body.kind,
            project_id=await _owned_project(body.project_id, principal_id),
            confirm=body.confirm,
        )
    except repo.MemoryConflict:
        raise HTTPException(
            status_code=409,
            detail={"code": "memory_exists", "message": "That is already a memory there."},
        ) from None
    if memory is None:
        raise HTTPException(status_code=404, detail="No such memory.")
    return memory.model_dump(mode="json")


# ── Research projects (P8) ──────────────────────────────────────────────────


@router.get("/projects")
async def list_projects(
    include_archived: bool = False,
    principal_id: str = Depends(require_principal),
) -> dict[str, Any]:
    projects = await repo.list_projects(principal_id, include_archived=include_archived)
    return {"projects": [p.model_dump() for p in projects]}


@router.post("/projects")
@limiter.limit("30/minute")
async def create_project(
    request: Request,
    body: CreateProjectRequest,
    principal_id: str = Depends(principal_or_new),
) -> dict[str, Any]:
    """Create a project. May mint a principal, like creating a session: a first-time
    visitor organising work before their first question is a normal start."""
    project = await repo.create_project(
        owner_id=principal_id, title=body.title, description=body.description
    )
    return {"project": project.model_dump()}


@router.get("/projects/{project_id}")
async def get_project(
    project_id: str,
    principal_id: str = Depends(require_principal),
) -> dict[str, Any]:
    """A project with its sessions, its papers and its own memories."""
    try:
        project = await repo.get_project(project_id, owner_id=principal_id)
    except repo.ProjectNotFound:
        raise HTTPException(status_code=404, detail="No such project.") from None
    sessions = await repo.list_sessions(principal_id, project_id=project_id, limit=200)
    papers = await repo.list_project_papers(project_id)
    memories = await repo.list_memories(
        principal_id, project_id=project_id, include_global=False
    )
    return {
        "project": project.model_dump(),
        "sessions": [s.model_dump() for s in sessions],
        "papers": [p.model_dump(mode="json") for p in papers],
        "memories": [m.model_dump(mode="json") for m in memories],
    }


@router.patch("/projects/{project_id}")
@limiter.limit("60/minute")
async def update_project(
    request: Request,
    project_id: str,
    body: UpdateProjectRequest,
    principal_id: str = Depends(require_principal),
) -> dict[str, Any]:
    try:
        project = await repo.update_project(
            project_id,
            owner_id=principal_id,
            title=body.title,
            description=body.description,
            status=body.status,
        )
    except repo.ProjectNotFound:
        raise HTTPException(status_code=404, detail="No such project.") from None
    return {"project": project.model_dump()}


@router.delete("/projects/{project_id}")
@limiter.limit("30/minute")
async def delete_project(
    request: Request,
    project_id: str,
    sessions: str = Query(default="keep", pattern="^(keep|delete)$"),
    principal_id: str = Depends(require_principal),
) -> dict[str, Any]:
    """Delete a project and its memories. Its conversations are moved out of it
    (`sessions=keep`) or deleted with it (`sessions=delete`)."""
    try:
        deleted = await data_lifecycle.delete_project(
            project_id, owner_id=principal_id, delete_sessions=sessions == "delete"
        )
    except repo.ProjectNotFound:
        raise HTTPException(status_code=404, detail="No such project.") from None
    return {"project_id": project_id, "deleted": True, "sessions_deleted": deleted}


@router.get("/search")
@limiter.limit("60/minute")
async def search_conversations(
    request: Request,
    q: str = Query(min_length=1, max_length=500),
    project_id: str | None = Query(default=None, max_length=64),
    limit: int = Query(default=10, ge=1, le=10),
    principal_id: str = Depends(require_principal),
) -> dict[str, Any]:
    """Search the caller's conversations — the same search the agent's recall uses.

    `project_id` narrows to one of the caller's projects. One that is not
    theirs is a 404, the same answer as one that does not exist.
    """
    if project_id:
        try:
            await repo.get_project(project_id, owner_id=principal_id)
        except repo.ProjectNotFound:
            raise HTTPException(status_code=404, detail="No such project.") from None
    turns = await conversation_recall.recall(
        principal_id, q, project_id=project_id or None, limit=limit
    )
    return {"query": q, "turns": [t.as_dict() for t in turns]}


@router.delete("/memories/{memory_id}")
async def delete_memory(
    memory_id: str,
    principal_id: str = Depends(require_principal),
) -> dict[str, Any]:
    removed = await repo.delete_memory(memory_id, owner_id=principal_id)
    if not removed:
        raise HTTPException(status_code=404, detail="No such memory.")
    return {"memory_id": memory_id, "deleted": True}


@router.post("/runs/{run_id}/cancel")
async def cancel_run(
    run_id: str,
    principal_id: str = Depends(require_principal),
) -> dict[str, Any]:
    """Request cancellation. Idempotent; safe to call on a finished run."""
    try:
        run = await repo.request_cancel(run_id, owner_id=principal_id)
    except repo.SessionNotFound:
        raise HTTPException(status_code=404, detail="No such run.") from None
    # The flag alone is only read between streaming steps. A turn waiting on a
    # download or a parse would otherwise ignore stop for minutes, so the task
    # is interrupted as well when this process is the one running it.
    interrupted = await agent_runner.request_stop(run_id)
    return {
        "run_id": run.run_id,
        "status": run.status.value,
        "cancel_requested": True,
        "interrupted": interrupted,
    }


@router.get("/runs/{run_id}/activity")
async def run_activity(
    run_id: str,
    principal_id: str = Depends(require_principal),
) -> dict[str, Any]:
    """What a past turn did, replayed from the durable event log.

    A finished turn's tool activity exists only in the stream that carried it,
    so reopening a session would otherwise show answers with no record of what
    was read to produce them. The client folds these events with the same
    reducer it applies live, which is the only way the restored view and the
    live view can be guaranteed to agree.

    `message.delta` is left out: the answer is already stored as a message.
    """
    try:
        run = await repo.get_run(run_id, owner_id=principal_id)
    except repo.SessionNotFound:
        raise HTTPException(status_code=404, detail="No such run.") from None
    events = await repo.list_run_events(run_id, exclude_types=("message.delta",))
    return {
        "run_id": run.run_id,
        "status": run.status.value,
        "error_code": run.error_code,
        "events": [e.model_dump(mode="json") for e in events],
    }


@router.post("/runs/{run_id}/stream-ticket")
async def stream_ticket(
    run_id: str,
    principal_id: str = Depends(require_principal),
) -> dict[str, Any]:
    """A short-lived ticket that opens this run's event stream.

    The stream is the one request that goes straight to the backend rather than
    through the same-origin proxy (which buffers it), so the login cookie does
    not reach it — and being HttpOnly, page scripts cannot put it in a URL
    either. The ticket names one run and expires in minutes, so a copy that
    lands in an access log opens nothing else, and nothing for long.
    """
    try:
        await repo.get_run(run_id, owner_id=principal_id)
    except repo.SessionNotFound:
        raise HTTPException(status_code=404, detail="No such run.") from None
    return {
        "ticket": identity.seal(
            "stream", {"p": principal_id, "r": run_id}, ttl_seconds=_STREAM_TICKET_SECONDS
        ),
        "expires_in": _STREAM_TICKET_SECONDS,
    }


async def _stream_principal(request: Request, run_id: str, ticket: str) -> str:
    """Who may open a stream: a ticket for this run, else the usual credential."""
    if ticket:
        payload = identity.unseal("stream", ticket)
        if not payload or payload.get("r") != run_id:
            raise HTTPException(status_code=401, detail="Invalid or expired stream ticket.")
        return str(payload["p"])
    caller = await resolve_caller(request)
    if caller is None:
        raise HTTPException(status_code=401, detail="Login required.")
    return caller.principal_id


@router.get("/runs/{run_id}/events")
async def stream_events(
    request: Request,
    run_id: str,
    after: int = Query(default=0, ge=0),
    ticket: str = Query(default=""),
    last_event_id: str = Header(default="", alias="Last-Event-ID"),
) -> StreamingResponse:
    """Stream a run's events, resuming from `after` or `Last-Event-ID`.

    `EventSource` cannot set headers, so access comes from `?ticket=` (see
    `stream_ticket`). The cookie, the header and the old `?token=` credential
    still work for same-origin and pre-accounts clients.
    """
    principal_id = await _stream_principal(request, run_id, ticket)
    try:
        run = await repo.get_run(run_id, owner_id=principal_id)
    except repo.SessionNotFound:
        raise HTTPException(status_code=404, detail="No such run.") from None

    resume_from = after
    if last_event_id.isdigit():
        resume_from = max(resume_from, int(last_event_id))

    async def generate():
        queue = agent_runner.subscribe(run.session_id)
        try:
            # Replay first, so a client that reconnects mid-run sees the gap
            # before it sees anything live.
            backlog = await repo.list_events(run.session_id, after_seq=resume_from)
            highest = resume_from
            for event in backlog:
                if event.run_id and event.run_id != run_id:
                    continue
                highest = max(highest, event.seq)
                yield _sse(event.to_sse())
                if _is_terminal(event.type.value):
                    return

            loop = asyncio.get_running_loop()
            last_sent = loop.time()
            while True:
                # The queue only hears a turn this process executes. One that
                # runs on another replica — or is closed by another replica's
                # recovery sweep — reaches this stream through the table.
                local = agent_runner.is_executing(run_id)
                try:
                    events = [
                        await asyncio.wait_for(
                            queue.get(),
                            timeout=_SSE_KEEPALIVE_SECONDS if local else _SSE_POLL_SECONDS,
                        )
                    ]
                except asyncio.TimeoutError:
                    events = (
                        []
                        if local
                        else await repo.list_events(run.session_id, after_seq=highest)
                    )
                    if not events and loop.time() - last_sent >= _SSE_KEEPALIVE_SECONDS:
                        last_sent = loop.time()
                        yield ": keepalive\n\n"
                for event in events:
                    if event.seq <= highest:
                        continue   # already replayed from the table
                    highest = event.seq
                    if event.run_id and event.run_id != run_id:
                        continue
                    last_sent = loop.time()
                    yield _sse(event.to_sse())
                    if _is_terminal(event.type.value):
                        return
        finally:
            agent_runner.unsubscribe(run.session_id, queue)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(frame: dict[str, str]) -> str:
    return f"id: {frame['id']}\nevent: {frame['event']}\ndata: {frame['data']}\n\n"


def _is_terminal(event_type: str) -> bool:
    return event_type in {"run.completed", "run.failed", "run.cancelled"}


@router.get("/evidence/{evidence_id}")
async def get_evidence(
    evidence_id: str,
    principal_id: str = Depends(require_principal),
) -> dict[str, Any]:
    """Resolve a citation to its paper, page and original excerpt."""
    try:
        evidence = await evidence_service.resolve(evidence_id, owner_id=principal_id)
    except repo.EvidenceNotFound:
        raise HTTPException(status_code=404, detail="No such evidence.") from None
    return await evidence_service.evidence_to_api(evidence)


def _run_budget(quota: accounts.QuotaUsage | None = None) -> dict[str, object]:
    """The ceilings a new turn runs under, never above what is left of today.

    The daily quota is checked when a turn is submitted; without this a reader
    with a few thousand tokens left could still start a turn allowed millions.
    A turn with little left closes early and says so, which is the wrap-up
    doing its job.
    """
    from app.agents.context import RunBudget

    budget = RunBudget.from_settings()
    if quota is not None and quota.tokens_limit > 0:
        budget.max_tokens = min(budget.max_tokens, max(1, quota.tokens_limit - quota.tokens))
    return budget.as_dict()


def _prompt_version() -> str:
    from app.agents.prompts import PROMPT_VERSION

    return PROMPT_VERSION
