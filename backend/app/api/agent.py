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

from app.api.deps import principal_or_new, require_principal
from app.db import agent_repository as repo
from app.db import database as db
from app.models.agent_models import (
    Availability,
    CreateSessionRequest,
    CreateSessionResponse,
    MessageRole,
    PostMessageRequest,
    PostMessageResponse,
    SessionDetailResponse,
)
from app.rate_limit import limiter
from app.services import agent_runner, evidence_service, paper_catalog
from app.services.identity import AGENT_TOKEN_HEADER

logger = logging.getLogger("scholar.api.agent")

router = APIRouter(prefix="/agent", tags=["agent"])

# How long an idle stream waits before sending a keepalive. Proxies routinely
# drop a silent connection at 60s, and a turn can think for longer than that.
_SSE_KEEPALIVE_SECONDS = 15.0


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
        await repo.attach_session_paper(
            session_id=session_id,
            literature_id=literature_id,
            paper_id=paper_id,
            availability=await _availability_of(paper_id),
            added_by="user",
        )
        attached.append(paper_id)
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

    The only route that mints a principal: a first-time client has no
    credential yet, and `principal_or_new` returns the new one in the
    `X-Agent-Token` response header.
    """
    session = await repo.create_session(
        owner_id=principal_id,
        title=body.title,
        language=body.language,
        llm_model=body.llm_model,
    )
    if body.paper_ids:
        await _attach_papers(session.session_id, body.paper_ids)
    return CreateSessionResponse(
        session_id=session.session_id,
        thread_id=session.thread_id,
        created_at=session.created_at,
    )


@router.get("/sessions")
async def list_sessions(
    request: Request,
    x_agent_token: str = Header(default="", alias=AGENT_TOKEN_HEADER),
) -> dict[str, Any]:
    """Recent sessions for the calling principal."""
    principal_id = await require_principal(x_agent_token)
    sessions = await repo.list_sessions(principal_id)
    return {"sessions": [s.model_dump() for s in sessions]}


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(
    session_id: str,
    x_agent_token: str = Header(default="", alias=AGENT_TOKEN_HEADER),
) -> SessionDetailResponse:
    """Messages, papers and run history for one session."""
    principal_id = await require_principal(x_agent_token)
    try:
        session = await repo.get_session(session_id, owner_id=principal_id)
    except repo.SessionNotFound:
        raise HTTPException(status_code=404, detail="No such session.") from None
    return SessionDetailResponse(
        session=session,
        messages=await repo.list_messages(session_id),
        papers=await repo.list_session_papers(session_id),
        runs=await repo.list_runs(session_id),
        last_event_seq=await repo.last_event_seq(session_id),
    )


@router.post("/sessions/{session_id}/messages", response_model=PostMessageResponse)
@limiter.limit("30/minute")
async def post_message(
    request: Request,
    session_id: str,
    body: PostMessageRequest,
    x_agent_token: str = Header(default="", alias=AGENT_TOKEN_HEADER),
) -> PostMessageResponse:
    """Submit a turn. Persists first, schedules second, returns a run id."""
    principal_id = await require_principal(x_agent_token)
    try:
        session = await repo.get_session(session_id, owner_id=principal_id)
    except repo.SessionNotFound:
        raise HTTPException(status_code=404, detail="No such session.") from None

    question = (body.content or "").strip()
    if not question:
        raise HTTPException(status_code=422, detail="content must not be empty.")

    if body.paper_ids:
        await _attach_papers(session_id, body.paper_ids)

    try:
        run, deduplicated = await repo.create_run(
            session_id=session_id,
            owner_id=principal_id,
            client_request_id=body.client_request_id,
            llm_model=session.llm_model,
            prompt_version=_prompt_version(),
            # Stored on the run, so an answer can later be audited against the
            # limits it actually ran under rather than today's settings.
            budget=_run_budget(),
        )
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
    await agent_runner.start_turn(session=session, run=run, question=question)
    return PostMessageResponse(
        run_id=run.run_id, session_id=session_id, status=run.status
    )


@router.post("/runs/{run_id}/cancel")
async def cancel_run(
    run_id: str,
    x_agent_token: str = Header(default="", alias=AGENT_TOKEN_HEADER),
) -> dict[str, Any]:
    """Request cancellation. Idempotent; safe to call on a finished run."""
    principal_id = await require_principal(x_agent_token)
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
    x_agent_token: str = Header(default="", alias=AGENT_TOKEN_HEADER),
) -> dict[str, Any]:
    """What a past turn did, replayed from the durable event log.

    A finished turn's tool activity exists only in the stream that carried it,
    so reopening a session would otherwise show answers with no record of what
    was read to produce them. The client folds these events with the same
    reducer it applies live, which is the only way the restored view and the
    live view can be guaranteed to agree.

    `message.delta` is left out: the answer is already stored as a message.
    """
    principal_id = await require_principal(x_agent_token)
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


@router.get("/runs/{run_id}/events")
async def stream_events(
    run_id: str,
    after: int = Query(default=0, ge=0),
    token: str = Query(default=""),
    last_event_id: str = Header(default="", alias="Last-Event-ID"),
    x_agent_token: str = Header(default="", alias=AGENT_TOKEN_HEADER),
) -> StreamingResponse:
    """Stream a run's events, resuming from `after` or `Last-Event-ID`.

    `EventSource` cannot set headers, so the credential may also arrive as a
    query parameter. That is the same credential either way; it is not a weaker
    check, though it does mean the value can land in an access log, which is why
    every other route takes the header.
    """
    principal_id = await require_principal(x_agent_token or token)
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

            while True:
                try:
                    event = await asyncio.wait_for(
                        queue.get(), timeout=_SSE_KEEPALIVE_SECONDS
                    )
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if event.run_id and event.run_id != run_id:
                    continue
                if event.seq <= highest:
                    continue   # already replayed from the table
                highest = event.seq
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
    x_agent_token: str = Header(default="", alias=AGENT_TOKEN_HEADER),
) -> dict[str, Any]:
    """Resolve a citation to its paper, page and original excerpt."""
    principal_id = await require_principal(x_agent_token)
    try:
        evidence = await evidence_service.resolve(evidence_id, owner_id=principal_id)
    except repo.EvidenceNotFound:
        raise HTTPException(status_code=404, detail="No such evidence.") from None
    return await evidence_service.evidence_to_api(evidence)


def _run_budget() -> dict[str, object]:
    from app.agents.context import RunBudget

    return RunBudget.from_settings().as_dict()


def _prompt_version() -> str:
    from app.agents.prompts import PROMPT_VERSION

    return PROMPT_VERSION
