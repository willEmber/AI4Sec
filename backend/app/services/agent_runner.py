"""Executes one conversational turn and turns it into durable business events.

Three responsibilities, deliberately kept apart from the agent itself:

*Translation.* Deep Agents' stream is framework detail. It is converted here
into the event types the API contract names, so the frontend never binds to
LangChain objects (development plan §3 rule 7).

*Durability.* Every event is written to `agent_events` before anyone publishes
it, which is what lets a reconnecting client replay what it missed. An
in-memory fan-out exists only to avoid polling for live listeners.

*Cancellation.* The flag is checked between steps. Work already handed to an
external service may still finish and land in a cache, but a cancelled run is
never resumed from it (development plan §7.2).

*Ownership.* A turn is executed by a task in this process, and says so: the run
carries this worker's id and a heartbeat. That is what makes "nobody is running
this" an observable fact rather than an assumption, and it is what
`agent_worker` acts on. The seam is still `start_turn` — nothing above it
assumes the work happens here — but the recovery story no longer depends on
moving it elsewhere.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from app.agents.context import AgentContext, BudgetUsage, RunBudget
from app.agents.harness import create_paper_agent
from app.agents.middleware import build_agent_middleware
from app.agents.model_factory import build_chat_model
from app.agents.prompts import PROMPT_VERSION, build_system_prompt, mode_instruction
from app.agents.tools import ALL_AGENT_TOOLS
from app.config import get_settings
from app.db import agent_repository as repo
from app.models.agent_models import (
    AgentEvent,
    AgentProject,
    AgentRun,
    AgentSession,
    ErrorCode,
    EventType,
    MessageRole,
    RunStatus,
    SessionPaper,
)
from app.services.agent_worker import HEARTBEAT_SECONDS, WORKER_ID
from app.services.llm_gateway import get_registry

logger = logging.getLogger("scholar.agent.runner")

# Live listeners per session. Purely an optimisation over polling the event
# table; a listener that arrives late reads the table instead and misses
# nothing.
_subscribers: dict[str, set[asyncio.Queue[AgentEvent]]] = {}
_tasks: dict[str, asyncio.Task[None]] = {}


def subscribe(session_id: str) -> asyncio.Queue[AgentEvent]:
    queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
    _subscribers.setdefault(session_id, set()).add(queue)
    return queue


def unsubscribe(session_id: str, queue: asyncio.Queue[AgentEvent]) -> None:
    listeners = _subscribers.get(session_id)
    if not listeners:
        return
    listeners.discard(queue)
    if not listeners:
        _subscribers.pop(session_id, None)


async def emit(
    *,
    session_id: str,
    run_id: str,
    type: EventType,
    payload: dict[str, Any] | None = None,
) -> AgentEvent:
    """Persist an event, then hand it to any live listener.

    Public because recovery writes events too: a run this process never
    executed still has to reach a terminal event, and it has to reach the
    browser watching it now rather than on the next reload.
    """
    event = await repo.append_event(
        session_id=session_id, run_id=run_id, type=type, payload=payload or {}
    )
    for queue in list(_subscribers.get(session_id, set())):
        queue.put_nowait(event)
    return event


class DeltaCoalescer:
    """Buffers one turn's answer text and emits it as fewer, larger deltas.

    Every event is a committed row, so emitting one per model chunk made a turn
    hundreds of writes. Text is flushed when the oldest buffered fragment is
    `flush_ms` old or the buffer reaches `flush_chars`, whichever comes first;
    a timer covers the case where the model pauses with text still buffered.

    Order matters more than batching: a delta must never land after an event
    the model produced later. Every other event of the turn therefore goes
    through :meth:`emit`, which flushes first, and flush and emit share a lock
    so the timer cannot interleave a stale delta between them.
    """

    def __init__(
        self,
        *,
        session_id: str,
        run_id: str,
        flush_ms: int,
        flush_chars: int,
    ) -> None:
        self._session_id = session_id
        self._run_id = run_id
        self._flush_seconds = max(0, flush_ms) / 1000.0
        self._flush_chars = max(1, flush_chars)
        self._parts: list[str] = []
        self._size = 0
        self._lock = asyncio.Lock()
        self._timer: asyncio.Task[None] | None = None

    async def add(self, text: str) -> None:
        if not text:
            return
        async with self._lock:
            self._parts.append(text)
            self._size += len(text)
            if self._size >= self._flush_chars or self._flush_seconds == 0:
                await self._flush_locked()
                return
        if self._timer is None or self._timer.done():
            self._timer = asyncio.create_task(self._flush_later())

    async def _flush_later(self) -> None:
        await asyncio.sleep(self._flush_seconds)
        async with self._lock:
            await self._flush_locked()

    async def _flush_locked(self) -> None:
        if not self._parts:
            return
        text = "".join(self._parts)
        self._parts.clear()
        self._size = 0
        await emit(
            session_id=self._session_id,
            run_id=self._run_id,
            type=EventType.MESSAGE_DELTA,
            payload={"text": text},
        )

    async def flush(self) -> None:
        async with self._lock:
            await self._flush_locked()

    async def emit(self, type: EventType, payload: dict[str, Any] | None = None) -> AgentEvent:
        """Emit a non-delta event of this turn, after any text that preceded it."""
        async with self._lock:
            await self._flush_locked()
            return await emit(
                session_id=self._session_id, run_id=self._run_id, type=type, payload=payload
            )

    async def close(self) -> None:
        """Flush what is left. Safe to call more than once.

        The timer is awaited rather than cancelled: cancelling it mid-write
        would drop text it had already taken out of the buffer, and it finishes
        within one flush window anyway.
        """
        timer, self._timer = self._timer, None
        if timer is not None and not timer.done():
            with contextlib.suppress(Exception):
                await timer
        await self.flush()


def is_executing(run_id: str) -> bool:
    """Whether this process is running that turn right now."""
    task = _tasks.get(run_id)
    return task is not None and not task.done()


async def request_stop(run_id: str) -> bool:
    """Interrupt a turn this process is executing.

    The cancel flag alone only takes effect between streaming steps, so a turn
    blocked inside a tool — a download, a parse, a slow provider — would keep
    going for minutes after the reader pressed stop. Cancelling the task ends
    the wait; the executor's `CancelledError` path closes the run and writes the
    terminal event (acceptance case A15).
    """
    task = _tasks.get(run_id)
    if task is None or task.done():
        return False
    task.cancel()
    return True


def _announced_calls(
    message: AIMessage | AIMessageChunk, seen: set[str]
) -> list[tuple[str, str]]:
    """Tool calls in this message that have not been announced yet.

    A streaming model delivers `tool_call_chunks`, whose name arrives in the
    first fragment; a non-streaming one delivers a finished `tool_calls` list.
    Both are read, and `seen` de-duplicates, so a call is announced exactly once
    however the model chose to send it.
    """
    calls: list[tuple[str, str]] = []
    chunks = getattr(message, "tool_call_chunks", None) or []
    raw = chunks or (getattr(message, "tool_calls", None) or [])
    for call in raw:
        name = call.get("name") or ""
        if not name:
            continue
        call_id = call.get("id") or ""
        key = call_id or f"{name}:{len(seen)}"
        if key in seen:
            continue
        seen.add(key)
        calls.append((name, call_id))
    return calls


def _text_of(content: Any) -> str:
    """Pull plain text out of a message, ignoring reasoning and tool blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
            parts.append(block.get("text", "") or "")
    return "".join(parts)


def _citations_in(text: str, known: list[str]) -> list[str]:
    """Which of the evidence ids handed to the model actually got cited.

    Scanning for known ids rather than parsing `[ev_...]` patterns means an id
    the model invented never enters the citation list — it simply is not in
    `known`, and the answer's claim is left uncited rather than pointing at
    something that does not exist.
    """
    return [eid for eid in dict.fromkeys(known) if eid in text]


async def _heartbeat(run_id: str) -> None:
    """Say this worker is still on the run, until the turn stops it.

    Without this the only signal of life is the run's status, which says
    "running" just as loudly after the process holding it is gone. Scoped to
    this worker, so a stamp from a process that has already been recovered does
    not resurrect the run.
    """
    while True:
        await asyncio.sleep(HEARTBEAT_SECONDS)
        with contextlib.suppress(Exception):
            await repo.heartbeat_run(run_id, worker_id=WORKER_ID)


async def start_turn(
    *, session: AgentSession, run: AgentRun, question: str, mode: str = "auto"
) -> None:
    """Schedule a turn. Returns as soon as the work is queued."""
    task = asyncio.create_task(
        _execute_turn(session=session, run=run, question=question, mode=mode)
    )
    _tasks[run.run_id] = task
    task.add_done_callback(lambda _t: _tasks.pop(run.run_id, None))


def _usage_of(message: AIMessage | AIMessageChunk) -> int:
    """Total tokens a model call reported on this message, or 0 when it did not."""
    meta = getattr(message, "usage_metadata", None) or {}
    try:
        return int(meta.get("total_tokens") or 0)
    except (TypeError, ValueError):
        return 0


def _summary_model(chat_model: Any) -> Any:
    """The model that compacts context: the utility model, whoever is answering.

    Compaction is a utility call, so every conversation is summarised the same
    way and an extension gateway's model is never asked to do it.
    """
    registry = get_registry()
    name = registry.utility_model
    if not name or registry.profile(name).request_model == getattr(chat_model, "model_name", ""):
        return chat_model
    try:
        return build_chat_model(name)
    except Exception:  # noqa: BLE001 — a turn must not fail over its summariser
        logger.warning("Utility model %r unavailable; summarising with the turn's model", name)
        return chat_model


def _new_usage(message: AIMessage | AIMessageChunk, counted: dict[str, int]) -> tuple[int, bool]:
    """Tokens on this message not yet counted, and whether it opens a new model call.

    One call's usage normally arrives once, on its last chunk. New-API sends it
    twice — on the last content chunk and again on the closing usage chunk —
    and summing both doubled a turn's token count. Chunks of one call share a
    message id, so each id is counted up to the largest total it reported.
    """
    reported = _usage_of(message)
    if not reported:
        return 0, False
    message_id = getattr(message, "id", None)
    if not message_id:
        return reported, True
    already = counted.get(message_id, 0)
    if reported <= already:
        return 0, False
    counted[message_id] = reported
    return reported - already, already == 0


def _model_input(question: str, mode: str, language: str) -> str:
    """The user turn as the model sees it: the question plus the chosen mode.

    The persisted user message stays the reader's own words; the mode note is
    an instruction about *this* turn, so it travels with the turn and never
    lands in the transcript the reader scrolls back through.
    """
    note = mode_instruction(mode, language)
    return f"{question}\n\n{note}" if note else question


async def _execute_turn(
    *, session: AgentSession, run: AgentRun, question: str, mode: str = "auto"
) -> None:
    """Run one turn to completion, emitting events as it goes."""
    started = time.perf_counter()
    usage = BudgetUsage()
    budget = RunBudget.from_settings(run.budget)
    collected_evidence: list[str] = []
    answer_parts: list[str] = []
    seen_calls: set[str] = set()
    counted_usage: dict[str, int] = {}
    heartbeat: asyncio.Task[None] | None = None
    settings = get_settings()
    stream = DeltaCoalescer(
        session_id=session.session_id,
        run_id=run.run_id,
        flush_ms=settings.agent_delta_flush_ms,
        flush_chars=settings.agent_delta_flush_chars,
    )

    try:
        await repo.claim_run(run.run_id, worker_id=WORKER_ID)
        heartbeat = asyncio.create_task(_heartbeat(run.run_id))
        await stream.emit(
            EventType.RUN_STARTED,
            {"question": question, "model": run.llm_model, "mode": mode},
        )

        papers = await repo.list_session_papers(session.session_id)
        project, project_papers = await _project_context(session)
        memories = await repo.list_memories_for_prompt(
            session.owner_id,
            project_id=project.project_id if project is not None else "",
            limit=settings.agent_memory_max_items,
        )
        context = AgentContext(
            owner_id=session.owner_id,
            session_id=session.session_id,
            run_id=run.run_id,
            thread_id=session.thread_id,
            language=session.language,
            llm_model=run.llm_model,
            owner_token=str((session.config or {}).get("owner_token") or ""),
            project_id=project.project_id if project is not None else "",
            budget=budget,
            usage=usage,
        )

        async def _on_compact(payload: dict[str, Any]) -> None:
            await stream.emit(EventType.CONTEXT_COMPACTED, payload)

        chat_model = build_chat_model(run.llm_model)
        # Off the loop: building the graph is synchronous work, and on a cold
        # process it also triggers the deepagents import. Neither should stop
        # the server answering anything else — a cancel for this very turn
        # included.
        agent = await asyncio.to_thread(
            create_paper_agent,
            tools=ALL_AGENT_TOOLS,
            system_prompt=build_system_prompt(
                language=session.language,
                papers=papers,
                memories=memories,
                project=project,
                project_papers=project_papers,
            ),
            model=chat_model,
            context_schema=AgentContext,
            checkpointer=_checkpointer(),
            middleware=build_agent_middleware(
                _summary_model(chat_model), on_compact=_on_compact
            ),
        )

        cancelled = False
        async for stream_mode, chunk in agent.astream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": _model_input(question, mode, session.language),
                    }
                ]
            },
            config=context.config(),
            context=context,
            stream_mode=["messages"],
        ):
            if stream_mode != "messages":
                continue
            if await repo.is_cancel_requested(run.run_id):
                cancelled = True
                break

            message, _meta = chunk
            if isinstance(message, ToolMessage):
                usage.tool_calls += 1
                payload = _tool_completion_payload(message)
                collected_evidence.extend(payload.pop("_evidence_ids", []))
                attached = payload.pop("_attached_paper", None)
                await stream.emit(
                    EventType.TOOL_FAILED
                    if payload.get("status") in {"error"}
                    else EventType.TOOL_COMPLETED,
                    payload,
                )
                if attached:
                    # The session gained a paper mid-turn. The client shows the
                    # paper list beside the conversation, so it has to hear about
                    # it now rather than on the next reload.
                    await stream.emit(EventType.PAPER_ADDED, attached)
                continue

            if isinstance(message, (AIMessage, AIMessageChunk)):
                for name, call_id in _announced_calls(message, seen_calls):
                    await stream.emit(
                        EventType.TOOL_STARTED, {"tool": name, "call_id": call_id}
                    )
                # Summed per run, so the token ceiling is enforced against what
                # was actually billed rather than left at "unknown".
                reported, new_call = _new_usage(message, counted_usage)
                if reported:
                    usage.tokens = (usage.tokens or 0) + reported
                if new_call:
                    usage.llm_calls += 1
                text = _text_of(message.content)
                if text:
                    answer_parts.append(text)
                    await stream.add(text)

            # Refreshed every step: the wall-clock ceiling is the only one that
            # keeps rising while nothing else happens, so a slow parse or a
            # stalled provider has to be able to end the turn.
            usage.wall_seconds = time.perf_counter() - started
            exceeded = usage.exceeded(budget)
            if exceeded:
                logger.warning("Run %s hit budget ceiling %s", run.run_id, exceeded)
                answer_parts.append(
                    f"\n\n_[stopped: {exceeded} reached; the findings above are what was gathered]_"
                )
                break

        await stream.close()
        usage.wall_seconds = time.perf_counter() - started
        answer = "".join(answer_parts).strip()
        citations = _citations_in(answer, collected_evidence)

        if cancelled:
            await repo.finish_run(
                run.run_id,
                status=RunStatus.CANCELLED,
                error_code=ErrorCode.CANCELLED.value,
                usage=usage.as_dict(),
            )
            await stream.emit(EventType.RUN_CANCELLED, {"usage": usage.as_dict()})
            return

        if answer:
            await repo.append_message(
                session_id=session.session_id,
                role=MessageRole.ASSISTANT,
                content=answer,
                run_id=run.run_id,
                citations=citations,
            )
            await repo.set_session_title(session.session_id, question[:60])

        await repo.finish_run(run.run_id, status=RunStatus.DONE, usage=usage.as_dict())
        await repo.touch_session(session.session_id)
        await stream.emit(
            EventType.RUN_COMPLETED, {"citations": citations, "usage": usage.as_dict()}
        )

    except asyncio.CancelledError:
        await repo.finish_run(
            run.run_id, status=RunStatus.CANCELLED, error_code=ErrorCode.CANCELLED.value
        )
        with contextlib.suppress(Exception):
            await stream.close()
            await stream.emit(EventType.RUN_CANCELLED, {})
        raise

    except Exception as exc:  # noqa: BLE001 — a failed turn must still close its run
        logger.exception("Agent run %s failed", run.run_id)
        usage.wall_seconds = time.perf_counter() - started
        usage.errors += 1
        await repo.finish_run(
            run.run_id,
            status=RunStatus.FAILED,
            error_code=ErrorCode.UPSTREAM_ERROR.value,
            error_msg=str(exc)[:500],
            usage=usage.as_dict(),
        )
        with contextlib.suppress(Exception):
            await stream.close()
            await stream.emit(
                EventType.RUN_FAILED,
                {
                    "code": ErrorCode.UPSTREAM_ERROR.value,
                    "error": str(exc)[:500],
                    "usage": usage.as_dict(),
                },
            )

    finally:
        if heartbeat is not None:
            heartbeat.cancel()


async def _project_context(
    session: AgentSession,
) -> tuple[AgentProject | None, list[SessionPaper]]:
    """The session's project and the project papers it has not attached, if any.

    A project that does not resolve for the session's owner is treated as
    none, so recall and project papers are never scoped to something the
    reader does not own.
    """
    if not session.project_id:
        return None, []
    try:
        project = await repo.get_project(session.project_id, owner_id=session.owner_id)
    except repo.ProjectNotFound:
        logger.warning(
            "Session %s names project %s, which its owner does not have",
            session.session_id,
            session.project_id,
        )
        return None, []
    papers = await repo.list_project_papers(
        project.project_id, exclude_session_id=session.session_id
    )
    return project, papers


def _tool_completion_payload(message: ToolMessage) -> dict[str, Any]:
    """Summarise a tool result for the event stream.

    The full result goes to the model, not to the client: it can be tens of
    kilobytes of paper text, and the UI only needs to show what ran and what
    came back.
    """
    import json

    payload: dict[str, Any] = {"tool": message.name or "", "status": "ok"}
    evidence_ids: list[str] = []
    try:
        parsed = json.loads(message.content if isinstance(message.content, str) else "{}")
    except (TypeError, ValueError):
        parsed = {}
    if isinstance(parsed, dict):
        payload["status"] = parsed.get("status", "ok")
        evidence_ids = list(parsed.get("evidence_ids") or [])
        payload["evidence_count"] = len(evidence_ids)
        if parsed.get("note"):
            payload["note"] = parsed["note"]
        error = parsed.get("error")
        if isinstance(error, dict):
            payload["error"] = {
                "code": error.get("code", ""),
                "message": error.get("message", ""),
                "retryable": error.get("retryable", False),
            }
        data = parsed.get("data")
        if isinstance(data, dict):
            payload["summary"] = _summarise(data)
            if data.get("paper_id") and data.get("literature_id"):
                payload["_attached_paper"] = {
                    "literature_id": data["literature_id"],
                    "paper_id": data["paper_id"],
                    "title": data.get("title", ""),
                    "availability": data.get("availability", ""),
                }
    payload["_evidence_ids"] = evidence_ids
    return payload


def _summarise(data: dict[str, Any]) -> dict[str, Any]:
    """A few scalars a UI can show without carrying the whole result."""
    summary: dict[str, Any] = {}
    for key in (
        "paper_id", "section", "question", "title", "page",
        "query", "venue", "relation", "of_paper", "source", "availability",
        "run_id", "mode", "reused", "report_url", "memory_id", "kind",
        "url", "domain", "provider", "topic", "cached",
    ):
        if key in data:
            summary[key] = data[key]
    for key in (
        "sections", "hits", "blocks", "results", "papers", "rankings", "unknown_year", "memories",
        "chunks", "unknown_date", "turns",
    ):
        if isinstance(data.get(key), list):
            summary[f"{key}_count"] = len(data[key])
    return summary


# ── Checkpointer lifetime ───────────────────────────────────────────────────
# Held open for the process rather than per run: the saver owns a connection
# pool, and it is safe to share between concurrent turns.
_checkpointer_cm = None
_checkpointer_instance = None


def _checkpointer():
    return _checkpointer_instance


async def warm_up_agent() -> None:
    """Compile the agent graph before a reader's question has to wait for it.

    Built with the same tools, context schema, checkpointer *and chat model* a
    turn uses, because the graph compiled depends on all of them — warming a
    different shape warms nothing. Off the event loop, since compilation is
    CPU-bound Python.

    This also matters for stopping: `asyncio.to_thread` cannot be cancelled once
    the thread is running, so a turn interrupted during its own first build
    finishes that build before it can unwind. Paying the cost here is what keeps
    stop responsive.

    A model that cannot be constructed — no gateway configured — still leaves
    most of the machinery worth warming, so the build goes ahead without one
    rather than skipping the warm-up entirely.
    """
    from app.agents.harness import warm_up

    try:
        model = build_chat_model("")
    except Exception:  # noqa: BLE001
        logger.warning("Warming up without a chat model; the first turn will be slower")
        model = None

    await asyncio.to_thread(
        warm_up,
        context_schema=AgentContext,
        checkpointer=_checkpointer(),
        model=model,
        middleware=build_agent_middleware(_summary_model(model) if model else None),
    )


async def open_agent_checkpointer() -> None:
    """Open the shared checkpointer. Called from the app lifespan."""
    global _checkpointer_cm, _checkpointer_instance
    if _checkpointer_instance is not None:
        return
    from app.agents.checkpointer import open_checkpointer

    _checkpointer_cm = open_checkpointer()
    _checkpointer_instance = await _checkpointer_cm.__aenter__()


async def close_agent_checkpointer() -> None:
    """Close the shared checkpointer. Called from the app lifespan."""
    global _checkpointer_cm, _checkpointer_instance
    if _checkpointer_cm is None:
        return
    cm, _checkpointer_cm, _checkpointer_instance = _checkpointer_cm, None, None
    with contextlib.suppress(Exception):
        await cm.__aexit__(None, None, None)
