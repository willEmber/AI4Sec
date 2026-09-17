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

Scheduling in this version is an in-process task, which is enough for a single
reading session. P4 replaces it with a leased worker; the seam is
`start_turn`, and nothing above it assumes the work happens in this process.
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
from app.agents.model_factory import build_chat_model
from app.agents.prompts import PROMPT_VERSION, build_system_prompt
from app.agents.tools import READING_TOOLS
from app.db import agent_repository as repo
from app.models.agent_models import (
    AgentEvent,
    AgentRun,
    AgentSession,
    ErrorCode,
    EventType,
    MessageRole,
    RunStatus,
)

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


async def _emit(
    *,
    session_id: str,
    run_id: str,
    type: EventType,
    payload: dict[str, Any] | None = None,
) -> AgentEvent:
    """Persist an event, then hand it to any live listener."""
    event = await repo.append_event(
        session_id=session_id, run_id=run_id, type=type, payload=payload or {}
    )
    for queue in list(_subscribers.get(session_id, set())):
        queue.put_nowait(event)
    return event


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


async def start_turn(
    *, session: AgentSession, run: AgentRun, question: str
) -> None:
    """Schedule a turn. Returns as soon as the work is queued."""
    task = asyncio.create_task(_execute_turn(session=session, run=run, question=question))
    _tasks[run.run_id] = task
    task.add_done_callback(lambda _t: _tasks.pop(run.run_id, None))


async def _execute_turn(
    *, session: AgentSession, run: AgentRun, question: str
) -> None:
    """Run one turn to completion, emitting events as it goes."""
    started = time.perf_counter()
    usage = BudgetUsage()
    budget = RunBudget.from_dict(run.budget)
    collected_evidence: list[str] = []
    answer_parts: list[str] = []
    seen_calls: set[str] = set()

    try:
        await repo.mark_run_running(run.run_id)
        await _emit(
            session_id=session.session_id,
            run_id=run.run_id,
            type=EventType.RUN_STARTED,
            payload={"question": question, "model": run.llm_model},
        )

        papers = await repo.list_session_papers(session.session_id)
        context = AgentContext(
            owner_id=session.owner_id,
            session_id=session.session_id,
            run_id=run.run_id,
            thread_id=session.thread_id,
            language=session.language,
            budget=budget,
            usage=usage,
        )
        agent = create_paper_agent(
            tools=READING_TOOLS,
            system_prompt=build_system_prompt(language=session.language, papers=papers),
            model=build_chat_model(run.llm_model),
            context_schema=AgentContext,
            checkpointer=_checkpointer(),
        )

        cancelled = False
        async for mode, chunk in agent.astream(
            {"messages": [{"role": "user", "content": question}]},
            config=context.config(),
            context=context,
            stream_mode=["messages"],
        ):
            if mode != "messages":
                continue
            if await repo.is_cancel_requested(run.run_id):
                cancelled = True
                break

            message, _meta = chunk
            if isinstance(message, ToolMessage):
                usage.tool_calls += 1
                payload = _tool_completion_payload(message)
                collected_evidence.extend(payload.pop("_evidence_ids", []))
                await _emit(
                    session_id=session.session_id,
                    run_id=run.run_id,
                    type=EventType.TOOL_FAILED
                    if payload.get("status") in {"error"}
                    else EventType.TOOL_COMPLETED,
                    payload=payload,
                )
                continue

            if isinstance(message, (AIMessage, AIMessageChunk)):
                for name, call_id in _announced_calls(message, seen_calls):
                    await _emit(
                        session_id=session.session_id,
                        run_id=run.run_id,
                        type=EventType.TOOL_STARTED,
                        payload={"tool": name, "call_id": call_id},
                    )
                text = _text_of(message.content)
                if text:
                    answer_parts.append(text)
                    await _emit(
                        session_id=session.session_id,
                        run_id=run.run_id,
                        type=EventType.MESSAGE_DELTA,
                        payload={"text": text},
                    )

            exceeded = usage.exceeded(budget)
            if exceeded:
                logger.warning("Run %s hit budget ceiling %s", run.run_id, exceeded)
                answer_parts.append(
                    f"\n\n_[stopped: {exceeded} reached; the findings above are what was gathered]_"
                )
                break

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
            await _emit(
                session_id=session.session_id,
                run_id=run.run_id,
                type=EventType.RUN_CANCELLED,
                payload={"usage": usage.as_dict()},
            )
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
        await _emit(
            session_id=session.session_id,
            run_id=run.run_id,
            type=EventType.RUN_COMPLETED,
            payload={"citations": citations, "usage": usage.as_dict()},
        )

    except asyncio.CancelledError:
        await repo.finish_run(
            run.run_id, status=RunStatus.CANCELLED, error_code=ErrorCode.CANCELLED.value
        )
        with contextlib.suppress(Exception):
            await _emit(
                session_id=session.session_id,
                run_id=run.run_id,
                type=EventType.RUN_CANCELLED,
                payload={},
            )
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
            await _emit(
                session_id=session.session_id,
                run_id=run.run_id,
                type=EventType.RUN_FAILED,
                payload={"error": str(exc)[:500], "usage": usage.as_dict()},
            )


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
    payload["_evidence_ids"] = evidence_ids
    return payload


def _summarise(data: dict[str, Any]) -> dict[str, Any]:
    """A few scalars a UI can show without carrying the whole result."""
    summary: dict[str, Any] = {}
    for key in ("paper_id", "section", "question", "title", "page"):
        if key in data:
            summary[key] = data[key]
    for key in ("sections", "hits", "blocks"):
        if isinstance(data.get(key), list):
            summary[f"{key}_count"] = len(data[key])
    return summary


# ── Checkpointer lifetime ───────────────────────────────────────────────────
# Held open for the process rather than per run: opening a SQLite connection
# per turn would serialise behind the previous one's WAL checkpoint, and the
# saver is safe to share.
_checkpointer_cm = None
_checkpointer_instance = None


def _checkpointer():
    return _checkpointer_instance


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
