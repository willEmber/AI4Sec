from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langgraph.graph.state import CompiledStateGraph

from app.db import database as db
from app.rate_limit import limiter
from app.models.schemas import RecentRunResponse, RunCreate, RunOutputResponse, RunResponse
from app.workflows.main_graph import build_main_graph
from app.workflows.progress import emit_progress
from app.workflows.state import MainGraphState

logger = logging.getLogger("scholar.runs")

router = APIRouter(tags=["runs"])

# In-memory queues for SSE progress per run
_run_queues: dict[str, asyncio.Queue] = {}

# Compiled graph (lazy init)
_compiled_graph: CompiledStateGraph | None = None

# Limit concurrent background run executions
_run_semaphore = asyncio.Semaphore(5)


def _get_graph() -> CompiledStateGraph:
    global _compiled_graph
    if _compiled_graph is None:
        graph = build_main_graph()
        _compiled_graph = graph.compile()
    return _compiled_graph


_VALID_INPUT_MODES = {"snap", "lens", "sphere", "auto"}
_MAX_QUESTION_LEN = 2000
_MAX_OWNER_TOKEN_LEN = 100

# A run still pending/running past this many seconds has lost its owning
# background task (the process restarted, or it hung well past the 30-min SSE
# window), so it can never finish on its own. It is reconciled to 'failed' on
# read so the recent-runs banner stops showing zombies. Comfortably beyond any
# legitimate run, which the SSE stream caps at 1800s.
_STALE_RUN_SECONDS = 3600

# Never surface runs older than this in the recent list — older history is not
# useful for recovery and only clutters the UI.
_RECENT_RUN_DAYS = 7


async def _reconcile_stale_runs() -> None:
    """Mark abandoned pending/running runs as failed (self-healing on read)."""
    await db.execute(
        "UPDATE runs SET status = 'failed', "
        "error_msg = 'Interrupted (task no longer running)', "
        "finished_at = datetime('now') "
        "WHERE status IN ('pending', 'running') "
        "AND started_at < datetime('now', ?)",
        (f"-{_STALE_RUN_SECONDS} seconds",),
    )


@router.post("/runs", response_model=RunResponse)
@limiter.limit("3/minute")
async def create_run(request: Request, req: RunCreate):
    # Verify paper exists
    paper = await db.fetch_one("SELECT paper_id FROM papers WHERE paper_id = ?", (req.paper_id,))
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    mode = req.mode if req.mode in _VALID_INPUT_MODES else "snap"
    language = req.language if req.language in ("en", "zh") else "en"
    question = (req.question or "").strip()[:_MAX_QUESTION_LEN]
    owner_token = (req.owner_token or "").strip()[:_MAX_OWNER_TOKEN_LEN]

    if mode == "auto" and not question:
        raise HTTPException(status_code=400, detail="Smart Q&A mode requires a non-empty question")

    run_id = uuid.uuid4().hex[:16]
    await db.execute(
        "INSERT INTO runs (run_id, paper_id, mode, llm_model, language, status, user_question, owner_token) "
        "VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)",
        (run_id, req.paper_id, mode, req.llm_model, language, question, owner_token),
    )

    # Create queue for SSE
    _run_queues[run_id] = asyncio.Queue()

    logger.info(
        f"[run:{run_id}] Created run paper={req.paper_id} mode={mode} model={req.llm_model or '(default)'} lang={language} q={'(yes)' if question else '(no)'}"
    )

    # Launch graph in background
    asyncio.create_task(_execute_run(run_id, req.paper_id, mode, req.llm_model, language, question))

    row = await db.fetch_one("SELECT * FROM runs WHERE run_id = ?", (run_id,))
    return RunResponse(**row)


async def _execute_run(
    run_id: str,
    paper_id: str,
    mode: str,
    llm_model: str,
    language: str = "en",
    user_question: str = "",
) -> None:
    """Execute the LangGraph workflow as a background task, bounded by semaphore."""
    queue = _run_queues.get(run_id)

    try:
        await asyncio.wait_for(_run_semaphore.acquire(), timeout=300.0)
    except asyncio.TimeoutError:
        logger.warning(f"[run:{run_id}] Timed out waiting for execution slot")
        await db.execute(
            "UPDATE runs SET status = 'failed', error_msg = 'Server busy, please retry later', finished_at = datetime('now') WHERE run_id = ?",
            (run_id,),
        )
        if queue:
            await queue.put({"event": "error", "data": {"error": "Server busy, please retry later"}})
            await queue.put(None)
        return

    t0 = time.perf_counter()
    try:
        await db.execute(
            "UPDATE runs SET status = 'running', started_at = datetime('now') WHERE run_id = ?",
            (run_id,),
        )

        if queue:
            await queue.put({"event": "status", "data": {"status": "running"}})

        logger.info(f"[run:{run_id}] ▶ Graph execution started")

        initial_state: MainGraphState = {
            "paper_id": paper_id,
            "run_id": run_id,
            "mode": mode,
            "llm_model": llm_model,
            "language": language,
            "user_question": user_question,
            "progress": [],
        }

        graph = _get_graph()

        # Stream through graph nodes
        final_state: dict[str, Any] = {}
        async for event in graph.astream(initial_state):
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
        if queue:
            error = final_state.get("error")
            if error:
                logger.error(f"[run:{run_id}] ✗ Graph failed at {elapsed:.1f}s — {error}")
                await queue.put({"event": "error", "data": {"error": error}})
            else:
                logger.info(f"[run:{run_id}] ✔ Graph completed at {elapsed:.1f}s")
                await queue.put({
                    "event": "done",
                    "data": {
                        "run_id": run_id,
                        "status": "done",
                    },
                })

    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.exception(f"[run:{run_id}] ✗ Graph exception at {elapsed:.1f}s — {e}")
        await db.execute(
            "UPDATE runs SET status = 'failed', error_msg = ?, finished_at = datetime('now') WHERE run_id = ?",
            (str(e), run_id),
        )
        if queue:
            await queue.put({"event": "error", "data": {"error": str(e)}})

    finally:
        _run_semaphore.release()
        if queue:
            await queue.put(None)  # Signal end of stream


@router.get("/runs/recent", response_model=list[RecentRunResponse])
@limiter.limit("30/minute")
async def list_recent_runs(
    request: Request,
    owner_token: str = "",
    limit: int = 20,
    active_only: bool = False,
):
    """Recent runs for one browser (scoped by `owner_token`), with paper title.

    Runs are scoped to the caller's `owner_token` so one browser never sees
    another's tasks. When `active_only=true`, only pending/running runs are
    returned — used by the upload page banner to surface tasks the user
    navigated away from. Declared before `/runs/{run_id}` so the literal path
    wins.

    Abandoned runs are reconciled to 'failed' first, and only runs from the
    last ``_RECENT_RUN_DAYS`` days are returned, so stale tasks never linger
    as perpetual "running" entries.
    """
    limit = max(1, min(limit, 100))
    owner_token = (owner_token or "").strip()[:_MAX_OWNER_TOKEN_LEN]
    await _reconcile_stale_runs()

    conds = ["r.owner_token = ?", "r.started_at >= datetime('now', ?)"]
    params: list[Any] = [owner_token, f"-{_RECENT_RUN_DAYS} days"]
    if active_only:
        conds.append("r.status IN ('pending', 'running')")
    where = "WHERE " + " AND ".join(conds)
    rows = await db.fetch_all(
        f"""
        SELECT r.run_id, r.paper_id, COALESCE(p.title, '') AS paper_title,
               r.mode, r.status, r.started_at, r.finished_at,
               COALESCE(r.current_step, '') AS current_step,
               COALESCE(r.user_question, '') AS user_question
          FROM runs r
          LEFT JOIN papers p ON p.paper_id = r.paper_id
          {where}
         ORDER BY r.started_at DESC
         LIMIT ?
        """,
        (*params, limit),
    )
    return [RecentRunResponse(**r) for r in rows]


@router.get("/runs/{run_id}", response_model=RunResponse)
@limiter.limit("30/minute")
async def get_run(request: Request, run_id: str):
    row = await db.fetch_one("SELECT * FROM runs WHERE run_id = ?", (run_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunResponse(**row)


@router.get("/runs/{run_id}/output", response_model=RunOutputResponse)
@limiter.limit("20/minute")
async def get_run_output(request: Request, run_id: str):
    row = await db.fetch_one("SELECT * FROM run_outputs WHERE run_id = ?", (run_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Run output not found")
    return RunOutputResponse(**row)


@router.post("/runs/{run_id}/dismiss", response_model=RunResponse)
@limiter.limit("30/minute")
async def dismiss_run(request: Request, run_id: str, owner_token: str = ""):
    """Manually clear a pending/running run the user abandoned.

    Only the owning browser (matching `owner_token`) may dismiss a run; legacy
    runs with no owner are dismissible by anyone. Marks the run failed so it
    leaves the active banner, and tears down any live SSE queue. Already-finished
    runs are returned unchanged.
    """
    owner_token = (owner_token or "").strip()[:_MAX_OWNER_TOKEN_LEN]
    row = await db.fetch_one("SELECT status, owner_token FROM runs WHERE run_id = ?", (run_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")
    # Don't reveal existence of runs the caller doesn't own.
    if row["owner_token"] and row["owner_token"] != owner_token:
        raise HTTPException(status_code=404, detail="Run not found")

    if row["status"] in ("pending", "running"):
        await db.execute(
            "UPDATE runs SET status = 'failed', error_msg = 'Dismissed by user', "
            "finished_at = datetime('now') WHERE run_id = ?",
            (run_id,),
        )
        queue = _run_queues.pop(run_id, None)
        if queue is not None:
            await queue.put({"event": "error", "data": {"error": "Dismissed by user"}})
            await queue.put(None)
        logger.info(f"[run:{run_id}] Dismissed by user")

    updated = await db.fetch_one("SELECT * FROM runs WHERE run_id = ?", (run_id,))
    return RunResponse(**updated)


@router.get("/runs/{run_id}/stream")
@limiter.limit("10/minute")
async def stream_run(request: Request, run_id: str):
    """SSE endpoint for streaming run progress."""
    queue = _run_queues.get(run_id)
    if not queue:
        raise HTTPException(status_code=404, detail="No active stream for this run")

    async def event_generator():
        try:
            while True:
                msg = await asyncio.wait_for(queue.get(), timeout=1800.0)
                if msg is None:
                    yield f"data: {json.dumps({'event': 'end'})}\n\n"
                    break
                yield f"data: {json.dumps(msg)}\n\n"
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'event': 'timeout'})}\n\n"
        finally:
            _run_queues.pop(run_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/papers/{paper_id}/runs", response_model=list[RunResponse])
@limiter.limit("20/minute")
async def list_paper_runs(request: Request, paper_id: str):
    rows = await db.fetch_all(
        "SELECT * FROM runs WHERE paper_id = ? ORDER BY started_at DESC",
        (paper_id,),
    )
    return [RunResponse(**r) for r in rows]
