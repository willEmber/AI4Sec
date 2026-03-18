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
from app.models.schemas import RunCreate, RunOutputResponse, RunResponse
from app.workflows.main_graph import build_main_graph
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


@router.post("/runs", response_model=RunResponse)
@limiter.limit("3/minute")
async def create_run(request: Request, req: RunCreate):
    # Verify paper exists
    paper = await db.fetch_one("SELECT paper_id FROM papers WHERE paper_id = ?", (req.paper_id,))
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    run_id = uuid.uuid4().hex[:16]
    language = req.language if req.language in ("en", "zh") else "en"
    await db.execute(
        "INSERT INTO runs (run_id, paper_id, mode, llm_model, language, status) VALUES (?, ?, ?, ?, ?, 'pending')",
        (run_id, req.paper_id, req.mode, req.llm_model, language),
    )

    # Create queue for SSE
    _run_queues[run_id] = asyncio.Queue()

    logger.info(f"[run:{run_id}] Created run paper={req.paper_id} mode={req.mode} model={req.llm_model or '(default)'} lang={language}")

    # Launch graph in background
    asyncio.create_task(_execute_run(run_id, req.paper_id, req.mode, req.llm_model, language))

    row = await db.fetch_one("SELECT * FROM runs WHERE run_id = ?", (run_id,))
    return RunResponse(**row)


async def _execute_run(run_id: str, paper_id: str, mode: str, llm_model: str, language: str = "en") -> None:
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
                if queue:
                    await queue.put({"event": "progress", "data": latest})

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
