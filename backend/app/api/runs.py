"""Reading, exporting, stopping and deleting mode reports.

A report is made inside a conversation (`agents/tools/modes.py`); there is no
endpoint that starts one on its own any more (P16). Reading a run is open to
anyone holding its id — that is how a report is shared — while stopping and
deleting are the owner's.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from app.api.deps import optional_caller
from app.config import get_settings
from app.db import database as db
from app.rate_limit import limiter
from app.models.schemas import RecentRunResponse, RunOutputResponse, RunResponse
from app.services import mode_runs, report_citations, report_images, zotero_export
from app.services.accounts import Caller

router = APIRouter(tags=["runs"])

_MAX_OWNER_TOKEN_LEN = 100

# Never surface runs older than this in the recent list — older history is not
# useful for recovery and only clutters the UI.
_RECENT_RUN_DAYS = 7


@router.get("/runs/recent", response_model=list[RecentRunResponse])
@limiter.limit("30/minute")
async def list_recent_runs(
    request: Request,
    owner_token: str = "",
    limit: int = 20,
    active_only: bool = False,
    days: int = _RECENT_RUN_DAYS,
    caller: Caller | None = Depends(optional_caller),
):
    """Recent runs for the caller, with paper title.

    Scoped by the server-verified principal (`owner_id`). Runs made before
    accounts carry only the browser's `owner_token`; the first principal that
    lists them with that token adopts them, which grants nothing the token did
    not already grant. When `active_only=true`, only pending/running runs are
    returned. Pass `days=0` to list runs across all dates (up to `limit`).
    """
    limit = max(1, min(limit, 100))
    days = max(0, min(days, 3650))
    owner_token = (owner_token or "").strip()[:_MAX_OWNER_TOKEN_LEN]

    if caller is not None and owner_token:
        await db.execute(
            "UPDATE runs SET owner_id = ? WHERE owner_id IS NULL AND owner_token = ?",
            (caller.principal_id, owner_token),
        )
    conds = [
        "(r.owner_id = ? OR (r.owner_id IS NULL AND r.owner_token = ?))",
    ]
    params: list[Any] = [
        caller.principal_id if caller is not None else "",
        owner_token,
    ]
    if days > 0:
        conds.append("r.started_at >= now() - make_interval(days => ?)")
        params.append(days)
    if active_only:
        conds.append("r.status IN ('pending', 'running')")
    where = "WHERE " + " AND ".join(conds)
    rows = await db.fetch_all(
        f"""
        SELECT r.run_id, r.paper_id, COALESCE(p.title, '') AS paper_title,
               r.mode, r.status, r.started_at, r.finished_at,
               COALESCE(r.current_step, '') AS current_step,
               COALESCE(r.user_question, '') AS user_question,
               COALESCE(r.agent_session_id, '') AS agent_session_id
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


def _exported_markdown(output: dict[str, Any] | None, run: dict[str, Any]) -> str:
    """The report's text as it leaves the application.

    A report written as an answer (P16) cites `[ev_…]`, which means nothing
    outside the conversation: it goes out numbered, with a references section
    built from the citation table frozen when it was saved. Older reports cite
    pages and go out as stored.
    """
    if not output:
        return ""
    markdown = output.get("markdown") or ""
    try:
        data = json.loads(output.get("json_data") or "{}")
    except (TypeError, ValueError):
        data = {}
    citations = data.get("citations") if isinstance(data, dict) else None
    if not isinstance(citations, dict) or not citations:
        return markdown
    return report_citations.to_numbered(
        markdown, citations, paper_id=run["paper_id"], language=run.get("language") or "en"
    )


@router.get("/runs/{run_id}/zotero-bundle")
@limiter.limit("20/minute")
async def get_run_zotero_bundle(request: Request, run_id: str):
    """Download a Zotero RDF import bundle (.zip) for this run.

    The bundle holds a ``.rdf`` (the paper as a journalArticle, the AI report as
    a child note) plus the original PDF under ``files/``. The user runs
    File → Import on the ``.rdf`` to add the item, note, and stored PDF straight
    into a *local* Zotero — no Zotero cloud account or running app required.
    """
    run = await db.fetch_one("SELECT * FROM runs WHERE run_id = ?", (run_id,))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    paper = await db.fetch_one(
        "SELECT * FROM papers WHERE paper_id = ?", (run["paper_id"],)
    )
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    output = await db.fetch_one(
        "SELECT markdown, json_data FROM run_outputs WHERE run_id = ?", (run_id,)
    )
    markdown_report = _exported_markdown(output, run)

    settings = get_settings()
    pdf_path = (settings.data_dir / paper["file_path"]).resolve()
    # Path traversal guard + existence: drop the PDF if anything is off; the
    # bundle is still useful with just the item + note.
    if not pdf_path.is_relative_to(settings.data_dir.resolve()) or not pdf_path.exists():
        pdf_path = None

    zip_bytes, filename = await zotero_export.build_zotero_bundle(
        paper=paper,
        markdown_report=markdown_report,
        pdf_path=pdf_path,
        run=run,
        data_dir=settings.data_dir,
    )
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/runs/{run_id}/markdown")
@limiter.limit("20/minute")
async def get_run_markdown_export(request: Request, run_id: str):
    """Download the report as a portable ``.md``.

    Internal figure links (``/api/papers/.../images/...``) are rewritten to their
    public object-storage URLs when R2 is configured (figures are uploaded on
    demand), so the file renders outside the app. When R2 is not configured the
    links are left unchanged — the original behaviour.
    """
    run = await db.fetch_one("SELECT * FROM runs WHERE run_id = ?", (run_id,))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    paper = await db.fetch_one(
        "SELECT * FROM papers WHERE paper_id = ?", (run["paper_id"],)
    )
    output = await db.fetch_one(
        "SELECT markdown, json_data FROM run_outputs WHERE run_id = ?", (run_id,)
    )
    if not output or not (output["markdown"] or "").strip():
        raise HTTPException(status_code=404, detail="Run output not found")

    settings = get_settings()
    markdown_text = await report_images.process_report_markdown(
        _exported_markdown(output, run),
        paper_id=run["paper_id"],
        data_dir=settings.data_dir,
        embed_fallback=False,
        drop_unresolved=False,
    )

    title = (paper.get("title") if paper else "") or run["paper_id"]
    base = zotero_export._safe_name(title, fallback=f"paper_{run['paper_id'][:8]}")
    filename = f"{base}_{run['mode']}.md"
    return Response(
        content=markdown_text,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


async def _owned_run(run_id: str, caller: Caller | None, owner_token: str) -> dict[str, Any]:
    """The run, if the caller may change it; 404 otherwise.

    Reading a run is open to anyone holding its id — that is how a report is
    shared. Changing one is the owner's: the principal for runs that have one,
    the browser's `owner_token` for runs from before accounts, and anyone for
    legacy runs with neither. A run that is not the caller's is reported as
    missing, so its existence is not revealed either.
    """
    owner_token = (owner_token or "").strip()[:_MAX_OWNER_TOKEN_LEN]
    row = await db.fetch_one(
        "SELECT status, owner_token, owner_id FROM runs WHERE run_id = ?", (run_id,)
    )
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")
    if row["owner_id"]:
        owned = caller is not None and caller.principal_id == row["owner_id"]
    else:
        owned = not row["owner_token"] or row["owner_token"] == owner_token
    if not owned:
        raise HTTPException(status_code=404, detail="Run not found")
    return row


async def _cancel_owned(run_id: str, caller: Caller | None, owner_token: str) -> RunResponse:
    await _owned_run(run_id, caller, owner_token)
    await mode_runs.cancel(run_id)
    updated = await db.fetch_one("SELECT * FROM runs WHERE run_id = ?", (run_id,))
    return RunResponse(**updated)


@router.post("/runs/{run_id}/cancel", response_model=RunResponse)
@limiter.limit("30/minute")
async def cancel_run(
    request: Request,
    run_id: str,
    owner_token: str = "",
    caller: Caller | None = Depends(optional_caller),
):
    """Stop a pending/running run. Only its owner may.

    The run is closed as `cancelled` and whoever is executing it is
    interrupted — on this replica at once, on another at its next heartbeat.
    A run that already finished is returned unchanged.
    """
    return await _cancel_owned(run_id, caller, owner_token)


@router.post("/runs/{run_id}/dismiss", response_model=RunResponse)
@limiter.limit("30/minute")
async def dismiss_run(
    request: Request,
    run_id: str,
    owner_token: str = "",
    caller: Caller | None = Depends(optional_caller),
):
    """The earlier name of `cancel`, kept for clients that still call it."""
    return await _cancel_owned(run_id, caller, owner_token)


@router.delete("/runs/{run_id}")
@limiter.limit("30/minute")
async def delete_run(
    request: Request,
    run_id: str,
    owner_token: str = "",
    caller: Caller | None = Depends(optional_caller),
):
    """Delete a run and its report. Only its owner may; links to it stop working."""
    await _owned_run(run_id, caller, owner_token)
    await mode_runs.delete(run_id)
    return {"deleted": run_id}


# How often the stream looks at the run's row, and how long it stays open.
_STREAM_POLL_SECONDS = 1.0
_STREAM_MAX_SECONDS = 1800.0


@router.get("/runs/{run_id}/stream")
@limiter.limit("10/minute")
async def stream_run(request: Request, run_id: str):
    """SSE endpoint for streaming run progress.

    The stream follows the run's row rather than anything in this process: it
    replays the steps recorded so far, then reports new ones until the run
    ends. So it can be opened on any replica, at any point in the run, and
    again after a reload.
    """
    if not await db.fetch_one("SELECT 1 AS hit FROM runs WHERE run_id = ?", (run_id,)):
        raise HTTPException(status_code=404, detail="Run not found")

    def frame(event: str, data: dict[str, Any] | None = None) -> str:
        body: dict[str, Any] = {"event": event}
        if data is not None:
            body["data"] = data
        return f"data: {json.dumps(body)}\n\n"

    async def event_generator():
        sent = 0
        announced_running = False
        deadline = asyncio.get_running_loop().time() + _STREAM_MAX_SECONDS
        while True:
            row = await db.fetch_one(
                "SELECT status, error_msg, progress_json FROM runs WHERE run_id = ?", (run_id,)
            )
            if row is None:   # deleted while we watched
                yield frame("error", {"error": "Run not found"})
                yield frame("end")
                return
            try:
                steps = json.loads(row["progress_json"] or "[]")
            except (TypeError, ValueError):
                steps = []
            if len(steps) < sent:   # a second attempt started the list over
                sent = 0
            if row["status"] == "running" and not announced_running:
                announced_running = True
                yield frame("status", {"status": "running"})
            for step in steps[sent:]:
                yield frame("progress", step)
            sent = len(steps)

            status = row["status"]
            if status == "done":
                yield frame("done", {"run_id": run_id, "status": "done"})
            elif status == "cancelled":
                yield frame("cancelled", {"run_id": run_id, "status": "cancelled"})
            elif status not in mode_runs.ACTIVE_STATUSES:
                yield frame("error", {"error": row["error_msg"] or "Run failed"})
            if status not in mode_runs.ACTIVE_STATUSES:
                yield frame("end")
                return
            if asyncio.get_running_loop().time() > deadline:
                yield frame("timeout")
                return
            if await request.is_disconnected():
                return
            await asyncio.sleep(_STREAM_POLL_SECONDS)

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
