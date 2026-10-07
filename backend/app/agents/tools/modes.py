"""The three reading modes, callable from inside a conversation.

Insight Snap, Logic Lens and Research Sphere were built as branches of the
upload pipeline. Here each becomes a tool, so the same analysis can be asked
for mid-conversation — by the reader picking a mode in the composer, or by the
agent deciding a question deserves the full treatment.

Three things are deliberately kept from the pipeline rather than reinvented:

*The report is a `runs` row.* A mode run made here writes the same `runs` and
`run_outputs` rows the upload page writes, stamped with the session and the
turn. The report page, the compare matrix and both exports keep working on it
without knowing where it came from.

*Progress is the pipeline's progress.* The subgraphs already report steps
through `workflows.progress`; those are forwarded as `tool.progress` events,
so the conversation shows the same step names the report page shows.

*Repeating the question costs nothing.* A report is keyed on paper, mode,
language and model through `agent_jobs`, so asking twice returns the first
report — and says so.

What the model receives is small on purpose: the report is already on the
reader's screen as a card, so the tool returns an excerpt for the model to
summarise from, not the whole document to repeat.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from typing import Any

from langchain.tools import ToolRuntime, tool

from app.agents.context import AgentContext
from app.agents.tools.acquisition import _budget_stop, _parsed_already
from app.agents.tools.reading import _authorize
from app.config import get_settings
from app.db import agent_repository as repo
from app.db import database as db
from app.models.agent_models import Availability, ErrorCode, EventType, ToolResult
from app.services import agent_jobs, mode_runs, paper_catalog, parse_service
from app.services.paper_ir_loader import PaperNotParsed, ensure_pub_rank, load_paper_ir
from app.workflows import progress

logger = logging.getLogger("scholar.agents.tools.modes")

# How much of the finished report the model is shown. Enough to summarise
# faithfully; far less than the report, which the reader already has.
REPORT_EXCERPT_CHARS = 6_000

MODE_LABELS = {
    "snap": {"zh": "快速洞察 (Insight Snap)", "en": "Insight Snap"},
    "lens": {"zh": "逻辑透镜 (Logic Lens)", "en": "Logic Lens"},
    "sphere": {"zh": "研究全景 (Research Sphere)", "en": "Research Sphere"},
}


def mode_job_key(*, paper_id: str, mode: str, language: str, model: str) -> str:
    return f"mode:{paper_id}:{mode}:{language}:{model or 'default'}"


async def _emit(ctx: AgentContext, type: EventType, payload: dict[str, Any]) -> None:
    """Publish an event for this turn. Lazy import: the runner imports the tools."""
    from app.services import agent_runner

    with contextlib.suppress(Exception):
        await agent_runner.emit(
            session_id=ctx.session_id, run_id=ctx.run_id, type=type, payload=payload
        )


async def _ensure_parsed(ctx: AgentContext, paper_id: str, *, tool_name: str) -> ToolResult | None:
    """Parse the paper if it is not parsed yet. Returns a denial, or None on success.

    Counts against the run's parse budget like `ensure_paper_parsed` does — a
    mode is not a way around the ceiling.
    """
    if await _parsed_already(paper_id):
        return None

    pdf_path = get_settings().data_dir / "papers" / paper_id / "original.pdf"
    if not pdf_path.exists():
        return ToolResult.unavailable(
            ErrorCode.FULLTEXT_UNAVAILABLE,
            f"No PDF is stored for {paper_id}; download it first.",
        )
    stop = _budget_stop(ctx, "parse")
    if stop is not None:
        return stop

    try:
        handle = await parse_service.run_agent_parse(
            ctx, paper_id, tool_name=tool_name,
        )
    except agent_jobs.JobFailed as exc:
        if exc.code == ErrorCode.TIMEOUT:
            return ToolResult.partial(
                {"paper_id": paper_id},
                note="MinerU has not finished within the waiting limit. The submitted batch "
                "is retained for a later turn; do not retry parsing or generate a full-text "
                "report in this turn. Explain the wait and use available information.",
            )
        return ToolResult.unavailable(
            ErrorCode.PARSE_FAILED, f"Parsing failed: {exc.message}"[:400]
        )
    if handle.status == "running":
        return ToolResult.partial(
            {"paper_id": paper_id, "job_id": handle.job_id},
            note="This paper is still being parsed by another turn. Answer from what "
            "you have and say the report will be possible once parsing finishes.",
        )
    if not handle.reused:
        ctx.usage.parses += 1

    literature_id = await paper_catalog.literature_for_paper(paper_id)
    if literature_id:
        await repo.set_session_paper_availability(
            session_id=ctx.session_id,
            literature_id=literature_id,
            availability=Availability.PARSED,
        )
    return None


async def _forward_progress(
    ctx: AgentContext, tool_name: str, queue: asyncio.Queue[Any]
) -> None:
    """Turn the run's step progress into `tool.progress` events."""
    while True:
        item = await queue.get()
        if item is None:
            return
        await _emit(ctx, EventType.TOOL_PROGRESS, {"tool": tool_name, **item})


async def _run_subgraph(mode: str, state: dict[str, Any]) -> dict[str, Any]:
    """Run one mode subgraph, then the pipeline's translation step."""
    if mode == "snap":
        from app.workflows.snap_subgraph import run_insight_snap as run
    elif mode == "lens":
        from app.workflows.lens_subgraph import run_logic_lens as run
    elif mode == "sphere":
        from app.workflows.sphere_subgraph import run_research_sphere as run
    else:  # pragma: no cover — guarded by the tool signatures
        raise ValueError(f"unknown mode {mode}")

    result = await run(state)
    state.update(result)
    if state.get("error"):
        return state
    from app.workflows.translate import translate_output

    translated = await translate_output(state)
    state.update(translated)
    return state


async def _produce_report(
    *, ctx: AgentContext, mode: str, paper_id: str, focus: str, tool_name: str
) -> dict[str, Any]:
    """The body of a mode job: make the legacy run row, run the analysis, persist it."""
    run_id = uuid.uuid4().hex[:16]
    t0 = time.perf_counter()

    await db.execute(
        """INSERT INTO runs
               (run_id, paper_id, mode, llm_model, language, status, user_question,
                owner_token, owner_id, agent_session_id, agent_run_id, started_at)
           VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?, ?, ?, now())""",
        (
            run_id,
            paper_id,
            mode,
            ctx.llm_model,
            ctx.language,
            focus[:2000],
            ctx.owner_token,
            ctx.owner_id,
            ctx.session_id,
            ctx.run_id,
        ),
    )

    # The subgraphs report each step as they go. Listening here is what lets
    # the conversation show the same steps the report page shows.
    queue: asyncio.Queue[Any] = progress.subscribe(run_id)
    forwarder = asyncio.create_task(_forward_progress(ctx, tool_name, queue))

    try:
        async with mode_runs.heartbeating(run_id):
            return await _report_body(
                ctx=ctx, mode=mode, paper_id=paper_id, focus=focus, run_id=run_id, t0=t0
            )
    except asyncio.CancelledError:
        # The reader stopped the turn (or the run itself). Conditional, so a
        # run already closed as cancelled keeps the reason it was given.
        await db.execute(
            "UPDATE runs SET status = 'cancelled', error_msg = 'Cancelled', "
            "finished_at = now() WHERE run_id = ? AND status IN ('pending', 'running')",
            (run_id,),
        )
        raise
    except Exception as exc:
        await db.execute(
            "UPDATE runs SET status = 'failed', error_msg = ?, finished_at = now() "
            "WHERE run_id = ? AND status IN ('pending', 'running')",
            (str(exc)[:500], run_id),
        )
        raise
    finally:
        progress.unsubscribe(run_id, queue)
        queue.put_nowait(None)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(forwarder, timeout=5)


async def _report_body(
    *, ctx: AgentContext, mode: str, paper_id: str, focus: str, run_id: str, t0: float
) -> dict[str, Any]:
    settings = get_settings()
    paper_ir = await load_paper_ir(paper_id)
    pdf_path = settings.data_dir / "papers" / paper_id / "original.pdf"
    state: dict[str, Any] = {
        "paper_id": paper_id,
        "run_id": run_id,
        "mode": mode,
        "llm_model": ctx.llm_model,
        "language": ctx.language,
        "user_question": focus,
        "pdf_path": str(pdf_path),
        "paper_ir_json": paper_ir.model_dump_json(),
        "progress": [],
    }
    if mode in ("snap", "lens"):
        pub_rank = await ensure_pub_rank(paper_id, paper_ir, pdf_path)
        state["pub_rank_json"] = json.dumps(pub_rank)

    state = await _run_subgraph(mode, state)
    if state.get("error"):
        raise RuntimeError(str(state["error"]))

    markdown = state.get("final_markdown", "") or ""
    json_data = state.get("final_json", "{}") or "{}"
    await db.execute(
        "INSERT INTO run_outputs (run_id, markdown, json_data) VALUES (?, ?, ?) "
        "ON CONFLICT (run_id) DO UPDATE SET markdown = excluded.markdown, "
        "json_data = excluded.json_data",
        (run_id, markdown, json_data),
    )
    closed = await db.execute(
        "UPDATE runs SET status = 'done', finished_at = now() "
        "WHERE run_id = ? AND status IN ('pending', 'running')",
        (run_id,),
    )
    if not closed:
        raise RuntimeError("The report was cancelled before it finished.")
    logger.info(
        "[%s] mode %s report %s done in %.1fs (%d chars)",
        paper_id, mode, run_id, time.perf_counter() - t0, len(markdown),
    )
    return {
        "run_id": run_id,
        "title": paper_ir.title,
        "markdown_chars": len(markdown),
    }


async def _stored_report(run_id: str) -> dict[str, Any] | None:
    row = await db.fetch_one(
        """SELECT r.run_id, r.paper_id, r.mode, r.language, r.status, o.markdown
             FROM runs r LEFT JOIN run_outputs o ON o.run_id = r.run_id
            WHERE r.run_id = ?""",
        (run_id,),
    )
    if row is None or row["status"] != "done" or not (row["markdown"] or "").strip():
        return None
    return row


async def _run_mode(
    ctx: AgentContext, *, mode: str, paper_id: str, focus: str, tool_name: str
) -> str:
    paper_id, denial = await _authorize(ctx, paper_id)
    if denial is not None:
        return denial.to_json()

    denial = await _ensure_parsed(ctx, paper_id, tool_name=tool_name)
    if denial is not None:
        return denial.to_json()

    focus = (focus or "").strip()
    key = mode_job_key(paper_id=paper_id, mode=mode, language=ctx.language, model=ctx.llm_model)

    # A stored report whose output has since gone missing must not be "reused".
    existing = await agent_jobs.get_job(key)
    if existing is not None and existing["status"] == "done":
        try:
            prior = json.loads(existing["result_json"] or "{}")
        except (TypeError, ValueError):
            prior = {}
        if not await _stored_report(str(prior.get("run_id", ""))):
            await agent_jobs.forget(key)

    async def _work(_job_id: str) -> dict[str, Any]:
        return await _produce_report(
            ctx=ctx, mode=mode, paper_id=paper_id, focus=focus, tool_name=tool_name
        )

    try:
        handle = await agent_jobs.run_once(
            kind="mode",
            idempotency_key=key,
            work=_work,
            session_id=ctx.session_id,
            run_id=ctx.run_id,
            request={"paper_id": paper_id, "mode": mode, "focus": focus},
        )
    except agent_jobs.JobFailed as exc:
        return ToolResult.failed(
            ErrorCode.UPSTREAM_ERROR,
            f"The {mode} analysis failed: {exc.message}"[:400],
            retryable=False,
        ).to_json()

    if handle.status == "running":
        return ToolResult.partial(
            {"paper_id": paper_id, "mode": mode, "job_id": handle.job_id},
            note="This report is already being generated by another turn. Do not start "
            "it again; tell the reader it is on its way.",
        ).to_json()

    run_id = str(handle.result.get("run_id", ""))
    report = await _stored_report(run_id)
    if report is None:
        return ToolResult.failed(
            ErrorCode.UPSTREAM_ERROR,
            "The report finished but its output could not be read back.",
            retryable=True,
        ).to_json()

    # Stamp the reused report onto this session too, so it shows in the
    # conversation's artifact list even though another session made it.
    if handle.reused:
        await db.execute(
            """UPDATE runs SET agent_session_id = CASE WHEN agent_session_id = '' THEN ? ELSE agent_session_id END
                WHERE run_id = ?""",
            (ctx.session_id, run_id),
        )

    title = str(handle.result.get("title") or "")
    if not title:
        row = await db.fetch_one("SELECT title FROM papers WHERE paper_id = ?", (paper_id,))
        title = (row or {}).get("title") or ""

    await _emit(
        ctx,
        EventType.ARTIFACT_CREATED,
        {
            "run_id": run_id,
            "paper_id": paper_id,
            "mode": mode,
            "title": title,
            "language": ctx.language,
            "reused": handle.reused,
        },
    )

    markdown = report["markdown"] or ""
    excerpt = markdown[:REPORT_EXCERPT_CHARS]
    truncated = len(markdown) > REPORT_EXCERPT_CHARS
    label = MODE_LABELS[mode]["zh" if ctx.language == "zh" else "en"]
    data = {
        "run_id": run_id,
        "paper_id": paper_id,
        "mode": mode,
        "mode_label": label,
        "title": title,
        "reused": handle.reused,
        "report_url": f"/paper/{paper_id}/run/{run_id}",
        "report_excerpt": excerpt,
        "excerpt_truncated": truncated,
    }
    note = (
        "The full report is already displayed to the reader as a card in this "
        "conversation. Summarise its main findings in a few sentences and point out "
        "anything that deserves attention; do not reproduce the report. Cite nothing "
        "from it with [ev_...] — it is a report, not paper text."
    )
    if handle.reused:
        note = "An identical report already existed and was shown again. " + note
    return ToolResult.ok(data, note=note).to_json()


@tool(parse_docstring=False)
async def run_insight_snap(
    runtime: ToolRuntime[AgentContext],
    paper_id: str = "",
    focus: str = "",
) -> str:
    """Produce the Insight Snap triage report for a paper and show it to the reader.

    Insight Snap is a fast, evidence-backed verdict: what the paper claims, how
    well the results support it, external signals (citations, venue, code,
    retraction) and whether it is worth a full read. Use it when the reader
    asks for a quick assessment, a triage, "is this worth reading", or picks the
    Snap mode. Takes a minute or two. Pass `focus` when the reader has a
    specific concern the review should weigh. Omit paper_id when the session
    holds a single paper.
    """
    return await _run_mode(
        runtime.context, mode="snap", paper_id=paper_id, focus=focus, tool_name="run_insight_snap"
    )


@tool(parse_docstring=False)
async def run_logic_lens(
    runtime: ToolRuntime[AgentContext],
    paper_id: str = "",
    focus: str = "",
) -> str:
    """Produce the Logic Lens deep-dive report for a paper and show it to the reader.

    Logic Lens walks the method in depth: the core idea, the pipeline, every
    equation typeset and explained, algorithms step by step, datasets and result
    tables, and reproducibility. Use it when the reader wants to understand how
    the method works in detail or picks the Lens mode. Takes several minutes.
    Omit paper_id when the session holds a single paper.
    """
    return await _run_mode(
        runtime.context, mode="lens", paper_id=paper_id, focus=focus, tool_name="run_logic_lens"
    )


@tool(parse_docstring=False)
async def run_research_sphere(
    runtime: ToolRuntime[AgentContext],
    paper_id: str = "",
) -> str:
    """Map the research landscape around a paper and show the report to the reader.

    Research Sphere expands the paper's citation graph, scores and gates the
    candidates, clusters the core set, compares the paper against its
    neighbours and names research gaps. Use it when the reader asks how this
    paper relates to the field, for related work, or picks the Sphere mode.
    This is the slowest mode — often ten minutes or more — so call it once and
    wait. Omit paper_id when the session holds a single paper.
    """
    return await _run_mode(
        runtime.context, mode="sphere", paper_id=paper_id, focus="", tool_name="run_research_sphere"
    )


MODE_TOOLS = [run_insight_snap, run_logic_lens, run_research_sphere]
