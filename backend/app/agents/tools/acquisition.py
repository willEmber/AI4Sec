"""Getting a paper into a state where it can actually be read.

These two tools are the bridge between "found it" and "read it": a candidate
from a search is a title and an abstract until someone fetches the PDF and
parses it. Both steps cost minutes and money, which shapes everything here.

*Budgets are enforced, not suggested.* A run has a ceiling on downloads and on
parses. The ceiling is checked in the tool, against the trusted runtime
context, so the model cannot talk its way past it (development plan §7.3).

*Work is keyed, not repeated.* Both go through `agent_jobs`, so asking twice
costs once — within a turn, across a retried request, and after a restart
(acceptance cases A12, A13).

*Failure is a finding.* "This paper's full text cannot be obtained" is a real
answer, and it comes back as `unavailable` with the routes that were tried, not
as an error to retry. The session's paper keeps an honest `availability`, so
every later answer knows it is working from an abstract (acceptance case A08).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from langchain.tools import ToolRuntime, tool

from app.agents.context import AgentContext
from app.config import get_settings
from app.db import agent_repository as repo
from app.db import database as db
from app.models.agent_models import Availability, ErrorCode, ToolResult
from app.services import agent_jobs, paper_acquisition, paper_catalog, parse_service

logger = logging.getLogger("scholar.agents.tools.acquisition")

# How long a failed acquisition stays cached. Long enough that one session does
# not re-try a paywalled paper on every turn; short enough that a paper which
# has since been posted openly is reachable again. A *successful* download never
# expires — the file is already on disk.
FAILED_DOWNLOAD_TTL_SECONDS = 24 * 3600


def _budget_stop(ctx: AgentContext, what: str) -> ToolResult | None:
    """Refuse a new download/parse once this run has spent its allowance."""
    if what == "download" and ctx.usage.downloads >= ctx.budget.max_downloads:
        return ToolResult.failed(
            ErrorCode.BUDGET_EXCEEDED,
            f"This run has already downloaded {ctx.usage.downloads} paper(s), which is "
            "its limit. Work with the papers you have, and say what you could not fetch.",
            retryable=False,
        )
    if what == "parse" and ctx.usage.parses >= ctx.budget.max_parses:
        return ToolResult.failed(
            ErrorCode.BUDGET_EXCEEDED,
            f"This run has already parsed {ctx.usage.parses} paper(s), which is its limit.",
            retryable=False,
        )
    return None


async def _parsed_already(paper_id: str) -> bool:
    row = await db.fetch_one(
        "SELECT 1 FROM paper_nodes WHERE paper_id = ? LIMIT 1", (paper_id,)
    )
    return row is not None


@tool(parse_docstring=False)
async def download_paper(
    runtime: ToolRuntime[AgentContext],
    literature_id: str = "",
    doi: str = "",
    arxiv_id: str = "",
) -> str:
    """Fetch a paper's PDF and attach it to this session.

    Takes the literature_id from search_papers, resolve_paper or
    get_related_papers — or a bare DOI / arXiv id. Works with no DOI are handled
    via arXiv. On success the paper still has to be parsed before it can be read;
    call ensure_paper_parsed next. When no full text can be obtained this returns
    unavailable, and you should continue from the abstract and say so.
    """
    ctx = runtime.context

    if not literature_id:
        if not (doi or arxiv_id):
            return ToolResult.failed(
                ErrorCode.INVALID_ARGUMENT,
                "Pass literature_id, or a doi / arxiv_id.",
                retryable=False,
            ).to_json()
        literature_id = await paper_catalog.upsert_literature_item(
            doi=doi, arxiv_id=arxiv_id, source="user"
        )

    item = await paper_catalog.get_literature_item(literature_id)
    if item is None:
        return ToolResult.unavailable(
            ErrorCode.PAPER_NOT_FOUND, f"No such work: {literature_id}"
        ).to_json()

    # A file already held costs nothing and does not touch the budget.
    existing = await paper_catalog.primary_paper_for_literature(literature_id)
    if existing and (get_settings().data_dir / "papers" / existing / "original.pdf").exists():
        await _attach(ctx, literature_id, existing, item)
        return ToolResult.ok(
            {
                "literature_id": literature_id,
                "paper_id": existing,
                "title": item.get("title") or "",
                "source": "local",
                "availability": (
                    Availability.PARSED.value
                    if await _parsed_already(existing)
                    else Availability.PDF_READY.value
                ),
            },
            note="Already held locally; nothing was downloaded.",
        ).to_json()

    stop = _budget_stop(ctx, "download")
    if stop is not None:
        return stop.to_json()

    await repo.attach_session_paper(
        session_id=ctx.session_id,
        literature_id=literature_id,
        availability=Availability.DOWNLOADING,
        added_by="agent",
    )

    idempotency_key = f"download:{literature_id}"
    await _expire_stale_failure(idempotency_key)

    async def _work(_job_id: str) -> dict[str, Any]:
        result = await paper_acquisition.acquire_pdf(literature_id)
        return {
            "ok": result.ok,
            "paper_id": result.paper_id,
            "source": result.source,
            "detail": result.detail,
            "attempts": result.attempts,
        }

    try:
        handle = await agent_jobs.run_once(
            kind="download",
            # Keyed on the work, so two turns asking for the same paper share
            # one download and the second pays nothing.
            idempotency_key=idempotency_key,
            work=_work,
            session_id=ctx.session_id,
            run_id=ctx.run_id,
            request={"literature_id": literature_id},
        )
    except agent_jobs.JobFailed as exc:
        await repo.set_session_paper_availability(
            session_id=ctx.session_id,
            literature_id=literature_id,
            availability=Availability.UNAVAILABLE,
            note=exc.message[:200],
        )
        return ToolResult.unavailable(
            ErrorCode.DOWNLOAD_FAILED,
            f"Could not download this paper: {exc.message}"[:400],
            data={"literature_id": literature_id, "attempts": exc.attempts},
        ).to_json()

    if handle.status == "running":
        return ToolResult.partial(
            {"literature_id": literature_id, "job_id": handle.job_id},
            note="A download for this paper is already in progress. Do not start "
            "another; continue with what you have and report it as pending.",
        ).to_json()

    result = handle.result
    if not result.get("ok"):
        await repo.set_session_paper_availability(
            session_id=ctx.session_id,
            literature_id=literature_id,
            availability=Availability.UNAVAILABLE,
            note=str(result.get("detail", ""))[:200],
        )
        return ToolResult.unavailable(
            ErrorCode.FULLTEXT_UNAVAILABLE,
            f"No full text could be obtained. Routes tried — {result.get('detail', 'none')}. "
            "Use the abstract instead, and say that is what you are working from.",
            data={
                "literature_id": literature_id,
                "title": item.get("title") or "",
                "attempts": result.get("attempts", []),
            },
        ).to_json()

    if not handle.reused:
        ctx.usage.downloads += 1

    paper_id = result["paper_id"]
    await _attach(ctx, literature_id, paper_id, item)
    return ToolResult.ok(
        {
            "literature_id": literature_id,
            "paper_id": paper_id,
            "title": item.get("title") or "",
            "source": result.get("source", ""),
            "availability": Availability.PDF_READY.value,
        },
        note="Downloaded. Call ensure_paper_parsed before reading it.",
    ).to_json()


async def _backfill_title(paper_id: str, literature_id: str) -> None:
    """Give the work the title the parse just recovered.

    A paper fetched from a bare arXiv id has no title on its work record — the
    agent had nothing but an identifier when it asked. Parsing extracts one into
    `papers.title`, but nothing carries it across, so the paper would read as
    "(untitled)" in the sidebar and in the prompt for the rest of the session.
    `upsert_literature_item` fills blanks only, so this never overwrites a title
    a metadata provider supplied.
    """
    item = await paper_catalog.get_literature_item(literature_id)
    if item is None or (item.get("title") or "").strip():
        return
    paper = await db.fetch_one("SELECT title FROM papers WHERE paper_id = ?", (paper_id,))
    title = ((paper or {}).get("title") or "").strip()
    if not title:
        return
    await paper_catalog.upsert_literature_item(
        title=title,
        doi=item.get("doi") or "",
        arxiv_id=item.get("arxiv_id") or "",
        source=item.get("source") or "parse",
    )


async def _expire_stale_failure(idempotency_key: str) -> None:
    """Let an old "no full text" answer be re-tested, but not a recent one."""
    row = await agent_jobs.get_job(idempotency_key)
    if row is None or row["status"] != "done":
        return
    try:
        result = json.loads(row["result_json"] or "{}")
    except (TypeError, ValueError):
        return
    if result.get("ok"):
        return
    try:
        recorded = datetime.fromisoformat(row["updated_at"].replace(" ", "T"))
    except (AttributeError, ValueError):
        return
    # Timestamps come back from the database as naive UTC; compare like with like.
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    age = (now - recorded).total_seconds()
    if age > FAILED_DOWNLOAD_TTL_SECONDS:
        await agent_jobs.forget(idempotency_key)


async def _attach(
    ctx: AgentContext, literature_id: str, paper_id: str, item: dict[str, Any]
) -> None:
    availability = (
        Availability.PARSED if await _parsed_already(paper_id) else Availability.PDF_READY
    )
    await repo.attach_session_paper(
        session_id=ctx.session_id,
        literature_id=literature_id,
        paper_id=paper_id,
        availability=availability,
        added_by="agent",
        note=item.get("title") or "",
    )


@tool(parse_docstring=False)
async def ensure_paper_parsed(runtime: ToolRuntime[AgentContext], paper_id: str) -> str:
    """Parse a downloaded PDF so its text, tables and equations can be read.

    Returns immediately when the paper is already parsed. Otherwise it runs the
    parse, which takes minutes — call it once and wait; calling it again while
    it runs starts nothing new. Only after this succeeds do get_paper_outline,
    search_paper_content and read_paper_section work on the paper.
    """
    ctx = runtime.context
    paper_id = (paper_id or "").strip()
    if not paper_id:
        return ToolResult.failed(
            ErrorCode.INVALID_ARGUMENT, "paper_id must not be empty.", retryable=False
        ).to_json()

    if not await repo.session_owns_paper(ctx.session_id, paper_id):
        return ToolResult.unavailable(
            ErrorCode.PAPER_NOT_FOUND, f"No paper {paper_id} in this session."
        ).to_json()

    # Before the branch below: a paper that is *already* parsed can be just as
    # nameless as one being parsed now, and the early return would skip this.
    existing_literature = await paper_catalog.literature_for_paper(paper_id)
    if existing_literature:
        await _backfill_title(paper_id, existing_literature)

    if await _parsed_already(paper_id):
        version = await paper_catalog.get_current_version(paper_id)
        return ToolResult.ok(
            {
                "paper_id": paper_id,
                "version_id": (version or {}).get("version_id", ""),
                "availability": Availability.PARSED.value,
            },
            note="Already parsed; nothing was submitted.",
        ).to_json()

    pdf_path = get_settings().data_dir / "papers" / paper_id / "original.pdf"
    if not pdf_path.exists():
        return ToolResult.unavailable(
            ErrorCode.FULLTEXT_UNAVAILABLE,
            f"No PDF is stored for {paper_id}; download it first.",
        ).to_json()

    stop = _budget_stop(ctx, "parse")
    if stop is not None:
        return stop.to_json()

    async def _work(job_id: str) -> dict[str, Any]:
        # The body lives in parse_service because the recovery sweep runs the
        # same one: "resume this parse" has to mean one thing, or a restart
        # would resubmit work a tool call would have rejoined.
        return await parse_service.run_parse_job(job_id=job_id, paper_id=paper_id)

    try:
        handle = await agent_jobs.run_once(
            kind="parse",
            idempotency_key=parse_service.parse_idempotency_key(paper_id),
            work=_work,
            session_id=ctx.session_id,
            run_id=ctx.run_id,
            request={"paper_id": paper_id},
        )
    except agent_jobs.JobFailed as exc:
        return ToolResult.unavailable(
            ErrorCode.PARSE_FAILED,
            f"Parsing failed: {exc.message}"[:400],
            data={"paper_id": paper_id, "attempts": exc.attempts},
        ).to_json()

    if handle.status == "running":
        return ToolResult.partial(
            {"paper_id": paper_id, "job_id": handle.job_id, "remote_id": handle.remote_id},
            note="A parse for this paper is already running. Do not start another; "
            "answer from what you have and say the full text is still being prepared.",
        ).to_json()

    if not handle.reused:
        ctx.usage.parses += 1

    literature_id = await paper_catalog.literature_for_paper(paper_id)
    if literature_id:
        await _backfill_title(paper_id, literature_id)
        await repo.set_session_paper_availability(
            session_id=ctx.session_id,
            literature_id=literature_id,
            availability=Availability.PARSED,
        )

    return ToolResult.ok(
        {
            "paper_id": paper_id,
            "version_id": handle.result.get("version_id", ""),
            "sections": handle.result.get("sections", 0),
            "availability": Availability.PARSED.value,
        },
        note="Parsed. You can now read this paper.",
    ).to_json()


ACQUISITION_TOOLS = [download_paper, ensure_paper_parsed]
