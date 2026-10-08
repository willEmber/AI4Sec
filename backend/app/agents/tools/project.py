"""Project tools: recall earlier conversations, and bring a project paper into the session (P8).

Both are scoped by the trusted runtime context, never by an argument: the
reader is `owner_id`, the project is the session's `project_id`. A model that
is told "search user X's sessions" or "open the paper from project Y" has no
parameter through which to do it.

What recall returns is a *lead*, not evidence about a paper. The earlier
answer's wording is shown so the agent knows what was concluded; the evidence
that answer cited is returned as evidence — immutable snapshots pinned to the
parse version read at the time — and only those ids become citable.
"""

from __future__ import annotations

import logging

from langchain.tools import ToolRuntime, tool

from app.agents.context import AgentContext
from app.db import agent_repository as repo
from app.models.agent_models import Availability, ErrorCode, ToolResult
from app.services import conversation_recall

logger = logging.getLogger("scholar.agents.tools.project")

MAX_RECALL_RESULTS = 8


@tool(parse_docstring=False)
async def recall_conversations(
    query: str,
    runtime: ToolRuntime[AgentContext],
    scope: str = "project",
    limit: int = 5,
) -> str:
    """Search the reader's earlier conversations for what was asked, answered and cited.

    Use this when the reader refers to something discussed before ("last time",
    "the number we found", "上次"), or when an earlier conversation in this
    project probably already settled the question. `query` is a few keywords;
    a turn matches on any of them, so give the terms in both Chinese and
    English when the earlier conversation may have used either. `scope` is "project" (this project's other
    conversations; the default) or "all" (every earlier conversation of this
    reader). Each result is one earlier turn: the question, an excerpt of the
    answer, and the evidence that answer cited. Cite that evidence by its id;
    never cite the earlier answer's wording as if it were the paper. The current
    conversation is not searched — it is already in your context.
    """
    ctx = runtime.context
    text = (query or "").strip()
    if not text:
        return ToolResult.failed(
            ErrorCode.INVALID_ARGUMENT, "query must not be empty.", retryable=False
        ).to_json()

    wanted = (scope or "project").strip().lower()
    notes: list[str] = []
    if wanted not in ("project", "all"):
        wanted = "project"
    if wanted == "project" and not ctx.project_id:
        wanted = "all"
        notes.append("This conversation is not in a project, so every earlier conversation was searched.")

    turns = await conversation_recall.recall(
        ctx.owner_id,
        text,
        project_id=ctx.project_id if wanted == "project" else None,
        exclude_session_id=ctx.session_id,
        limit=max(1, min(int(limit or 5), MAX_RECALL_RESULTS)),
    )
    evidence_ids = list(dict.fromkeys(e.evidence_id for t in turns for e in t.evidence))
    if not turns:
        notes.append(
            "No earlier conversation matched. Try other keywords, scope=\"all\", "
            "or answer from the papers directly."
        )
    else:
        notes.append(
            "Earlier answers are leads, not evidence: cite only the evidence ids listed. "
            "Re-read the paper when a figure or formula matters and the evidence may be stale."
        )
    return ToolResult.ok(
        {"scope": wanted, "query": text, "turns": [t.as_dict() for t in turns]},
        evidence_ids=evidence_ids,
        note=" ".join(notes),
    ).to_json()


@tool(parse_docstring=False)
async def open_project_paper(paper: str, runtime: ToolRuntime[AgentContext]) -> str:
    """Add one of this project's other papers to the current conversation so it can be read.

    The system prompt lists the project's papers that this conversation does not
    have yet. Pass the paper's `paper_id` or `literature_id` exactly as listed.
    After this, the reading tools accept it like any paper of this conversation.
    Only papers already in this project can be opened; to find new papers, use
    search_papers.
    """
    ctx = runtime.context
    handle = (paper or "").strip()
    if not handle:
        return ToolResult.failed(
            ErrorCode.INVALID_ARGUMENT, "paper must not be empty.", retryable=False
        ).to_json()
    if not ctx.project_id:
        return ToolResult.unavailable(
            ErrorCode.PAPER_NOT_FOUND,
            "This conversation is not in a project, so there are no project papers to open.",
        ).to_json()

    found = await repo.find_project_paper(ctx.project_id, handle)
    if found is None:
        # The same answer whether the paper does not exist or belongs elsewhere.
        return ToolResult.unavailable(
            ErrorCode.PAPER_NOT_FOUND, f"No paper {handle} in this project."
        ).to_json()

    await repo.attach_session_paper(
        session_id=ctx.session_id,
        literature_id=found.literature_id,
        paper_id=found.paper_id,
        availability=found.availability,
        added_by="agent",
    )
    readable = bool(found.paper_id) and found.availability == Availability.PARSED
    note = (
        "Added to this conversation; the reading tools can use it now."
        if readable
        else "Added to this conversation, but its full text is not parsed yet: "
        "use ensure_paper_parsed (or download_paper) before reading it."
    )
    return ToolResult.ok(
        {
            "literature_id": found.literature_id,
            "paper_id": found.paper_id,
            "title": found.title,
            "availability": found.availability.value,
        },
        note=note,
    ).to_json()


PROJECT_TOOLS = [recall_conversations, open_project_paper]
