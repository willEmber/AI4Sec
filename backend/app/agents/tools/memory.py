"""Long-term memory: what the agent may keep about a reader across sessions.

Memory here is deliberately narrow. It holds durable facts about the *reader*
— how they want answers, what they are working on, standing instructions —
never paper content, which lives in evidence and is re-read rather than
remembered. Each memory belongs to a principal (`owner_id`), is injected into
the prompt as reference data, and can be inspected and deleted by the reader.

A memory is global or belongs to one research project (P8). A project memory
reaches only that project's sessions, so "I am working on X" kept in one
project is not wrong context in the next. The project is the session's, taken
from the trusted context; the model can only choose between "this project"
and "everywhere".

Three boundaries are enforced in code, not asked of the model:

* Memories are written under the identity in the trusted runtime context. The
  model cannot choose whose memory it writes to.
* Anything shaped like a credential is refused. A reader pasting an API key
  into a chat must not find it in their memory list a week later.
* A memory has to be traceable to something the reader said in this
  conversation. One that is not — a sentence a paper or a web page told the
  agent to remember, say — is kept as *pending*: no turn sees it until the
  reader confirms it. A memory outlives the conversation, so an instruction
  planted in a document would otherwise steer every later one.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

from langchain.tools import ToolRuntime, tool

from app.agents.context import AgentContext
from app.db import agent_repository as repo
from app.models.agent_models import (
    AgentMemory,
    ErrorCode,
    EventType,
    MemoryKind,
    MessageRole,
    ToolResult,
)

logger = logging.getLogger("scholar.agents.tools.memory")

MAX_MEMORY_CHARS = 500

# Coarse on purpose: a false positive costs one refused memory, a false
# negative leaks a secret. Two shapes are let through because a reader of ML
# papers writes them all the time: a setting with a plain number ("token =
# 100000") and a long identifier with no digits in it.
_KEY_PREFIX_RE = re.compile(r"sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}")
_ASSIGNED_RE = re.compile(
    r"\b(?:api[_-]?key|secret|token|password|passwd)\b\s*[:=]\s*(\S{6,})", re.IGNORECASE
)
_LONG_RUN_RE = re.compile(r"\b[A-Za-z0-9_-]{40,}\b")
_PLAIN_NUMBER_RE = re.compile(r"[\d.,_%]+[A-Za-z]{0,2}[.,;，。；]?")

# How much of a memory's wording has to come from the reader's own messages.
# Below half: the memory is a rewording, and "the reader prefers …" around
# what they said must not sink it.
GROUNDED_SHARE = 0.4
_WORD_RE = re.compile(r"[a-z0-9]+")
_HAN_RE = re.compile(r"[\u4e00-\u9fff]+")
# How a memory talks *about* the reader, which the reader never says.
_FRAMING_RE = re.compile(r"用户|读者|希望|喜欢|偏好|要求|习惯")
_STOPWORDS = frozenset(
    "the and for with that this from are was were has have had not but you your "
    "they them their she her his him its our who what when how any all can will "
    "would should about into over than then also only reader user readers users "
    "prefer prefers preferred want wants like likes need needs ask asks asked "
    "always answer answers".split()
)


def looks_like_secret(text: str, known_ids: Iterable[str] = ()) -> bool:
    """Whether `text` holds something shaped like a credential.

    `known_ids` are handles that are allowed to appear: a `paper_id` is forty
    hex characters, which is exactly what the token rule looks for. Only ids
    the caller already knows are exempt — a forty-character run that is not
    one of them may well be a key.
    """
    text = text or ""
    for known in known_ids:
        if known:
            text = text.replace(known, "")
    if _KEY_PREFIX_RE.search(text):
        return True
    if any(not _PLAIN_NUMBER_RE.fullmatch(m.group(1)) for m in _ASSIGNED_RE.finditer(text)):
        return True
    return any(
        re.search(r"\d", run) and re.search(r"[A-Za-z]", run)
        for run in _LONG_RUN_RE.findall(text)
    )


def memory_text_problem(text: str) -> str:
    """Why `text` cannot be a memory at all, or `""`. The secret check is separate."""
    if not text:
        return "content must not be empty."
    if len(text) > MAX_MEMORY_CHARS:
        return f"A memory must be under {MAX_MEMORY_CHARS} characters; keep it to one durable point."
    return ""


def _terms(text: str) -> set[str]:
    """Words and Han bigrams of `text`, the units grounding is counted in."""
    text = _FRAMING_RE.sub(" ", (text or "").lower())
    terms = set()
    for word in _WORD_RE.findall(text):
        if len(word) > 3 and word.endswith("s"):
            word = word[:-1]
        if len(word) > 2 and word not in _STOPWORDS:
            terms.add(word)
    for run in _HAN_RE.findall(text):
        terms.update(run[i : i + 2] for i in range(max(len(run) - 1, 1)))
    return terms


def grounded_in(text: str, said: Iterable[str]) -> bool:
    """Whether enough of `text` repeats what the reader `said`."""
    terms = _terms(text)
    if not terms:
        return False
    heard: set[str] = set()
    for message in said:
        heard |= _terms(message)
    return len(terms & heard) / len(terms) >= GROUNDED_SHARE


async def _reader_said(ctx: AgentContext) -> list[str]:
    """The reader's own messages in this conversation, from the stored rows.

    Not from the model's message state: that is where paper and web text sit
    too, and a compaction may have summarised the reader's words away.
    """
    return [
        m.content
        for m in await repo.list_messages(ctx.session_id)
        if m.role == MessageRole.USER
    ]


async def _paper_handles(ctx: AgentContext) -> set[str]:
    """The `paper_id`s this turn can name: the session's and its project's."""
    papers = list(await repo.list_session_papers(ctx.session_id))
    if ctx.project_id:
        papers += await repo.list_project_papers(
            ctx.project_id, exclude_session_id=ctx.session_id
        )
    return {p.paper_id for p in papers if p.paper_id}


async def _emit_saved(ctx: AgentContext, memory: AgentMemory, replaced: str = "") -> None:
    from app.services import agent_runner

    try:
        await agent_runner.emit(
            session_id=ctx.session_id,
            run_id=ctx.run_id,
            type=EventType.MEMORY_SAVED,
            payload={
                "memory_id": memory.memory_id,
                "kind": memory.kind.value,
                "preview": memory.content[:120],
                "scope": "project" if memory.project_id else "global",
                "pending": memory.pending,
                "replaced": replaced,
            },
        )
    except Exception:  # noqa: BLE001 — the memory is saved; the notice is best effort
        logger.debug("memory.saved event not published for %s", memory.memory_id)


@tool(parse_docstring=False)
async def save_memory(
    content: str,
    runtime: ToolRuntime[AgentContext],
    kind: str = "preference",
    scope: str = "auto",
    replaces: str = "",
) -> str:
    """Remember something durable about the reader for future conversations.

    Use this when the reader states how they want answers (language, format,
    depth), what they are working on, or gives a standing instruction — or when
    they explicitly ask you to remember something. `kind` is one of
    preference, fact, project, instruction. `scope` is "project" (only this
    research project's conversations will see it), "global" (every
    conversation), or "auto": a `project` memory stays in the project, anything
    else is global. Do not store paper content, one-off requests, or anything
    transient; do not store credentials. Write the memory as a short,
    self-contained sentence in the reader's own language and close to their
    own words: a memory that cannot be traced to something the reader said in
    this conversation is not used until they confirm it. When this corrects or
    supersedes a memory you already hold, pass that memory's id as `replaces`
    and it is forgotten in the same call.
    """
    ctx = runtime.context
    text = (content or "").strip()
    problem = memory_text_problem(text)
    if problem:
        return ToolResult.failed(ErrorCode.INVALID_ARGUMENT, problem, retryable=False).to_json()
    if looks_like_secret(text) and looks_like_secret(text, await _paper_handles(ctx)):
        return ToolResult.failed(
            ErrorCode.FORBIDDEN,
            "That looks like a credential. Credentials are never stored; tell the reader so.",
            retryable=False,
        ).to_json()
    try:
        memory_kind = MemoryKind(kind.strip().lower())
    except ValueError:
        memory_kind = MemoryKind.FACT

    project_id, scope_note = _memory_project(ctx, memory_kind, scope)
    wanted = (replaces or "").strip()
    try:
        memory, was_replaced = await repo.save_memory(
            owner_id=ctx.owner_id,
            content=text,
            kind=memory_kind,
            source_session_id=ctx.session_id,
            source_run_id=ctx.run_id,
            project_id=project_id,
            pending=not grounded_in(text, await _reader_said(ctx)),
            replaces=wanted,
            reach_project_id=ctx.project_id,
        )
    except repo.MemoryLimitReached:
        return ToolResult.failed(
            ErrorCode.FORBIDDEN,
            f"The reader already has {repo.MAX_MEMORIES_PER_OWNER} memories, the most one may "
            "hold. Tell them; nothing more is saved until some are deleted.",
            retryable=False,
        ).to_json()
    replaced = wanted if was_replaced else ""
    await _emit_saved(ctx, memory, replaced)

    data = {
        "memory_id": memory.memory_id,
        "kind": memory.kind.value,
        "content": memory.content,
        "scope": "project" if memory.project_id else "global",
        "pending": memory.pending,
        "replaced": replaced,
    }
    if memory.pending:
        return ToolResult.ok(
            data,
            note=(
                "Not in use yet: this could not be traced to something the reader said in "
                "this conversation, so it waits for them to confirm it in the memory panel"
                + (f", and {wanted} stays until they do" if wanted else "")
                + f". Tell the reader that; do not save it again. {scope_note}"
            ).strip(),
        ).to_json()
    if wanted:
        scope_note = (
            f"It replaces {wanted}. {scope_note}"
            if was_replaced
            else f"No memory {wanted} in reach, so nothing was replaced. {scope_note}"
        )
    where = (
        "future conversations in this project" if memory.project_id else "future conversations"
    )
    return ToolResult.ok(
        data,
        note=f"Saved. It will be available in {where}; the reader can delete it. {scope_note}".strip(),
    ).to_json()


def _memory_project(ctx: AgentContext, kind: MemoryKind, scope: str) -> tuple[str, str]:
    """`(project_id, note)` for a memory being saved in this turn.

    The project can only ever be the session's own; `scope` merely says
    whether to use it.
    """
    wanted = (scope or "auto").strip().lower()
    if wanted == "auto":
        wanted = "project" if kind == MemoryKind.PROJECT else "global"
    if wanted != "project":
        return "", ""
    if not ctx.project_id:
        return "", "This conversation is not in a project, so it was saved as a global memory."
    return ctx.project_id, ""


@tool(parse_docstring=False)
async def list_memories(runtime: ToolRuntime[AgentContext]) -> str:
    """List what you currently remember about the reader, with memory ids.

    The active memories are already in your instructions, ids included; call
    this only when the instructions say some were left out, or the reader asks
    what you remember.
    """
    ctx = runtime.context
    # Pending ones are left out: they are not the agent's until confirmed.
    memories = await repo.list_memories(
        ctx.owner_id, project_id=ctx.project_id, include_pending=False
    )
    return ToolResult.ok(
        {
            "memories": [
                {
                    "memory_id": m.memory_id,
                    "kind": m.kind.value,
                    "content": m.content,
                    "scope": "project" if m.project_id else "global",
                }
                for m in memories
            ]
        }
    ).to_json()


@tool(parse_docstring=False)
async def forget_memory(memory_id: str, runtime: ToolRuntime[AgentContext]) -> str:
    """Forget one memory, by id, when the reader asks you to or corrects it.

    The id is beside each memory in your instructions. To correct a memory,
    call save_memory with `replaces` instead.
    """
    ctx = runtime.context
    memory_id = (memory_id or "").strip()
    if not memory_id:
        return ToolResult.failed(
            ErrorCode.INVALID_ARGUMENT, "memory_id must not be empty.", retryable=False
        ).to_json()
    removed = await repo.delete_memory(
        memory_id, owner_id=ctx.owner_id, reach_project_id=ctx.project_id
    )
    if not removed:
        return ToolResult.unavailable(
            ErrorCode.EVIDENCE_NOT_FOUND, f"No memory {memory_id} in reach."
        ).to_json()
    return ToolResult.ok({"memory_id": memory_id, "forgotten": True}).to_json()


MEMORY_TOOLS = [save_memory, list_memories, forget_memory]
