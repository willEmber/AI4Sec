"""Long-term memory: what the agent may keep about a reader across sessions.

Memory here is deliberately narrow. It holds durable facts about the *reader*
— how they want answers, what they are working on, standing instructions —
never paper content, which lives in evidence and is re-read rather than
remembered. Each memory belongs to a principal (`owner_id`), is injected into
the prompt as reference data, and can be inspected and deleted by the reader.

Two boundaries are enforced in code, not asked of the model:

* Memories are written under the identity in the trusted runtime context. The
  model cannot choose whose memory it writes to.
* Anything shaped like a credential is refused. A reader pasting an API key
  into a chat must not find it in their memory list a week later.
"""

from __future__ import annotations

import logging
import re

from langchain.tools import ToolRuntime, tool

from app.agents.context import AgentContext
from app.db import agent_repository as repo
from app.models.agent_models import ErrorCode, EventType, MemoryKind, ToolResult

logger = logging.getLogger("scholar.agents.tools.memory")

MAX_MEMORY_CHARS = 500

# Token-like runs and the usual key prefixes. Coarse on purpose: a false
# positive costs one refused memory, a false negative leaks a secret.
_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|"
    r"\b(api[_-]?key|secret|token|password|passwd)\b\s*[:=]\s*\S{6,}|"
    r"\b[A-Za-z0-9_-]{40,}\b)",
    re.IGNORECASE,
)


def looks_like_secret(text: str) -> bool:
    return _SECRET_RE.search(text or "") is not None


async def _emit_saved(ctx: AgentContext, memory_id: str, kind: str, content: str) -> None:
    from app.services import agent_runner

    try:
        await agent_runner.emit(
            session_id=ctx.session_id,
            run_id=ctx.run_id,
            type=EventType.MEMORY_SAVED,
            payload={"memory_id": memory_id, "kind": kind, "preview": content[:120]},
        )
    except Exception:  # noqa: BLE001 — the memory is saved; the notice is best effort
        logger.debug("memory.saved event not published for %s", memory_id)


@tool(parse_docstring=False)
async def save_memory(
    content: str,
    runtime: ToolRuntime[AgentContext],
    kind: str = "preference",
) -> str:
    """Remember something durable about the reader for future conversations.

    Use this when the reader states how they want answers (language, format,
    depth), what they are working on, or gives a standing instruction — or when
    they explicitly ask you to remember something. `kind` is one of
    preference, fact, project, instruction. Do not store paper content, one-off
    requests, or anything transient; do not store credentials. Write the memory
    as a short, self-contained sentence.
    """
    ctx = runtime.context
    text = (content or "").strip()
    if not text:
        return ToolResult.failed(
            ErrorCode.INVALID_ARGUMENT, "content must not be empty.", retryable=False
        ).to_json()
    if len(text) > MAX_MEMORY_CHARS:
        return ToolResult.failed(
            ErrorCode.INVALID_ARGUMENT,
            f"A memory must be under {MAX_MEMORY_CHARS} characters; keep it to one durable point.",
            retryable=False,
        ).to_json()
    if looks_like_secret(text):
        return ToolResult.failed(
            ErrorCode.FORBIDDEN,
            "That looks like a credential. Credentials are never stored; tell the reader so.",
            retryable=False,
        ).to_json()
    try:
        memory_kind = MemoryKind(kind.strip().lower())
    except ValueError:
        memory_kind = MemoryKind.FACT

    memory = await repo.add_memory(
        owner_id=ctx.owner_id,
        content=text,
        kind=memory_kind,
        source_session_id=ctx.session_id,
        source_run_id=ctx.run_id,
    )
    await _emit_saved(ctx, memory.memory_id, memory.kind.value, memory.content)
    return ToolResult.ok(
        {"memory_id": memory.memory_id, "kind": memory.kind.value, "content": memory.content},
        note="Saved. It will be available in future conversations; the reader can delete it.",
    ).to_json()


@tool(parse_docstring=False)
async def list_memories(runtime: ToolRuntime[AgentContext]) -> str:
    """List what you currently remember about the reader, with memory ids.

    The active memories are already in your instructions; call this only when
    you need an id — for example to forget one the reader has asked you to drop.
    """
    ctx = runtime.context
    memories = await repo.list_memories(ctx.owner_id)
    return ToolResult.ok(
        {
            "memories": [
                {"memory_id": m.memory_id, "kind": m.kind.value, "content": m.content}
                for m in memories
            ]
        }
    ).to_json()


@tool(parse_docstring=False)
async def forget_memory(memory_id: str, runtime: ToolRuntime[AgentContext]) -> str:
    """Forget one memory, by id, when the reader asks you to or corrects it.

    Get the id from list_memories. To replace a memory, forget the old one and
    save the corrected one.
    """
    ctx = runtime.context
    memory_id = (memory_id or "").strip()
    if not memory_id:
        return ToolResult.failed(
            ErrorCode.INVALID_ARGUMENT, "memory_id must not be empty.", retryable=False
        ).to_json()
    removed = await repo.deactivate_memory(memory_id, owner_id=ctx.owner_id)
    if not removed:
        return ToolResult.unavailable(
            ErrorCode.EVIDENCE_NOT_FOUND, f"No active memory {memory_id}."
        ).to_json()
    return ToolResult.ok({"memory_id": memory_id, "forgotten": True}).to_json()


MEMORY_TOOLS = [save_memory, list_memories, forget_memory]
