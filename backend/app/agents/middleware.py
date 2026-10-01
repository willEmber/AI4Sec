"""Context management for the paper agent: compaction and tool-result eviction.

A reading conversation grows in a particular way. Each question pulls in a few
tool results, and the big ones — a section read, a report excerpt — are worth
their size exactly once. Two turns later the model needs to know *that* it
read the section and which evidence ids came back, not the section itself.
Two middleware handle the two halves of that:

*`ToolResultEvictionMiddleware`* cuts old, large tool results down to a stub
before each model call. The stub keeps the tool's `status`, `note` and every
`evidence_id`, so an answer can still cite what was read; the text is gone,
and re-reading is one cheap tool call away. Only the request is changed — the
checkpoint keeps the full result, so nothing is lost for replay or audit.

*`ScholarSummarizationMiddleware`* is deepagents' summarisation with three
things changed: the trigger is an absolute token count from settings (the
gateway models publish no profile, so fraction triggers never fire), the
summary prompt is written for a reading assistant (it must carry forward
paper handles, evidence ids and the reader's stated preferences), and each
compaction is reported to a hook so the turn can tell the reader it happened.
The class reports `name == "SummarizationMiddleware"` so `create_deep_agent`
replaces its default instance in place rather than running both.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately

from deepagents.backends import StateBackend
from deepagents.middleware.summarization import (
    SUMMARIZATION_EVENT_KEY,
    SummarizationMiddleware,
)

logger = logging.getLogger("scholar.agents.middleware")

CompactHook = Callable[[dict[str, Any]], Awaitable[None]]

# Tools whose results are the paper text itself. These are the ones worth
# evicting; a search hit list or a rank lookup is small and stays.
EVICTABLE_TOOLS = frozenset(
    {
        "read_paper_section",
        "search_paper_content",
        "run_insight_snap",
        "run_logic_lens",
        "run_research_sphere",
    }
)

# How much of the summariser's own input to keep. LangChain's default (4000
# tokens) was tuned for terse coding sessions; a reading session's evicted
# window carries several section reads, and cutting them to 4k would summarise
# mostly the beginning of the oldest one.
SUMMARY_INPUT_TOKENS = 24_000

# `.format(messages=...)` is applied to this, so every literal brace is doubled.
SCHOLAR_SUMMARY_PROMPT = """<role>
You are compacting the working memory of a paper-reading assistant.
</role>

<objective>
The conversation below is about to be replaced by what you write. The assistant
must be able to keep answering the reader's questions about the same papers
without re-doing work it has already done, and without losing the reader's
stated wishes. Write in the language the reader has been using.
</objective>

<instructions>
Produce these sections; write "None" where a section is empty.

## Reader's goal and preferences
What the reader is trying to find out, and every preference or instruction they
stated (language, format, depth, what to skip). These must survive verbatim in
spirit.

## Papers in play
One line per paper: title, its `paper_id` or `literature_id` handle exactly as
it appeared, its availability (parsed / abstract only / unavailable), and what
has already been read from it (sections, tables).

## Established findings
Concrete facts established from the papers so far — numbers, setups, claims,
comparisons — each followed by the `[ev_...]` evidence ids that support it,
copied exactly. Do not invent ids; an unsupported finding is listed without one
and marked as such. Distinguish what a paper states from what was inferred.

## Reports produced
Any Insight Snap / Logic Lens / Research Sphere report already generated, with
its `run_id`, so it is not generated again.

## Open threads
Questions asked but not fully answered, tool calls that reported partial or
unavailable results, budget ceilings reached, and anything the assistant said
it would come back to.
</instructions>

Respond ONLY with the sections above. Do not add commentary before or after.

<messages>
Messages to summarize:
{messages}
</messages>"""


class ScholarSummarizationMiddleware(SummarizationMiddleware):
    """deepagents summarisation with a reading-oriented prompt and a compaction hook."""

    def __init__(
        self,
        model: BaseChatModel,
        *,
        trigger_tokens: int,
        keep_messages: int,
        on_compact: CompactHook | None = None,
        evictor: Callable[[ModelRequest], ModelRequest] | None = None,
    ) -> None:
        super().__init__(
            model,
            backend=StateBackend(),
            trigger=("tokens", int(trigger_tokens)),
            keep=("messages", int(keep_messages)),
            token_counter=count_tokens_approximately,
            summary_prompt=SCHOLAR_SUMMARY_PROMPT,
            trim_tokens_to_summarize=SUMMARY_INPUT_TOKENS,
        )
        self._on_compact = on_compact
        # Applied to the request *before* tokens are counted, so cheap trimming
        # gets a chance to make the expensive summarisation unnecessary. Done
        # here rather than as a separate middleware because deepagents places
        # caller middleware inside its own stack, where it would run after the
        # count had already been taken.
        self._evictor = evictor
        self._trigger_tokens = int(trigger_tokens)
        self._keep_messages = int(keep_messages)

    @property
    def name(self) -> str:
        # The public alias is what `create_deep_agent` matches on to replace its
        # own instance; a subclass would otherwise report its class name and be
        # *added* beside the default, summarising twice.
        return "SummarizationMiddleware"

    def _compaction_payload(
        self, before: list[AnyMessage], response: Any
    ) -> dict[str, Any] | None:
        command = getattr(response, "command", None)
        update = getattr(command, "update", None) if command is not None else None
        if not isinstance(update, dict) or SUMMARIZATION_EVENT_KEY not in update:
            return None
        event = update[SUMMARIZATION_EVENT_KEY] or {}
        cutoff = int(event.get("cutoff_index", 0) or 0)
        return {
            "summarized_messages": max(0, cutoff),
            "kept_messages": max(0, len(before) - cutoff),
            "tokens_before": self._count_tokens(before, None, None),
            "trigger_tokens": self._trigger_tokens,
        }

    async def awrap_model_call(  # type: ignore[override]
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> Any:
        if self._evictor is not None:
            request = self._evictor(request)
        before = list(request.messages)
        response = await super().awrap_model_call(request, handler)
        payload = self._compaction_payload(before, response)
        if payload is not None:
            logger.info("Conversation compacted: %s", payload)
            if self._on_compact is not None:
                try:
                    await self._on_compact(payload)
                except Exception:  # noqa: BLE001 — reporting must not fail the turn
                    logger.exception("compaction hook failed")
        return response

    def wrap_model_call(  # type: ignore[override]
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> Any:
        if self._evictor is not None:
            request = self._evictor(request)
        before = list(request.messages)
        response = super().wrap_model_call(request, handler)
        payload = self._compaction_payload(before, response)
        if payload is not None:
            logger.info("Conversation compacted (sync): %s", payload)
        return response


def _evidence_ids_of(content: str) -> list[str]:
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, dict):
        return []
    ids = parsed.get("evidence_ids") or []
    return [str(i) for i in ids if isinstance(i, str)]


def _stub_for(message: ToolMessage, content: str) -> str:
    """A compact stand-in for a large tool result, keeping what an answer can still use."""
    status = "ok"
    note = ""
    summary: dict[str, Any] = {}
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            status = str(parsed.get("status", "ok"))
            note = str(parsed.get("note", "") or "")
            data = parsed.get("data")
            if isinstance(data, dict):
                for key in ("paper_id", "section", "question", "title", "run_id", "mode", "page"):
                    if key in data:
                        summary[key] = data[key]
                for key in ("hits", "blocks", "papers"):
                    if isinstance(data.get(key), list):
                        summary[f"{key}_count"] = len(data[key])
    except (TypeError, ValueError):
        pass
    stub = {
        "status": status,
        "evicted": True,
        "note": (
            "Full result removed from context to save space. The evidence ids below "
            "remain citable; call the tool again if you need the text itself. "
            + note
        ).strip(),
        "data": summary,
        "evidence_ids": _evidence_ids_of(content),
        "original_chars": len(content),
    }
    return json.dumps(stub, ensure_ascii=False)


def evict_tool_results(
    messages: list[AnyMessage], *, keep_recent: int, max_chars: int
) -> tuple[list[AnyMessage], int]:
    """Replace old, large tool results with stubs. Returns `(messages, evicted_count)`.

    "Old" means not among the last `keep_recent` tool messages; "large" means
    longer than `max_chars`. Only tools in `EVICTABLE_TOOLS` are touched — the
    others are small and their exact content matters (a rank, a candidate list).
    """
    tool_positions = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    protected = set(tool_positions[-keep_recent:]) if keep_recent > 0 else set()
    out: list[AnyMessage] = list(messages)
    evicted = 0
    for index in tool_positions:
        if index in protected:
            continue
        message = messages[index]
        if (message.name or "") not in EVICTABLE_TOOLS:
            continue
        content = message.content if isinstance(message.content, str) else ""
        if len(content) <= max_chars:
            continue
        if '"evicted": true' in content:
            continue
        replacement = message.model_copy()
        replacement.content = _stub_for(message, content)
        out[index] = replacement
        evicted += 1
    return out, evicted


class ToolResultEvictionMiddleware(AgentMiddleware):
    """Trim old, large tool results out of the request before each model call."""

    def __init__(self, *, keep_recent: int, max_chars: int) -> None:
        super().__init__()
        self._keep_recent = max(0, int(keep_recent))
        self._max_chars = max(200, int(max_chars))

    @property
    def name(self) -> str:
        return "ToolResultEvictionMiddleware"

    def _apply(self, request: ModelRequest) -> ModelRequest:
        messages, evicted = evict_tool_results(
            list(request.messages), keep_recent=self._keep_recent, max_chars=self._max_chars
        )
        if evicted:
            logger.debug("Evicted %d old tool results from the model request", evicted)
            return request.override(messages=messages)
        return request

    def wrap_model_call(  # type: ignore[override]
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._apply(request))

    async def awrap_model_call(  # type: ignore[override]
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._apply(request))


def build_agent_middleware(
    model: BaseChatModel | None,
    *,
    on_compact: CompactHook | None = None,
) -> list[AgentMiddleware]:
    """The middleware a turn (and the warm-up) hands to `create_paper_agent`.

    With a model, eviction is folded into the summarisation middleware so it
    runs before the token count; without one (a warm-up on a process with no
    gateway configured) eviction stands alone and summarisation is left to the
    deepagents default.
    """
    from app.config import get_settings

    settings = get_settings()
    eviction = ToolResultEvictionMiddleware(
        keep_recent=settings.agent_tool_result_keep,
        max_chars=settings.agent_tool_result_max_chars,
    )
    if model is None:
        return [eviction]
    return [
        ScholarSummarizationMiddleware(
            model,
            trigger_tokens=settings.agent_context_trigger_tokens,
            keep_messages=settings.agent_context_keep_messages,
            on_compact=on_compact,
            evictor=eviction._apply,
        )
    ]
