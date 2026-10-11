"""The turn guard: what happens when a turn runs out of something (P13).

One middleware around every tool call and every model call. Like the tools, it
acts on the trusted `AgentContext`, never on anything the model wrote.

*Tool calls* pass four checks, outermost first:

1. **Exceptions.** A tool that raises becomes an `error` result the model can
   read. LangGraph's tool node re-raises anything but an argument error, so
   one database hiccup used to fail the whole turn.
2. **Time.** Each tool has a ceiling. The executor checks the wall clock only
   when the stream yields, and a blocked tool yields nothing.
3. **Budget.** A call takes its slot *before* it runs, so five calls issued
   together cannot all run on two remaining slots.
4. **Repeats.** The third identical call is answered without running: nothing
   has changed since the second.

*Model calls*: once a turn ceiling is reached the next call is the closing
one — made without tools, told to answer from what was gathered. Before this,
reaching a ceiling cut the turn off where it stood, and a paper parsed on the
last allowed call was never read. A call that returns nothing is made once
more; a call that fails before writing anything moves the turn to the fallback
model when one is configured.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents.context import AgentContext
from app.agents.errors import EmptyModelResponse, classify_error
from app.agents.model_factory import STREAM_PROBE, StreamProbe
from app.models.agent_models import ErrorCode, EventType, ToolResult

logger = logging.getLogger("scholar.agents.guard")

# Seconds a tool may take. Tools that read the local database answer in
# milliseconds; search tools wait on providers that pace their callers (DBLP
# alone takes 5–10 s, and S2's one request a second is shared by every turn).
# A parse and the three reports keep their own, much longer limits: `None`.
LOCAL_TOOL_SECONDS = 30.0
REMOTE_TOOL_SECONDS = 120.0
TOOL_SECONDS: dict[str, float | None] = {
    "get_paper_outline": LOCAL_TOOL_SECONDS,
    "search_paper_content": LOCAL_TOOL_SECONDS,
    "read_paper_section": LOCAL_TOOL_SECONDS,
    "save_memory": LOCAL_TOOL_SECONDS,
    "list_memories": LOCAL_TOOL_SECONDS,
    "forget_memory": LOCAL_TOOL_SECONDS,
    "recall_conversations": LOCAL_TOOL_SECONDS,
    "open_project_paper": LOCAL_TOOL_SECONDS,
    "download_paper": 180.0,
    "ensure_paper_parsed": None,
    "run_insight_snap": None,
    "run_logic_lens": None,
    "run_research_sphere": None,
}

# The identical call that is no longer run.
REPEAT_LIMIT = 3
# Failed tool results in a row after which the turn stops gathering.
FAILURE_LIMIT = 5

# Tools whose answer depends on something outside the conversation, so asking
# again is how the model finds out whether it changed.
REPEAT_EXEMPT = frozenset({"ensure_paper_parsed", "list_memories", "open_project_paper"})
# Tools that change what the others can read; a success resets the repeat
# counts, because the same question may now have a different answer.
STATE_CHANGING = frozenset(
    {"download_paper", "ensure_paper_parsed", "open_project_paper", "save_memory", "forget_memory"}
)

# Failures a different model might not have: the gateway was busy, slow or
# broken. A request that was too long or was filtered fails the same way anywhere.
_FALLBACK_CODES = frozenset(
    {ErrorCode.RATE_LIMITED, ErrorCode.TIMEOUT, ErrorCode.UPSTREAM_ERROR}
)

_LIMIT_NAMES = {
    "max_tool_calls": ("工具调用次数", "tool calls"),
    "max_llm_calls": ("模型调用次数", "model calls"),
    "max_tokens": ("token 预算", "token budget"),
    "max_wall_seconds": ("时间上限", "time limit"),
    "tool_failures": ("连续失败的工具调用", "consecutive failed tool calls"),
}


def _limit_name(reason: str, language: str) -> str:
    zh, en = _LIMIT_NAMES.get(reason, (reason, reason))
    return zh if language == "zh" else en


def closing_note(reason: str, language: str) -> str:
    """What the closing model call is told. Request only; never checkpointed."""
    limit = _limit_name(reason, language)
    if language == "zh":
        return (
            f"[系统提示] 本轮已达到{limit}上限，不能再调用任何工具。"
            "请只依据上面已经取得的证据直接回答读者的问题，照常标注引用；"
            "并说明哪些部分因为没来得及查而缺失。不要输出工具调用。"
        )
    return (
        f"[System notice] This turn has reached its {limit} ceiling and no further tool "
        "can be called. Answer the reader's question now from the evidence gathered above, "
        "citing it as usual, and say which parts are missing because there was no room to "
        "look them up. Do not emit tool calls."
    )


def _context_of(runtime: Any) -> AgentContext | None:
    context = getattr(runtime, "context", None)
    return context if isinstance(context, AgentContext) else None


def _call_key(name: str, args: Any) -> str:
    try:
        canonical = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        canonical = repr(args)
    return hashlib.sha1(f"{name}\n{canonical}".encode()).hexdigest()


def _status_of(message: Any) -> str:
    """The `ToolResult` status inside a tool message, '' when it is not one."""
    if not isinstance(message, ToolMessage) or not isinstance(message.content, str):
        return ""
    try:
        parsed = json.loads(message.content)
    except (TypeError, ValueError):
        return ""
    return str(parsed.get("status") or "") if isinstance(parsed, dict) else ""


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        block.get("text", "") or ""
        for block in content
        if isinstance(block, dict) and block.get("type") in {"text", "output_text"}
    )


def _answer_of(response: Any) -> AIMessage | None:
    result = getattr(response, "result", None)
    if isinstance(response, AIMessage):
        return response
    for message in reversed(result or []):
        if isinstance(message, AIMessage):
            return message
    return None


def _is_empty(response: Any, *, text_only: bool) -> bool:
    """Whether a model call produced nothing the turn can continue from."""
    message = _answer_of(response)
    if message is None:
        return False
    if _text_of(message.content).strip():
        return False
    if text_only:
        return True
    return not (message.tool_calls or getattr(message, "invalid_tool_calls", None))


def _without_tool_calls(response: Any) -> Any:
    """The response with its tool calls removed, so the graph ends on it.

    A gateway may answer a request that carried no tools with a tool call
    anyway. Left in, the graph would run the tool node and come back for a
    second closing call.
    """
    message = _answer_of(response)
    if message is None or not (
        message.tool_calls or getattr(message, "invalid_tool_calls", None)
    ):
        return response
    plain = AIMessage(
        content=_text_of(message.content),
        id=message.id,
        usage_metadata=message.usage_metadata,
        response_metadata=dict(message.response_metadata or {}),
    )
    if isinstance(response, AIMessage):
        return plain
    response.result = [plain if m is message else m for m in response.result]
    return response


class TurnGuardMiddleware(AgentMiddleware):
    """Budgets, time limits and fallbacks for one turn's tool and model calls."""

    def __init__(
        self,
        *,
        fallback_model: BaseChatModel | None = None,
        fallback_name: str = "",
    ) -> None:
        super().__init__()
        self._fallback = fallback_model
        self._fallback_name = fallback_name

    @property
    def name(self) -> str:
        return "TurnGuardMiddleware"

    # ── Tool calls ──────────────────────────────────────────────────────────

    def _refuse(self, ctx: AgentContext, request: Any, result: ToolResult) -> ToolMessage:
        ctx.usage.refused_tool_calls += 1
        call = request.tool_call
        return ToolMessage(
            content=result.to_json(), name=call.get("name") or "", tool_call_id=call.get("id") or ""
        )

    def _admit(self, ctx: AgentContext, request: Any) -> ToolMessage | None:
        """Take a slot for this call, or say why it does not get one.

        Synchronous on purpose: calls issued together are admitted one after
        another with nothing awaited in between, so the count cannot be read
        twice before it is written.
        """
        call = request.tool_call
        name = call.get("name") or ""
        ctx.elapsed()
        if not ctx.wrapup:
            reason = ""
            if ctx.usage.tool_calls >= ctx.budget.max_tool_calls:
                reason = "max_tool_calls"
            elif ctx.usage.wall_seconds >= ctx.budget.max_wall_seconds:
                reason = "max_wall_seconds"
            elif ctx.usage.tokens is not None and ctx.usage.tokens >= ctx.budget.max_tokens:
                reason = "max_tokens"
            if reason:
                ctx.begin_wrapup(reason)
        if ctx.wrapup:
            return self._refuse(
                ctx,
                request,
                ToolResult.unavailable(
                    ErrorCode.BUDGET_EXCEEDED,
                    f"This turn has reached its {_limit_name(ctx.wrapup, 'en')} ceiling; "
                    "no further tool will run. Answer from what you already have and say "
                    "what is missing.",
                ),
            )

        ctx.usage.tool_calls += 1
        if name not in REPEAT_EXEMPT:
            key = _call_key(name, call.get("args"))
            seen = ctx.call_counts.get(key, 0) + 1
            ctx.call_counts[key] = seen
            if seen >= REPEAT_LIMIT:
                self._note_failure(ctx)
                return self._refuse(
                    ctx,
                    request,
                    ToolResult.unavailable(
                        ErrorCode.REPEATED_CALL,
                        f"`{name}` has already been called {seen - 1} times this turn with "
                        "exactly these arguments, and nothing it reads has changed since. "
                        "The earlier result stands: change the arguments, use another tool, "
                        "or answer.",
                    ),
                )
        return None

    def _note_failure(self, ctx: AgentContext) -> None:
        ctx.consecutive_failures += 1
        if ctx.consecutive_failures >= FAILURE_LIMIT:
            ctx.begin_wrapup("tool_failures")

    def _observe(self, ctx: AgentContext, name: str, result: Any) -> None:
        status = _status_of(result)
        if status in {"error", "unavailable"}:
            self._note_failure(ctx)
            return
        ctx.consecutive_failures = 0
        if name in STATE_CHANGING and status in {"ok", "partial"}:
            ctx.call_counts.clear()

    async def awrap_tool_call(  # type: ignore[override]
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        ctx = _context_of(request.runtime)
        if ctx is None:
            return await handler(request)
        refusal = self._admit(ctx, request)
        if refusal is not None:
            return refusal

        call = request.tool_call
        name = call.get("name") or ""
        seconds = TOOL_SECONDS.get(name, REMOTE_TOOL_SECONDS)
        failure: ToolResult
        try:
            async with asyncio.timeout(seconds) as deadline:
                result = await handler(request)
            self._observe(ctx, name, result)
            return result
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            if not deadline.expired():
                # The tool's own timeout, not ours; it is still a timeout.
                logger.warning("Tool %s timed out internally", name, exc_info=True)
            else:
                logger.warning("Tool %s exceeded its %.0fs limit", name, seconds or 0)
            ctx.usage.tool_timeouts += 1
            failure = ToolResult.unavailable(
                ErrorCode.TIMEOUT,
                f"`{name}` did not answer in time. Do not call it again with the same "
                "arguments this turn; use what you have or try a different route.",
            )
        except Exception:  # noqa: BLE001 — a tool fault must not fail the turn
            # The traceback goes to the log only: a tool result reaches the
            # model's context and the event stream.
            logger.exception("Tool %s raised", name)
            ctx.usage.errors += 1
            failure = ToolResult.failed(
                ErrorCode.INTERNAL_ERROR,
                f"`{name}` failed inside the service. Try a different approach, or say "
                "plainly that this could not be looked up.",
                retryable=False,
            )
        self._note_failure(ctx)
        return ToolMessage(
            content=failure.to_json(), name=name, tool_call_id=call.get("id") or ""
        )

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:  # type: ignore[override]
        # The synchronous path is the tool-surface probe and nothing else; a
        # turn always runs asynchronously.
        return handler(request)

    # ── Model calls ─────────────────────────────────────────────────────────

    async def _notify(self, ctx: AgentContext, type: EventType, payload: dict[str, Any]) -> None:
        if ctx.notify is None:
            return
        try:
            await ctx.notify(type.value, payload)
        except Exception:  # noqa: BLE001 — reporting must not fail the turn
            logger.exception("turn guard could not emit %s", type.value)

    async def _call_once(
        self,
        ctx: AgentContext,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """One model call, moved to the fallback model if it fails unwritten."""
        if ctx.on_fallback and self._fallback is not None:
            request = request.override(model=self._fallback)
        probe = StreamProbe()
        token = STREAM_PROBE.set(probe)
        try:
            ctx.usage.llm_calls += 1
            try:
                return await handler(request)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — decided just below
                if (
                    self._fallback is None
                    or ctx.on_fallback
                    or probe.text
                    or classify_error(exc) not in _FALLBACK_CODES
                ):
                    raise
                code = classify_error(exc)
                logger.warning(
                    "Run %s: %s failed (%s); continuing on %s",
                    ctx.run_id, ctx.llm_model, code.value, self._fallback_name, exc_info=True,
                )
                ctx.on_fallback = True
                ctx.usage.model_used = self._fallback_name
                await self._notify(
                    ctx,
                    EventType.MODEL_FALLBACK,
                    {"from": ctx.llm_model, "to": self._fallback_name, "code": code.value},
                )
                ctx.usage.llm_calls += 1
                return await handler(request.override(model=self._fallback))
        finally:
            STREAM_PROBE.reset(token)

    async def awrap_model_call(  # type: ignore[override]
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> Any:
        ctx = _context_of(request.runtime)
        if ctx is None:
            return await handler(request)

        ctx.elapsed()
        if not ctx.wrapup:
            reason = ctx.usage.exceeded(ctx.budget)
            if reason:
                logger.warning("Run %s reached %s; closing the turn", ctx.run_id, reason)
                ctx.begin_wrapup(reason)
        closing = bool(ctx.wrapup)
        if closing:
            ctx.wrapup_call_started = True
            request = request.override(
                tools=[],
                messages=[
                    *request.messages,
                    HumanMessage(content=closing_note(ctx.wrapup, ctx.language)),
                ],
            )

        response = await self._call_once(ctx, request, handler)
        if _is_empty(response, text_only=closing):
            logger.warning("Run %s: the model returned nothing; asking once more", ctx.run_id)
            response = await self._call_once(ctx, request, handler)
            if _is_empty(response, text_only=False):
                raise EmptyModelResponse("the model returned neither text nor a tool call")
        return _without_tool_calls(response) if closing else response

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Any]) -> Any:  # type: ignore[override]
        return handler(request)
