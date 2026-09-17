"""Agent construction with the host / delegation boundary enforced.

`create_deep_agent` ships a general-purpose subagent and a virtual-filesystem
toolset. Both are deliberately switched off here:

* The default `task` delegation tool is dropped, because the first version runs
  a single agent. Passing ``subagents=[]`` does **not** do this — deepagents
  auto-inserts a general-purpose subagent unless the harness profile disables
  it, which is what `GeneralPurposeSubagentProfile(enabled=False)` below does.
* The seven built-in filesystem tools are stripped from the model-visible tool
  list, so the paper agent only ever sees domain tools.

The default backend is `StateBackend`: its files live in graph state, not on
the host. Verified in `tests/test_agent_harness.py` — a hallucinated
``write_file`` lands in state and never touches disk, ``read_file`` cannot see
host paths, and ``execute`` is refused because the backend implements no
sandbox protocol. Together with the trimmed surface that is the boundary the
development plan requires: no host filesystem, no shell, no env, no DB handle.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool

from app.agents.model_factory import SCHOLAR_PROVIDER, build_chat_model

logger = logging.getLogger("scholar.agents.harness")

# Built-in tools deepagents injects for its virtual workspace. Excluded from the
# model-visible surface; `execute` is listed too so a backend swap can never
# quietly hand the model a shell.
BUILTIN_HOST_TOOLS = frozenset(
    {"ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "execute"}
)

# The delegation tool `SubAgentMiddleware` installs. Disabling the
# general-purpose subagent removes it; kept here so tests can assert on it.
DELEGATION_TOOL = "task"

_profile_lock = threading.Lock()
_profile_registered = False


def register_scholar_harness_profile() -> None:
    """Register the paper agent's harness profile. Idempotent and thread-safe."""
    global _profile_registered
    with _profile_lock:
        if _profile_registered:
            return
        from deepagents import (
            GeneralPurposeSubagentProfile,
            HarnessProfile,
            register_harness_profile,
        )

        register_harness_profile(
            SCHOLAR_PROVIDER,
            HarnessProfile(
                excluded_tools=frozenset(BUILTIN_HOST_TOOLS),
                general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
            ),
        )
        _profile_registered = True
        logger.info(
            "Registered harness profile for %s: built-in tools excluded, "
            "general-purpose subagent disabled",
            SCHOLAR_PROVIDER,
        )


def warm_up(
    *,
    context_schema: type | None = None,
    checkpointer: Any = None,
    model: BaseChatModel | None = None,
) -> None:
    """Build one throwaway agent so no request has to be the first.

    Two costs hide behind `create_paper_agent`, and both used to land on whoever
    asked the first question after a restart: importing the deepagents stack,
    and compiling the graph, which builds a great deal of pydantic machinery. On
    a cold filesystem that measured ~28s of a completely unresponsive server,
    and the request it would not answer was the cancel for the very turn causing
    it.

    Everything that changes what gets compiled is a parameter, because warming a
    different shape warms nothing. That was learned twice, measured each time:
    omitting `context_schema` and `checkpointer` left the first real turn paying
    9.0s, and warming with the probe model still left it paying 8.3s — binding
    the tool schemas to a real chat model is most of the cost, and the probe
    does not do it. With the real model passed, a second build costs 0.01s.

    The model is never called: the agent is constructed and discarded.
    """
    from app.agents.tools import RESEARCH_TOOLS

    register_scholar_harness_profile()
    create_paper_agent(
        tools=RESEARCH_TOOLS,
        system_prompt="warm-up",
        model=model if model is not None else _ToolSurfaceProbe(),
        context_schema=context_schema,
        checkpointer=checkpointer,
    )


def create_paper_agent(
    *,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]],
    system_prompt: str,
    model: BaseChatModel | str | None = None,
    checkpointer: Any = None,
    context_schema: type | None = None,
    middleware: Sequence[Any] = (),
    **kwargs: Any,
) -> Any:
    """Build the paper-reading agent.

    Args:
        tools: Domain tools. These are the only tools the model will see.
        system_prompt: Reading goals, evidence rules and stop conditions.
        model: A chat model, a model id, or `None` for the configured default.
        checkpointer: LangGraph checkpointer; `None` leaves the agent stateless.
        context_schema: Trusted runtime context injected per run.
        middleware: Extra middleware (budget, call logging, result limits).
        **kwargs: Passed through to `create_deep_agent`.
    """
    from deepagents import create_deep_agent

    register_scholar_harness_profile()

    if model is None or isinstance(model, str):
        model = build_chat_model(model or "")

    agent_kwargs: dict[str, Any] = {
        "model": model,
        "tools": list(tools),
        "system_prompt": system_prompt,
        "middleware": list(middleware),
    }
    if checkpointer is not None:
        agent_kwargs["checkpointer"] = checkpointer
    if context_schema is not None:
        agent_kwargs["context_schema"] = context_schema
    agent_kwargs.update(kwargs)
    return create_deep_agent(**agent_kwargs)


class _ToolSurfaceProbe(BaseChatModel):
    """Chat model that records the tools bound to it and then stops.

    Used to assert what the real agent would expose; the tool list handed to the
    model is the authoritative boundary, since deepagents' `ToolNode` keeps the
    built-ins registered even when the profile hides them.
    """

    recorded: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "scholar-tool-surface-probe"

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> Any:
        params = super()._get_ls_params(stop=stop, **kwargs)
        try:
            params["ls_provider"] = SCHOLAR_PROVIDER
        except TypeError:  # pragma: no cover
            pass
        return params

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:  # type: ignore[override]
        names: list[str] = []
        for t in tools:
            name = getattr(t, "name", None)
            if name is None and isinstance(t, dict):
                name = t.get("name") or (t.get("function") or {}).get("name") or t.get("type")
            names.append(str(name or getattr(t, "__name__", t)))
        self.recorded = names
        return self

    def _generate(
        self, messages: list[Any], stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])


def model_visible_tool_names(
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]],
    *,
    system_prompt: str = "probe",
) -> list[str]:
    """Return the tool names the model would actually be offered.

    Builds the agent exactly as `create_paper_agent` does but swaps in a probe
    model, so the answer reflects the live profile rather than a restatement of
    it.
    """
    probe = _ToolSurfaceProbe()
    agent = create_paper_agent(tools=tools, system_prompt=system_prompt, model=probe)
    agent.invoke(
        {"messages": [{"role": "user", "content": "probe"}]},
        config={"configurable": {"thread_id": "tool-surface-probe"}},
    )
    return list(probe.recorded)
