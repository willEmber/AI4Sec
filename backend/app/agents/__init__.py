"""Deep Agents layer for the conversational paper-reading agent.

`model_factory` builds the chat model, `harness` builds the agent with the
host/delegation boundary enforced, and `checkpointer` supplies the durable
LangGraph checkpointer that makes a session resumable across restarts.

Attributes resolve lazily: pulling in `deepagents` drags along langchain plus
three provider SDKs, which is slow enough to notice at API startup (~45s on a
WSL `/mnt` checkout, where the cost is filesystem I/O rather than CPU). Nothing
here is imported until the agent is actually built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from app.agents.checkpointer import agent_checkpoint_db_path, open_checkpointer
    from app.agents.harness import (
        BUILTIN_HOST_TOOLS,
        DELEGATION_TOOL,
        create_paper_agent,
        model_visible_tool_names,
        register_scholar_harness_profile,
    )
    from app.agents.model_factory import (
        SCHOLAR_PROVIDER,
        ScholarChatModel,
        build_chat_model,
        resolve_model_name,
    )

_EXPORTS = {
    "agent_checkpoint_db_path": "app.agents.checkpointer",
    "open_checkpointer": "app.agents.checkpointer",
    "BUILTIN_HOST_TOOLS": "app.agents.harness",
    "DELEGATION_TOOL": "app.agents.harness",
    "create_paper_agent": "app.agents.harness",
    "model_visible_tool_names": "app.agents.harness",
    "register_scholar_harness_profile": "app.agents.harness",
    "SCHOLAR_PROVIDER": "app.agents.model_factory",
    "ScholarChatModel": "app.agents.model_factory",
    "build_chat_model": "app.agents.model_factory",
    "resolve_model_name": "app.agents.model_factory",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_path), name)


def __dir__() -> list[str]:
    return __all__
