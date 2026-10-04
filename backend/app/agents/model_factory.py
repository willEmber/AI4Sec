"""Chat model for the paper agent, built from the model's registry profile.

The profile (`app.services.llm_gateway`) decides the gateway and the protocol:
the base gateway speaks the Responses API, an extension gateway speaks
chat/completions. Either way the model is a `ScholarChatModel`, so the harness
profile and the tool surface do not depend on which one answers.

P0 protocol findings for the base gateway (verified against the configured gateway on 2026-09-17,
see `scripts/verify_agent_protocol.py`):

* ``POST {base}/chat/completions`` answers ``404 {"message": "not support"}``,
  so the ordinary OpenAI chat-completions path is unusable. Only
  ``POST {base}/responses`` exists, hence ``use_responses_api=True``.
* ``/responses`` *does* accept local ``{"type": "function", ...}`` tools and
  returns ``function_call`` items carrying a ``call_id`` and JSON arguments,
  so no bespoke tool-call protocol is needed — the stock ``langchain-openai``
  Responses implementation parses them into ``AIMessage.tool_calls``.
* Reasoning is on by default there; ``enable_thinking`` is not required and is
  therefore not sent. Reasoning items need not be echoed back on the next turn.
"""

from __future__ import annotations

import logging
from typing import Any

from collections.abc import Callable, Sequence

from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.services.llm_gateway import get_registry

logger = logging.getLogger("scholar.agents.model")

# Provider key this model reports to deepagents. It must not be "openai":
# deepagents resolves a `HarnessProfile` through `_get_ls_params()["ls_provider"]`
# and registering the paper agent's profile under the shared "openai" key would
# silently reconfigure every other OpenAI-compatible model in the process.
SCHOLAR_PROVIDER = "scholar_maas"


def _clean(value: str | None) -> str:
    """Strip whitespace / CRLF / quotes that leak from Windows .env files."""
    return (value or "").strip().strip('"').strip("'").strip("\r").strip()


def flatten_nullable(schema: Any) -> Any:
    """Rewrite `anyOf [X, {"type": "null"}]` as `X`, at any depth.

    That is what pydantic emits for `list[str] | None`, and Gemini rejects the
    whole request over it ("schema didn't specify the schema type field"). The
    parameter stays optional: it keeps its default and is not in `required`.
    A union of several real types is left alone.
    """
    if isinstance(schema, list):
        return [flatten_nullable(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    node = schema
    options = node.get("anyOf")
    if isinstance(options, list):
        real = [o for o in options if not (isinstance(o, dict) and o.get("type") == "null")]
        if len(real) == 1 and len(real) < len(options) and isinstance(real[0], dict):
            node = {**{k: v for k, v in node.items() if k != "anyOf"}, **real[0]}
    return {key: flatten_nullable(value) for key, value in node.items()}


class ScholarChatModel(ChatOpenAI):
    """`ChatOpenAI` reporting its own provider key.

    Subclassed to report `SCHOLAR_PROVIDER` and, for gateways that need it, to
    flatten nullable tool parameters; every other protocol detail is handled by
    the stock integration.
    """

    flatten_nullable_tool_params: bool = False

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        **kwargs: Any,
    ) -> Any:
        if self.flatten_nullable_tool_params:
            # Only the schema the model sees changes; the tool node still runs
            # the original tools.
            tools = [flatten_nullable(convert_to_openai_tool(tool)) for tool in tools]
        return super().bind_tools(tools, **kwargs)

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> Any:
        params = super()._get_ls_params(stop=stop, **kwargs)
        try:
            params["ls_provider"] = SCHOLAR_PROVIDER
        except TypeError:  # pragma: no cover — defensive, params is a TypedDict
            pass
        return params


def resolve_model_name(model: str = "") -> str:
    """Pick the requested model, else `AGENT_MODELNAME`, else the first agent model."""
    name = _clean(model) or get_registry().agent_default_model
    if not name:
        raise RuntimeError(
            "No model configured. Set AGENT_MODELNAME or THINKING_MODELNAME in .env, "
            "or pass an explicit model."
        )
    return name


def build_chat_model(
    model: str = "",
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    **kwargs: Any,
) -> ScholarChatModel:
    """Build the chat model the paper agent runs on.

    Args:
        model: Model id; empty picks the default agent model.
        temperature: Sampling temperature; `None` leaves the gateway default.
        max_tokens: Output-token cap; `None` leaves the gateway default.
        timeout: Per-request timeout in seconds; defaults to the configured
            `AGENT_REQUEST_TIMEOUT_SECONDS`.
        max_retries: Transport-level retries inside the OpenAI SDK.
        **kwargs: Passed through to `ChatOpenAI`.
    """
    settings = get_settings()
    name = resolve_model_name(model)
    profile = get_registry().profile(name)
    provider = profile.provider
    if not provider.configured:
        raise RuntimeError(
            f"The gateway serving {name!r} has no base URL or API key configured."
        )

    params: dict[str, Any] = {
        "model": profile.request_model,
        "base_url": provider.base_url,
        "api_key": provider.api_key,
        "use_responses_api": profile.uses_responses_api,
        "flatten_nullable_tool_params": profile.flatten_nullable_tool_params,
        "timeout": timeout if timeout is not None else float(settings.agent_request_timeout_seconds),
        "max_retries": max_retries if max_retries is not None else settings.agent_max_retries,
    }
    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None and max_tokens > 0:
        params["max_tokens"] = max_tokens
    params.update(kwargs)

    logger.info(
        "Agent model: %s via %s (%s, timeout=%ss, retries=%s)",
        name,
        params["base_url"],
        "responses api" if params["use_responses_api"] else "chat completions",
        params["timeout"],
        params["max_retries"],
    )
    return ScholarChatModel(**params)
