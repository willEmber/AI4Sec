"""Chat model for the paper agent, bound to the MaaS Responses gateway.

P0 protocol findings (verified against the configured gateway on 2026-09-17,
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

from langchain_openai import ChatOpenAI

from app.config import get_settings

logger = logging.getLogger("scholar.agents.model")

# Provider key this model reports to deepagents. It must not be "openai":
# deepagents resolves a `HarnessProfile` through `_get_ls_params()["ls_provider"]`
# and registering the paper agent's profile under the shared "openai" key would
# silently reconfigure every other OpenAI-compatible model in the process.
SCHOLAR_PROVIDER = "scholar_maas"


def _clean(value: str | None) -> str:
    """Strip whitespace / CRLF / quotes that leak from Windows .env files."""
    return (value or "").strip().strip('"').strip("'").strip("\r").strip()


class ScholarChatModel(ChatOpenAI):
    """`ChatOpenAI` pinned to the Responses endpoint and to its own provider key.

    Subclassed only to report `SCHOLAR_PROVIDER`; every protocol detail is
    handled by the stock integration.
    """

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> Any:
        params = super()._get_ls_params(stop=stop, **kwargs)
        try:
            params["ls_provider"] = SCHOLAR_PROVIDER
        except TypeError:  # pragma: no cover — defensive, params is a TypedDict
            pass
        return params


def resolve_model_name(model: str = "") -> str:
    """Pick the requested model, else `AGENT_MODELNAME`, else the first thinking model."""
    settings = get_settings()
    name = _clean(model) or _clean(settings.agent_model) or settings.default_thinking_model
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
        model: Model id; empty picks the first entry of `THINKING_MODELNAME`.
        temperature: Sampling temperature; `None` leaves the gateway default.
        max_tokens: Output-token cap; `None` leaves the gateway default.
        timeout: Per-request timeout in seconds; defaults to the configured
            `AGENT_REQUEST_TIMEOUT_SECONDS`.
        max_retries: Transport-level retries inside the OpenAI SDK.
        **kwargs: Passed through to `ChatOpenAI`.
    """
    settings = get_settings()
    base_url = _clean(settings.llm_base_url).rstrip("/")
    api_key = _clean(settings.llm_api_key)
    if not base_url or not api_key:
        raise RuntimeError("LLM_BASEURL and LLM_APIKEY must be set to build the agent model.")

    name = resolve_model_name(model)
    params: dict[str, Any] = {
        "model": name,
        "base_url": base_url,
        "api_key": api_key,
        "use_responses_api": True,
        "timeout": timeout if timeout is not None else float(settings.agent_request_timeout_seconds),
        "max_retries": max_retries if max_retries is not None else settings.agent_max_retries,
    }
    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None and max_tokens > 0:
        params["max_tokens"] = max_tokens
    params.update(kwargs)

    logger.info(
        "Agent model: %s via %s (responses api, timeout=%ss, retries=%s)",
        name,
        base_url,
        params["timeout"],
        params["max_retries"],
    )
    return ScholarChatModel(**params)
