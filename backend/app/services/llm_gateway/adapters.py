"""The two wire protocols `LLMService` speaks.

An adapter turns one chat call into a request body and reads the answer back
out: text, why generation stopped early, and token usage. Retries, timeouts and
the empty-answer rule stay in `LLMService`, so both protocols fail the same way.
"""

from __future__ import annotations

from typing import Any

from app.services.llm_gateway.registry import ModelProfile


class ResponsesAdapter:
    """OpenAI Responses API, as the Bailian MaaS gateway serves it."""

    path = "/responses"

    @staticmethod
    def build_payload(
        profile: ModelProfile,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        enable_thinking: bool,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": profile.request_model,
            "input": messages,
            "temperature": temperature,
        }
        if profile.sends_enable_thinking:
            # Thinking is on by default on the gateway; sending the flag keeps
            # the request explicit and is the only way to switch it off.
            payload["enable_thinking"] = bool(enable_thinking)
        if tools:
            payload["tools"] = tools
        if max_tokens and max_tokens > 0:
            payload["max_output_tokens"] = max_tokens
        return payload

    @staticmethod
    def extract_text(data: dict[str, Any]) -> str:
        """Pull final assistant text out of a Responses API payload.

        Thinking models return a ``reasoning`` item before the ``message`` item;
        tool calls may also appear. Only message text is returned.
        """
        content = ""
        for item in data.get("output", []) or []:
            if item.get("type") != "message":
                continue
            parts = item.get("content", []) or []
            if isinstance(parts, str):
                content += parts
                break
            for part in parts:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type", "")
                if ptype in {"output_text", "text"}:
                    content += part.get("text", "") or ""
                elif "text" in part and isinstance(part.get("text"), str):
                    content += part["text"]
            break

        if content:
            return content

        # Fallbacks seen on some gateways
        for key in ("output_text", "text"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val
        return ""

    @staticmethod
    def truncation_reason(data: dict[str, Any], *, max_tokens: int) -> str:
        """Return why generation stopped early, or ``""`` if it ended normally.

        The Responses API marks a budget-exhausted answer with
        ``status=incomplete`` plus ``incomplete_details.reason``; some gateways
        only set a per-item ``finish_reason``. When neither is present, output
        tokens landing exactly on the ceiling means the same thing, so that is
        the last check rather than the first.
        """
        details = data.get("incomplete_details")
        if isinstance(details, dict):
            reason = str(details.get("reason") or "").strip()
            if reason:
                return reason
        if str(data.get("status") or "").strip() == "incomplete":
            return "incomplete"
        for item in data.get("output", []) or []:
            if not isinstance(item, dict):
                continue
            finish = str(item.get("finish_reason") or "").strip()
            if finish and finish != "stop":
                return finish
        if max_tokens > 0:
            produced = usage_tokens(data)[1]
            if isinstance(produced, int) and produced >= max_tokens:
                return "max_output_tokens"
        return ""


class ChatCompletionsAdapter:
    """OpenAI chat/completions, as New-API serves every upstream."""

    path = "/chat/completions"

    @staticmethod
    def build_payload(
        profile: ModelProfile,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        enable_thinking: bool,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        # No thinking parameter: these models think by default, and the one
        # portable switch (`reasoning_effort`) is rejected at "none" by Gemini.
        payload: dict[str, Any] = {
            "model": profile.request_model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
        if max_tokens and max_tokens > 0:
            # Reasoning is billed against this ceiling here too.
            payload["max_tokens"] = max_tokens
        return payload

    @staticmethod
    def extract_text(data: dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return ""
        content = (choices[0].get("message") or {}).get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                part.get("text", "") or ""
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        return ""

    @staticmethod
    def truncation_reason(data: dict[str, Any], *, max_tokens: int) -> str:
        choices = data.get("choices") or []
        if choices and isinstance(choices[0], dict):
            finish = str(choices[0].get("finish_reason") or "").strip()
            if finish and finish not in {"stop", "tool_calls"}:
                return finish
        if max_tokens > 0:
            produced = usage_tokens(data)[1]
            if isinstance(produced, int) and produced >= max_tokens:
                return "max_tokens"
        return ""


def usage_tokens(data: dict[str, Any]) -> tuple[Any, Any]:
    """`(input, output)` token counts under either protocol's field names.

    New-API returns both spellings and leaves the Responses ones at 0, so a
    present-but-zero field must not hide the populated one.
    """
    usage = data.get("usage", {}) or {}
    return (
        usage.get("input_tokens") or usage.get("prompt_tokens"),
        usage.get("output_tokens") or usage.get("completion_tokens"),
    )


def adapter_for(profile: ModelProfile) -> type[ResponsesAdapter] | type[ChatCompletionsAdapter]:
    return ResponsesAdapter if profile.uses_responses_api else ChatCompletionsAdapter
