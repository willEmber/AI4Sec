from __future__ import annotations

import contextlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
from langchain_core.messages import AIMessageChunk

from app.agents.model_factory import build_chat_model, flatten_nullable
from app.services.agent_runner import _new_usage
from app.services.llm_gateway import (
    CHAT_COMPLETIONS,
    RESPONSES,
    RegistryConfigError,
    get_registry,
)
from app.services.llm_gateway.adapters import (
    ChatCompletionsAdapter,
    ResponsesAdapter,
    usage_tokens,
)
from app.services.llm_service import LLMEmptyResponseError, LLMService


_REGISTRY_TOML = """
[providers.newapi]
base_url_env = "TEST_NEWAPI_BASEURL"
api_key_env = "TEST_NEWAPI_APIKEY"
protocol = "chat_completions"

[models."gemini-flash"]
provider = "newapi"

[models."gemini-pro"]
provider = "newapi"
"""

_ENV = {"TEST_NEWAPI_BASEURL": "https://newapi.test", "TEST_NEWAPI_APIKEY": "ext-key"}


def _settings(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "llm_base_url": "https://base.test/v1",
        "llm_api_key": "base-key",
        "thinking_model": "qwen-max, qwen-plus",
        "base_model": "qwen-flash",
        "agent_model": "",
        "llm_registry_file": "",
        "agent_request_timeout_seconds": 300,
        "agent_max_retries": 3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _Configured:
    """Settings plus a registry file and gateway credentials, for one `with` block."""

    def __init__(
        self, toml: str | None = _REGISTRY_TOML, env: dict[str, str] | None = None, **overrides: Any
    ) -> None:
        self._toml = toml
        self._env = _ENV if env is None else env
        self._overrides = overrides
        self._stack = contextlib.ExitStack()

    def __enter__(self) -> SimpleNamespace:
        directory = Path(self._stack.enter_context(tempfile.TemporaryDirectory()))
        path = directory / "models.toml"
        if self._toml is not None:
            path.write_text(self._toml, encoding="utf-8")
        settings = _settings(llm_registry_file=str(path), **self._overrides)
        self._stack.enter_context(
            patch("app.services.llm_gateway.registry.get_settings", lambda: settings)
        )
        # An absent variable must not be found in the developer's real .env.
        self._stack.enter_context(
            patch("app.services.llm_gateway.registry._ENV_FILE", directory / "no.env")
        )
        self._stack.enter_context(patch.dict(os.environ, self._env))
        return settings

    def __exit__(self, *exc: Any) -> None:
        self._stack.close()


def _with_settings(**overrides: Any) -> _Configured:
    return _Configured(**overrides)


class RegistryTests(unittest.TestCase):
    def test_without_a_registry_file_there_is_only_the_base_gateway(self) -> None:
        with _Configured(toml=None, base_model=""):
            registry = get_registry()
        self.assertEqual(registry.selectable_models(), ["qwen-max", "qwen-plus"])
        self.assertEqual(registry.agent_models(), ["qwen-max", "qwen-plus"])
        self.assertEqual(registry.default_model, "qwen-max")
        self.assertEqual(registry.utility_model, "qwen-max")
        profile = registry.profile("")
        self.assertEqual(profile.name, "qwen-max")
        self.assertEqual(profile.provider.protocol, RESPONSES)
        self.assertTrue(profile.sends_enable_thinking)

    def test_extension_models_follow_the_base_ones_and_are_report_only(self) -> None:
        with _Configured():
            registry = get_registry()
        self.assertEqual(
            registry.selectable_models(), ["qwen-max", "qwen-plus", "gemini-flash", "gemini-pro"]
        )
        self.assertEqual(registry.agent_models(), ["qwen-max", "qwen-plus"])
        profile = registry.profile("gemini-flash")
        self.assertEqual(profile.provider.protocol, CHAT_COMPLETIONS)
        self.assertEqual(profile.provider.base_url, "https://newapi.test/v1")
        self.assertEqual(profile.provider.api_key, "ext-key")
        self.assertFalse(profile.sends_enable_thinking)
        self.assertTrue(profile.flatten_nullable_tool_params)
        self.assertEqual(profile.request_model, "gemini-flash")

    def test_agent_flag_opens_an_extension_model_for_conversations(self) -> None:
        toml = _REGISTRY_TOML.replace(
            '[models."gemini-flash"]\nprovider = "newapi"',
            '[models."gemini-flash"]\nprovider = "newapi"\nagent = true',
        )
        with _Configured(toml=toml):
            registry = get_registry()
        self.assertEqual(registry.agent_models(), ["qwen-max", "qwen-plus", "gemini-flash"])

    def test_model_entry_can_alias_the_upstream_name_and_override_defaults(self) -> None:
        toml = _REGISTRY_TOML + (
            '\n[models."Fast"]\nprovider = "newapi"\nupstream_id = "vendor/fast-001"\n'
            "flatten_nullable_tool_params = false\n"
        )
        with _Configured(toml=toml):
            profile = get_registry().profile("Fast")
        self.assertEqual(profile.request_model, "vendor/fast-001")
        self.assertFalse(profile.flatten_nullable_tool_params)

    def test_utility_model_is_served_by_the_base_gateway_and_not_offered(self) -> None:
        with _Configured():
            registry = get_registry()
        self.assertEqual(registry.utility_model, "qwen-flash")
        self.assertNotIn("qwen-flash", registry.selectable_models())
        self.assertEqual(registry.profile("qwen-flash").provider.base_url, "https://base.test/v1")

    def test_models_of_a_gateway_without_credentials_are_not_offered(self) -> None:
        with _Configured(env={"TEST_NEWAPI_BASEURL": "https://newapi.test"}):
            registry = get_registry()
        self.assertEqual(registry.selectable_models(), ["qwen-max", "qwen-plus"])

    def test_a_name_the_base_gateway_already_serves_stays_there(self) -> None:
        toml = _REGISTRY_TOML + '\n[models."qwen-max"]\nprovider = "newapi"\n'
        with _Configured(toml=toml):
            registry = get_registry()
        self.assertEqual(registry.profile("qwen-max").provider.protocol, RESPONSES)

    def test_base_url_already_ending_in_v1_is_kept(self) -> None:
        env = {**_ENV, "TEST_NEWAPI_BASEURL": "https://newapi.test/v1/"}
        with _Configured(env=env):
            registry = get_registry()
        self.assertEqual(registry.profile("gemini-flash").provider.base_url, "https://newapi.test/v1")

    def test_undeclared_provider_is_an_error_not_a_missing_model(self) -> None:
        with _Configured(toml='[models."x"]\nprovider = "nowhere"\n'):
            with self.assertRaises(RegistryConfigError):
                get_registry()

    def test_unknown_protocol_and_malformed_file_are_errors(self) -> None:
        with _Configured(toml='[providers.p]\nprotocol = "grpc"\n'):
            with self.assertRaises(RegistryConfigError):
                get_registry()
        with _Configured(toml="[providers.p\n"):
            with self.assertRaises(RegistryConfigError):
                get_registry()


class AdapterTests(unittest.TestCase):
    def _profiles(self):
        with _with_settings():
            registry = get_registry()
        return registry.profile("qwen-max"), registry.profile("gemini-flash")

    def test_responses_payload_keeps_the_base_gateway_vocabulary(self) -> None:
        base, _ = self._profiles()
        payload = ResponsesAdapter.build_payload(
            base,
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.2,
            max_tokens=100,
            enable_thinking=True,
            tools=None,
        )
        self.assertEqual(payload["input"], [{"role": "user", "content": "hi"}])
        self.assertEqual(payload["max_output_tokens"], 100)
        self.assertIs(payload["enable_thinking"], True)

    def test_chat_completions_payload_sends_no_thinking_parameter(self) -> None:
        _, ext = self._profiles()
        payload = ChatCompletionsAdapter.build_payload(
            ext,
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.2,
            max_tokens=100,
            enable_thinking=True,
            tools=None,
        )
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(payload["max_tokens"], 100)
        self.assertNotIn("enable_thinking", payload)
        self.assertNotIn("reasoning_effort", payload)
        self.assertNotIn("input", payload)

    def test_chat_completions_reads_text_and_truncation(self) -> None:
        data = {
            "choices": [{"finish_reason": "length", "message": {"content": "partial"}}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 196},
        }
        self.assertEqual(ChatCompletionsAdapter.extract_text(data), "partial")
        self.assertEqual(ChatCompletionsAdapter.truncation_reason(data, max_tokens=200), "length")

    def test_usage_is_read_past_zeroed_fields_of_the_other_protocol(self) -> None:
        data = {
            "choices": [{"finish_reason": "stop", "message": {"content": "x"}}],
            "usage": {"input_tokens": 0, "output_tokens": 0, "prompt_tokens": 9, "completion_tokens": 200},
        }
        self.assertEqual(usage_tokens(data), (9, 200))
        self.assertEqual(ChatCompletionsAdapter.truncation_reason(data, max_tokens=200), "max_tokens")

    def test_chat_completions_normal_stop_is_not_truncation(self) -> None:
        data = {
            "choices": [{"finish_reason": "stop", "message": {"content": "done"}}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 20},
        }
        self.assertEqual(ChatCompletionsAdapter.truncation_reason(data, max_tokens=200), "")


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers: dict[str, str] = {}
        self.text = str(payload)

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error", request=httpx.Request("POST", "https://x.test"), response=self  # type: ignore[arg-type]
            )


class _RecordingClient:
    calls: list[dict[str, Any]] = []
    response: _FakeResponse = _FakeResponse(200, {})

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> "_RecordingClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]) -> _FakeResponse:
        type(self).calls.append({"url": url, "headers": headers, "json": json})
        return type(self).response


class LLMServiceRoutingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        _RecordingClient.calls = []

    async def _chat(self, model: str, response: _FakeResponse) -> str:
        _RecordingClient.response = response
        with _with_settings(), patch(
            "app.services.llm_service.httpx.AsyncClient", _RecordingClient
        ):
            return await LLMService(retry_base_delay=0.0).chat(
                [{"role": "user", "content": "hi"}], model=model, max_tokens=100
            )

    async def test_extension_model_goes_to_its_own_gateway_over_chat_completions(self) -> None:
        text = await self._chat(
            "gemini-flash",
            _FakeResponse(200, {"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}),
        )
        self.assertEqual(text, "ok")
        call = _RecordingClient.calls[0]
        self.assertEqual(call["url"], "https://newapi.test/v1/chat/completions")
        self.assertEqual(call["headers"]["Authorization"], "Bearer ext-key")
        self.assertEqual(call["json"]["model"], "gemini-flash")

    async def test_base_and_utility_models_stay_on_the_base_gateway(self) -> None:
        answer = _FakeResponse(
            200, {"output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]}
        )
        for model in ("qwen-max", "qwen-flash", ""):
            _RecordingClient.calls = []
            await self._chat(model, answer)
            call = _RecordingClient.calls[0]
            self.assertEqual(call["url"], "https://base.test/v1/responses")
            self.assertEqual(call["headers"]["Authorization"], "Bearer base-key")

    async def test_empty_chat_completions_answer_raises_with_the_reason(self) -> None:
        with self.assertRaises(LLMEmptyResponseError) as ctx:
            await self._chat(
                "gemini-flash",
                _FakeResponse(
                    200,
                    {
                        "choices": [{"finish_reason": "length", "message": {"content": ""}}],
                        "usage": {"completion_tokens": 100},
                    },
                ),
            )
        self.assertEqual(ctx.exception.reason, "length")

    async def test_model_not_found_is_not_retried_despite_its_503(self) -> None:
        with self.assertRaises(httpx.HTTPStatusError):
            await self._chat(
                "gemini-flash",
                _FakeResponse(503, {"error": {"code": "model_not_found", "message": "no channel"}}),
            )
        self.assertEqual(len(_RecordingClient.calls), 1)


class FlattenNullableTests(unittest.TestCase):
    def test_nullable_list_becomes_a_plain_list_and_keeps_its_default(self) -> None:
        schema = {"anyOf": [{"items": {"type": "string"}, "type": "array"}, {"type": "null"}], "default": None}
        self.assertEqual(
            flatten_nullable(schema),
            {"default": None, "items": {"type": "string"}, "type": "array"},
        )

    def test_nested_properties_are_flattened(self) -> None:
        tool = {
            "type": "function",
            "function": {
                "name": "t",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "q": {"type": "string"},
                        "year": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None},
                    },
                    "required": ["q"],
                },
            },
        }
        flat = flatten_nullable(tool)
        self.assertEqual(
            flat["function"]["parameters"]["properties"]["year"], {"default": None, "type": "integer"}
        )
        self.assertEqual(flat["function"]["parameters"]["required"], ["q"])

    def test_union_of_real_types_is_left_alone(self) -> None:
        schema = {"anyOf": [{"type": "integer"}, {"type": "string"}]}
        self.assertEqual(flatten_nullable(schema), schema)
        nullable_union = {"anyOf": [{"type": "integer"}, {"type": "string"}, {"type": "null"}]}
        self.assertEqual(flatten_nullable(nullable_union), nullable_union)


class BuildChatModelTests(unittest.TestCase):
    def test_protocol_and_flattening_come_from_the_profile(self) -> None:
        with _with_settings() as settings, patch(
            "app.agents.model_factory.get_settings", lambda: settings
        ):
            base = build_chat_model("qwen-max")
            ext = build_chat_model("gemini-flash")
        self.assertTrue(base.use_responses_api)
        self.assertFalse(base.flatten_nullable_tool_params)
        self.assertFalse(ext.use_responses_api)
        self.assertTrue(ext.flatten_nullable_tool_params)
        self.assertEqual(str(ext.openai_api_base), "https://newapi.test/v1")

    def test_flattening_reaches_the_tools_the_model_is_bound_to(self) -> None:
        def search(query: str, venues: list[str] | None = None) -> str:
            """Search papers."""
            return query

        with _with_settings() as settings, patch(
            "app.agents.model_factory.get_settings", lambda: settings
        ):
            bound = build_chat_model("gemini-flash").bind_tools([search])
        venues = bound.kwargs["tools"][0]["function"]["parameters"]["properties"]["venues"]
        self.assertNotIn("anyOf", venues)
        self.assertEqual(venues["type"], "array")


class UsageDeduplicationTests(unittest.TestCase):
    @staticmethod
    def _chunk(message_id: str | None, total: int) -> AIMessageChunk:
        return AIMessageChunk(
            content="",
            id=message_id,
            usage_metadata={"input_tokens": 0, "output_tokens": total, "total_tokens": total},
        )

    def test_usage_repeated_on_two_chunks_of_one_call_counts_once(self) -> None:
        counted: dict[str, int] = {}
        self.assertEqual(_new_usage(self._chunk("m1", 392), counted), (392, True))
        self.assertEqual(_new_usage(self._chunk("m1", 392), counted), (0, False))

    def test_each_call_is_counted(self) -> None:
        counted: dict[str, int] = {}
        self.assertEqual(_new_usage(self._chunk("m1", 100), counted), (100, True))
        self.assertEqual(_new_usage(self._chunk("m2", 250), counted), (250, True))

    def test_a_larger_later_total_adds_only_the_difference(self) -> None:
        counted: dict[str, int] = {}
        _new_usage(self._chunk("m1", 100), counted)
        self.assertEqual(_new_usage(self._chunk("m1", 130), counted), (30, False))

    def test_chunk_without_usage_counts_nothing(self) -> None:
        self.assertEqual(_new_usage(AIMessageChunk(content="x", id="m1"), {}), (0, False))


if __name__ == "__main__":
    unittest.main()
