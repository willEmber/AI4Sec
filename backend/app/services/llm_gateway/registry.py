"""Model registry: one base gateway plus optional extension gateways.

The base gateway (`LLM_BASEURL`, Alibaba Bailian) stays the only source of
rerank, embeddings and every *utility* call — intent classification, digest and
evidence extraction, rank extraction, context summarisation. Those outputs are
consumed by code, their prompts and budgets were tuned against one model, and a
failure there degrades silently, so they do not follow the reader's choice.

Utility calls also run with thinking off (`enable_thinking=False` at each call
site). Measured on a Lens run: with thinking, the evidence pass was cut off at
its 8192-token ceiling on qwen3.8-max and qwen3.8-flash alike and the digest
came back empty at 12288; without it, both finished in under a minute.

An extension gateway (New-API) only adds chat models the reader may pick for a
report or a conversation. If it is down, the runs that picked one of its models
fail and nothing else does.

The base gateway is configured in the environment, as it always was
(`LLM_BASEURL`, `LLM_APIKEY`, `THINKING_MODELNAME`, `BASE_MODELNAME`).
Extension gateways and their models are declared in `backend/models.toml`
(`LLM_REGISTRY_FILE` overrides the path), which names the environment variables
holding each gateway's URL and key rather than the secrets themselves. Without
that file there is only the base gateway.

Protocol findings behind the profile fields (2026-10-04, see
DEV_DOC/P10_模型网关与多模型接入.md):

* Bailian serves `/responses` only; New-API answers `/responses` with
  "not implemented" for Gemini channels and serves `/chat/completions`.
* Gemini rejects a tool parameter declared as `anyOf [X, null]`, which is what
  every `list[str] | None` argument becomes, so those are flattened to `X`.
* `enable_thinking` is a Bailian parameter. It is sent there and nowhere else.
"""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from app.config import get_settings

logger = logging.getLogger("scholar.llm.registry")

RESPONSES = "responses"
CHAT_COMPLETIONS = "chat_completions"
PROTOCOLS = (RESPONSES, CHAT_COMPLETIONS)

BASE_PROVIDER = "base"

_BACKEND_DIR = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY_FILE = _BACKEND_DIR / "models.toml"
_ENV_FILE = _BACKEND_DIR.parent / ".env"

# Parsed registry file by (path, mtime): the registry is rebuilt per call so a
# settings change is seen at once, but the file is read only when it changes.
_file_cache: dict[tuple[str, float], dict[str, Any]] = {}
_warned: set[str] = set()


class RegistryConfigError(RuntimeError):
    """`models.toml` is malformed. Raised rather than skipped: a typo there would
    otherwise make a model silently disappear from the list."""


def _clean(value: str | None) -> str:
    """Strip whitespace / CRLF / quotes that leak from Windows .env files."""
    return (value or "").strip().strip('"').strip("'").strip("\r").strip()


def _names(value: str | None) -> list[str]:
    seen: list[str] = []
    for part in (value or "").split(","):
        name = _clean(part)
        if name and name not in seen:
            seen.append(name)
    return seen


def _warn_once(message: str) -> None:
    if message not in _warned:
        _warned.add(message)
        logger.warning(message)


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    api_key: str
    protocol: str

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key)


@dataclass(frozen=True)
class ModelProfile:
    # The name readers see and sessions store.
    name: str
    provider: Provider
    # The name the gateway knows the model by; the same unless aliased.
    upstream_id: str = ""
    # Whether `enable_thinking` is part of this model's request vocabulary.
    sends_enable_thinking: bool = False
    flatten_nullable_tool_params: bool = False
    # Offered for conversations; a model is opened to the agent only after it
    # has been verified against the real tool surface.
    agent: bool = False

    @property
    def request_model(self) -> str:
        return self.upstream_id or self.name

    @property
    def uses_responses_api(self) -> bool:
        return self.provider.protocol == RESPONSES


@dataclass(frozen=True)
class Registry:
    base: Provider
    models: dict[str, ModelProfile] = field(default_factory=dict)
    default_model: str = ""
    agent_default_model: str = ""
    utility_model: str = ""

    def selectable_models(self) -> list[str]:
        """Models a reader may pick for a report, in configured order."""
        return list(self.models)

    def agent_models(self) -> list[str]:
        """Models a reader may pick for a conversation."""
        return [name for name, profile in self.models.items() if profile.agent]

    def profile(self, model: str = "") -> ModelProfile:
        """Profile for `model`, or for the default when it is empty.

        A name nobody registered is served by the base gateway, as every name
        was before there was a registry: the utility model is not selectable,
        and scripts pass models that are not in the list.
        """
        name = _clean(model) or self.default_model
        known = self.models.get(name)
        if known is not None:
            return known
        return _profile(name, self.base, {})


def _profile(name: str, provider: Provider, spec: dict[str, Any], *, agent: bool = False) -> ModelProfile:
    """A model's profile: protocol defaults, overridden by what its entry says."""
    chat_completions = provider.protocol == CHAT_COMPLETIONS
    return ModelProfile(
        name=name,
        provider=provider,
        upstream_id=_clean(str(spec.get("upstream_id") or "")),
        sends_enable_thinking=bool(spec.get("enable_thinking_param", not chat_completions)),
        flatten_nullable_tool_params=bool(spec.get("flatten_nullable_tool_params", chat_completions)),
        agent=bool(spec.get("agent", agent)),
    )


def _env_value(name: str) -> str:
    """An environment variable, or its value in the project's `.env`.

    Settings load `.env` without exporting it, so a variable a registry entry
    names is usually not in `os.environ`.
    """
    if not name:
        return ""
    value = os.environ.get(name)
    if value is None and _ENV_FILE.is_file():
        value = dotenv_values(_ENV_FILE).get(name)
    return _clean(value)


def _v1(base_url: str) -> str:
    """Gateways are often configured by origin; the OpenAI routes live under /v1."""
    url = _clean(base_url).rstrip("/")
    if not url or url.endswith("/v1"):
        return url
    return f"{url}/v1"


def _load_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    key = (str(path), path.stat().st_mtime)
    cached = _file_cache.get(key)
    if cached is None:
        try:
            cached = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise RegistryConfigError(f"{path}: {exc}") from exc
        _file_cache.clear()
        _file_cache[key] = cached
    return cached


def _extension_providers(data: dict[str, Any], path: Path) -> dict[str, Provider]:
    providers: dict[str, Provider] = {}
    for name, spec in (data.get("providers") or {}).items():
        if name == BASE_PROVIDER:
            raise RegistryConfigError(
                f"{path}: provider {BASE_PROVIDER!r} is the gateway configured by LLM_BASEURL "
                "and cannot be redefined"
            )
        protocol = str(spec.get("protocol") or CHAT_COMPLETIONS)
        if protocol not in PROTOCOLS:
            raise RegistryConfigError(
                f"{path}: provider {name!r} has unknown protocol {protocol!r}; "
                f"use one of {', '.join(PROTOCOLS)}"
            )
        base_url = _env_value(str(spec.get("base_url_env") or "")) or str(spec.get("base_url") or "")
        providers[name] = Provider(
            name=name,
            base_url=_v1(base_url),
            api_key=_env_value(str(spec.get("api_key_env") or "")),
            protocol=protocol,
        )
    return providers


def get_registry() -> Registry:
    """Build the registry from settings and the registry file."""
    settings = get_settings()
    base = Provider(
        name=BASE_PROVIDER,
        base_url=_clean(settings.llm_base_url).rstrip("/"),
        api_key=_clean(settings.llm_api_key),
        protocol=RESPONSES,
    )
    models: dict[str, ModelProfile] = {
        name: _profile(name, base, {}, agent=True) for name in _names(settings.thinking_model)
    }

    path = Path(_clean(getattr(settings, "llm_registry_file", "")) or DEFAULT_REGISTRY_FILE)
    data = _load_file(path)
    providers = {BASE_PROVIDER: base, **_extension_providers(data, path)}
    for name, spec in (data.get("models") or {}).items():
        provider_name = str(spec.get("provider") or "")
        provider = providers.get(provider_name)
        if provider is None:
            raise RegistryConfigError(
                f"{path}: model {name!r} names provider {provider_name!r}, which is not declared"
            )
        if name in models:
            _warn_once(f"Model {name!r} is already served by the base gateway; its entry in {path.name} is ignored")
            continue
        if not provider.configured:
            _warn_once(
                f"Gateway {provider.name!r} has no base URL or API key; model {name!r} is not offered"
            )
            continue
        models[name] = _profile(name, provider, spec, agent=provider is base)

    default_model = next(iter(models), "")
    agent_default = _clean(settings.agent_model) or next(
        (name for name, profile in models.items() if profile.agent), default_model
    )
    return Registry(
        base=base,
        models=models,
        default_model=default_model,
        agent_default_model=agent_default,
        utility_model=_clean(settings.base_model) or default_model,
    )
