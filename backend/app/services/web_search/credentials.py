"""Web search provider keys, read in one place.

Every provider is configured as a pool: `.env` holds a comma-separated list per
provider, and the pool rotates past a key that is refused or out of credit.
The accepted variable names live on `AppSettings` (plural first, then the
singular spellings found in deployed files), so this module only splits.
"""

from __future__ import annotations

import re

from app.config import get_settings

TAVILY = "Tavily"
EXA = "Exa"
FIRECRAWL = "Firecrawl"

_SEPARATORS = re.compile(r"[,;\s]+")


def parse_key_list(raw: str | None) -> tuple[str, ...]:
    """Split a key list, dropping quotes, blanks and repeats but keeping order.

    Order matters: the first key is the one tried first, so an operator can put
    the paid key ahead of the free ones.
    """
    seen: dict[str, None] = {}
    for part in _SEPARATORS.split(raw or ""):
        key = part.strip().strip('"').strip("'")
        if key:
            seen.setdefault(key, None)
    return tuple(seen)


def provider_keys(provider: str) -> tuple[str, ...]:
    settings = get_settings()
    raw = {
        TAVILY: settings.tavily_api_keys,
        EXA: settings.exa_api_keys,
        FIRECRAWL: settings.firecrawl_api_keys,
    }.get(provider, "")
    return parse_key_list(raw)


def web_search_enabled() -> bool:
    return bool(get_settings().web_search_enabled)


def mask_key(key: str) -> str:
    """Enough of a key to tell two apart in a log line, never enough to use."""
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}…{key[-2:]}"
