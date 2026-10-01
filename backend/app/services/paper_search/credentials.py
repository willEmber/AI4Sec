"""Provider credentials, read in one place.

Every module that talks to Semantic Scholar or OpenAlex used to read its own
environment variable, and when the S2 key was renamed half of them kept sending
the old one — S2 answers a pruned key with 403, which the search fan-out then
swallowed as "no results". Reading the keys here means a rename is one edit.
"""

from __future__ import annotations

import os

# The canonical names come first; the others are spellings already found in
# deployed `.env` files and are accepted so a key is never silently ignored.
_S2_KEY_VARS = ("S2_API_KEY", "S2_API_Key")
_OPENALEX_KEY_VARS = ("OPENALEX_KEY", "OPENALEX_API_KEY")
_OPENREVIEW_USER_VARS = ("OPENREVIEW_USERNAME",)
_OPENREVIEW_PASSWORD_VARS = ("OPENREVIEW_PASSWORD",)


def _first_env(names: tuple[str, ...]) -> str:
    for name in names:
        value = (os.getenv(name) or "").strip().strip('"').strip("'")
        if value:
            return value
    return ""


def s2_api_key() -> str:
    return _first_env(_S2_KEY_VARS)


def s2_headers() -> dict[str, str]:
    key = s2_api_key()
    return {"x-api-key": key} if key else {}


def openalex_api_key() -> str:
    return _first_env(_OPENALEX_KEY_VARS)


def openalex_auth_params(mailto: str = "") -> dict[str, str]:
    """Query parameters that identify us to OpenAlex.

    OpenAlex ignores `mailto` now and meters by API key: without one the daily
    budget is a tenth of the keyed one (about a hundred searches). `mailto` is
    still sent when there is no key, since it costs nothing.
    """
    key = openalex_api_key()
    if key:
        return {"api_key": key}
    return {"mailto": mailto} if mailto else {}


def openreview_credentials() -> tuple[str, str]:
    return _first_env(_OPENREVIEW_USER_VARS), _first_env(_OPENREVIEW_PASSWORD_VARS)
