"""Request-scoped identity.

The identity a handler acts on comes from here and nowhere else. A request may
carry a `user_id`, an `owner_token` or any other self-declared field; none of
them is trusted. Only a credential this server issued resolves to a principal
(development plan §6; accounts in P7.5 I2).

Where a credential is looked for, in order:

1. the `scholar_auth` cookie — HttpOnly, so page scripts cannot read it; it
   holds either a login session (`st_…`) or an anonymous credential;
2. the `X-Agent-Token` header — how clients before accounts sent the anonymous
   credential from localStorage. A valid one is copied into the cookie, so
   those visitors move over without noticing;
3. `?token=` — the old event-stream form, kept for clients mid-upgrade.

In `single_user` mode there is nothing to check: every request is the local
principal, and an anonymous credential an old browser still sends is merged
into it once, so nothing made before the switch goes missing.
"""

from __future__ import annotations

import logging

from fastapi import HTTPException, Request, Response

from app.config import get_settings
from app.services import accounts, identity
from app.services.accounts import Caller

logger = logging.getLogger("scholar.api.deps")

AUTH_COOKIE = "scholar_auth"
# Anonymous credentials do not expire on their own; the cookie carrying one
# lives a year and is refreshed whenever it is re-issued.
_ANONYMOUS_COOKIE_SECONDS = 365 * 86400


def set_auth_cookie(response: Response, value: str, *, max_age: int) -> None:
    response.set_cookie(
        AUTH_COOKIE,
        value,
        max_age=max_age,
        path="/api",
        httponly=True,
        samesite="lax",
        secure=get_settings().auth_cookie_secure,
    )


def clear_auth_cookie(response: Response) -> None:
    response.delete_cookie(
        AUTH_COOKIE,
        path="/api",
        httponly=True,
        samesite="lax",
        secure=get_settings().auth_cookie_secure,
    )


def _presented(request: Request) -> list[tuple[str, str]]:
    return [
        (request.cookies.get(AUTH_COOKIE, ""), "cookie"),
        (request.headers.get(identity.AGENT_TOKEN_HEADER, ""), "header"),
        (request.query_params.get("token", ""), "query"),
    ]


async def resolve_caller(request: Request) -> Caller | None:
    """Who the request is, or `None` when it proves nobody."""
    settings = get_settings()
    if settings.auth_mode == "single_user":
        for value, _ in _presented(request):
            if value and not accounts.is_login_token(value):
                anonymous_id = await identity.resolve_principal(value)
                if anonymous_id:
                    await accounts.merge_anonymous(anonymous_id, accounts.LOCAL_PRINCIPAL_ID)
        return Caller(accounts.LOCAL_PRINCIPAL_ID, "user", "single_user")

    for value, via in _presented(request):
        value = (value or "").strip()
        if not value:
            continue
        if accounts.is_login_token(value):
            principal_id = await accounts.resolve_login_session(value)
            if principal_id:
                return Caller(principal_id, "user", via)
            continue
        principal_id = await identity.resolve_principal(value)
        if principal_id and settings.auth_allow_anonymous:
            return Caller(principal_id, "anonymous", via)
    return None


def _unauthenticated() -> HTTPException:
    return HTTPException(status_code=401, detail={"code": "login_required", "message": "Login required."})


def adopt_header_credential(request: Request, response: Response, caller: Caller) -> None:
    """A pre-accounts client sent its credential as a header: from now on the
    browser carries it as the cookie."""
    if caller.kind == "anonymous" and caller.via == "header":
        set_auth_cookie(
            response,
            request.headers.get(identity.AGENT_TOKEN_HEADER, "").strip(),
            max_age=_ANONYMOUS_COOKIE_SECONDS,
        )


async def require_caller(request: Request, response: Response) -> Caller:
    """The caller, or 401. For every route that reads existing data."""
    caller = await resolve_caller(request)
    if caller is None:
        raise _unauthenticated()
    adopt_header_credential(request, response, caller)
    return caller


async def require_principal(request: Request, response: Response) -> str:
    return (await require_caller(request, response)).principal_id


async def optional_caller(request: Request) -> Caller | None:
    """The caller if there is one; never mints, never rejects."""
    return await resolve_caller(request)


async def caller_or_new(request: Request, response: Response) -> Caller:
    """The caller, minting an anonymous principal for a first visit.

    Used by routes that may legitimately be someone's first contact (creating a
    session, uploading, starting a mode run). When anonymous use is off, a
    visitor who has not logged in gets 401 instead.
    """
    caller = await resolve_caller(request)
    if caller is not None:
        adopt_header_credential(request, response, caller)
        return caller
    if not get_settings().auth_allow_anonymous:
        raise _unauthenticated()
    principal_id, credential = await identity.create_principal()
    set_auth_cookie(response, credential, max_age=_ANONYMOUS_COOKIE_SECONDS)
    # Also in a header, for clients that still keep it themselves. Never in a
    # body, where it would land in logs and caches beside ordinary data.
    response.headers[identity.AGENT_TOKEN_HEADER] = credential
    logger.info("Issued a new anonymous principal %s", principal_id)
    return Caller(principal_id, "anonymous", "new")


async def principal_or_new(request: Request, response: Response) -> str:
    return (await caller_or_new(request, response)).principal_id
