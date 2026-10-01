"""Login, logout and "who am I" (P7.5 I2).

The OAuth round trip keeps its state in a short-lived signed cookie
(`scholar_oauth`) rather than in the database: the `state` the provider echoes,
the PKCE verifier, where to go afterwards, and the anonymous principal the
visitor was before logging in. The callback merges that principal into the
account, which is how a conversation started before login is still there after
it — and because the cookie is sealed, a callback cannot name somebody else's
anonymous principal to merge.
"""

from __future__ import annotations

import hmac
import logging
import secrets
from typing import Any
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from app.api.deps import (
    AUTH_COOKIE,
    adopt_header_credential,
    clear_auth_cookie,
    optional_caller,
    resolve_caller,
    set_auth_cookie,
)
from app.config import get_settings
from app.rate_limit import limiter
from app.services import accounts, identity, oauth_providers
from app.services.accounts import Caller

logger = logging.getLogger("scholar.api.auth")

router = APIRouter(prefix="/auth", tags=["auth"])

OAUTH_COOKIE = "scholar_oauth"
_OAUTH_COOKIE_PATH = "/api/auth"
_OAUTH_TTL_SECONDS = 600


class LoginRequest(BaseModel):
    next: str = "/"


def safe_next(target: str) -> str:
    """A same-site path to return to; anything else becomes `/`.

    `//host` and `/\\host` are protocol-relative to a browser, so an open
    redirect hides behind a leading slash unless both are refused.
    """
    target = (target or "").strip()
    if not target.startswith("/") or target.startswith("//") or "\\" in target:
        return "/"
    return target[:500]


def _providers() -> list[dict[str, str]]:
    return [{"id": p.id, "label": p.label} for p in oauth_providers.enabled_providers()]


@router.get("/providers")
async def providers() -> dict[str, Any]:
    settings = get_settings()
    return {
        "mode": settings.auth_mode,
        "allow_anonymous": settings.auth_allow_anonymous,
        "providers": _providers(),
    }


@router.get("/me")
async def me(
    request: Request,
    response: Response,
    caller: Caller | None = Depends(optional_caller),
) -> dict[str, Any]:
    """The caller, their account if any, and today's quota. Never mints.

    A legacy header credential is copied into the cookie here too: this is the
    first call a page makes, and the client forgets its stored copy once this
    answers with a principal.
    """
    settings = get_settings()
    body: dict[str, Any] = {
        "mode": settings.auth_mode,
        "allow_anonymous": settings.auth_allow_anonymous,
        "providers": _providers(),
        "authenticated": False,
        "principal_id": None,
        "kind": None,
        "user": None,
        "quota": None,
    }
    if caller is None:
        return body
    adopt_header_credential(request, response, caller)
    body.update(principal_id=caller.principal_id, kind=caller.kind)
    if caller.kind == "user":
        user = await accounts.get_user(caller.principal_id)
        if user is not None:
            body["authenticated"] = True
            body["user"] = {
                "display_name": user["display_name"],
                "email": user["email"],
                "avatar_url": user["avatar_url"],
                "role": user["role"],
            }
    body["quota"] = (await accounts.daily_usage(caller)).as_dict()
    return body


@router.post("/login/{provider}")
@limiter.limit("20/minute")
async def login(
    request: Request, response: Response, provider: str, body: LoginRequest
) -> dict[str, str]:
    """Start a login: returns the provider URL to send the browser to."""
    if get_settings().auth_mode == "single_user":
        raise HTTPException(status_code=400, detail="Login is disabled in single_user mode.")
    spec = oauth_providers.get_provider(provider)
    if spec is None:
        raise HTTPException(status_code=404, detail="Unknown or unconfigured login provider.")

    caller = await resolve_caller(request)
    state = secrets.token_urlsafe(24)
    verifier, challenge = oauth_providers.new_pkce()
    sealed = identity.seal(
        "oauth",
        {
            "provider": spec.id,
            "state": state,
            "verifier": verifier,
            "next": safe_next(body.next),
            "anon": caller.principal_id if caller and caller.kind == "anonymous" else "",
        },
        ttl_seconds=_OAUTH_TTL_SECONDS,
    )
    response.set_cookie(
        OAUTH_COOKIE,
        sealed,
        max_age=_OAUTH_TTL_SECONDS,
        path=_OAUTH_COOKIE_PATH,
        httponly=True,
        # Lax, not Strict: the callback is a top-level navigation coming back
        # from the provider's site, which Strict would strip the cookie from.
        samesite="lax",
        secure=get_settings().auth_cookie_secure,
    )
    return {"authorize_url": oauth_providers.authorize_url(spec, state=state, code_challenge=challenge)}


def _back_to(target: str, error: str = "") -> RedirectResponse:
    if error:
        sep = "&" if "?" in target else "?"
        target = f"{target}{sep}{urlencode({'auth_error': error})}"
    response = RedirectResponse(target, status_code=302)
    response.delete_cookie(OAUTH_COOKIE, path=_OAUTH_COOKIE_PATH)
    return response


@router.get("/callback/{provider}")
@limiter.limit("20/minute")
async def callback(
    request: Request,
    provider: str,
    code: str = "",
    state: str = "",
    error: str = "",
) -> RedirectResponse:
    """Where the provider sends the browser back. Always redirects into the app."""
    pending = identity.unseal("oauth", request.cookies.get(OAUTH_COOKIE, ""))
    if not pending or pending.get("provider") != provider:
        return _back_to("/", "login_expired")
    target = safe_next(str(pending.get("next") or "/"))
    if error:
        return _back_to(target, "denied" if error == "access_denied" else "provider_error")
    if not code or not hmac.compare_digest(state, str(pending.get("state") or "")):
        return _back_to(target, "state_mismatch")
    spec = oauth_providers.get_provider(provider)
    if spec is None:
        return _back_to(target, "provider_disabled")

    try:
        token = await oauth_providers.exchange_code(
            spec, code=code, code_verifier=str(pending.get("verifier") or "")
        )
        profile = await oauth_providers.fetch_profile(spec, token)
        principal_id = await accounts.upsert_user(profile)
    except oauth_providers.OAuthError as exc:
        logger.warning("Login via %s failed: %s", provider, exc)
        return _back_to(target, exc.code)
    except accounts.AccountDisabled:
        return _back_to(target, "account_disabled")

    anonymous_id = str(pending.get("anon") or "")
    if anonymous_id:
        await accounts.merge_anonymous(anonymous_id, principal_id)
    session_token = await accounts.create_login_session(
        principal_id, user_agent=request.headers.get("user-agent", "")
    )
    response = _back_to(target)
    set_auth_cookie(
        response, session_token, max_age=get_settings().auth_session_days * 86400
    )
    logger.info("Login via %s as %s", provider, principal_id)
    return response


@router.post("/logout")
async def logout(request: Request, response: Response) -> dict[str, bool]:
    """End the login session. The browser becomes a new visitor."""
    token = request.cookies.get(AUTH_COOKIE, "")
    await accounts.revoke_login_session(token)
    clear_auth_cookie(response)
    return {"ok": True}
