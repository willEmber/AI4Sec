"""OAuth 2.0 login providers: GitHub, Google, LINUX DO Connect.

Authorization-code flow with PKCE (S256) for all three. The browser is sent to
the provider, comes back to `/api/auth/callback/{provider}` on the *public*
origin (`PUBLIC_BASE_URL`, proxied to the backend), and the code is exchanged
here, server side — the client secret and the access token never reach the
browser. The access token is used once, to read the profile, and not stored.

A provider is offered only when both its client id and secret are configured.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import secrets
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx

from app.config import get_settings
from app.services.accounts import ExternalProfile

logger = logging.getLogger("scholar.oauth")

_TIMEOUT = httpx.Timeout(15.0)
_USER_AGENT = "scholar-oauth"


class OAuthError(Exception):
    """A login that cannot complete. `code` is safe to show the user."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    label: str
    authorize_url: str
    token_url: str
    userinfo_url: str
    scope: str
    # How the client authenticates at the token endpoint: form fields
    # (`client_secret_post`) or HTTP Basic (`client_secret_basic`).
    token_auth: str = "post"


PROVIDERS: dict[str, ProviderSpec] = {
    "github": ProviderSpec(
        id="github",
        label="GitHub",
        authorize_url="https://github.com/login/oauth/authorize",
        token_url="https://github.com/login/oauth/access_token",
        userinfo_url="https://api.github.com/user",
        scope="read:user user:email",
    ),
    "google": ProviderSpec(
        id="google",
        label="Google",
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        userinfo_url="https://openidconnect.googleapis.com/v1/userinfo",
        scope="openid email profile",
    ),
    "linuxdo": ProviderSpec(
        id="linuxdo",
        label="LINUX DO",
        authorize_url="https://connect.linux.do/oauth2/authorize",
        token_url="https://connect.linux.do/oauth2/token",
        userinfo_url="https://connect.linux.do/api/user",
        scope="user",
        token_auth="basic",
    ),
}


def client_credentials(provider: str) -> tuple[str, str]:
    settings = get_settings()
    client_id = str(getattr(settings, f"oauth_{provider}_client_id", "") or "").strip()
    secret = str(getattr(settings, f"oauth_{provider}_client_secret", "") or "").strip()
    return client_id, secret


def enabled_providers() -> list[ProviderSpec]:
    return [spec for spec in PROVIDERS.values() if all(client_credentials(spec.id))]


def get_provider(provider: str) -> ProviderSpec | None:
    spec = PROVIDERS.get(provider)
    return spec if spec is not None and all(client_credentials(spec.id)) else None


def redirect_uri(provider: str) -> str:
    base = get_settings().public_base_url.rstrip("/")
    return f"{base}/api/auth/callback/{provider}"


def new_pkce() -> tuple[str, str]:
    """`(verifier, S256 challenge)` per RFC 7636."""
    verifier = secrets.token_urlsafe(48)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return verifier, base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def authorize_url(spec: ProviderSpec, *, state: str, code_challenge: str) -> str:
    client_id, _ = client_credentials(spec.id)
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri(spec.id),
        "response_type": "code",
        "scope": spec.scope,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    if spec.id == "google":
        params["prompt"] = "select_account"
    return f"{spec.authorize_url}?{urlencode(params)}"


async def exchange_code(spec: ProviderSpec, *, code: str, code_verifier: str) -> str:
    """Trade the callback's code for an access token."""
    client_id, secret = client_credentials(spec.id)
    form = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri(spec.id),
        "code_verifier": code_verifier,
    }
    auth: tuple[str, str] | None = None
    if spec.token_auth == "basic":
        auth = (client_id, secret)
    else:
        form.update(client_id=client_id, client_secret=secret)
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                spec.token_url,
                data=form,
                auth=auth,
                headers={"Accept": "application/json", "User-Agent": _USER_AGENT},
            )
    except httpx.HTTPError as exc:
        raise OAuthError("provider_unreachable", str(exc)) from exc
    try:
        body = resp.json()
    except ValueError:
        body = {}
    token = body.get("access_token") if isinstance(body, dict) else None
    if resp.status_code != 200 or not token:
        # GitHub answers 200 with `error` in the body; others use 4xx.
        reason = (body.get("error") if isinstance(body, dict) else "") or f"http_{resp.status_code}"
        logger.warning("Token exchange with %s failed: %s", spec.id, reason)
        raise OAuthError("token_exchange_failed", str(reason))
    return str(token)


async def _get_json(client: httpx.AsyncClient, url: str, token: str) -> Any:
    resp = await client.get(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": _USER_AGENT,
        },
    )
    if resp.status_code != 200:
        raise OAuthError("profile_unavailable", f"{url} answered {resp.status_code}")
    return resp.json()


async def fetch_profile(spec: ProviderSpec, access_token: str) -> ExternalProfile:
    """Who the token belongs to, normalised across providers."""
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            data = await _get_json(client, spec.userinfo_url, access_token)
            if not isinstance(data, dict):
                raise OAuthError("profile_unavailable", "profile is not an object")
            if spec.id == "github":
                return await _github_profile(client, data, access_token)
            if spec.id == "google":
                return _google_profile(data)
            return _linuxdo_profile(data)
    except httpx.HTTPError as exc:
        raise OAuthError("provider_unreachable", str(exc)) from exc


async def _github_profile(
    client: httpx.AsyncClient, data: dict[str, Any], token: str
) -> ExternalProfile:
    email = ""
    # `email` on /user is the *public* address, unverified as far as this
    # response says; the verified primary comes from /user/emails.
    try:
        emails = await _get_json(client, "https://api.github.com/user/emails", token)
    except OAuthError:
        emails = []
    for entry in emails if isinstance(emails, list) else []:
        if isinstance(entry, dict) and entry.get("primary") and entry.get("verified"):
            email = str(entry.get("email") or "")
            break
    return ExternalProfile(
        provider="github",
        subject=_subject(data.get("id")),
        email=email,
        display_name=str(data.get("name") or data.get("login") or ""),
        avatar_url=str(data.get("avatar_url") or ""),
        raw={k: data.get(k) for k in ("id", "login", "name", "html_url")},
    )


def _google_profile(data: dict[str, Any]) -> ExternalProfile:
    return ExternalProfile(
        provider="google",
        subject=_subject(data.get("sub")),
        email=str(data.get("email") or "") if data.get("email_verified") else "",
        display_name=str(data.get("name") or ""),
        avatar_url=str(data.get("picture") or ""),
        raw={k: data.get(k) for k in ("sub", "name", "locale")},
    )


def _linuxdo_profile(data: dict[str, Any]) -> ExternalProfile:
    if data.get("active") is False or data.get("silenced") is True:
        raise OAuthError("account_inactive", "LINUX DO reports the account inactive")
    avatar = str(data.get("avatar_url") or "")
    if not avatar and data.get("avatar_template"):
        avatar = str(data["avatar_template"]).replace("{size}", "120")
        if avatar.startswith("/"):
            avatar = f"https://linux.do{avatar}"
    return ExternalProfile(
        provider="linuxdo",
        subject=_subject(data.get("id")),
        # Connect returns no verification flag; an address is kept for display
        # only and never used to match accounts.
        email=str(data.get("email") or ""),
        display_name=str(data.get("name") or data.get("username") or ""),
        avatar_url=avatar,
        raw={k: data.get(k) for k in ("id", "username", "name", "trust_level")},
    )


def _subject(value: Any) -> str:
    subject = str(value if value is not None else "").strip()
    if not subject:
        raise OAuthError("profile_unavailable", "provider returned no user id")
    return subject
