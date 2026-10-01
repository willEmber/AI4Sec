"""Server-verified anonymous identity for agent resources.

The existing `owner_token` is a UUID the browser generates and stores in
`localStorage`. It scopes a listing, but any client can send any value, so it
cannot decide who may read a session, its events, its papers or its evidence.

This issues a credential the *server* signs: `<principal_id>.<hmac>`. A client
keeps it and sends it back; nothing else is trusted. That is enough to isolate
sessions (acceptance case A16) without building accounts — the development plan
explicitly allows a server-verified anonymous credential for the first version.

Accounts (P7.5 I2) did not reuse this shape: an account is a `kind='user'`
principal reached through a login session (`app/services/accounts.py`), and
this credential proves only an *anonymous* principal that has not been merged
into one. Once a visitor logs in and their data moves to the account, the old
credential stops working rather than remaining a second key to it.

`seal` / `unseal` sign short-lived payloads (the event-stream ticket, the OAuth
state) with keys derived per purpose, so no sealed value can stand in for a
credential or for a value of another purpose.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
import uuid
from pathlib import Path

from app.config import get_settings
from app.db import database as db

logger = logging.getLogger("scholar.identity")

_SECRET_FILENAME = ".agent_identity_secret"
_cached_secret: bytes | None = None

# Header the client returns the credential in.
AGENT_TOKEN_HEADER = "X-Agent-Token"


def _secret_path() -> Path:
    return get_settings().data_dir / _SECRET_FILENAME


def get_secret() -> bytes:
    """Resolve the signing secret, generating and persisting one if needed."""
    global _cached_secret
    if _cached_secret is not None:
        return _cached_secret

    configured = (get_settings().agent_identity_secret or "").strip()
    if configured:
        _cached_secret = configured.encode("utf-8")
        return _cached_secret

    path = _secret_path()
    if path.exists():
        _cached_secret = path.read_bytes().strip()
        if _cached_secret:
            return _cached_secret

    generated = secrets.token_bytes(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(generated)
    try:
        path.chmod(0o600)
    except OSError:  # pragma: no cover — best effort on filesystems without modes
        logger.warning("Could not restrict permissions on %s", path)
    logger.info(
        "Generated an agent identity secret at %s. Set AGENT_IDENTITY_SECRET to "
        "keep credentials valid across deployments and instances.",
        path,
    )
    _cached_secret = generated
    return _cached_secret


def reset_secret_cache() -> None:
    """Drop the cached secret. For tests that swap the data dir."""
    global _cached_secret
    _cached_secret = None


def _sign(principal_id: str) -> str:
    digest = hmac.new(get_secret(), principal_id.encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _purpose_key(purpose: str) -> bytes:
    return hmac.new(get_secret(), f"seal:{purpose}".encode("utf-8"), hashlib.sha256).digest()


def seal(purpose: str, payload: dict, *, ttl_seconds: int) -> str:
    """Sign `payload` for `purpose`, valid for `ttl_seconds`. Not encrypted."""
    body = _b64(
        json.dumps({**payload, "exp": int(time.time()) + ttl_seconds}, separators=(",", ":")).encode()
    )
    signature = _b64(hmac.new(_purpose_key(purpose), body.encode("ascii"), hashlib.sha256).digest())
    return f"{body}.{signature}"


def unseal(purpose: str, token: str) -> dict | None:
    """The payload `seal` signed for this purpose, or `None` if forged or expired."""
    body, _, signature = (token or "").strip().rpartition(".")
    if not body or not signature:
        return None
    expected = _b64(hmac.new(_purpose_key(purpose), body.encode("ascii"), hashlib.sha256).digest())
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        payload = json.loads(_unb64(body))
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or int(payload.get("exp") or 0) < time.time():
        return None
    return payload


def issue_credential(principal_id: str) -> str:
    """Return the signed credential for a principal."""
    return f"{principal_id}.{_sign(principal_id)}"


def verify_credential(token: str) -> str | None:
    """Return the principal id a credential proves, or `None` if it does not.

    Rejects anything whose signature does not match, which is what stops a
    client from naming someone else's principal.
    """
    token = (token or "").strip()
    if not token or "." not in token:
        return None
    principal_id, _, signature = token.rpartition(".")
    if not principal_id or not signature:
        return None
    if not hmac.compare_digest(signature, _sign(principal_id)):
        return None
    return principal_id


async def create_principal(kind: str = "anonymous", label: str = "") -> tuple[str, str]:
    """Create a principal row and return `(principal_id, credential)`.

    Only an anonymous principal gets a credential; an account is reached through
    a login session, so its credential is `""`.
    """
    principal_id = f"pr_{uuid.uuid4().hex[:24]}"
    await db.execute(
        "INSERT INTO agent_principals (principal_id, kind, label) VALUES (?, ?, ?)",
        (principal_id, kind, label),
    )
    return principal_id, issue_credential(principal_id) if kind == "anonymous" else ""


async def resolve_principal(token: str) -> str | None:
    """Verify a credential and confirm the principal still exists.

    A valid signature over a deleted principal is rejected: the signature proves
    the server issued the id, not that the id is still in use. So is one over an
    account, or over an anonymous principal already merged into an account —
    the signature is not a key to either.
    """
    principal_id = verify_credential(token)
    if principal_id is None:
        return None
    row = await db.fetch_one(
        "SELECT kind, merged_into FROM agent_principals WHERE principal_id = ?", (principal_id,)
    )
    if row is None or row["kind"] != "anonymous" or row["merged_into"]:
        return None
    await db.execute(
        "UPDATE agent_principals SET last_seen_at = now() WHERE principal_id = ?",
        (principal_id,),
    )
    return principal_id


async def resolve_or_create_principal(token: str) -> tuple[str, str | None]:
    """Resolve a credential, minting a new principal when there is none.

    Returns `(principal_id, new_credential_or_None)`. A new credential is
    returned only when one was minted, so the caller can hand it to the client
    exactly once.
    """
    principal_id = await resolve_principal(token)
    if principal_id is not None:
        return principal_id, None
    principal_id, credential = await create_principal()
    return principal_id, credential
