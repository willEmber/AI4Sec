"""Server-verified anonymous identity for agent resources.

The existing `owner_token` is a UUID the browser generates and stores in
`localStorage`. It scopes a listing, but any client can send any value, so it
cannot decide who may read a session, its events, its papers or its evidence.

This issues a credential the *server* signs: `<principal_id>.<hmac>`. A client
keeps it and sends it back; nothing else is trusted. That is enough to isolate
sessions (acceptance case A16) without building accounts — the development plan
explicitly allows a server-verified anonymous credential for the first version.

Upgrading later means adding a real `kind='user'` principal and issuing the
same credential shape after a login; callers keep working unchanged.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import secrets
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
    """Create a principal row and return `(principal_id, credential)`."""
    principal_id = f"pr_{uuid.uuid4().hex[:24]}"
    await db.execute(
        "INSERT INTO agent_principals (principal_id, kind, label) VALUES (?, ?, ?)",
        (principal_id, kind, label),
    )
    return principal_id, issue_credential(principal_id)


async def resolve_principal(token: str) -> str | None:
    """Verify a credential and confirm the principal still exists.

    A valid signature over a deleted principal is rejected: the signature proves
    the server issued the id, not that the id is still in use.
    """
    principal_id = verify_credential(token)
    if principal_id is None:
        return None
    row = await db.fetch_one(
        "SELECT principal_id FROM agent_principals WHERE principal_id = ?", (principal_id,)
    )
    if row is None:
        return None
    await db.execute(
        "UPDATE agent_principals SET last_seen_at = datetime('now') WHERE principal_id = ?",
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
