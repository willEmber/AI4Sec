"""Accounts: login sessions, users, merging anonymous data, daily quotas.

A user is a principal of kind `user` (P7.5 I2). Everything that owns data keeps
pointing at `agent_principals`, so the agent layer never learns about accounts;
what changes is how a request proves which principal it is:

- a login session — the cookie holds `st_<random>`, the server stores only its
  SHA-256, and it can expire or be revoked;
- an anonymous credential (`identity.py`), only while anonymous use is allowed
  and only until the visitor logs in.

Besides OAuth there is one password login: the administrator configured by
`ADMIN_USERNAME` / `ADMIN_PASSWORD`, for local testing and personal deployments.

Logging in moves an anonymous visitor's sessions, runs, memories and evidence
to the account in one transaction, then marks the anonymous principal merged,
which is what makes its old credential stop resolving.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.config import get_settings
from app.db import database as db

logger = logging.getLogger("scholar.accounts")

LOGIN_TOKEN_PREFIX = "st_"
# The one principal of a single_user deployment.
LOCAL_PRINCIPAL_ID = "pr_local"
# Renewing a login session is a write; doing it at most this often keeps a
# busy page from writing on every request.
_RENEW_AFTER_SECONDS = 3600

# Tables whose `owner_id` moves with a merge. `runs` (classic mode reports) is
# handled beside them because its column is nullable.
_OWNED_TABLES = (
    "agent_sessions",
    "agent_runs",
    "agent_memories",
    "evidence",
    "agent_projects",
    "quota_charges",
)

# The configured administrator is one `user_identities` row. Its subject is
# fixed rather than the username, so renaming the admin keeps its data.
ADMIN_PROVIDER = "admin"
_ADMIN_SUBJECT = "admin"
_SCRYPT = {"n": 2**14, "r": 8, "p": 1}


class AccountDisabled(Exception):
    """The account exists but an operator disabled it."""


@dataclass(frozen=True)
class ExternalProfile:
    """What a provider says about the person who just logged in."""

    provider: str
    subject: str
    email: str = ""          # only an address the provider reports as verified
    display_name: str = ""
    avatar_url: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Caller:
    """Who a request is, and how it proved it."""

    principal_id: str
    kind: str                # anonymous | user
    via: str                 # cookie | header | query | single_user | new


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def is_login_token(token: str) -> bool:
    return (token or "").startswith(LOGIN_TOKEN_PREFIX)


# ── login sessions ──────────────────────────────────────────────────────────


async def create_login_session(principal_id: str, *, user_agent: str = "") -> str:
    """Start a login session and return the cookie value. Shown once, stored hashed."""
    token = LOGIN_TOKEN_PREFIX + secrets.token_urlsafe(32)
    await db.execute(
        """INSERT INTO auth_sessions (token_hash, principal_id, expires_at, user_agent)
           VALUES (?, ?, now() + make_interval(days => ?), ?)""",
        (hash_token(token), principal_id, get_settings().auth_session_days, user_agent[:300]),
    )
    await db.execute(
        "UPDATE users SET last_login_at = now() WHERE principal_id = ?", (principal_id,)
    )
    return token


async def resolve_login_session(token: str) -> str | None:
    """The account a session cookie belongs to, renewing it while it is in use."""
    if not is_login_token(token):
        return None
    token_hash = hash_token(token)
    row = await db.fetch_one(
        """SELECT s.principal_id,
                  s.last_seen_at < now() - make_interval(secs => ?) AS stale
             FROM auth_sessions s
             JOIN users u ON u.principal_id = s.principal_id
            WHERE s.token_hash = ? AND s.revoked_at IS NULL AND s.expires_at > now()
              AND u.status = 'active'""",
        (_RENEW_AFTER_SECONDS, token_hash),
    )
    if row is None:
        return None
    if row["stale"]:
        await db.execute(
            """UPDATE auth_sessions
                  SET last_seen_at = now(), expires_at = now() + make_interval(days => ?)
                WHERE token_hash = ?""",
            (get_settings().auth_session_days, token_hash),
        )
    return row["principal_id"]


async def revoke_login_session(token: str) -> None:
    if is_login_token(token):
        await db.execute(
            "UPDATE auth_sessions SET revoked_at = now() WHERE token_hash = ? AND revoked_at IS NULL",
            (hash_token(token),),
        )


# ── users ───────────────────────────────────────────────────────────────────


async def upsert_user(profile: ExternalProfile) -> str:
    """The account for `(provider, subject)`, created on first login.

    Matching is by the provider's subject only. Joining two logins because they
    report the same email would let whoever controls an address on one provider
    take over an account made through another.
    """
    profile_json = json.dumps(profile.raw, ensure_ascii=False, default=str)[:20_000]
    async with db.transaction() as tx:
        row = await tx.fetch_one(
            """SELECT i.principal_id, u.status
                 FROM user_identities i JOIN users u ON u.principal_id = i.principal_id
                WHERE i.provider = ? AND i.subject = ?
                FOR UPDATE OF i""",
            (profile.provider, profile.subject),
        )
        if row is not None:
            if row["status"] != "active":
                raise AccountDisabled(row["principal_id"])
            principal_id = row["principal_id"]
            await tx.execute(
                """UPDATE user_identities
                      SET email = ?, profile_json = ?, last_login_at = now()
                    WHERE provider = ? AND subject = ?""",
                (profile.email, profile_json, profile.provider, profile.subject),
            )
            # Name and picture follow the provider; an empty value never wipes
            # one we already have.
            await tx.execute(
                """UPDATE users
                      SET display_name = COALESCE(NULLIF(?, ''), display_name),
                          avatar_url = COALESCE(NULLIF(?, ''), avatar_url),
                          email = COALESCE(NULLIF(?, ''), email)
                    WHERE principal_id = ?""",
                (profile.display_name, profile.avatar_url, profile.email, principal_id),
            )
            return principal_id

        principal_id = f"pr_{uuid.uuid4().hex[:24]}"
        await tx.execute(
            "INSERT INTO agent_principals (principal_id, kind, label) VALUES (?, 'user', ?)",
            (principal_id, f"{profile.provider}:{profile.subject}"[:200]),
        )
        await tx.execute(
            """INSERT INTO users (principal_id, email, display_name, avatar_url)
               VALUES (?, ?, ?, ?)""",
            (principal_id, profile.email, profile.display_name, profile.avatar_url),
        )
        await tx.execute(
            """INSERT INTO user_identities (provider, subject, principal_id, email, profile_json)
               VALUES (?, ?, ?, ?, ?)""",
            (profile.provider, profile.subject, principal_id, profile.email, profile_json),
        )
    logger.info("Created account %s via %s", principal_id, profile.provider)
    return principal_id


async def get_user(principal_id: str) -> dict[str, Any] | None:
    return await db.fetch_one(
        """SELECT principal_id, email, display_name, avatar_url, role, status,
                  created_at, last_login_at
             FROM users WHERE principal_id = ?""",
        (principal_id,),
    )


async def ensure_local_principal() -> str:
    """The single_user principal, created on first start."""
    await db.execute(
        """INSERT INTO agent_principals (principal_id, kind, label)
           VALUES (?, 'user', 'local') ON CONFLICT (principal_id) DO NOTHING""",
        (LOCAL_PRINCIPAL_ID,),
    )
    await db.execute(
        """INSERT INTO users (principal_id, display_name, role)
           VALUES (?, 'local', 'admin') ON CONFLICT (principal_id) DO NOTHING""",
        (LOCAL_PRINCIPAL_ID,),
    )
    return LOCAL_PRINCIPAL_ID


# ── the configured administrator ────────────────────────────────────────────


def admin_login_enabled() -> bool:
    settings = get_settings()
    return (
        settings.auth_mode == "multi_user"
        and bool(settings.admin_username.strip())
        and bool(settings.admin_password)
    )


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def check_admin_credentials(username: str, password: str) -> bool:
    """Whether a login form names the configured admin. Constant-time."""
    if not admin_login_enabled():
        return False
    settings = get_settings()
    # Both comparisons always run, so timing does not tell which one failed.
    user_ok = hmac.compare_digest(_digest(username.strip()), _digest(settings.admin_username.strip()))
    password_ok = hmac.compare_digest(_digest(password), _digest(settings.admin_password))
    return user_ok and password_ok


def _credential_material(username: str, password: str) -> bytes:
    return f"{username.strip()}\0{password}".encode("utf-8")


def _fingerprint(username: str, password: str) -> str:
    """A salted scrypt of the configured credentials, so a read of the database
    does not reveal the password but a restart can tell whether it changed."""
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(_credential_material(username, password), salt=salt, **_SCRYPT)
    return f"scrypt${salt.hex()}${digest.hex()}"


def _fingerprint_matches(stored: str, username: str, password: str) -> bool:
    try:
        scheme, salt_hex, digest_hex = stored.split("$")
        if scheme != "scrypt":
            return False
        digest = hashlib.scrypt(
            _credential_material(username, password), salt=bytes.fromhex(salt_hex), **_SCRYPT
        )
    except ValueError:
        return False
    return hmac.compare_digest(digest.hex(), digest_hex)


async def ensure_admin_account() -> str | None:
    """Bring the admin account in line with the configuration; its principal id.

    Creates it on first start and keeps its role `admin`. When the configured
    username or password differs from the one its sessions were opened with —
    or the admin login was switched off — every session it has is revoked:
    changing a leaked password must also end the logins made with it.
    Returns None when no admin is configured and none ever was.
    """
    settings = get_settings()
    enabled = admin_login_enabled()
    username = settings.admin_username.strip()
    password = settings.admin_password
    async with db.transaction() as tx:
        # Replicas starting together would otherwise both create the account.
        await tx.execute("SELECT pg_advisory_xact_lock(hashtextextended('scholar.admin', 0))")
        row = await tx.fetch_one(
            "SELECT principal_id, profile_json FROM user_identities WHERE provider = ? AND subject = ?",
            (ADMIN_PROVIDER, _ADMIN_SUBJECT),
        )
        if row is None:
            if not enabled:
                return None
            principal_id = f"pr_{uuid.uuid4().hex[:24]}"
            profile_json = json.dumps({"credential": _fingerprint(username, password)})
            await tx.execute(
                "INSERT INTO agent_principals (principal_id, kind, label) VALUES (?, 'user', 'admin')",
                (principal_id,),
            )
            await tx.execute(
                "INSERT INTO users (principal_id, display_name, role) VALUES (?, ?, 'admin')",
                (principal_id, username),
            )
            await tx.execute(
                """INSERT INTO user_identities (provider, subject, principal_id, profile_json)
                   VALUES (?, ?, ?, ?)""",
                (ADMIN_PROVIDER, _ADMIN_SUBJECT, principal_id, profile_json),
            )
            logger.info("Created the admin account %s", principal_id)
            return principal_id

        principal_id = row["principal_id"]
        try:
            stored = str(json.loads(row["profile_json"] or "{}").get("credential") or "")
        except ValueError:
            stored = ""
        unchanged = enabled and _fingerprint_matches(stored, username, password)
        if not unchanged:
            revoked = await tx.execute(
                "UPDATE auth_sessions SET revoked_at = now() WHERE principal_id = ? AND revoked_at IS NULL",
                (principal_id,),
            )
            if revoked:
                logger.warning("Admin credentials changed: revoked %d admin login session(s)", revoked)
        if not enabled:
            return principal_id
        if not unchanged:
            await tx.execute(
                "UPDATE user_identities SET profile_json = ? WHERE provider = ? AND subject = ?",
                (json.dumps({"credential": _fingerprint(username, password)}), ADMIN_PROVIDER, _ADMIN_SUBJECT),
            )
        await tx.execute(
            "UPDATE users SET role = 'admin', display_name = ? WHERE principal_id = ?",
            (username, principal_id),
        )
    return principal_id


# ── merging ─────────────────────────────────────────────────────────────────


async def merge_anonymous(anonymous_id: str, account_id: str) -> bool:
    """Move an anonymous principal's data into an account. True if anything moved.

    Only an unmerged anonymous principal can be merged, so one account can
    never swallow another, and a second merge of the same visitor is a no-op.
    The row lock serialises two logins racing with the same anonymous cookie.
    """
    if not anonymous_id or anonymous_id == account_id:
        return False
    async with db.transaction() as tx:
        row = await tx.fetch_one(
            "SELECT kind, merged_into FROM agent_principals WHERE principal_id = ? FOR UPDATE",
            (anonymous_id,),
        )
        if row is None or row["kind"] != "anonymous" or row["merged_into"]:
            return False
        moved: dict[str, int] = {}
        for table in _OWNED_TABLES:
            moved[table] = await tx.execute(
                f"UPDATE {table} SET owner_id = ? WHERE owner_id = ?", (account_id, anonymous_id)
            )
        moved["runs"] = await tx.execute(
            "UPDATE runs SET owner_id = ? WHERE owner_id = ?", (account_id, anonymous_id)
        )
        await tx.execute(
            "UPDATE agent_principals SET merged_into = ? WHERE principal_id = ?",
            (account_id, anonymous_id),
        )
    logger.info("Merged %s into %s: %s", anonymous_id, account_id, moved)
    return True


# ── quotas ──────────────────────────────────────────────────────────────────


async def record_charge(caller: Caller, kind: str) -> None:
    """Count one costly action that has no row of its own against today."""
    await db.execute(
        "INSERT INTO quota_charges (owner_id, kind) VALUES (?, ?)", (caller.principal_id, kind)
    )


@dataclass(frozen=True)
class QuotaUsage:
    runs: int
    tokens: int
    runs_limit: int          # 0 = unlimited
    tokens_limit: int

    @property
    def exceeded(self) -> bool:
        return bool(
            (self.runs_limit and self.runs >= self.runs_limit)
            or (self.tokens_limit and self.tokens >= self.tokens_limit)
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "runs": self.runs,
            "runs_limit": self.runs_limit,
            "tokens": self.tokens,
            "tokens_limit": self.tokens_limit,
        }


async def is_admin(caller: Caller) -> bool:
    if caller.kind != "user":
        return False
    row = await db.fetch_one("SELECT role FROM users WHERE principal_id = ?", (caller.principal_id,))
    return row is not None and row["role"] == "admin"


async def daily_usage(caller: Caller) -> QuotaUsage:
    """Runs and tokens since UTC midnight, against the caller's limits.

    A run is an agent turn, a mode run started from the upload page, or a
    library question. Turns and mode runs are counted from their own rows
    rather than a counter table, so the count cannot drift from what actually
    ran; a mode run made inside a turn carries `agent_run_id` and is that
    turn, not a second run. The tokens of a turn still in flight are not in
    yet; the run count is what stops a burst.
    """
    settings = get_settings()
    if settings.auth_mode == "single_user" or await is_admin(caller):
        runs_limit = tokens_limit = 0
    elif caller.kind == "user":
        runs_limit, tokens_limit = settings.quota_user_daily_runs, settings.quota_user_daily_tokens
    else:
        runs_limit, tokens_limit = settings.quota_anon_daily_runs, settings.quota_anon_daily_tokens
    row = await db.fetch_one(
        """SELECT COUNT(*) AS runs,
                  COALESCE(SUM(CASE WHEN jsonb_typeof(usage_json::jsonb -> 'tokens') = 'number'
                                    THEN (usage_json::jsonb ->> 'tokens')::numeric
                                    ELSE 0 END), 0) AS tokens
             FROM agent_runs
            WHERE owner_id = ? AND started_at >= date_trunc('day', now())""",
        (caller.principal_id,),
    )
    other = await db.fetch_one(
        """SELECT (SELECT COUNT(*) FROM runs
                    WHERE owner_id = ? AND COALESCE(agent_run_id, '') = ''
                      AND started_at >= date_trunc('day', now()))
                + (SELECT COUNT(*) FROM quota_charges
                    WHERE owner_id = ? AND created_at >= date_trunc('day', now())) AS runs""",
        (caller.principal_id, caller.principal_id),
    )
    return QuotaUsage(
        runs=int(row["runs"] or 0) + int(other["runs"] or 0),
        tokens=int(row["tokens"] or 0),
        runs_limit=max(0, runs_limit),
        tokens_limit=max(0, tokens_limit),
    )
