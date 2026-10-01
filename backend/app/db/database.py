"""PostgreSQL access for the application.

Callers use five functions — `fetch_one`, `fetch_all`, `execute`,
`execute_returning`, `execute_many` — plus `transaction()` when several
statements must commit together. Three things are done here so that no caller
has to know about them:

*Placeholders.* SQL in this codebase is written with `?` placeholders. They are
rewritten to psycopg's `%s` before a statement is sent, and every literal `%`
is doubled, so `LIKE '%http%'` keeps working. A `?` inside a quoted literal,
a quoted identifier or a comment is left alone. (PostgreSQL's jsonb `?`
operators are therefore unavailable; use `jsonb_exists()` instead.)

*Time.* Every timestamp column is `timestamptz` and every session runs in UTC.
Rows come back with timestamps formatted as UTC `YYYY-MM-DD HH:MM:SS` — the
exact shape SQLite's `datetime('now')` produced — so the API contract and the
frontend did not change when the database did.

*Numbers.* `SUM`, `EXTRACT` and friends return `numeric`; they come back as
`int` when integral and `float` otherwise, never as `Decimal`.

The pool is process-wide and opened by `init_db()` (or `open_pool()`); tests
point it at a throwaway schema through the URL's `search_path` option.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from decimal import Decimal
from functools import lru_cache
from typing import Any

import psycopg
from psycopg import postgres
from psycopg.adapt import Dumper
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.db.migrations import apply_migrations

logger = logging.getLogger("scholar.db")

# Raised for unique, foreign-key and check violations. Callers catch this
# rather than a driver-specific class.
IntegrityError = psycopg.IntegrityError

_url: str = ""
_pool: AsyncConnectionPool | None = None
_pool_min = 1
_pool_max = 10


# ── Configuration and lifetime ──────────────────────────────────────────────


def configure(url: str, *, min_size: int = 1, max_size: int = 10) -> None:
    """Point the module at a database. Takes effect when the pool is (re)opened."""
    global _url, _pool_min, _pool_max
    _url = url
    _pool_min = max(1, int(min_size))
    _pool_max = max(_pool_min, int(max_size))


def get_database_url() -> str:
    if not _url:
        raise RuntimeError("Database not configured. Call database.configure() first.")
    return _url


class _BoolAsIntDumper(Dumper):
    """Send Python booleans as 0/1.

    Flags in this schema are INTEGER columns, as they were in SQLite, which
    stored `True` as 1 without complaint; PostgreSQL refuses a boolean for an
    integer column. No column here is boolean, so this is safe globally.
    """

    oid = postgres.types["int4"].oid

    def dump(self, obj: bool) -> bytes:
        return b"1" if obj else b"0"


def _adapt(conn: psycopg.BaseConnection[Any]) -> None:
    conn.adapters.register_dumper(bool, _BoolAsIntDumper)


async def _configure_connection(conn: psycopg.AsyncConnection) -> None:
    _adapt(conn)
    # Session-level, so it must be committed rather than left in the implicit
    # transaction the SET opened.
    await conn.execute("SET TIME ZONE 'UTC'")
    await conn.commit()


async def open_pool() -> AsyncConnectionPool:
    """Open the shared pool if it is not open yet, and return it."""
    global _pool
    if _pool is not None:
        return _pool
    pool = AsyncConnectionPool(
        get_database_url(),
        min_size=_pool_min,
        max_size=_pool_max,
        kwargs={"row_factory": dict_row},
        configure=_configure_connection,
        open=False,
        name="scholar",
    )
    await pool.open(wait=True, timeout=30)
    _pool = pool
    return pool


async def close_pool() -> None:
    global _pool
    pool, _pool = _pool, None
    if pool is not None:
        await pool.close()


def _require_pool() -> AsyncConnectionPool:
    if _pool is None:
        raise RuntimeError("Database pool is not open. Call init_db() or open_pool() first.")
    return _pool


async def init_db() -> None:
    """Open the pool, apply pending migrations, and close out orphaned legacy runs."""
    pool = await open_pool()
    async with pool.connection() as conn:
        applied = await apply_migrations(conn)
        if applied:
            logger.info("Applied %d schema migration(s): %s", len(applied), ", ".join(applied))
        # A legacy (non-agent) run still pending/running at startup lost the
        # background task that owned it when the previous process exited, so
        # it can never finish. Agent runs are not touched here: they carry a
        # worker heartbeat, and `agent_worker` decides from that.
        cursor = await conn.execute(
            "UPDATE runs SET status = 'failed', "
            "error_msg = 'Interrupted (server restarted)', finished_at = now() "
            "WHERE status IN ('pending', 'running')"
        )
        if cursor.rowcount:
            logger.info("Reconciled %d interrupted run(s) on startup", cursor.rowcount)


# ── SQL and row translation ─────────────────────────────────────────────────


@lru_cache(maxsize=1024)
def to_pg_sql(sql: str) -> str:
    """Rewrite `?` placeholders to `%s` and escape literal `%`.

    `%` is doubled everywhere, literals included: psycopg scans the whole string
    for placeholders whenever parameters are passed, and they always are.
    """
    out: list[str] = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch == "%":
            out.append("%%")
            i += 1
        elif ch == "?":
            out.append("%s")
            i += 1
        elif ch in ("'", '"'):
            # Quoted literal or identifier; a doubled quote is an escaped one.
            j = i + 1
            while j < n:
                if sql[j] == ch:
                    if j + 1 < n and sql[j + 1] == ch:
                        j += 2
                        continue
                    break
                j += 1
            out.append(sql[i : j + 1].replace("%", "%%"))
            i = j + 1
        elif ch == "-" and sql.startswith("--", i):
            j = sql.find("\n", i)
            j = n if j == -1 else j
            out.append(sql[i:j].replace("%", "%%"))
            i = j
        elif ch == "/" and sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            j = n if j == -1 else j + 2
            out.append(sql[i:j].replace("%", "%%"))
            i = j
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def format_timestamp(value: datetime) -> str:
    """UTC `YYYY-MM-DD HH:MM:SS`, the shape every timestamp in the API has."""
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _value(value: Any) -> Any:
    if isinstance(value, datetime):
        return format_timestamp(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    return value


def _row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: _value(value) for key, value in row.items()}


def _params(params: Sequence[Any] | None) -> tuple[Any, ...]:
    return tuple(params or ())


# ── Statements ──────────────────────────────────────────────────────────────


class Transaction:
    """Statements on one connection inside one transaction (see `transaction()`)."""

    def __init__(self, conn: psycopg.AsyncConnection) -> None:
        self._conn = conn

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        cursor = await self._conn.execute(to_pg_sql(sql), _params(params))
        return cursor.rowcount

    async def fetch_one(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        cursor = await self._conn.execute(to_pg_sql(sql), _params(params))
        return _row(await cursor.fetchone())

    async def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        cursor = await self._conn.execute(to_pg_sql(sql), _params(params))
        return [_row(r) for r in await cursor.fetchall()]  # type: ignore[misc]

    execute_returning = fetch_one

    async def execute_many(self, sql: str, params_seq: Iterable[Sequence[Any]]) -> None:
        async with self._conn.cursor() as cursor:
            await cursor.executemany(to_pg_sql(sql), [_params(p) for p in params_seq])


@asynccontextmanager
async def transaction() -> AsyncIterator[Transaction]:
    """Run several statements atomically: all commit, or none do."""
    async with _require_pool().connection() as conn:
        async with conn.transaction():
            yield Transaction(conn)


async def execute(sql: str, params: Sequence[Any] = ()) -> int:
    """Run a write and commit it. Returns the affected row count."""
    async with _require_pool().connection() as conn:
        cursor = await conn.execute(to_pg_sql(sql), _params(params))
        return cursor.rowcount


async def execute_returning(sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
    """Run a write with a RETURNING clause, commit it, and return the first row."""
    async with _require_pool().connection() as conn:
        cursor = await conn.execute(to_pg_sql(sql), _params(params))
        return _row(await cursor.fetchone())


async def execute_many(sql: str, params_seq: Iterable[Sequence[Any]]) -> None:
    async with _require_pool().connection() as conn:
        async with conn.cursor() as cursor:
            await cursor.executemany(to_pg_sql(sql), [_params(p) for p in params_seq])


async def fetch_one(sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
    async with _require_pool().connection() as conn:
        cursor = await conn.execute(to_pg_sql(sql), _params(params))
        return _row(await cursor.fetchone())


async def fetch_all(sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
    async with _require_pool().connection() as conn:
        cursor = await conn.execute(to_pg_sql(sql), _params(params))
        return [_row(r) for r in await cursor.fetchall()]  # type: ignore[misc]


def execute_sync(sql: str, params: Sequence[Any] = ()) -> int:
    """A write from a worker thread that has no event loop.

    Opens a short-lived connection rather than borrowing from the async pool,
    which belongs to the main loop. Only for low-rate bookkeeping (MinerU poll
    diagnostics); anything on a request path uses the async functions.
    """
    with psycopg.connect(get_database_url(), connect_timeout=10) as conn:
        _adapt(conn)
        conn.execute("SET TIME ZONE 'UTC'")
        cursor = conn.execute(to_pg_sql(sql), _params(params))
        return cursor.rowcount


# ── Small domain helpers that need more than one statement ──────────────────


async def record_traffic_visit(visitor_hash: str, path: str) -> dict[str, Any]:
    """Upsert one anonymous visit and return the current traffic totals."""
    async with transaction() as tx:
        visitor = await tx.fetch_one(
            "INSERT INTO traffic_visitors "
            "(visitor_hash, first_seen_at, last_seen_at, visit_count, last_path) "
            "VALUES (?, now(), now(), 1, ?) "
            "ON CONFLICT (visitor_hash) DO UPDATE SET "
            "last_seen_at = now(), "
            "visit_count = traffic_visitors.visit_count + 1, "
            "last_path = excluded.last_path "
            "RETURNING visit_count",
            (visitor_hash, path),
        )
        totals = await tx.fetch_one(
            "SELECT COUNT(*) AS unique_users, "
            "COALESCE(SUM(visit_count), 0) AS total_visits "
            "FROM traffic_visitors"
        )

    return {
        "new_user": bool(visitor and visitor["visit_count"] == 1),
        "unique_users": int(totals["unique_users"]) if totals else 0,
        "total_visits": int(totals["total_visits"]) if totals else 0,
    }
