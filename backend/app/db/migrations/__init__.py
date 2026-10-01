"""Numbered schema migrations (PostgreSQL).

`0001_baseline.sql` is the whole schema as of the move off SQLite; anything
later ships as the next numbered file. Each file is applied once, inside a
transaction, and recorded in `schema_migrations`. Files are ordered by their
numeric prefix.

Two instances starting at once must not both migrate. A session-level advisory
lock serialises them: the second waits, then finds every version applied.

Migration files are executed as written — no placeholder translation — so they
may use `%`, `?` operators and dollar-quoted bodies freely.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import psycopg

logger = logging.getLogger("scholar.db.migrations")

_MIGRATIONS_DIR = Path(__file__).parent
_NAME_RE = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")

# Arbitrary but fixed: every process migrating this database contends on it.
MIGRATION_LOCK_KEY = 0x5C0_1A12


def discover_migrations() -> list[tuple[str, Path]]:
    """Return `(version, path)` for every migration file, lowest version first."""
    found: list[tuple[str, Path]] = []
    for path in sorted(_MIGRATIONS_DIR.glob("*.sql")):
        match = _NAME_RE.match(path.name)
        if not match:
            raise RuntimeError(
                f"Migration {path.name!r} does not match NNNN_lower_snake_case.sql"
            )
        found.append((match.group(1), path))
    versions = [v for v, _ in found]
    duplicates = {v for v in versions if versions.count(v) > 1}
    if duplicates:
        raise RuntimeError(f"Duplicate migration version(s): {sorted(duplicates)}")
    return found


async def apply_migrations(conn: psycopg.AsyncConnection) -> list[str]:
    """Apply every unapplied migration. Returns the versions applied this call."""
    await conn.execute("SELECT pg_advisory_lock(%s)", (MIGRATION_LOCK_KEY,))
    await conn.commit()
    try:
        await conn.execute(
            """CREATE TABLE IF NOT EXISTS schema_migrations (
                   version    TEXT PRIMARY KEY,
                   name       TEXT NOT NULL,
                   applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
               )"""
        )
        await conn.commit()
        cursor = await conn.execute("SELECT version FROM schema_migrations")
        applied = {row["version"] if isinstance(row, dict) else row[0] for row in await cursor.fetchall()}
        await conn.commit()

        newly_applied: list[str] = []
        for version, path in discover_migrations():
            if version in applied:
                continue
            sql = path.read_text(encoding="utf-8")
            try:
                async with conn.transaction():
                    # No parameters: sent as a simple query, so a file may hold
                    # many statements.
                    await conn.execute(sql)  # type: ignore[arg-type]
                    await conn.execute(
                        "INSERT INTO schema_migrations (version, name) VALUES (%s, %s)",
                        (version, path.name),
                    )
            except Exception:
                logger.error("Migration %s failed; database left at the previous version", path.name)
                raise
            newly_applied.append(version)
            logger.info("Applied migration %s", path.name)
        return newly_applied
    finally:
        await conn.execute("SELECT pg_advisory_unlock(%s)", (MIGRATION_LOCK_KEY,))
        await conn.commit()
