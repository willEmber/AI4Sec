"""Incremental schema migrations.

`schema.sql` stays the baseline for a fresh database. Anything added after a
release ships as a numbered `.sql` file here instead, so an existing deployment
moves forward without the ad-hoc `ALTER TABLE ... except: pass` pattern in
`database.py` — those silently swallow real errors and cannot express a new
table plus its backfill as one unit.

Each file is applied once, inside a transaction, and recorded in
`schema_migrations`. Files are ordered by their numeric prefix.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import aiosqlite

logger = logging.getLogger("scholar.db.migrations")

_MIGRATIONS_DIR = Path(__file__).parent
_NAME_RE = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")


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


async def apply_migrations(db: aiosqlite.Connection) -> list[str]:
    """Apply every unapplied migration. Returns the versions applied this call."""
    await db.execute(
        """CREATE TABLE IF NOT EXISTS schema_migrations (
               version    TEXT PRIMARY KEY,
               name       TEXT NOT NULL,
               applied_at TEXT NOT NULL DEFAULT (datetime('now'))
           )"""
    )
    await db.commit()

    async with db.execute("SELECT version FROM schema_migrations") as cursor:
        applied = {row[0] for row in await cursor.fetchall()}

    newly_applied: list[str] = []
    for version, path in discover_migrations():
        if version in applied:
            continue
        sql = path.read_text(encoding="utf-8")
        try:
            await db.executescript(sql)
            await db.execute(
                "INSERT INTO schema_migrations (version, name) VALUES (?, ?)",
                (version, path.name),
            )
            await db.commit()
        except Exception:
            await db.rollback()
            logger.error("Migration %s failed; database left at the previous version", path.name)
            raise
        newly_applied.append(version)
        logger.info("Applied migration %s", path.name)
    return newly_applied
