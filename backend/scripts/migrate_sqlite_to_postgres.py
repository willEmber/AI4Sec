"""Copy an existing SQLite deployment into PostgreSQL.

What moves:

- every table of `app.db` that the PostgreSQL baseline also has, column by
  column (the intersection — a column that only one side has is reported, not
  guessed at). `paper_node_fts` is not copied: `paper_nodes.search_tsv` is a
  generated column and rebuilds itself from the rows;
- timestamps, which SQLite stored as text (`datetime('now')` naive UTC, or ISO
  8601 with an offset), parsed as UTC;
- the agent checkpoints, whole: every checkpoint of every thread with its
  writes, oldest first, keeping each one's parent. A latest-only copy is not
  enough — LangGraph stores `messages` as a `DeltaChannel`, which a checkpoint
  does not hold by value; it is rebuilt by replaying writes along the parent
  chain, so dropping the chain silently empties the conversation.

What is checked:

- the target must be empty (refused otherwise; `--truncate` empties it first);
- rows whose foreign key points at a row that does not exist — SQLite never
  enforced them — are skipped and counted per table, rather than aborting the
  whole copy;
- identity sequences are moved past the copied ids.

Run (from backend/):
    uv run python -m scripts.migrate_sqlite_to_postgres \\
        --sqlite data/app.db --checkpoints data/agent_checkpoints.db
`--database-url` defaults to DATABASE_URL.
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Parents before children, so foreign keys can be checked against what is
# already there.
TABLE_ORDER = [
    "papers",
    "mineru_parses",
    "blocks",
    "paper_nodes",
    "runs",
    "run_outputs",
    "paper_signals",
    "sphere_nodes",
    "sphere_edges",
    "traffic_visitors",
    "agent_principals",
    "agent_sessions",
    "agent_runs",
    "agent_messages",
    "agent_events",
    "literature_items",
    "literature_files",
    "session_papers",
    "paper_versions",
    "evidence",
    "agent_jobs",
    "agent_memories",
]

BATCH = 500


def _parse_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    text = str(value).strip().replace(" ", "T", 1)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _fallback(data_type: str) -> Any:
    """A value for a NOT NULL column SQLite left NULL."""
    if data_type == "timestamp with time zone":
        return datetime.now(timezone.utc)
    if data_type in ("integer", "bigint", "smallint"):
        return 0
    if data_type in ("double precision", "real", "numeric"):
        return 0.0
    return ""


def _convert(value: Any, data_type: str, nullable: bool) -> Any:
    if data_type == "timestamp with time zone":
        try:
            value = _parse_timestamp(value)
        except ValueError:
            value = None
    elif data_type in ("integer", "bigint", "smallint") and value not in (None, ""):
        value = int(value)
    elif data_type in ("double precision", "real") and value not in (None, ""):
        value = float(value)
    elif data_type == "text" and value is not None and not isinstance(value, str):
        value = str(value)
    if value in (None, "") and data_type not in ("text",):
        value = None
    if value is None and not nullable:
        value = _fallback(data_type)
    return value


async def _pg_columns(tx: Any, table: str) -> dict[str, tuple[str, bool, bool]]:
    """`name -> (data_type, nullable, generated)` for a table in the current schema."""
    rows = await tx.fetch_all(
        """SELECT column_name, data_type, is_nullable, is_generated
             FROM information_schema.columns
            WHERE table_schema = current_schema() AND table_name = ?
            ORDER BY ordinal_position""",
        (table,),
    )
    return {
        r["column_name"]: (r["data_type"], r["is_nullable"] == "YES", r["is_generated"] == "ALWAYS")
        for r in rows
    }


async def _foreign_keys(tx: Any, table: str) -> list[tuple[list[str], str, list[str]]]:
    rows = await tx.fetch_all(
        """SELECT con.conname,
                  array_agg(src.attname ORDER BY k.ord) AS columns,
                  ref.relname AS ref_table,
                  array_agg(dst.attname ORDER BY k.ord) AS ref_columns
             FROM pg_constraint con
             JOIN pg_class tbl ON tbl.oid = con.conrelid
             JOIN pg_class ref ON ref.oid = con.confrelid
             JOIN pg_namespace ns ON ns.oid = tbl.relnamespace
             CROSS JOIN LATERAL unnest(con.conkey, con.confkey) WITH ORDINALITY AS k(src_num, dst_num, ord)
             JOIN pg_attribute src ON src.attrelid = con.conrelid AND src.attnum = k.src_num
             JOIN pg_attribute dst ON dst.attrelid = con.confrelid AND dst.attnum = k.dst_num
            WHERE con.contype = 'f' AND tbl.relname = ? AND ns.nspname = current_schema()
            GROUP BY con.conname, ref.relname""",
        (table,),
    )
    return [(list(r["columns"]), r["ref_table"], list(r["ref_columns"])) for r in rows]


async def copy_tables(sqlite_path: Path, *, truncate: bool) -> None:
    from app.db import database as db

    src = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    sqlite_tables = {
        r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }

    async with db.transaction() as tx:
        present = [t for t in TABLE_ORDER if (await _pg_columns(tx, t))]
        counts = {
            t: (await tx.fetch_one(f"SELECT COUNT(*) AS n FROM {t}"))["n"] for t in present
        }
        non_empty = {t: n for t, n in counts.items() if n}
        if non_empty and not truncate:
            raise SystemExit(
                f"Target is not empty: {non_empty}. Re-run with --truncate to replace it."
            )
        if non_empty:
            await tx.execute(f"TRUNCATE {', '.join(present)} CASCADE")
            print(f"Truncated {len(present)} tables")

        for table in TABLE_ORDER:
            if table not in sqlite_tables:
                print(f"  {table:<18} not in SQLite, skipped")
                continue
            pg_cols = await _pg_columns(tx, table)
            src_cols = [r[1] for r in src.execute(f"PRAGMA table_info({table})")]
            columns = [c for c in src_cols if c in pg_cols and not pg_cols[c][2]]
            only_sqlite = sorted(set(src_cols) - set(pg_cols))
            if only_sqlite:
                print(f"  {table:<18} columns not in PostgreSQL, dropped: {only_sqlite}")

            # Foreign keys SQLite never enforced: load the referenced keys once.
            fk_checks: list[tuple[list[int], set[tuple[Any, ...]]]] = []
            for fk_cols, ref_table, ref_cols in await _foreign_keys(tx, table):
                if not all(c in columns for c in fk_cols):
                    continue
                keys = {
                    tuple(r[c] for c in ref_cols)
                    for r in await tx.fetch_all(
                        f"SELECT {', '.join(ref_cols)} FROM {ref_table}"
                    )
                }
                fk_checks.append(([columns.index(c) for c in fk_cols], keys))

            placeholders = ", ".join("?" for _ in columns)
            insert = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
            copied = 0
            orphans: dict[str, int] = defaultdict(int)
            batch: list[tuple[Any, ...]] = []
            for row in src.execute(f"SELECT {', '.join(columns)} FROM {table}"):
                values = tuple(
                    _convert(row[c], pg_cols[c][0], pg_cols[c][1]) for c in columns
                )
                orphan = False
                for positions, keys in fk_checks:
                    key = tuple(values[i] for i in positions)
                    if key not in keys:
                        orphans[",".join(columns[i] for i in positions)] += 1
                        orphan = True
                        break
                if orphan:
                    continue
                batch.append(values)
                if len(batch) >= BATCH:
                    await tx.execute_many(insert, batch)
                    copied += len(batch)
                    batch = []
            if batch:
                await tx.execute_many(insert, batch)
                copied += len(batch)

            note = f", skipped orphans {dict(orphans)}" if orphans else ""
            print(f"  {table:<18} {copied} rows{note}")

        for table, column in (("blocks", "block_id"), ("sphere_edges", "id")):
            await tx.execute(
                f"SELECT setval(pg_get_serial_sequence('{table}', '{column}'), "
                f"COALESCE((SELECT MAX({column}) FROM {table}), 0) + 1, false)"
            )
    src.close()


async def copy_checkpoints(path: Path) -> None:
    """Every checkpoint of every thread, oldest first, with writes and parents."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from app.agents.checkpointer import open_checkpointer

    if not path.exists():
        print(f"No checkpoint database at {path}; skipped")
        return
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    threads = [r[0] for r in conn.execute("SELECT DISTINCT thread_id FROM checkpoints")]
    conn.close()

    checkpoints = writes = 0
    async with AsyncSqliteSaver.from_conn_string(str(path)) as source, open_checkpointer() as target:
        for thread_id in threads:
            history = [t async for t in source.alist({"configurable": {"thread_id": thread_id}})]
            # `alist` is newest first; parents must exist before their children.
            for tup in sorted(history, key=lambda t: t.checkpoint["id"]):
                configurable = tup.config["configurable"]
                parent_id = (
                    tup.parent_config["configurable"].get("checkpoint_id")
                    if tup.parent_config
                    else None
                )
                put_config: dict[str, Any] = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": configurable.get("checkpoint_ns", ""),
                    }
                }
                if parent_id:
                    put_config["configurable"]["checkpoint_id"] = parent_id
                stored = await target.aput(
                    put_config,
                    tup.checkpoint,
                    tup.metadata,
                    dict(tup.checkpoint.get("channel_versions") or {}),
                )
                by_task: dict[str, list[tuple[str, Any]]] = defaultdict(list)
                for task_id, channel, value in tup.pending_writes or []:
                    by_task[task_id].append((channel, value))
                for task_id, task_writes in by_task.items():
                    await target.aput_writes(stored, task_writes, task_id)
                    writes += len(task_writes)
                checkpoints += 1
    print(f"  checkpoints        {checkpoints} checkpoints, {writes} writes, {len(threads)} threads")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sqlite", type=Path, default=Path("data/app.db"))
    parser.add_argument("--checkpoints", type=Path, default=Path("data/agent_checkpoints.db"))
    parser.add_argument("--database-url", default="")
    parser.add_argument("--truncate", action="store_true", help="empty the target first")
    args = parser.parse_args()

    from app.config import get_settings
    from app.db import database as db

    if not args.sqlite.exists():
        raise SystemExit(f"No SQLite database at {args.sqlite}")
    db.configure(args.database_url or get_settings().database_url)
    await db.init_db()
    try:
        print(f"Copying {args.sqlite} -> PostgreSQL")
        await copy_tables(args.sqlite, truncate=args.truncate)
        await copy_checkpoints(args.checkpoints)
    finally:
        await db.close_pool()
    print("Done.")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
