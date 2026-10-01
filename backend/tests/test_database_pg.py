"""The PostgreSQL access layer: translation, row shapes, concurrency, migrations."""

from __future__ import annotations

import asyncio
import re
import unittest

from tests.pg_support import open_fresh_database


class SqlTranslationTests(unittest.TestCase):
    def test_placeholders_and_percent_signs(self) -> None:
        from app.db.database import to_pg_sql

        self.assertEqual(
            to_pg_sql("SELECT * FROM t WHERE a = ? AND b LIKE '%x?%'"),
            "SELECT * FROM t WHERE a = %s AND b LIKE '%%x?%%'",
        )

    def test_quoted_identifiers_comments_and_escaped_quotes(self) -> None:
        from app.db.database import to_pg_sql

        sql = "SELECT \"we?ird\" -- what?\nFROM t WHERE c = 'it''s ?' AND d = ? /* ? */"
        self.assertEqual(
            to_pg_sql(sql),
            "SELECT \"we?ird\" -- what?\nFROM t WHERE c = 'it''s ?' AND d = %s /* ? */",
        )


class DatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await open_fresh_database(self)

    async def test_timestamps_keep_the_sqlite_shape(self) -> None:
        from app.db import database as db

        await db.execute(
            "INSERT INTO papers (paper_id, file_path, title) VALUES (?, ?, ?)",
            ("p1", "papers/p1/original.pdf", "T"),
        )
        row = await db.fetch_one("SELECT created_at FROM papers WHERE paper_id = ?", ("p1",))
        self.assertRegex(row["created_at"], r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")

    async def test_booleans_and_numerics(self) -> None:
        from app.db import database as db

        await db.execute(
            "INSERT INTO agent_memories (memory_id, owner_id, content, active) VALUES (?, ?, ?, ?)",
            ("m1", "pr_1", "x", True),
        )
        row = await db.fetch_one(
            "SELECT active, SUM(active) AS total, EXTRACT(EPOCH FROM interval '90 minutes') / 3600.0 AS h "
            "FROM agent_memories GROUP BY active"
        )
        self.assertEqual(row["active"], 1)
        self.assertIsInstance(row["total"], int)
        self.assertAlmostEqual(row["h"], 1.5)

    async def test_transaction_rolls_back_as_a_unit(self) -> None:
        from app.db import database as db

        with self.assertRaises(db.IntegrityError):
            async with db.transaction() as tx:
                await tx.execute(
                    "INSERT INTO papers (paper_id, file_path) VALUES (?, ?)", ("p2", "a")
                )
                await tx.execute(
                    "INSERT INTO papers (paper_id, file_path) VALUES (?, ?)", ("p2", "b")
                )
        self.assertIsNone(await db.fetch_one("SELECT 1 FROM papers WHERE paper_id = ?", ("p2",)))

    async def test_concurrent_appends_never_share_a_seq(self) -> None:
        """SQLite serialised these for free; PostgreSQL needs the advisory lock."""
        from app.db import agent_repository as repo
        from app.models.agent_models import EventType, MessageRole
        from app.services import identity

        owner, _ = await identity.create_principal()
        session = await repo.create_session(owner_id=owner)
        await asyncio.gather(
            *(
                repo.append_event(
                    session_id=session.session_id,
                    type=EventType.MESSAGE_DELTA,
                    payload={"text": str(i)},
                )
                for i in range(40)
            ),
            *(
                repo.append_message(
                    session_id=session.session_id, role=MessageRole.USER, content=str(i)
                )
                for i in range(20)
            ),
        )
        events = await repo.list_events(session.session_id)
        messages = await repo.list_messages(session.session_id)
        self.assertEqual([e.seq for e in events], list(range(1, 41)))
        self.assertEqual([m.seq for m in messages], list(range(1, 21)))

    async def test_migrations_are_recorded_and_idempotent(self) -> None:
        from app.db import database as db

        before = await db.fetch_all("SELECT version FROM schema_migrations ORDER BY version")
        await db.init_db()
        after = await db.fetch_all("SELECT version FROM schema_migrations ORDER BY version")
        self.assertEqual(before, after)
        self.assertTrue(all(re.fullmatch(r"\d{4}", r["version"]) for r in after))

    async def test_full_text_search_column_follows_the_row(self) -> None:
        from app.db import database as db

        await db.execute("INSERT INTO papers (paper_id, file_path) VALUES (?, ?)", ("p3", "x"))
        await db.execute(
            "INSERT INTO paper_nodes (node_id, paper_id, node_type, title_path, text_for_search) "
            "VALUES (?, ?, 'chunk', ?, ?)",
            ("n1", "p3", "Method", "contrastive-loss objective"),
        )
        rows = await db.fetch_all(
            "SELECT node_id FROM paper_nodes "
            "WHERE search_tsv @@ websearch_to_tsquery('simple', ?)",
            ('"objective" or "nothing"',),
        )
        self.assertEqual([r["node_id"] for r in rows], ["n1"])


if __name__ == "__main__":
    unittest.main()
