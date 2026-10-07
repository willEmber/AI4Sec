"""Deleting data: what a reader removes, and what the retention pass collects.

Against a throwaway PostgreSQL schema, on the P8 fixture: one owner, one
session, one parsed paper attached to it.
"""

from __future__ import annotations

from pathlib import Path

from tests.test_agent_p8 import P8TestCase


async def _count(table: str, where: str = "TRUE", params: tuple = ()) -> int:
    from app.db import database as db

    row = await db.fetch_one(f"SELECT COUNT(*) AS n FROM {table} WHERE {where}", params)
    return int(row["n"])


class LifecycleTestCase(P8TestCase):
    async def _finished_turn(self, session_id: str, owner_id: str, *, days_ago: int = 0) -> str:
        """A finished turn with one streamed fragment and one tool event."""
        from app.db import database as db

        run_id = await self._turn(session_id, "what is the top-2 result?", "It is 71.3.")
        await db.execute(
            "INSERT INTO agent_runs (run_id, session_id, owner_id, status, finished_at) "
            "VALUES (?, ?, ?, 'done', now() - make_interval(days => ?))",
            (run_id, session_id, owner_id, days_ago),
        )
        for seq, kind in ((9001, "message.delta"), (9002, "tool.completed")):
            await db.execute(
                "INSERT INTO agent_events (session_id, seq, run_id, type) VALUES (?, ?, ?, ?)",
                (session_id, seq, run_id, kind),
            )
        return run_id

    async def _report(self, run_id: str, *, session_id: str = "", owner_id: str = "") -> None:
        from app.db import database as db

        await db.execute(
            "INSERT INTO runs (run_id, paper_id, status, owner_id, agent_session_id) "
            "VALUES (?, 'paper1', 'done', ?, ?)",
            (run_id, owner_id or self.owner, session_id),
        )
        await db.execute(
            "INSERT INTO run_outputs (run_id, markdown) VALUES (?, '# report')", (run_id,)
        )


class ReaderDeletionTests(LifecycleTestCase):
    async def test_a_deleted_session_takes_what_was_made_in_it(self) -> None:
        from app.db import agent_repository as repo
        from app.services import data_lifecycle, identity

        sid = self.session.session_id
        await self._finished_turn(sid, self.owner)
        await self._evidence(self.owner, sid, "top-2 accuracy is 71.3")
        await self._report("r_in_session", session_id=sid)
        await self._report("r_standalone")
        kept = await repo.create_session(owner_id=self.owner, language="en")
        await self._turn(kept.session_id, "another question", "another answer")

        other, _ = await identity.create_principal()
        with self.assertRaises(repo.SessionNotFound):
            await data_lifecycle.delete_session(sid, owner_id=other)
        self.assertEqual(await _count("agent_messages", "session_id = ?", (sid,)), 2)

        await data_lifecycle.delete_session(sid, owner_id=self.owner)

        for table in ("agent_sessions", "agent_messages", "agent_events", "agent_runs",
                      "session_papers", "evidence"):
            self.assertEqual(await _count(table, "session_id = ?", (sid,)), 0, table)
        self.assertEqual(await _count("runs", "run_id = 'r_in_session'"), 0)
        self.assertEqual(await _count("run_outputs", "run_id = 'r_in_session'"), 0)
        # What was not made in that conversation stays.
        self.assertEqual(await _count("runs", "run_id = 'r_standalone'"), 1)
        self.assertEqual(await _count("agent_messages", "session_id = ?", (kept.session_id,)), 2)
        self.assertEqual(await _count("papers", "paper_id = 'paper1'"), 1)

    async def test_a_project_is_deleted_with_or_without_its_sessions(self) -> None:
        from app.db import agent_repository as repo
        from app.services import data_lifecycle, identity

        sid = self.session.session_id
        project = await repo.create_project(owner_id=self.owner, title="MoE")
        await repo.set_session_project(sid, owner_id=self.owner, project_id=project.project_id)

        other, _ = await identity.create_principal()
        with self.assertRaises(repo.ProjectNotFound):
            await data_lifecycle.delete_project(
                project.project_id, owner_id=other, delete_sessions=True
            )

        deleted = await data_lifecycle.delete_project(
            project.project_id, owner_id=self.owner, delete_sessions=False
        )
        self.assertEqual(deleted, 0)
        self.assertEqual(await _count("agent_projects"), 0)
        self.assertEqual((await repo.get_session(sid, owner_id=self.owner)).project_id, "")

        project = await repo.create_project(owner_id=self.owner, title="MoE again")
        await repo.set_session_project(sid, owner_id=self.owner, project_id=project.project_id)
        deleted = await data_lifecycle.delete_project(
            project.project_id, owner_id=self.owner, delete_sessions=True
        )
        self.assertEqual(deleted, 1)
        self.assertEqual(await _count("agent_sessions", "session_id = ?", (sid,)), 0)

    async def test_a_deleted_principal_leaves_nothing_and_cannot_come_back(self) -> None:
        from app.db import agent_repository as repo
        from app.services import data_lifecycle, identity

        await self._finished_turn(self.session.session_id, self.owner)
        await self._report("r_mine")
        await repo.create_project(owner_id=self.owner, title="MoE")
        self.assertEqual(
            await data_lifecycle.owned_counts(self.owner),
            {"sessions": 1, "runs": 1, "projects": 1},
        )

        removed = await data_lifecycle.delete_principal(self.owner)

        self.assertEqual(removed, {"sessions": 1, "runs": 1})
        for table in ("agent_sessions", "agent_runs", "agent_projects", "runs", "agent_principals"):
            self.assertEqual(await _count(table), 0, table)
        self.assertIsNone(await identity.resolve_principal(self.credential))


class RetentionTests(LifecycleTestCase):
    async def test_only_fragments_of_turns_long_finished_are_pruned(self) -> None:
        from app.db import agent_repository as repo
        from app.services import data_lifecycle

        old = await self._finished_turn(self.session.session_id, self.owner, days_ago=30)
        recent_session = await repo.create_session(owner_id=self.owner, language="en")
        recent = await self._finished_turn(recent_session.session_id, self.owner, days_ago=1)

        self.assertEqual(await data_lifecycle.prune_answer_fragments(7, dry_run=True), 1)
        self.assertEqual(await _count("agent_events"), 4)   # a dry run removes nothing
        self.assertEqual(await data_lifecycle.prune_answer_fragments(7), 1)

        self.assertEqual(await _count("agent_events", "run_id = ?", (old,)), 1)
        self.assertEqual(
            await _count("agent_events", "run_id = ? AND type = 'tool.completed'", (old,)), 1
        )
        self.assertEqual(await _count("agent_events", "run_id = ?", (recent,)), 2)

    async def test_an_idle_visitor_goes_and_their_paper_becomes_an_orphan(self) -> None:
        from app.config import get_settings
        from app.db import database as db
        from app.services import data_lifecycle, identity

        paper_dir = get_settings().data_dir / "papers" / "paper1"
        (paper_dir / "mineru" / "raw").mkdir(parents=True)
        (paper_dir / "original.pdf").write_bytes(b"%PDF-1.4 " + b"x" * 100)
        await db.execute(
            "INSERT INTO agent_jobs (job_id, kind, idempotency_key, status) "
            "VALUES ('j1', 'parse', 'parse:paper1:vlm', 'done')"
        )
        await db.execute("UPDATE papers SET created_at = now() - make_interval(days => 60)")
        active, _ = await identity.create_principal()

        # Referred to by a conversation: old, but nobody's orphan.
        self.assertEqual((await data_lifecycle.collect_orphan_papers(30))["papers"], 0)
        # Seen recently: not idle.
        self.assertEqual(await data_lifecycle.purge_idle_visitors(90), 0)

        await db.execute(
            "UPDATE agent_principals SET last_seen_at = now() - make_interval(days => 120) "
            "WHERE principal_id = ?",
            (self.owner,),
        )
        self.assertEqual(await data_lifecycle.purge_idle_visitors(90, dry_run=True), 1)
        self.assertEqual(await _count("agent_sessions"), 1)
        self.assertEqual(await data_lifecycle.purge_idle_visitors(90), 1)
        self.assertEqual(await _count("agent_sessions"), 0)
        self.assertEqual(await _count("agent_principals", "principal_id = ?", (active,)), 1)

        dry = await data_lifecycle.collect_orphan_papers(30, dry_run=True)
        self.assertEqual(dry["papers"], 1)
        self.assertGreater(dry["bytes"], 100)
        self.assertTrue(paper_dir.exists())

        self.assertEqual((await data_lifecycle.collect_orphan_papers(30))["papers"], 1)
        for table in ("papers", "paper_nodes", "agent_jobs"):
            self.assertEqual(await _count(table), 0, table)
        self.assertFalse(paper_dir.exists())

    async def test_an_extracted_archive_and_the_pdf_copy_are_dropped(self) -> None:
        from app.services.mineru_adapter import drop_redundant_parse_files

        raw = Path(self._tmp.name) / "papers" / "p" / "mineru" / "raw"
        (raw / "data1" / "images").mkdir(parents=True)
        (raw / "data1.zip").write_bytes(b"z" * 10)
        (raw / "data1" / "abc_origin.pdf").write_bytes(b"p" * 20)
        (raw / "data1" / "content_list.json").write_text("[]")
        (raw / "data2.zip").write_bytes(b"z" * 40)   # downloaded, not extracted yet

        self.assertEqual(drop_redundant_parse_files(raw, dry_run=True), 30)
        self.assertTrue((raw / "data1.zip").exists())
        self.assertEqual(drop_redundant_parse_files(raw), 30)
        self.assertEqual(
            sorted(p.relative_to(raw).as_posix() for p in raw.rglob("*") if p.is_file()),
            ["data1/content_list.json", "data2.zip"],
        )

    async def test_a_pass_skips_rules_that_are_switched_off(self) -> None:
        import os

        from app.config import get_settings
        from app.services import data_lifecycle

        os.environ.update(RETENTION_ANON_DAYS="0", RETENTION_ORPHAN_PAPER_DAYS="0")
        self.addCleanup(os.environ.pop, "RETENTION_ANON_DAYS", None)
        self.addCleanup(os.environ.pop, "RETENTION_ORPHAN_PAPER_DAYS", None)
        get_settings.cache_clear()

        report = await data_lifecycle.run_retention(dry_run=True)
        self.assertEqual(
            sorted(report), ["answer_fragments", "log_rows", "parse_archive_bytes"]
        )
