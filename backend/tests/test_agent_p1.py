"""P1: sessions, identity, literature mapping, parse versions and evidence.

Everything runs against a throwaway PostgreSQL schema with no network access. The
exit criteria the development plan sets for P1 are covered directly:

* a candidate paper can be linked to a local PDF (`LiteratureCatalogTests`),
* a citation resolves (`EvidenceTests.test_evidence_resolves_with_locator`),
* evidence survives a re-parse (`EvidenceTests.test_evidence_survives_reparse`).
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest


class AgentDataTestCase(unittest.IsolatedAsyncioTestCase):
    """Boots a temp database and applies schema + migrations once per test."""

    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["DATA_DIR"] = self._tmp.name

        from app.config import get_settings

        get_settings.cache_clear()

        from app.services import identity
        from tests.pg_support import open_fresh_database

        identity.reset_secret_cache()
        await open_fresh_database(self)

    async def asyncTearDown(self) -> None:
        os.environ.pop("DATA_DIR", None)
        from app.config import get_settings
        from app.services import identity

        identity.reset_secret_cache()
        get_settings.cache_clear()
        self._tmp.cleanup()

    async def _principal(self) -> str:
        from app.services import identity

        principal_id, _ = await identity.create_principal()
        return principal_id

    async def _paper(self, paper_id: str, *, title: str = "A paper", doi: str = "") -> None:
        from app.db import database as db

        await db.execute(
            "INSERT INTO papers (paper_id, file_path, title, doi) VALUES (?, ?, ?, ?)",
            (paper_id, f"papers/{paper_id}/original.pdf", title, doi),
        )


class MigrationTests(AgentDataTestCase):
    async def test_agent_tables_exist_and_are_recorded(self) -> None:
        from app.db import database as db

        rows = await db.fetch_all(
            "SELECT table_name AS name FROM information_schema.tables "
            "WHERE table_schema = current_schema() ORDER BY table_name"
        )
        names = {r["name"] for r in rows}
        for table in (
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
        ):
            self.assertIn(table, names)

        applied = await db.fetch_all("SELECT version FROM schema_migrations")
        self.assertIn("0001", {r["version"] for r in applied})

    async def test_migrations_are_idempotent(self) -> None:
        from app.db import database as db
        from app.db.migrations import apply_migrations

        async with (await db.open_pool()).connection() as conn:
            second_pass = await apply_migrations(conn)
        self.assertEqual(second_pass, [], "a re-run re-applied a migration")


class IdentityTests(AgentDataTestCase):
    async def test_credential_round_trip(self) -> None:
        from app.services import identity

        principal_id, credential = await identity.create_principal()
        self.assertEqual(await identity.resolve_principal(credential), principal_id)

    async def test_forged_credential_is_rejected(self) -> None:
        from app.services import identity

        _, credential = await identity.create_principal()
        victim, _ = await identity.create_principal()
        signature = credential.rpartition(".")[2]
        # Someone else's id with a signature that was issued for a different id.
        self.assertIsNone(identity.verify_credential(f"{victim}.{signature}"))
        self.assertIsNone(identity.verify_credential(victim))
        self.assertIsNone(identity.verify_credential(""))

    async def test_valid_signature_over_deleted_principal_is_rejected(self) -> None:
        from app.db import database as db
        from app.services import identity

        principal_id, credential = await identity.create_principal()
        await db.execute(
            "DELETE FROM agent_principals WHERE principal_id = ?", (principal_id,)
        )
        self.assertIsNone(await identity.resolve_principal(credential))


class SessionTests(AgentDataTestCase):
    async def test_sessions_are_isolated_by_owner(self) -> None:
        from app.db import agent_repository as repo

        alice = await self._principal()
        bob = await self._principal()
        session = await repo.create_session(owner_id=alice, title="Alice's work")

        self.assertEqual(
            (await repo.get_session(session.session_id, owner_id=alice)).title,
            "Alice's work",
        )
        with self.assertRaises(repo.SessionNotFound):
            await repo.get_session(session.session_id, owner_id=bob)
        self.assertEqual(await repo.list_sessions(bob), [])

    async def test_session_gets_a_thread_id(self) -> None:
        from app.db import agent_repository as repo

        owner = await self._principal()
        a = await repo.create_session(owner_id=owner)
        b = await repo.create_session(owner_id=owner)
        self.assertTrue(a.thread_id)
        self.assertNotEqual(a.thread_id, b.thread_id)


class RunTests(AgentDataTestCase):
    async def _session(self):
        from app.db import agent_repository as repo

        owner = await self._principal()
        return owner, await repo.create_session(owner_id=owner)

    async def test_same_client_request_id_returns_the_same_run(self) -> None:
        from app.db import agent_repository as repo

        owner, session = await self._session()
        first, dedup_first = await repo.create_run(
            session_id=session.session_id, owner_id=owner, client_request_id="req-1"
        )
        second, dedup_second = await repo.create_run(
            session_id=session.session_id, owner_id=owner, client_request_id="req-1"
        )
        self.assertFalse(dedup_first)
        self.assertTrue(dedup_second)
        self.assertEqual(first.run_id, second.run_id)

    async def test_second_concurrent_run_conflicts(self) -> None:
        from app.db import agent_repository as repo

        owner, session = await self._session()
        await repo.create_run(
            session_id=session.session_id, owner_id=owner, client_request_id="req-1"
        )
        with self.assertRaises(repo.ActiveRunConflict):
            await repo.create_run(
                session_id=session.session_id, owner_id=owner, client_request_id="req-2"
            )

    async def test_a_finished_run_frees_the_session(self) -> None:
        from app.db import agent_repository as repo
        from app.models.agent_models import RunStatus

        owner, session = await self._session()
        first, _ = await repo.create_run(session_id=session.session_id, owner_id=owner)
        await repo.finish_run(first.run_id, status=RunStatus.DONE)
        second, _ = await repo.create_run(session_id=session.session_id, owner_id=owner)
        self.assertNotEqual(first.run_id, second.run_id)

    async def test_cancel_is_idempotent_and_owner_scoped(self) -> None:
        from app.db import agent_repository as repo

        owner, session = await self._session()
        intruder = await self._principal()
        run, _ = await repo.create_run(session_id=session.session_id, owner_id=owner)

        self.assertTrue((await repo.request_cancel(run.run_id, owner_id=owner)).cancel_requested)
        self.assertTrue((await repo.request_cancel(run.run_id, owner_id=owner)).cancel_requested)
        self.assertTrue(await repo.is_cancel_requested(run.run_id))
        with self.assertRaises(repo.SessionNotFound):
            await repo.request_cancel(run.run_id, owner_id=intruder)


class EventTests(AgentDataTestCase):
    async def test_sequence_is_monotonic_per_session_and_resumable(self) -> None:
        from app.db import agent_repository as repo
        from app.models.agent_models import EventType

        owner = await self._principal()
        session = await repo.create_session(owner_id=owner)
        other = await repo.create_session(owner_id=owner)

        seqs = []
        for _ in range(5):
            event = await repo.append_event(
                session_id=session.session_id, type=EventType.TOOL_STARTED
            )
            seqs.append(event.seq)
        self.assertEqual(seqs, [1, 2, 3, 4, 5])

        # A second session numbers its own events from one.
        first_elsewhere = await repo.append_event(
            session_id=other.session_id, type=EventType.RUN_STARTED
        )
        self.assertEqual(first_elsewhere.seq, 1)

        resumed = await repo.list_events(session.session_id, after_seq=3)
        self.assertEqual([e.seq for e in resumed], [4, 5])
        self.assertEqual(await repo.last_event_seq(session.session_id), 5)

    async def test_concurrent_appends_do_not_collide(self) -> None:
        from app.db import agent_repository as repo
        from app.models.agent_models import EventType

        owner = await self._principal()
        session = await repo.create_session(owner_id=owner)
        events = await asyncio.gather(
            *(
                repo.append_event(session_id=session.session_id, type=EventType.MESSAGE_DELTA)
                for _ in range(10)
            )
        )
        self.assertEqual(sorted(e.seq for e in events), list(range(1, 11)))

    async def test_sse_frame_carries_seq_as_id(self) -> None:
        from app.db import agent_repository as repo
        from app.models.agent_models import EventType

        owner = await self._principal()
        session = await repo.create_session(owner_id=owner)
        event = await repo.append_event(
            session_id=session.session_id, type=EventType.RUN_STARTED, payload={"x": 1}
        )
        frame = event.to_sse()
        self.assertEqual(frame["id"], str(event.seq))
        self.assertEqual(frame["event"], "run.started")


class LiteratureCatalogTests(AgentDataTestCase):
    async def test_same_doi_merges_into_one_work(self) -> None:
        from app.services import paper_catalog

        first = await paper_catalog.upsert_literature_item(
            title="Attention Is All You Need", doi="10.1000/ABC", source="arxiv"
        )
        second = await paper_catalog.upsert_literature_item(
            title="Attention is all you need",
            doi="https://doi.org/10.1000/abc",
            venue="NeurIPS",
            year=2017,
            source="openalex",
        )
        self.assertEqual(first, second)

        item = await paper_catalog.get_literature_item(first)
        self.assertEqual(item["venue"], "NeurIPS")
        self.assertEqual(item["year"], 2017)
        self.assertTrue(item["year_known"])

    async def test_enrichment_never_blanks_a_known_field(self) -> None:
        from app.services import paper_catalog

        lit = await paper_catalog.upsert_literature_item(
            title="A paper", doi="10.1/x", venue="NeurIPS", year=2020
        )
        await paper_catalog.upsert_literature_item(title="A paper", doi="10.1/x")
        item = await paper_catalog.get_literature_item(lit)
        self.assertEqual(item["venue"], "NeurIPS")
        self.assertEqual(item["year"], 2020)

    async def test_unknown_year_stays_distinguishable_from_zero(self) -> None:
        from app.services import paper_catalog

        lit = await paper_catalog.upsert_literature_item(title="Undated", doi="10.1/y")
        item = await paper_catalog.get_literature_item(lit)
        self.assertEqual(item["year"], 0)
        self.assertFalse(item["year_known"])

    async def test_candidate_links_to_a_local_pdf(self) -> None:
        """P1 exit criterion: a candidate can be attached to a downloaded file."""
        from app.services import paper_catalog

        await self._paper("sha1aaa", title="Attention Is All You Need")
        lit = await paper_catalog.upsert_literature_item(
            title="Attention Is All You Need", arxiv_id="arXiv:1706.03762v5"
        )
        await paper_catalog.link_paper_file(
            literature_id=lit, paper_id="sha1aaa", origin="arxiv"
        )

        self.assertEqual(await paper_catalog.literature_for_paper("sha1aaa"), lit)
        self.assertEqual(await paper_catalog.primary_paper_for_literature(lit), "sha1aaa")

        # A second file for the same work: the newest primary wins, both stay.
        await self._paper("sha1bbb", title="Attention Is All You Need (VoR)")
        await paper_catalog.link_paper_file(
            literature_id=lit, paper_id="sha1bbb", origin="oa_resolver"
        )
        self.assertEqual(await paper_catalog.primary_paper_for_literature(lit), "sha1bbb")
        self.assertEqual(await paper_catalog.literature_for_paper("sha1aaa"), lit)

    async def test_upload_without_a_doi_still_gets_a_work_record(self) -> None:
        from app.services import paper_catalog

        await self._paper("sha1ccc", title="Untitled upload")
        lit = await paper_catalog.ensure_literature_for_local_paper("sha1ccc")
        self.assertTrue(lit)
        # Calling again is stable rather than creating a second work.
        self.assertEqual(await paper_catalog.ensure_literature_for_local_paper("sha1ccc"), lit)


class PaperVersionTests(AgentDataTestCase):
    async def test_identical_reparse_reuses_the_version(self) -> None:
        from app.services import paper_catalog

        await self._paper("sha1ddd")
        first = await paper_catalog.register_paper_version(
            paper_id="sha1ddd", content_hash="hash-1", parser_config={"backend": "vlm"}
        )
        again = await paper_catalog.register_paper_version(
            paper_id="sha1ddd", content_hash="hash-1", parser_config={"backend": "vlm"}
        )
        self.assertEqual(first, again)
        self.assertEqual(len(await paper_catalog.list_versions("sha1ddd")), 1)

    async def test_new_content_supersedes_the_previous_version(self) -> None:
        from app.services import paper_catalog

        await self._paper("sha1eee")
        old = await paper_catalog.register_paper_version(
            paper_id="sha1eee", content_hash="hash-1"
        )
        new = await paper_catalog.register_paper_version(
            paper_id="sha1eee", content_hash="hash-2"
        )
        self.assertNotEqual(old, new)

        current = await paper_catalog.get_current_version("sha1eee")
        self.assertEqual(current["version_id"], new)
        # Exactly one current version, enforced by a partial unique index.
        versions = await paper_catalog.list_versions("sha1eee")
        self.assertEqual(sum(1 for v in versions if v["is_current"]), 1)
        self.assertEqual(
            next(v for v in versions if v["version_id"] == old)["status"], "superseded"
        )


class EvidenceTests(AgentDataTestCase):
    async def _fixture(self):
        from app.db import agent_repository as repo
        from app.services import paper_catalog

        owner = await self._principal()
        session = await repo.create_session(owner_id=owner)
        await self._paper("sha1fff", title="MoE Routing")
        lit = await paper_catalog.upsert_literature_item(
            title="MoE Routing", doi="10.1/moe", year=2024
        )
        await paper_catalog.link_paper_file(literature_id=lit, paper_id="sha1fff")
        version = await paper_catalog.register_paper_version(
            paper_id="sha1fff", content_hash="parse-v1"
        )
        return owner, session, lit, version

    async def test_evidence_resolves_with_locator(self) -> None:
        """P1 exit criterion: a citation resolves to a paper, page and quote."""
        from app.models.evidence_models import Locator
        from app.services import evidence_service

        owner, session, _lit, version = await self._fixture()
        stored = await evidence_service.record_fulltext_evidence(
            paper_id="sha1fff",
            quote="We propose a top-2 gated router.",
            owner_id=owner,
            session_id=session.session_id,
            locator=Locator(section_path="2 Method", page_index=2, block_start=10, block_end=12),
        )

        resolved = await evidence_service.resolve(stored.evidence_id, owner_id=owner)
        self.assertEqual(resolved.quote, "We propose a top-2 gated router.")
        self.assertEqual(resolved.parse_version, version)
        self.assertEqual(resolved.paper_id, "sha1fff")
        # Stored 0-based, shown 1-based — the contract's single conversion point.
        self.assertEqual(resolved.locator.page_index, 2)
        self.assertEqual(resolved.locator.page_label, 3)
        self.assertEqual(resolved.to_api()["locator"]["page_label"], 3)

        payload = await evidence_service.evidence_to_api(resolved)
        self.assertEqual(payload["literature"]["doi"], "10.1/moe")

    async def test_recording_the_same_passage_twice_yields_one_row(self) -> None:
        from app.models.evidence_models import Locator
        from app.services import evidence_service

        owner, session, _lit, _version = await self._fixture()
        locator = Locator(section_path="2 Method", page_index=2)
        kwargs = dict(
            paper_id="sha1fff",
            quote="Same passage.",
            owner_id=owner,
            session_id=session.session_id,
            locator=locator,
        )
        first = await evidence_service.record_fulltext_evidence(**kwargs)
        second = await evidence_service.record_fulltext_evidence(**kwargs)
        self.assertEqual(first.evidence_id, second.evidence_id)

        changed = await evidence_service.record_fulltext_evidence(
            **{**kwargs, "quote": "A different passage."}
        )
        self.assertNotEqual(first.evidence_id, changed.evidence_id)

    async def test_evidence_survives_reparse(self) -> None:
        """P1 exit criterion: an old citation still resolves after a re-parse.

        The re-parse wipes `blocks` and registers a new version, exactly as the
        real pipeline does. The old evidence must keep its quote, its locator
        and its own version, and must report that it predates the current parse.
        """
        from app.db import database as db
        from app.models.evidence_models import Locator
        from app.services import evidence_service, paper_catalog

        owner, session, _lit, old_version = await self._fixture()
        quote = "The router reaches 29.7 BLEU on WMT14 En-De."
        await db.execute(
            "INSERT INTO blocks (paper_id, type, page_idx, text, section_path, order_idx) "
            "VALUES (?, 'text', 5, ?, '3 Experiments', 42)",
            ("sha1fff", quote),
        )
        stored = await evidence_service.record_fulltext_evidence(
            paper_id="sha1fff",
            quote=quote,
            owner_id=owner,
            session_id=session.session_id,
            locator=Locator(section_path="3 Experiments", page_index=5, block_start=42),
        )

        # Re-parse: blocks are deleted and reinserted, so any block_id-based
        # pointer would now be dangling.
        await db.execute("DELETE FROM blocks WHERE paper_id = ?", ("sha1fff",))
        await db.execute(
            "INSERT INTO blocks (paper_id, type, page_idx, text, section_path, order_idx) "
            "VALUES (?, 'text', 5, ?, '3 Experiments', 7)",
            ("sha1fff", quote),
        )
        new_version = await paper_catalog.register_paper_version(
            paper_id="sha1fff", content_hash="parse-v2"
        )
        self.assertNotEqual(old_version, new_version)

        resolved = await evidence_service.resolve(stored.evidence_id, owner_id=owner)
        self.assertEqual(resolved.quote, quote)
        self.assertEqual(resolved.parse_version, old_version)
        self.assertEqual(resolved.locator.page_label, 6)

        checks = await evidence_service.validate_citations(
            [stored.evidence_id], owner_id=owner
        )
        self.assertTrue(checks[0].ok)

        status = await evidence_service.check_against_current_parse(resolved)
        self.assertFalse(status["is_current_version"])
        self.assertEqual(status["current_version"], new_version)
        self.assertTrue(status["still_present"])

    async def test_reparse_that_drops_the_text_is_reported(self) -> None:
        from app.db import database as db
        from app.models.evidence_models import Locator
        from app.services import evidence_service, paper_catalog

        owner, session, _lit, _old = await self._fixture()
        quote = "A sentence the next parser will mangle beyond recognition."
        await db.execute(
            "INSERT INTO blocks (paper_id, type, page_idx, text, order_idx) "
            "VALUES (?, 'text', 1, ?, 1)",
            ("sha1fff", quote),
        )
        stored = await evidence_service.record_fulltext_evidence(
            paper_id="sha1fff",
            quote=quote,
            owner_id=owner,
            session_id=session.session_id,
            locator=Locator(page_index=1),
        )
        await db.execute("DELETE FROM blocks WHERE paper_id = ?", ("sha1fff",))
        await paper_catalog.register_paper_version(
            paper_id="sha1fff", content_hash="parse-v3"
        )

        resolved = await evidence_service.resolve(stored.evidence_id, owner_id=owner)
        # The citation still resolves — the snapshot is the point.
        self.assertEqual(resolved.quote, quote)
        status = await evidence_service.check_against_current_parse(resolved)
        self.assertFalse(status["still_present"])

    async def test_evidence_is_owner_scoped(self) -> None:
        from app.db import agent_repository as repo
        from app.services import evidence_service

        owner, session, _lit, _version = await self._fixture()
        intruder = await self._principal()
        stored = await evidence_service.record_fulltext_evidence(
            paper_id="sha1fff",
            quote="Private reading.",
            owner_id=owner,
            session_id=session.session_id,
        )
        with self.assertRaises(repo.EvidenceNotFound):
            await evidence_service.resolve(stored.evidence_id, owner_id=intruder)

        checks = await evidence_service.validate_citations(
            [stored.evidence_id], owner_id=intruder
        )
        self.assertFalse(checks[0].ok)
        self.assertTrue(checks[0].exists)
        self.assertFalse(checks[0].authorized)

    async def test_unknown_citation_is_reported_not_raised(self) -> None:
        from app.services import evidence_service

        owner, _session, _lit, _version = await self._fixture()
        checks = await evidence_service.validate_citations(["ev_nope"], owner_id=owner)
        self.assertFalse(checks[0].ok)
        self.assertFalse(checks[0].exists)

    async def test_metadata_evidence_carries_no_page(self) -> None:
        from app.db import agent_repository as repo
        from app.services import evidence_service

        owner, session, lit, _version = await self._fixture()
        stored = await evidence_service.record_metadata_evidence(
            literature_id=lit,
            quote="Cited by 1,203 works (OpenAlex, 2026-09-17).",
            owner_id=owner,
            session_id=session.session_id,
            provider="openalex",
            source_url="https://api.openalex.org/works/W1",
        )
        resolved = await repo.get_evidence(stored.evidence_id, owner_id=owner)
        self.assertIsNone(resolved.locator.page_index)
        self.assertIsNone(resolved.locator.page_label)
        self.assertEqual(resolved.source_level.value, "metadata")
        self.assertEqual(resolved.paper_id, "")


class SessionPaperTests(AgentDataTestCase):
    async def test_attachment_tracks_availability(self) -> None:
        from app.db import agent_repository as repo
        from app.models.agent_models import Availability
        from app.services import paper_catalog

        owner = await self._principal()
        session = await repo.create_session(owner_id=owner)
        lit = await paper_catalog.upsert_literature_item(title="Candidate", doi="10.1/c")

        await repo.attach_session_paper(session_id=session.session_id, literature_id=lit)
        papers = await repo.list_session_papers(session.session_id)
        self.assertEqual(papers[0].availability, Availability.CANDIDATE)
        self.assertEqual(papers[0].paper_id, "")
        self.assertFalse(papers[0].year_known)

        await self._paper("sha1ggg")
        await repo.attach_session_paper(
            session_id=session.session_id,
            literature_id=lit,
            paper_id="sha1ggg",
            availability=Availability.PARSED,
        )
        papers = await repo.list_session_papers(session.session_id)
        self.assertEqual(len(papers), 1, "re-attaching created a duplicate row")
        self.assertEqual(papers[0].paper_id, "sha1ggg")
        self.assertEqual(papers[0].availability, Availability.PARSED)
        self.assertTrue(await repo.session_owns_paper(session.session_id, "sha1ggg"))

    async def test_unavailable_fulltext_is_expressible(self) -> None:
        from app.db import agent_repository as repo
        from app.models.agent_models import Availability
        from app.services import paper_catalog

        owner = await self._principal()
        session = await repo.create_session(owner_id=owner)
        lit = await paper_catalog.upsert_literature_item(title="Paywalled", doi="10.1/p")
        await repo.attach_session_paper(
            session_id=session.session_id,
            literature_id=lit,
            availability=Availability.UNAVAILABLE,
            note="No open-access copy; abstract only.",
        )
        papers = await repo.list_session_papers(session.session_id)
        self.assertEqual(papers[0].availability, Availability.UNAVAILABLE)
        self.assertIn("abstract only", papers[0].note)


if __name__ == "__main__":
    unittest.main()
