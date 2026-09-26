"""P3: discovery, acquisition, comparison, and the boundaries an answer must respect.

No network. Every provider is stubbed, because what these tests are about is
not whether OpenAlex answers — it is what the agent is told when it does not,
and whether the tools keep the distinctions that make an answer checkable: an
unknown year is not a year, an abstract is not the full text, a web-search
ranking is not a ranking database, and a paper that is silent does not disagree.

Whether a real model acts on those distinctions is the other half of P3 and
lives in `scripts/verify_research_agent.py`.
"""

from __future__ import annotations

import json
from typing import Any
from unittest import mock

from tests.test_agent_p2 import AgentP2TestCase


class _Meta:
    """Stands in for citation_graph.PaperMetadata."""

    def __init__(self, **kw: Any) -> None:
        from app.services.citation_graph import PaperMetadata

        self._m = PaperMetadata(**kw)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._m, name)


def _metadata(**kw: Any):
    from app.services.citation_graph import PaperMetadata

    return PaperMetadata(**kw)


def _outcome(rows: list[dict[str, Any]]):
    """What `search_papers_detailed` returns, built from plain rows."""
    from app.services.paper_search import Paper, PlatformStatus, SearchOutcome

    papers = [Paper(**row) for row in rows]
    return SearchOutcome(
        papers=papers,
        platforms=[PlatformStatus(platform="OpenAlex", status="ok", count=len(papers))],
    )


class SearchToolTests(AgentP2TestCase):
    async def test_year_filter_excludes_out_of_range_and_isolates_unknown_years(self) -> None:
        """A date filter must not silently swallow papers whose year is unknown."""
        from app.agents.tools import search_papers

        payload = _outcome(
            [
                {"title": "In range", "abstract": "a", "year": 2023, "doi": "10.1/a",
                 "url": "http://x/a", "authors": "A", "source_platform": "OpenAlex"},
                {"title": "Too old", "abstract": "b", "year": 2015, "doi": "10.1/b",
                 "url": "http://x/b", "authors": "B", "source_platform": "OpenAlex"},
                {"title": "No year at all", "abstract": "c", "year": 0, "doi": "10.1/c",
                 "url": "http://x/c", "authors": "C", "source_platform": "arXiv"},
            ]
        )
        with mock.patch(
            "app.agents.tools.discovery.run_search",
            new=mock.AsyncMock(return_value=payload),
        ):
            result = await self._call(
                search_papers, query="sparse routing", year_from=2020, limit=5
            )

        data = result["data"]
        self.assertEqual([p["title"] for p in data["results"]], ["In range"])
        self.assertEqual(data["counts"]["excluded_by_year"], 1)
        self.assertEqual([p["title"] for p in data["unknown_year"]], ["No year at all"])
        # Partial, not ok: the filter could not be checked against every candidate.
        self.assertEqual(result["status"], "partial")
        self.assertIn("year", result["note"].lower())

    async def test_year_is_null_not_zero_when_unknown(self) -> None:
        from app.agents.tools import search_papers

        payload = _outcome(
            [{"title": "Undated work", "abstract": "x", "year": 0, "doi": "",
              "url": "", "authors": "", "source_platform": "arXiv"}]
        )
        with mock.patch(
            "app.agents.tools.discovery.run_search",
            new=mock.AsyncMock(return_value=payload),
        ):
            result = await self._call(search_papers, query="anything")

        paper = result["data"]["results"][0]
        self.assertIsNone(paper["year"])
        self.assertFalse(paper["year_known"])

    async def test_results_are_abstract_evidence_never_fulltext(self) -> None:
        """A search result must not be citable as though the paper had been read."""
        from app.agents.tools import search_papers
        from app.services import evidence_service

        payload = _outcome(
            [{"title": "Some paper", "abstract": "The abstract text.", "year": 2024,
              "doi": "10.1/z", "url": "http://x/z", "authors": "Z",
              "source_platform": "OpenAlex"}]
        )
        with mock.patch(
            "app.agents.tools.discovery.run_search",
            new=mock.AsyncMock(return_value=payload),
        ):
            result = await self._call(search_papers, query="q")

        evidence_id = result["data"]["results"][0]["evidence_id"]
        evidence = await evidence_service.resolve(evidence_id, owner_id=self.owner)
        self.assertEqual(evidence.source_level.value, "abstract")
        self.assertEqual(evidence.locator.page_label, None)

    async def test_empty_query_is_rejected_without_calling_the_provider(self) -> None:
        from app.agents.tools import search_papers

        search = mock.AsyncMock()
        with mock.patch("app.agents.tools.discovery.run_search", new=search):
            result = await self._call(search_papers, query="   ")

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "invalid_argument")
        search.assert_not_awaited()

    async def test_provider_failure_is_retryable_not_fatal(self) -> None:
        from app.agents.tools import search_papers

        with mock.patch(
            "app.agents.tools.discovery.run_search",
            new=mock.AsyncMock(side_effect=RuntimeError("upstream 503")),
        ):
            result = await self._call(search_papers, query="q")

        self.assertEqual(result["status"], "error")
        self.assertTrue(result["error"]["retryable"])


class ResolveToolTests(AgentP2TestCase):
    async def test_resolve_registers_the_work_and_returns_its_id(self) -> None:
        from app.agents.tools import resolve_paper
        from app.services import paper_catalog

        meta = _metadata(
            title="Attention Is All You Need", doi="10.1/attn",
            openalex_id="W1", year=2017, venue="NeurIPS", abstract_text="We propose…",
        )
        with (
            mock.patch(
                "app.services.citation_graph.openalex_resolve_id",
                new=mock.AsyncMock(return_value="W1"),
            ),
            mock.patch(
                "app.services.citation_graph.openalex_get_metadata",
                new=mock.AsyncMock(return_value=meta),
            ),
        ):
            result = await self._call(resolve_paper, doi="10.1/attn")

        self.assertEqual(result["status"], "ok")
        literature_id = result["data"]["literature_id"]
        stored = await paper_catalog.get_literature_item(literature_id)
        self.assertEqual(stored["title"], "Attention Is All You Need")
        self.assertEqual(stored["year"], 2017)
        self.assertTrue(stored["year_known"])

    async def test_unindexed_identifier_still_yields_a_workable_record(self) -> None:
        """An arXiv-only paper no index knows is still downloadable."""
        from app.agents.tools import resolve_paper

        with (
            mock.patch(
                "app.services.citation_graph.openalex_resolve_id",
                new=mock.AsyncMock(return_value=None),
            ),
            mock.patch(
                "app.services.citation_graph.s2_resolve_id",
                new=mock.AsyncMock(return_value=None),
            ),
        ):
            result = await self._call(resolve_paper, arxiv_id="2401.01234")

        self.assertEqual(result["status"], "partial")
        self.assertTrue(result["data"]["literature_id"])
        self.assertFalse(result["data"]["year_known"])
        self.assertIn("unverified", result["note"].lower())

    async def test_an_incomplete_record_names_what_is_missing(self) -> None:
        """A partial record must say which field is absent, not just be terse."""
        from app.agents.tools import resolve_paper

        meta = _metadata(title="Undated preprint", arxiv_id="2402.09999", openalex_id="W7")
        with (
            mock.patch(
                "app.services.citation_graph.openalex_resolve_id",
                new=mock.AsyncMock(return_value="W7"),
            ),
            mock.patch(
                "app.services.citation_graph.openalex_get_metadata",
                new=mock.AsyncMock(return_value=meta),
            ),
        ):
            result = await self._call(resolve_paper, arxiv_id="2402.09999")

        self.assertEqual(result["status"], "partial")
        self.assertIn("publication year", result["note"])
        self.assertIn("venue", result["note"])

    async def test_nothing_to_go_on_is_an_argument_error(self) -> None:
        from app.agents.tools import resolve_paper

        result = await self._call(resolve_paper)
        self.assertEqual(result["error"]["code"], "invalid_argument")


class RelatedPapersTests(AgentP2TestCase):
    async def test_citations_are_registered_as_candidates(self) -> None:
        from app.agents.tools import get_related_papers
        from app.services import paper_catalog

        seed = await paper_catalog.upsert_literature_item(
            title="Seed paper", doi="10.1/seed", openalex_id="W9", source="test"
        )
        neighbours = [
            _metadata(title="Citing one", doi="10.1/n1", year=2024, cited_by_count=12),
            _metadata(title="Citing two", doi="10.1/n2", year=2023, cited_by_count=3),
        ]
        with mock.patch(
            "app.services.citation_graph.openalex_get_cited_by",
            new=mock.AsyncMock(return_value=neighbours),
        ):
            result = await self._call(
                get_related_papers, literature_id=seed, relation="citations"
            )

        self.assertEqual(result["status"], "ok")
        titles = [p["title"] for p in result["data"]["papers"]]
        self.assertEqual(titles, ["Citing one", "Citing two"])
        self.assertTrue(all(p["evidence_id"] for p in result["data"]["papers"]))

    async def test_unknown_relation_is_rejected(self) -> None:
        from app.agents.tools import get_related_papers

        result = await self._call(
            get_related_papers, literature_id="lit_x", relation="enemies"
        )
        self.assertEqual(result["error"]["code"], "invalid_argument")

    async def test_paper_outside_the_session_is_not_traversable(self) -> None:
        from app.agents.tools import get_related_papers

        result = await self._call(get_related_papers, paper_id="someone-elses-paper")
        self.assertEqual(result["error"]["code"], "paper_not_found")

    async def test_no_indexed_neighbours_is_unavailable_not_empty_success(self) -> None:
        from app.agents.tools import get_related_papers
        from app.services import paper_catalog

        seed = await paper_catalog.upsert_literature_item(
            title="Very new paper", doi="10.1/new", openalex_id="W8", source="test"
        )
        with (
            mock.patch(
                "app.services.citation_graph.openalex_get_cited_by",
                new=mock.AsyncMock(return_value=[]),
            ),
            mock.patch(
                "app.services.citation_graph.s2_resolve_id",
                new=mock.AsyncMock(return_value=None),
            ),
        ):
            result = await self._call(
                get_related_papers, literature_id=seed, relation="citations"
            )

        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["error"]["code"], "metadata_incomplete")


class PublicationRankTests(AgentP2TestCase):
    def _result(self, **kw: Any):
        from app.services.publication_rank import PublicationRankResult

        return PublicationRankResult(**kw)

    async def test_the_three_systems_stay_separate(self) -> None:
        from app.agents.tools import query_publication_rank

        stub = self._result(
            name="IEEE TIP", sci="Q1", ccf="A", success=True, source="easyscholar",
            extra={"sciUp": "计算机科学1区", "sciUpSmall": "计算机：人工智能1区",
                   "sciUpTop": "计算机科学TOP", "sciif": "15.3"},
        )
        with mock.patch(
            "app.agents.tools.discovery.lookup_rank",
            new=mock.AsyncMock(return_value=stub),
        ):
            result = await self._call(query_publication_rank, venue="IEEE TIP")

        systems = {e["ranking_system"]: e for e in result["data"]["rankings"]}
        self.assertEqual(systems["JCR"]["rank"], "Q1")
        self.assertEqual(systems["CAS"]["rank"], "计算机：人工智能1区")
        self.assertEqual(systems["CCF"]["rank"], "A")
        self.assertTrue(systems["CAS"]["top"])
        self.assertEqual(result["data"]["verification_status"], "verified")
        self.assertEqual(result["status"], "ok")

    async def test_web_search_rank_is_flagged_unverified(self) -> None:
        from app.agents.tools import query_publication_rank

        stub = self._result(name="Some Venue", sci="Q2", success=True, source="llm_websearch")
        with mock.patch(
            "app.agents.tools.discovery.lookup_rank",
            new=mock.AsyncMock(return_value=stub),
        ):
            result = await self._call(query_publication_rank, venue="Some Venue")

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["data"]["verification_status"], "unverified")
        self.assertIn("unverified", result["note"].lower())

    async def test_unlisted_venue_reports_unavailable(self) -> None:
        from app.agents.tools import query_publication_rank

        stub = self._result(name="Workshop X", success=False, error="not indexed")
        with mock.patch(
            "app.agents.tools.discovery.lookup_rank",
            new=mock.AsyncMock(return_value=stub),
        ):
            result = await self._call(query_publication_rank, venue="Workshop X")

        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["data"]["verification_status"], "unknown")


class AcquisitionTests(AgentP2TestCase):
    async def _candidate(self, **kw: Any) -> str:
        from app.services import paper_catalog

        return await paper_catalog.upsert_literature_item(source="test", **kw)

    async def test_no_full_text_is_unavailable_with_the_routes_tried(self) -> None:
        """A08: the agent must be told to fall back to the abstract, not to retry."""
        from app.agents.tools import download_paper
        from app.services.paper_acquisition import AcquisitionResult

        literature_id = await self._candidate(title="Paywalled", doi="10.1/pay")
        with mock.patch(
            "app.services.paper_acquisition.acquire_pdf",
            new=mock.AsyncMock(
                return_value=AcquisitionResult(
                    ok=False,
                    detail="oa_resolver: no OA copy",
                    attempts=[{"source": "oa_resolver", "detail": "no OA copy"}],
                )
            ),
        ):
            result = await self._call(download_paper, literature_id=literature_id)

        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["error"]["code"], "fulltext_unavailable")
        self.assertIn("abstract", result["error"]["message"].lower())

        # And the session remembers, so later turns do not imply they read it.
        from app.db import agent_repository as repo

        papers = await repo.list_session_papers(self.session.session_id)
        entry = next(p for p in papers if p.literature_id == literature_id)
        self.assertEqual(entry.availability.value, "unavailable")

    async def test_download_is_performed_once_per_work(self) -> None:
        """A12: a repeated request reuses the stored result and re-fetches nothing."""
        from app.agents.tools import download_paper
        from app.services.paper_acquisition import AcquisitionResult

        literature_id = await self._candidate(title="Open paper", arxiv_id="2401.00001")
        acquire = mock.AsyncMock(
            return_value=AcquisitionResult(ok=True, paper_id="paper1", source="arxiv")
        )
        with mock.patch("app.services.paper_acquisition.acquire_pdf", new=acquire):
            first = await self._call(download_paper, literature_id=literature_id)
            second = await self._call(download_paper, literature_id=literature_id)

        self.assertEqual(first["status"], "ok")
        self.assertEqual(second["status"], "ok")
        self.assertEqual(acquire.await_count, 1)
        # Only the real fetch counted against the budget.
        self.assertEqual(self.ctx.usage.downloads, 1)

    async def test_a_stale_unavailable_verdict_is_retried(self) -> None:
        """"No OA copy" is true on the day it is recorded, not forever."""
        from app.agents.tools import download_paper
        from app.agents.tools.acquisition import FAILED_DOWNLOAD_TTL_SECONDS
        from app.db import database as db
        from app.services.paper_acquisition import AcquisitionResult

        literature_id = await self._candidate(title="Later posted openly", doi="10.1/late")
        acquire = mock.AsyncMock(
            return_value=AcquisitionResult(ok=False, detail="oa_resolver: no OA copy")
        )
        with mock.patch("app.services.paper_acquisition.acquire_pdf", new=acquire):
            await self._call(download_paper, literature_id=literature_id)

            # Same day: the recorded answer stands, nothing is re-fetched.
            await self._call(download_paper, literature_id=literature_id)
            self.assertEqual(acquire.await_count, 1)

            # Age it past the TTL.
            await db.execute(
                "UPDATE agent_jobs SET updated_at = datetime('now', ?) "
                "WHERE idempotency_key = ?",
                (f"-{FAILED_DOWNLOAD_TTL_SECONDS + 3600} seconds", f"download:{literature_id}"),
            )
            await self._call(download_paper, literature_id=literature_id)

        self.assertEqual(acquire.await_count, 2)

    async def test_download_stops_at_the_run_budget(self) -> None:
        """A18: the ceiling is enforced in code, not asked of the model."""
        from app.agents.tools import download_paper

        literature_id = await self._candidate(title="One more", doi="10.1/more")
        self.ctx.usage.downloads = self.ctx.budget.max_downloads

        acquire = mock.AsyncMock()
        with mock.patch("app.services.paper_acquisition.acquire_pdf", new=acquire):
            result = await self._call(download_paper, literature_id=literature_id)

        self.assertEqual(result["error"]["code"], "budget_exceeded")
        self.assertFalse(result["error"]["retryable"])
        acquire.assert_not_awaited()

    async def test_parsing_an_already_parsed_paper_submits_nothing(self) -> None:
        from app.agents.tools import ensure_paper_parsed

        parse = mock.AsyncMock()
        with mock.patch("app.services.mineru_adapter.parse_pdf", new=parse):
            result = await self._call(ensure_paper_parsed, paper_id="paper1")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["data"]["availability"], "parsed")
        parse.assert_not_awaited()
        self.assertEqual(self.ctx.usage.parses, 0)

    async def test_parsing_backfills_a_title_the_work_never_had(self) -> None:
        """A paper fetched from a bare arXiv id must not stay "(untitled)"."""
        from app.agents.tools import ensure_paper_parsed
        from app.db import agent_repository as repo
        from app.models.agent_models import Availability
        from app.services import paper_catalog

        # A work known only by its identifier, with a PDF already on disk.
        literature_id = await paper_catalog.upsert_literature_item(
            arxiv_id="2403.05555", source="user"
        )
        await self._seed_paper("paper3", title="Recovered From The PDF")
        await paper_catalog.link_paper_file(
            literature_id=literature_id, paper_id="paper3", origin="arxiv"
        )
        await repo.attach_session_paper(
            session_id=self.session.session_id,
            literature_id=literature_id,
            paper_id="paper3",
            availability=Availability.PDF_READY,
        )
        self.assertEqual(
            (await paper_catalog.get_literature_item(literature_id))["title"], ""
        )

        result = await self._call(ensure_paper_parsed, paper_id="paper3")
        self.assertEqual(result["status"], "ok")

        stored = await paper_catalog.get_literature_item(literature_id)
        self.assertEqual(stored["title"], "Recovered From The PDF")
        papers = await repo.list_session_papers(self.session.session_id)
        entry = next(p for p in papers if p.literature_id == literature_id)
        self.assertEqual(entry.title, "Recovered From The PDF")

    async def test_parsing_a_paper_outside_the_session_is_refused(self) -> None:
        from app.agents.tools import ensure_paper_parsed

        result = await self._call(ensure_paper_parsed, paper_id="not-mine")
        self.assertEqual(result["error"]["code"], "paper_not_found")

    async def test_parsing_without_a_stored_pdf_is_unavailable(self) -> None:
        from app.agents.tools import ensure_paper_parsed
        from app.db import database as db

        await self._seed_paper("paper2", title="Second work")
        await self._attach("paper2", self.session.session_id)
        # Make it look downloaded-but-unparsed.
        await db.execute("DELETE FROM paper_nodes WHERE paper_id = ?", ("paper2",))

        result = await self._call(ensure_paper_parsed, paper_id="paper2")
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["error"]["code"], "fulltext_unavailable")


class CrossPaperTests(AgentP2TestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        await self._seed_paper("paper2", title="Dense Baseline Transformer")
        await self._attach("paper2", self.session.session_id)

    async def test_comparison_groups_evidence_by_paper(self) -> None:
        """A06: each paper answers the same question in its own right."""
        from app.agents.tools import search_paper_content

        result = await self._call(
            search_paper_content,
            question="BLEU on WMT14",
            paper_ids=["paper1", "paper2"],
        )

        groups = {g["paper_id"]: g for g in result["data"]["papers"]}
        self.assertEqual(set(groups), {"paper1", "paper2"})
        self.assertTrue(groups["paper1"]["hits"])
        self.assertTrue(groups["paper2"]["hits"])

    async def test_each_hit_carries_its_own_paper_identity(self) -> None:
        """A07: two papers can both cite page 3 and still resolve separately."""
        from app.agents.tools import search_paper_content
        from app.services import evidence_service

        result = await self._call(
            search_paper_content, question="router", paper_ids=["paper1", "paper2"]
        )
        by_paper = {}
        for group in result["data"]["papers"]:
            for hit in group["hits"]:
                evidence = await evidence_service.resolve(
                    hit["evidence_id"], owner_id=self.owner
                )
                by_paper.setdefault(evidence.paper_id, set()).add(
                    evidence.locator.page_label
                )

        self.assertEqual(set(by_paper), {"paper1", "paper2"})
        # Same page number in both papers, distinguished by paper_id alone.
        self.assertTrue(by_paper["paper1"] & by_paper["paper2"])

    async def test_a_silent_paper_is_named_not_treated_as_disagreement(self) -> None:
        from app.agents.tools import search_paper_content

        result = await self._call(
            search_paper_content,
            question="quantum annealing schedules",
            paper_ids=["paper1", "paper2"],
        )
        self.assertIn(result["status"], {"ok", "partial"})
        if result["status"] == "partial":
            self.assertIn("silence", result["note"].lower())

    async def test_papers_outside_the_session_are_dropped_from_the_scope(self) -> None:
        from app.agents.tools import search_paper_content

        result = await self._call(
            search_paper_content,
            question="router",
            paper_ids=["paper1", "someone-elses-paper"],
        )
        ids = {g["paper_id"] for g in result["data"]["papers"]}
        self.assertEqual(ids, {"paper1"})

    async def test_a_scope_of_only_foreign_papers_is_refused(self) -> None:
        from app.agents.tools import search_paper_content

        result = await self._call(
            search_paper_content, question="router", paper_ids=["nope-1", "nope-2"]
        )
        self.assertEqual(result["error"]["code"], "paper_not_found")


class JobIdempotencyTests(AgentP2TestCase):
    async def test_concurrent_requests_run_the_work_once(self) -> None:
        """Whatever the timing, one execution — and every caller gets its result.

        The hard case is not two simultaneous callers but a late one arriving
        just as the work finishes: it must reuse the stored result rather than
        re-claim a job that is already done.
        """
        import asyncio

        from app.services import agent_jobs

        for index, delay in enumerate((0.0, 0.002, 0.05)):
            calls: list[str] = []

            async def work(job_id: str, _d: float = delay) -> dict[str, Any]:
                calls.append(job_id)
                if _d:
                    await asyncio.sleep(_d)
                return {"value": 1}

            key = f"k{index}"
            handles = await asyncio.gather(
                *[
                    agent_jobs.run_once(kind="parse", idempotency_key=key, work=work)
                    for _ in range(5)
                ]
            )
            self.assertEqual(len(calls), 1, f"delay={delay}")
            self.assertTrue(
                all(h.result == {"value": 1} for h in handles), f"delay={delay}"
            )

    async def test_a_persistently_failing_job_stops_being_retried(self) -> None:
        from app.services import agent_jobs

        calls: list[str] = []

        async def work(job_id: str) -> dict[str, Any]:
            calls.append(job_id)
            raise RuntimeError("provider down")

        for _ in range(agent_jobs.MAX_ATTEMPTS + 2):
            with self.assertRaises(agent_jobs.JobFailed):
                await agent_jobs.run_once(kind="download", idempotency_key="d", work=work)

        self.assertEqual(len(calls), agent_jobs.MAX_ATTEMPTS)

    async def test_the_external_task_id_is_recorded_for_recovery(self) -> None:
        """A13: a restart looks the parse up instead of resubmitting it."""
        from app.services import agent_jobs

        async def work(job_id: str) -> dict[str, Any]:
            await agent_jobs.set_remote_id(job_id, "mineru-batch-42")
            return {"ok": True}

        await agent_jobs.run_once(kind="parse", idempotency_key="p", work=work)
        row = await agent_jobs.get_job("p")
        self.assertEqual(row["remote_id"], "mineru-batch-42")
        self.assertEqual(row["status"], "done")


class BudgetConfigurationTests(AgentP2TestCase):
    async def test_ceilings_come_from_settings_not_from_code(self) -> None:
        """§7.3: what a run may spend is a deployment decision."""
        from app.agents.context import RunBudget
        from app.config import get_settings

        with mock.patch.dict("os.environ", {"AGENT_MAX_DOWNLOADS": "2"}):
            get_settings.cache_clear()
            try:
                self.assertEqual(RunBudget.from_settings().max_downloads, 2)
                # A run's own stored budget still wins, so an old run keeps the
                # limits it actually ran under.
                self.assertEqual(
                    RunBudget.from_settings({"max_downloads": 7}).max_downloads, 7
                )
            finally:
                get_settings.cache_clear()

    async def test_a_run_records_the_budget_it_ran_under(self) -> None:
        from app.agents.context import RunBudget
        from app.db import agent_repository as repo

        run, _ = await repo.create_run(
            session_id=self.session.session_id,
            owner_id=self.owner,
            budget=RunBudget.from_settings().as_dict(),
        )
        self.assertEqual(run.budget["max_parses"], RunBudget.from_settings().max_parses)

    async def test_the_wall_clock_ceiling_is_reachable(self) -> None:
        """It was previously unreachable: usage.wall_seconds was only set after the loop."""
        from app.agents.context import BudgetUsage, RunBudget

        budget = RunBudget.from_settings({"max_wall_seconds": 1})
        usage = BudgetUsage()
        self.assertEqual(usage.exceeded(budget), "")
        usage.wall_seconds = 2.0
        self.assertEqual(usage.exceeded(budget), "max_wall_seconds")
