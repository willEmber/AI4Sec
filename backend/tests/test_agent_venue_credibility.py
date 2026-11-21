"""Unit tests for P17 Phase 3: agent venue credibility integration & tool contracts."""
from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

from app.agents.context import AgentContext
from app.agents.prompts import _format_papers
from app.agents.tools import discovery
from app.agents.tools.triage import snap_signal_facts
from app.models.agent_models import Availability, SessionPaper
from app.models.evidence_models import SourceLevel
from app.services.paper_search.models import Paper, PlatformStatus, SearchOutcome
from app.services.paper_search.search import _dedupe_keep_first


def _fake_evidence(eid: str, level: SourceLevel = SourceLevel.METADATA) -> SimpleNamespace:
    return SimpleNamespace(evidence_id=eid, source_level=level)


class AgentVenueCredibilityPureTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.ctx = AgentContext(owner_id="pr_test", session_id="s_test", run_id="r_test")
        self.runtime = SimpleNamespace(context=self.ctx)
        self.recorded_metadata: list[dict[str, Any]] = []
        self._seq = 0

        async def fake_record_meta(**kwargs: Any) -> SimpleNamespace:
            self._seq += 1
            eid = f"ev_{self._seq:020d}"
            self.recorded_metadata.append({"id": eid, **kwargs})
            return _fake_evidence(eid, SourceLevel.METADATA)

        async def fake_record_abs(**kwargs: Any) -> SimpleNamespace:
            self._seq += 1
            return _fake_evidence(f"ev_{self._seq:020d}", SourceLevel.ABSTRACT)

        self._items: dict[str, dict[str, Any]] = {}

        async def fake_upsert(**kwargs: Any) -> str:
            lit_id = f"lit_{len(self._items) + 1}"
            self._items[lit_id] = {
                "literature_id": lit_id,
                "title": kwargs.get("title") or "",
                "authors_json": json.dumps(kwargs.get("authors") or []),
                "year": kwargs.get("year") or 0,
                "year_known": 1 if kwargs.get("year") else 0,
                "venue": kwargs.get("venue") or "",
                "doi": kwargs.get("doi") or "",
                "arxiv_id": kwargs.get("arxiv_id") or "",
                "url": kwargs.get("url") or "",
                "oa_pdf_url": kwargs.get("oa_pdf_url") or "",
                "abstract": kwargs.get("abstract") or "",
            }
            return lit_id

        async def fake_get_item(lit_id: str) -> dict[str, Any] | None:
            return self._items.get(lit_id)

        for p in (
            mock.patch.object(discovery.evidence_service, "record_metadata_evidence", side_effect=fake_record_meta),
            mock.patch.object(discovery.evidence_service, "record_abstract_evidence", side_effect=fake_record_abs),
            mock.patch.object(discovery.paper_catalog, "upsert_literature_item", side_effect=fake_upsert),
            mock.patch.object(discovery.paper_catalog, "get_literature_item", side_effect=fake_get_item),
        ):
            p.start()
            self.addCleanup(p.stop)

    async def test_search_papers_attaches_venue_rank_and_promotes_accepted_arxiv(self) -> None:
        papers = [
            Paper(
                title="Accepted Vision Transformer Paper",
                abstract="We propose a vision architecture.",
                url="https://arxiv.org/abs/2401.12345",
                doi="",
                authors="Alice; Bob",
                source_platform="arXiv",
                year=2025,
                venue="arXiv",
                arxiv_id="2401.12345",
                acceptance="ICLR 2025 Oral",
                citation_counts={"SemanticScholar": 42},
            ),
            Paper(
                title="Questionable Paper in Warned Outlet",
                abstract="An empirical study.",
                url="https://doi.org/10.1016/j.compeleceng.2025.1",
                doi="10.1016/j.compeleceng.2025.1",
                authors="Carol",
                source_platform="Crossref",
                year=2025,
                venue="Computers & Electrical Engineering",
            ),
        ]
        outcome = SearchOutcome(
            papers=papers,
            platforms=[PlatformStatus(platform="SemanticScholar", status="ok", count=2)],
            candidates=2,
        )
        with mock.patch.object(discovery, "run_search", new=mock.AsyncMock(return_value=outcome)):
            raw = await discovery.search_papers.coroutine("vision transformer", runtime=self.runtime)

        payload = json.loads(raw)
        self.assertEqual(payload["status"], "ok")
        results = payload["data"]["results"]
        self.assertEqual(len(results), 2)

        # 1) Accepted arXiv paper promoted to ICLR with CCF A + CORE A* + Oral tier + preprint_file_only
        iclr_hit = results[0]
        self.assertEqual(iclr_hit["venue"], "ICLR")
        self.assertEqual(iclr_hit["venue_rank"]["ranks"]["CCF"], "A (2026)")
        self.assertEqual(iclr_hit["venue_rank"]["ranks"]["CORE"], "A* (2026)")
        self.assertEqual(iclr_hit["venue_rank"]["presentation_tier"], "oral")
        self.assertTrue(iclr_hit["venue_rank"]["preprint_file_only"])
        self.assertTrue(iclr_hit.get("metadata_evidence_id"))

        # 2) CAS Early Warning journal flagged on search result & recorded as metadata evidence
        warn_hit = results[1]
        self.assertIn("预警", warn_hit["venue_rank"]["cas_warning"])
        self.assertTrue(warn_hit.get("metadata_evidence_id"))
        quotes = "\n".join(m["quote"] for m in self.recorded_metadata)
        self.assertIn("CCF A (2026)", quotes)
        self.assertIn("CAS International Journal Early Warning", quotes)

    async def test_query_publication_rank_single_and_batch_modes(self) -> None:
        # Single lookup hitting L1 local catalog -> verified + multi-system + edition_year
        raw_single = await discovery.query_publication_rank.coroutine(
            runtime=self.runtime,
            venue="IEEE Transactions on Image Processing",
        )
        single = json.loads(raw_single)
        self.assertEqual(single["status"], "ok")
        self.assertEqual(single["data"]["verification_status"], "verified")
        self.assertEqual(single["data"]["source"], "local_catalog")
        systems = {e["ranking_system"]: e for e in single["data"]["rankings"]}
        self.assertEqual(systems["CCF"]["rank"], "A")
        self.assertEqual(systems["CCF"]["edition_year"], "2026")
        self.assertEqual(systems["JCR"]["rank"], "Q1")
        self.assertIn("CAS", systems)

        # Batch lookup across top conf + Chinese T1 journal + CAS warning journal
        raw_batch = await discovery.query_publication_rank.coroutine(
            runtime=self.runtime,
            venues=["CVPR", "软件学报", "Computers & Electrical Engineering"],
        )
        batch = json.loads(raw_batch)
        self.assertEqual(batch["status"], "ok")
        self.assertEqual(len(batch["data"]["venues"]), 3)
        self.assertEqual(len(batch["evidence_ids"]), 3)
        by_query = {v["query"]: v for v in batch["data"]["venues"]}
        self.assertEqual(by_query["CVPR"]["verification_status"], "verified")
        self.assertIn("预警", by_query["Computers & Electrical Engineering"]["cas_warning"])

    def test_search_dedupe_replaces_lncs_shell_with_dblp_conference(self) -> None:
        crossref_rec = Paper(
            title="Masked Autoencoders Are Scalable Vision Learners",
            abstract="Full abstract.",
            url="https://doi.org/10.1007/978-3-031-19806-9_1",
            doi="10.1007/978-3-031-19806-9_1",
            authors="Kaiming He",
            source_platform="Crossref",
            year=2022,
            venue="Lecture Notes in Computer Science",
            issns=[],
        )
        dblp_rec = Paper(
            title="Masked Autoencoders Are Scalable Vision Learners",
            abstract="",
            url="https://dblp.org/rec/conf/eccv/He22",
            doi="10.1007/978-3-031-19806-9_1",
            authors="Kaiming He",
            source_platform="DBLP",
            year=2022,
            venue="ECCV",
            dblp_key="conf/eccv/He22",
        )
        merged = _dedupe_keep_first([crossref_rec, dblp_rec])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].venue, "ECCV")
        self.assertEqual(merged[0].dblp_key, "conf/eccv/He22")

    def test_format_papers_renders_venue_badges_without_altering_bare_entries(self) -> None:
        bare = SessionPaper(
            session_id="s1",
            literature_id="lit_1",
            paper_id="p_bare",
            availability=Availability.PARSED,
            title="Untitled Bare Paper",
            year=2024,
            year_known=True,
        )
        ranked = SessionPaper(
            session_id="s1",
            literature_id="lit_2",
            paper_id="p_cvpr",
            availability=Availability.PARSED,
            title="Deep Residual Learning",
            year=2016,
            year_known=True,
            venue="IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
        )
        warned = SessionPaper(
            session_id="s1",
            literature_id="lit_3",
            paper_id="p_warn",
            availability=Availability.CANDIDATE,
            title="Suspicious Study",
            year=2025,
            year_known=True,
            venue="Computers & Electrical Engineering",
        )
        text = _format_papers([bare, ranked, warned], "zh")
        self.assertIn("- Untitled Bare Paper (2024) — paper_id=p_bare, parsed", text)
        self.assertIn("- Deep Residual Learning (2016, CVPR · CCF A) — paper_id=p_cvpr, parsed", text)
        self.assertIn("预警", text)

        accepted_preprint = SessionPaper(
            session_id="s1",
            literature_id="lit_4",
            paper_id="p_iclr",
            availability=Availability.PARSED,
            title="Accepted Preprint",
            year=2025,
            year_known=True,
            venue="ICLR",
            ccf_rank="A",
            venue_ranks={"presentation_tier": "oral", "preprint_file_only": True},
        )
        en_text = _format_papers([accepted_preprint], "en")
        self.assertIn("ICLR · CCF A / Oral [preprint PDF]", en_text)

    def test_snap_signal_facts_includes_multi_system_and_cas_warning(self) -> None:
        facts = snap_signal_facts(
            {
                "venue": "ECCV",
                "venue_normalized": "ECCV",
                "year": 2024,
                "ccf_rank": "B",
                "venue_ranks": {"CORE": "A* (2026)", "TH_CPL": "A (2024)"},
                "cas_warning": "2025年预警（论文工厂）",
                "provenance": {"venue": "DBLP", "ccf_rank": "Venue Catalog", "venue_ranks": "Venue Catalog"},
            },
            "My Paper",
        )
        joined = "\n".join(f"{fact} [{src}]" for fact, src in facts)
        self.assertIn("ECCV — CCF B [Venue Catalog]", joined)
        self.assertIn("ECCV — CORE A* (2026) [Venue Catalog]", joined)
        self.assertIn("ECCV — TH_CPL A (2024) [Venue Catalog]", joined)
        self.assertIn("CAS International Journal Early Warning: 2025年预警（论文工厂）", joined)


if __name__ == "__main__":
    unittest.main()
