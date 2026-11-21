"""Unit and PostgreSQL integration tests for P18 (venue tiers as citable evidence & report adaptation)."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from app.agents.prompts import PROMPT_VERSION, _paper_venue_badge
from app.agents.tools.literature_map import _metadata_quote, _node_profile, _paper_entry
from app.agents.tools.triage import snap_signal_facts
from app.models.agent_models import Availability, SessionPaper
from app.models.evidence_models import Evidence, Locator, SourceLevel
from app.models.paper_ir import PaperIR
from app.models.snap_models import TriageSignals
from app.models.sphere_models import CandidateSource, RelationType, SphereConfig, SphereNode, SphereState
from app.services import paper_metadata, report_citations
from app.services.publication_rank.paper_venue import (
    StoredVenue,
    parse_compact_rank,
    pub_rank_from_profile,
    venue_profile,
)
from app.services.triage_signals import (
    _rank_source_label,
    apply_venue_to_signals,
    compute_external_score,
    get_or_collect_triage_signals,
)
from app.services.triage_verdict import ContentReview, synthesize_verdict
from evals.rank_claims import audit_rank_claims, extract_rank_claims

try:
    from tests.pg_support import open_fresh_database
except ImportError:
    from pg_support import open_fresh_database  # type: ignore[no-redef]


class VenueProfilePriorityTests(unittest.TestCase):
    """P18 §2.1: six-rule deterministic read-time resolution."""

    def test_rule1_workshop_clears_legacy_main_venue_ranks(self) -> None:
        prof = venue_profile(
            StoredVenue(
                venue="CVPRW 2024",
                ccf_rank="A",
                sci_rank="Q1",
                extra={"ranks": {"CCF": "A (2026)", "CORE": "A* (2026)"}},
            )
        )
        self.assertEqual(prof.subtrack, "workshop")
        self.assertEqual(prof.rankings, [])
        self.assertEqual(prof.ccf_rank, "")
        self.assertEqual(prof.sci_rank, "")
        self.assertFalse(prof.has_verified_rank)

    def test_rule1_stored_findings_subtrack_is_preserved_without_main_tier(self) -> None:
        prof = venue_profile(
            StoredVenue(
                venue="ACL",
                ccf_rank="A",
                extra={"subtrack": "findings"},
            )
        )
        self.assertEqual(prof.subtrack, "findings")
        self.assertEqual(prof.rankings, [])
        self.assertIn("Findings", prof.display_venue)

    def test_rule2_catalog_overrides_stale_stored_tier(self) -> None:
        # HPDC was upgraded from CCF B to CCF A in the 7th edition (2026).
        prof = venue_profile(StoredVenue(venue="HPDC", ccf_rank="B"))
        self.assertEqual(prof.ccf_rank, "A")
        self.assertTrue(prof.has_verified_rank)
        self.assertEqual(prof.source, "local_catalog")

        # Prompt badge and pub_rank projection agree identically
        sp = SessionPaper(
            session_id="s1",
            literature_id="lit_hpdc",
            availability=Availability.PARSED,
            venue="HPDC",
            ccf_rank="B",
        )
        self.assertIn("CCF A", _paper_venue_badge(sp, "zh"))
        pr = pub_rank_from_profile(prof, raw_venue="HPDC", year=2025)
        self.assertEqual(pr["ccf"], "A")

    def test_rule3_mixed_catalog_and_unverified_stored_systems_stay_isolated(self) -> None:
        # WACV is CORE A in the local catalog with no CCF rating; add an unverified
        # web-search CCF tier on the stored row and check they do not pollute each other.
        prof = venue_profile(
            StoredVenue(
                venue="WACV",
                extra={
                    "ranks": {"CCF": "B"},
                    "source": "llm_websearch",
                    "verification_status": "unverified",
                },
            )
        )
        # Catalog supplies CORE A (verified); web-search supplied CCF B (unverified).
        ver_map = prof.rank_verified_map()
        self.assertTrue(ver_map.get("CORE"))
        self.assertFalse(ver_map.get("CCF"))

        # `verified_fact_clauses()` must emit ONLY verified systems (CORE A), never unverified CCF!
        clauses_text = "; ".join(prof.verified_fact_clauses())
        self.assertIn("CORE A", clauses_text)
        self.assertNotIn("CCF B", clauses_text)

    def test_rule4_legacy_row_without_source_is_unverified_not_easyscholar(self) -> None:
        self.assertEqual(_rank_source_label(""), "stored record (unverified)")
        self.assertEqual(_rank_source_label("legacy"), "stored record (unverified)")

        prof = venue_profile(
            StoredVenue(
                venue="Journal of Fictional Quantum Basketry",
                sci_rank="Q2",
            )
        )
        self.assertEqual(prof.sci_rank, "Q2")
        self.assertEqual(prof.verification_status, "unverified")
        self.assertFalse(prof.has_verified_rank)
        self.assertEqual(prof.verified_fact_clauses(), [])
        self.assertIn("未核实", prof.prompt_badge("zh"))

    def test_rule5_and_6_warnings_and_preprint_pdf_flag(self) -> None:
        prof = venue_profile(
            StoredVenue(
                venue="CVPR",
                extra={"cas_warning": "custom-flag", "presentation_tier": "oral"},
                doi="",
                arxiv_id="2311.14760",
            )
        )
        self.assertTrue(prof.is_preprint_file)
        self.assertEqual(prof.openreview_tier, "oral")
        self.assertEqual(prof.warning, "custom-flag")
        badge = prof.prompt_badge("zh")
        self.assertIn("Oral", badge)
        self.assertIn("[预印本文件]", badge)
        self.assertIn("预警:custom-flag", badge)

    def test_compact_rank_roundtrip(self) -> None:
        jcr = parse_compact_rank("JCR", "Q1 (IF 10.6)", source="local_catalog", verified=True)
        assert jcr is not None
        self.assertEqual(jcr.compact_label(), "Q1 (IF 10.6)")

        cas = parse_compact_rank("CAS", "1区 Top (2025)", source="local_catalog", verified=True)
        assert cas is not None
        self.assertTrue(cas.top)
        self.assertEqual(cas.rank, "1区")
        self.assertEqual(cas.edition_year, "2025")
        self.assertEqual(cas.compact_label(), "1区 Top (2025)")


class TriageAndVerdictP18Tests(unittest.IsolatedAsyncioTestCase):
    """P18 §2.1, §2.3, §2.4: Snap signals, cached re-resolution, facts, and verdict drivers."""

    def test_prompt_version_bumped(self) -> None:
        self.assertEqual(PROMPT_VERSION, "p18-venue-1")

    async def test_cached_signals_reapply_catalog_and_clear_workshop_tiers(self) -> None:
        paper_ir = PaperIR(paper_id="p_ws", title="Workshop Paper")
        stale_ws = TriageSignals(
            resolved=True,
            venue="CVPRW 2024",
            ccf_rank="A",
            venue_ranks={"CCF": "A"},
        )
        with patch(
            "app.services.triage_signals.load_cached_signals",
            new=AsyncMock(return_value=stale_ws),
        ):
            refreshed = await get_or_collect_triage_signals(paper_ir, paper_id="p_ws", ttl_hours=168)
        self.assertEqual(refreshed.venue_subtrack, "workshop")
        self.assertEqual(refreshed.ccf_rank, "")
        self.assertEqual(refreshed.venue_ranks, {})
        self.assertAlmostEqual(refreshed.venue_score, 0.2)

        stale_cvpr = TriageSignals(
            resolved=True,
            venue="CVPR",
            ccf_rank="A",
        )
        with patch(
            "app.services.triage_signals.load_cached_signals",
            new=AsyncMock(return_value=stale_cvpr),
        ):
            upgraded = await get_or_collect_triage_signals(paper_ir, paper_id="p_cvpr", ttl_hours=168)
        self.assertIn("CCF", upgraded.venue_ranks)
        self.assertIn("CORE", upgraded.venue_ranks)
        self.assertEqual(upgraded.provenance.get("ccf_rank"), "Venue Catalog")

    def test_snap_signal_facts_subtracks_editions_and_unverified_labels(self) -> None:
        ws_signals = TriageSignals(venue="CVPR Workshops", year=2024)
        apply_venue_to_signals(ws_signals)
        compute_external_score(ws_signals)
        ws_facts = snap_signal_facts(ws_signals.model_dump(mode="json"), "TinyNet")
        self.assertTrue(any("workshop track; main-venue tier does not apply" in f for f, _ in ws_facts))
        self.assertFalse(any("CCF" in f for f, _ in ws_facts))

        unv_signals = TriageSignals(venue="Unlisted Mystery Conf")
        apply_venue_to_signals(
            unv_signals,
            {
                "venue": "Unlisted Mystery Conf",
                "ccf": "B",
                "ranks": {"CCF": "B"},
                "source": "llm_websearch",
                "verification_status": "unverified",
            },
        )
        unv_facts = snap_signal_facts(unv_signals.model_dump(mode="json"), "MysteryPaper")
        self.assertTrue(
            any("CCF B [unverified (web search)]" in f for f, _ in unv_facts),
            msg=f"Unexpected facts: {unv_facts}",
        )

    def test_synthesize_verdict_drivers_cover_core_warning_and_subtracks(self) -> None:
        review = ContentReview(
            contribution_strength=2,
            evidence_strength=2,
            novelty=2,
            reproducibility=2,
            limitation_severity=1,
            available=True,
        )
        # CORE-only conference (ICLR) gets `venue_rank` driver
        iclr_sig = TriageSignals(venue="ICLR", resolved=True)
        apply_venue_to_signals(iclr_sig)
        compute_external_score(iclr_sig)
        v_iclr = synthesize_verdict(review, iclr_sig)
        self.assertIn("venue_rank", v_iclr.drivers)

        # CAS early-warning journal gets `cas_warning` driver, NOT `venue_rank`, and is NOT hard-capped to skip
        warn_sig = TriageSignals(venue="Some Journal", resolved=True)
        apply_venue_to_signals(warn_sig, {"venue": "Some Journal", "sci": "Q2", "cas_warning": "2024 预警"})
        compute_external_score(warn_sig)
        strong_review = ContentReview(
            contribution_strength=3,
            evidence_strength=3,
            novelty=3,
            reproducibility=3,
            limitation_severity=0,
            available=True,
        )
        v_warn = synthesize_verdict(strong_review, warn_sig)
        self.assertIn("cas_warning", v_warn.drivers)
        self.assertNotIn("venue_rank", v_warn.drivers)
        self.assertNotEqual(v_warn.tier, "skip")

        # Workshop gets `workshop_track` driver
        ws_sig = TriageSignals(venue="ECCVW 2024", resolved=True)
        apply_venue_to_signals(ws_sig)
        compute_external_score(ws_sig)
        v_ws = synthesize_verdict(review, ws_sig)
        self.assertIn("workshop_track", v_ws.drivers)
        self.assertNotIn("venue_rank", v_ws.drivers)

        # Findings gets `findings_track` driver
        fd_sig = TriageSignals(venue="Findings of ACL 2024", resolved=True)
        apply_venue_to_signals(fd_sig)
        compute_external_score(fd_sig)
        v_fd = synthesize_verdict(review, fd_sig)
        self.assertIn("findings_track", v_fd.drivers)


class SphereAndLiteratureMapP18Tests(unittest.IsolatedAsyncioTestCase):
    """P18 §2.5: Research Sphere T3/Frontier gates, cached map re-resolution, and center record."""

    async def test_sphere_t3_and_frontier_gates_accept_core_and_reject_workshop(self) -> None:
        from app.workflows.sphere_subgraph import step_score_and_rank, step_select_core_set

        center = SphereNode(node_id="c0", title="Center Paper", year=2023, venue="CVPR")
        wacv_t3 = SphereNode(
            node_id="n_wacv",
            title="Fresh WACV Frontier Method",
            year=2025,
            venue="WACV",
            cited_by_count=1,
            tier=3,
            sources=[CandidateSource.QUERY_SEARCH],
            relation_type=RelationType.FOLLOW_UP,
            relevance=3,
        )
        ws_t3 = SphereNode(
            node_id="n_ws",
            title="Low-Citation Workshop Note",
            year=2025,
            venue="CVPRW 2025",
            ccf_rank="A",  # stale rank must be wiped!
            cited_by_count=1,
            tier=3,
            sources=[CandidateSource.QUERY_SEARCH],
            relation_type=RelationType.FOLLOW_UP,
            relevance=3,
        )
        sphere = SphereState(
            config=SphereConfig(
                venue_rank_enabled=False,
                t3_min_citations=5,
                frontier_min_citations=10,
                quota_follow_up=0,  # force selection via frontier quota!
                quota_frontier=2,
            ),
            center_node=center,
            nodes={"c0": center, "n_wacv": wacv_t3, "n_ws": ws_t3},
        )
        state = {"paper_id": "p_center", "language": "zh", "persist_sphere": False}
        sphere = await step_score_and_rank(sphere, state, "")
        # WACV (CORE A, no CCF/SCI) survives T3 gate despite 1 citation; CVPRW is dropped!
        self.assertIn("n_wacv", sphere.nodes)
        self.assertNotIn("n_ws", sphere.nodes)
        self.assertEqual(sphere.center_node.ccf_rank, "A")
        self.assertIn("CORE", sphere.center_node.venue_ranks)

        sphere = await step_select_core_set(sphere, state, "")
        part_map = {p.key: p.node_ids for p in sphere.output.partitions}
        self.assertIn("n_wacv", part_map.get("frontier", []))

    def test_literature_map_resolves_pre_p18_cached_nodes_on_read(self) -> None:
        # Simulate a node deserialized from an older `sphere_map:` job with empty `venue_ranks`
        old_node = SphereNode(
            node_id="n1",
            title="ResShift",
            year=2023,
            venue="NeurIPS",
            cited_by_count=250,
            relation_type=RelationType.FOUNDATION,
        )
        prof = _node_profile(old_node)
        quote = _metadata_quote(old_node, as_of="2026-11-22", profile=prof)
        self.assertIn("CCF A", quote)
        self.assertIn("CORE A*", quote)
        entry = _paper_entry(
            1,
            {
                "group": "foundation",
                "node": old_node,
                "profile": prof,
                "literature_id": "lit_resshift",
                "abstract_id": "ev_abs",
                "metadata_id": "ev_meta",
            },
            language="zh",
        )
        self.assertIn("CCF A", entry["rank"])
        self.assertIn("CORE A*", entry["rank"])


class EnrichMetadataSingleflightTests(unittest.IsolatedAsyncioTestCase):
    """P18 §2.8: in-process singleflight deduplication and shield protection."""

    async def test_concurrent_default_calls_share_one_lookup_and_survive_waiter_cancel(self) -> None:
        calls = 0
        gate = asyncio.Event()

        async def _fake_once(paper_id: str, **_kwargs):
            nonlocal calls
            calls += 1
            await gate.wait()
            return {"venue": "CVPR", "year": 2024, "ccf": "A", "sci": "", "doi": "10.1109/x"}

        with patch("app.services.paper_metadata._enrich_metadata_once", side_effect=_fake_once):
            t1 = asyncio.create_task(paper_metadata.enrich_metadata("p_same"))
            t2 = asyncio.create_task(paper_metadata.enrich_metadata("p_same"))
            await asyncio.sleep(0.01)
            self.assertEqual(calls, 1)

            # Cancelling waiter t1 must NOT cancel the shared underlying lookup for t2!
            t1.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await t1

            gate.set()
            res2 = await t2
            self.assertEqual(res2["venue"], "CVPR")
            self.assertEqual(calls, 1)

            # Dry-run (`persist=False`) runs independently without joining singleflight
            await paper_metadata.enrich_metadata("p_same", reuse_stored=False, persist=False)
            self.assertEqual(calls, 2)


class RankClaimsAuditTests(unittest.TestCase):
    """P18 §3.2: deterministic 4-way classification and anti-false-positive guards."""

    def test_four_way_outcomes_and_partition_identity(self) -> None:
        ev1 = "ev_" + "1" * 20
        ev2 = "ev_" + "2" * 20
        ev3 = "ev_" + "3" * 20
        evidence = {
            ev1: {"quote": "SinSR — CVPR: CCF A (2026), CORE A* (2026) (source: venue catalog)"},
            ev2: {"quote": "SinSR improves PSNR to 24.56 on ImageNet-Test."},
            ev3: {"quote": "TinySR — venue: CVPRW (workshop track; main-venue tier does not apply)"},
        }
        answer = (
            f"SinSR 发表在 CVPR（CCF A、CORE A*）[{ev1}]。"
            f"另一处从徽章抄来中科院 1 区却只挂了正文实验引用 [{ev2}]。"
            "还有一句完全没写引用的 JCR Q1 结论。"
            f"这一句把等级误写成了 CCF B [{ev1}]。"
            f"最后一句把 Workshop 说成了 CCF A [{ev3}]。"
        )
        audit = audit_rank_claims(answer, evidence)
        self.assertEqual(
            audit["claims"],
            audit["supported"] + audit["unsupported"] + audit["uncited"] + audit["contradicted"],
        )
        self.assertEqual(audit["supported"], 2)     # CCF A, CORE A*
        self.assertEqual(audit["unsupported"], 1)   # 中科院 1 区 with fulltext citation
        self.assertEqual(audit["uncited"], 1)       # JCR Q1 without citation
        self.assertEqual(audit["contradicted"], 2)  # CCF B vs CCF A, and CCF A vs workshop

    def test_comparison_sentence_and_markdown_table_do_not_false_contradict(self) -> None:
        ev_a = "ev_" + "a" * 20
        ev_c = "ev_" + "c" * 20
        evidence = {
            ev_a: {"quote": "SinSR — CVPR: CCF A (2026) (source: venue catalog)"},
            ev_c: {"quote": "BaselineSR — BMVC: CCF C (2026) (source: venue catalog)"},
        }
        prose_and_table = (
            f"SinSR 发表于 CVPR（CCF A）[{ev_a}]，而 BaselineSR 发表于 BMVC（CCF C）[{ev_c}]。\n\n"
            "| 论文 | 场所与等级 |\n"
            "|---|---|\n"
            f"| SinSR | CVPR · CCF A [{ev_a}] |\n"
            f"| BaselineSR | BMVC · CCF C [{ev_c}] |\n"
        )
        audit = audit_rank_claims(prose_and_table, evidence)
        self.assertEqual(audit["contradicted"], 0)
        self.assertEqual(audit["supported"], 4)

    def test_no_false_positives_on_math_negated_warnings_or_experimental_findings(self) -> None:
        benign = (
            "Key experimental findings show that at time steps $T_1$, $T_2$ and $T_3$ "
            "(see T1 in Table 1), the model converges. "
            "未发现撤稿记录，也未列入中科院国际期刊预警名单；not on the CAS early-warning list."
        )
        self.assertEqual(extract_rank_claims(benign), [])


class PostgresP18IntegrationTests(unittest.IsolatedAsyncioTestCase):
    """Real PostgreSQL tests for frozen report citations, work sync, and owner paper queries."""

    async def asyncSetUp(self) -> None:
        from app.db import database

        await open_fresh_database(self)
        self.db = database

    async def test_freeze_citations_and_work_venue_join_and_owner_papers(self) -> None:
        from app.db import agent_repository as repo
        from app.services import evidence_service, identity, paper_catalog

        owner_id, _ = await identity.create_principal(label="test:p18")
        session = await repo.create_session(owner_id=owner_id, title="P18 Session", language="zh")

        # 1) Center paper (SinSR): first indexed as arXiv in `literature_items`,
        #    then enriched to CVPR 2024 on `papers`!
        lit_center = await paper_catalog.upsert_literature_item(
            title="SinSR: Diffusion-Based Image Super-Resolution in a Single Step",
            arxiv_id="2311.14760",
            year=2023,
            venue="arXiv",
            source="arxiv",
        )
        await self.db.execute(
            "INSERT INTO papers (paper_id, file_path, title, doi, venue, year, ccf_rank) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("paper_sinsr", "papers/paper_sinsr/original.pdf", "SinSR", "10.1109/CVPR.2024.1", "CVPR", 2024, "A"),
        )
        await paper_catalog.link_paper_file(literature_id=lit_center, paper_id="paper_sinsr")
        await repo.attach_session_paper(
            session_id=session.session_id,
            literature_id=lit_center,
            paper_id="paper_sinsr",
            availability=Availability.PARSED,
        )

        # `list_owner_papers` (GET /agent/papers) must surface `CVPR` and `ccf_rank="A"`!
        owner_papers = await repo.list_owner_papers(owner_id)
        self.assertEqual(len(owner_papers), 1)
        self.assertEqual(owner_papers[0].venue, "CVPR")
        self.assertEqual(owner_papers[0].ccf_rank, "A")

        # 2) Related work in NeurIPS (abstract evidence) + unreviewed preprint
        lit_resshift = await paper_catalog.upsert_literature_item(
            title="ResShift",
            doi="10.5555/neurips.2023.1",
            year=2023,
            venue="NeurIPS",
            abstract="Residual shifting for diffusion super-resolution.",
            source="openalex",
        )
        lit_preprint = await paper_catalog.upsert_literature_item(
            title="NewArxivSR",
            arxiv_id="2501.99999",
            year=2025,
            venue="arXiv",
            abstract="A very recent preprint on SR.",
            source="arxiv",
        )

        ev_own_ft: Evidence = await evidence_service.record_fulltext_evidence(
            paper_id="paper_sinsr",
            quote="SinSR distills 15 steps into 1 step.",
            owner_id=owner_id,
            session_id=session.session_id,
            locator=Locator(page_index=2, section_path="3 Method"),
            parse_version="pv_1",
            literature_id=lit_center,
        )
        ev_own_meta: Evidence = await evidence_service.record_metadata_evidence(
            literature_id=lit_center,
            quote="SinSR — published at CVPR, 2024",
            owner_id=owner_id,
            session_id=session.session_id,
            provider="Venue Catalog",
        )
        ev_ext_abs: Evidence = await evidence_service.record_abstract_evidence(
            literature_id=lit_resshift,
            quote="Residual shifting for diffusion super-resolution.",
            owner_id=owner_id,
            session_id=session.session_id,
            provider="research_sphere",
        )
        ev_pre_abs: Evidence = await evidence_service.record_abstract_evidence(
            literature_id=lit_preprint,
            quote="A very recent preprint on SR.",
            owner_id=owner_id,
            session_id=session.session_id,
            provider="research_sphere",
        )

        md = (
            f"本文一步完成超分 [{ev_own_ft.evidence_id}]，发表于 CVPR [{ev_own_meta.evidence_id}]；"
            f"前序工作 ResShift [{ev_ext_abs.evidence_id}] 与近期工作 NewArxivSR [{ev_pre_abs.evidence_id}]。"
        )
        frozen_md, table = await report_citations.freeze(
            md, owner_id=owner_id, paper_id="paper_sinsr", language="zh"
        )
        self.assertEqual(frozen_md, md)
        # Center paper's own fulltext has NO venue badge
        self.assertNotIn("venue", table[ev_own_ft.evidence_id])
        # Center paper's metadata citation resolves `CVPR` (joining `papers` past `li.venue='arXiv'`)!
        self.assertEqual(table[ev_own_meta.evidence_id]["venue"]["name"], "CVPR")
        self.assertEqual(table[ev_own_meta.evidence_id]["venue"]["year"], 2024)
        self.assertIn("CCF A", table[ev_own_meta.evidence_id]["venue"]["ranks"])

        # External NeurIPS work has frozen venue + tier
        self.assertEqual(table[ev_ext_abs.evidence_id]["venue"]["name"], "NeurIPS")
        self.assertIn("CCF A", table[ev_ext_abs.evidence_id]["venue"]["ranks"])

        # Preprint has subtrack='preprint'
        self.assertEqual(table[ev_pre_abs.evidence_id]["venue"]["subtrack"], "preprint")

        exported = report_citations.to_numbered(
            frozen_md, table, paper_id="paper_sinsr", language="zh"
        )
        self.assertIn("ResShift (NeurIPS 2023 · CCF A / CORE A*), 摘要", exported)
        self.assertIn("NewArxivSR (2025) [预印本], 摘要", exported)


if __name__ == "__main__":
    unittest.main()
