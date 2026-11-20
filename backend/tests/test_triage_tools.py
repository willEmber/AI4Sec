"""The Insight Snap tools a report written as the answer uses (P16 phase 4).

`get_triage_signals` registers the paper's external record as metadata
evidence; `review_paper_blind` runs the blind review once per paper, language,
model and focus, and combines it with the signals into the verdict tier. Both
hand their structured results to an open Snap report.
"""

from __future__ import annotations

import unittest
from unittest import mock

from app.models.paper_ir import PaperIR
from app.models.snap_models import TriageSignals
from app.services.triage_verdict import ContentReview
from tests.test_agent_p2 import AgentP2TestCase


def _signals() -> TriageSignals:
    return TriageSignals(
        resolved=True,
        venue="IEEE/CVF Conference on Computer Vision and Pattern Recognition",
        venue_normalized="CVPR",
        year=2024,
        ccf_rank="A",
        cited_by_count=120,
        citations_per_year=60.0,
        is_open_access=True,
        oa_url="https://arxiv.org/abs/2311.14760",
        external_score=0.8,
        provenance={"venue": "OpenAlex", "ccf_rank": "EasyScholar", "citations": "Semantic Scholar",
                    "open_access": "OpenAlex"},
        unavailable=["intents"],
    )


def _review(**overrides) -> ContentReview:
    fields = dict(
        contribution_strength=3, evidence_strength=2, novelty=3, reproducibility=2,
        limitation_severity=1, target_readers="People accelerating diffusion SR.",
        reasons=["Single-step inference matches 15-step quality [p.7]"],
        red_flags=[], available=True,
    )
    fields.update(overrides)
    return ContentReview(**fields)


class TriageToolTests(AgentP2TestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        from app.agents.tools import triage
        from app.workflows import snap_subgraph

        self.reviews: list[dict] = []

        async def fake_review(_llm, *, context, question, language, model, log_label):
            self.reviews.append({"question": question, "language": language})
            return self.next_review

        self.next_review = _review()
        for patcher in (
            mock.patch.object(triage, "load_paper_ir", return_value=PaperIR(paper_id="paper1", title="SparseMoE Routing")),
            mock.patch.object(triage, "ensure_pub_rank", return_value={}),
            mock.patch.object(triage, "build_triage_context", side_effect=lambda *_a, **_k: _Context()),
            mock.patch.object(snap_subgraph, "collect_signals_safe", side_effect=self._collect),
            mock.patch.object(snap_subgraph, "run_content_review", side_effect=fake_review),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.lookups = 0

    async def _collect(self, *_args, **_kwargs) -> TriageSignals:
        import asyncio

        self.lookups += 1
        await asyncio.sleep(0.3)  # the indexes take seconds; calls made together overlap
        return _signals()

    def _open_snap(self) -> None:
        from app.services.answer_reports import ReportTurn

        self.ctx.report = ReportTurn(run_id="r1", mode="snap", paper_id="paper1")

    async def test_signals_are_facts_with_their_sources(self) -> None:
        from app.agents.tools import get_triage_signals
        from app.services import evidence_service

        self._open_snap()
        result = await self._call(get_triage_signals)
        self.assertEqual(result["status"], "ok")
        facts = result["data"]["facts"]
        self.assertTrue(any("CCF A" in f["fact"] and "EasyScholar" in f["fact"] for f in facts))
        self.assertTrue(any("cited by 120" in f["fact"] and "Semantic Scholar" in f["fact"] for f in facts))
        self.assertEqual(result["data"]["unavailable"], ["intents"])
        stored = await evidence_service.resolve_many(result["evidence_ids"], owner_id=self.owner)
        self.assertEqual({e.source_level.value for e in stored}, {"metadata"})
        # The scorecard gets the signals as the tool found them.
        self.assertEqual(self.ctx.report.parts["signals"]["ccf_rank"], "A")

    async def test_the_blind_review_runs_once_and_gives_the_verdict(self) -> None:
        from app.agents.tools import review_paper_blind

        self._open_snap()
        first = await self._call(review_paper_blind)
        self.assertEqual(first["status"], "ok")
        data = first["data"]
        self.assertEqual(data["scores"]["novelty"], 3)
        self.assertEqual(data["verdict"]["tier"], "must_read")
        self.assertEqual(data["verdict"]["label"], "必读")  # the context defaults to zh
        self.assertFalse(data["reused"])
        self.assertEqual(first["evidence_ids"], [])  # a judgment, not evidence
        parts = self.ctx.report.parts
        self.assertEqual(parts["verdict"]["tier"], "must_read")
        self.assertTrue(parts["content_review"]["available"])
        self.assertEqual(parts["signals"]["venue_normalized"], "CVPR")

        again = await self._call(review_paper_blind)
        self.assertTrue(again["data"]["reused"])
        self.assertEqual(len(self.reviews), 1)

        # A concern is part of the review's key: it is weighed anew.
        focused = await self._call(review_paper_blind, focus="Does it hold on real-world data?")
        self.assertFalse(focused["data"]["reused"])
        self.assertEqual(self.reviews[-1]["question"], "Does it hold on real-world data?")

    async def test_an_unusable_review_is_not_kept(self) -> None:
        from app.agents.tools import review_paper_blind

        self.next_review = ContentReview()  # available=False
        failed = await self._call(review_paper_blind)
        self.assertEqual(failed["status"], "error")
        self.next_review = _review()
        retried = await self._call(review_paper_blind)
        self.assertEqual(retried["status"], "ok")
        self.assertEqual(len(self.reviews), 2)

    async def test_the_two_tools_called_together_share_one_lookup(self) -> None:
        import asyncio

        from app.agents.tools import get_triage_signals, review_paper_blind

        signals, review = await asyncio.gather(
            self._call(get_triage_signals), self._call(review_paper_blind)
        )
        self.assertEqual((signals["status"], review["status"]), ("ok", "ok"))
        self.assertEqual(self.lookups, 1)

    async def test_a_report_of_another_mode_is_left_alone(self) -> None:
        from app.agents.tools import get_triage_signals
        from app.services.answer_reports import ReportTurn

        self.ctx.report = ReportTurn(run_id="r1", mode="lens", paper_id="paper1")
        await self._call(get_triage_signals)
        self.assertEqual(self.ctx.report.parts, {})


class _Context:
    text = "Abstract. We route tokens to two experts [p.1]."
    total_budget = 100_000
    used: dict = {}
    tables_used = 0
    figures_used = 0
    equations_used = 0
    truncated: list = []


if __name__ == "__main__":
    unittest.main()
