"""The literature map a Research Sphere written as the answer stands on (P16 phase 5).

`build_literature_map` runs the Sphere funnel once per paper and language,
registers each core paper and hub as a work with its abstract and metadata as
evidence, attaches them to the session as candidates without touching a paper
the session already holds, and hands the map to an open Sphere report.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest import mock

from app.models.paper_ir import PaperIR
from app.models.sphere_models import (
    CorePartition,
    KeyHub,
    RelationType,
    SphereNode,
    SphereOutput,
    SphereState,
)
from tests.test_agent_p2 import AgentP2TestCase

ABSTRACT = "We shorten the diffusion chain for super-resolution by shifting residuals between images."


def _map() -> SphereState:
    center = SphereNode(node_id="c", title="SparseMoE Routing", year=2024, venue="CVPR", cited_by_count=40)
    nodes = {
        "c": center,
        "f1": SphereNode(node_id="f1", title="Denoising Diffusion Probabilistic Models", year=2020,
                         doi="10.5555/ddpm", venue="NeurIPS", ccf_rank="A", cited_by_count=20000,
                         abstract_text="We present high quality image synthesis results using diffusion "
                                       "probabilistic models.",
                         relation_type=RelationType.FOUNDATION, relation_reason="The model it builds on.",
                         method_summary="Iterative denoising.", cluster_id=0, layer=1),
        "m1": SphereNode(node_id="m1", title="ResShift: Efficient Diffusion for Super-Resolution", year=2023,
                         arxiv_id="2307.12348", abstract_text=ABSTRACT, cited_by_count=300,
                         relation_type=RelationType.COMPETITOR, cluster_id=0, layer=1),
        "m2": SphereNode(node_id="m2", title="A Paper Without An Abstract", year=2022,
                         relation_type=RelationType.METHOD_NEIGHBOR, cluster_id=1, layer=1),
        "h1": SphereNode(node_id="h1", title="High-Resolution Image Synthesis with Latent Diffusion Models",
                         year=2022, cited_by_count=9000, abstract_text="Latent diffusion models reach a "
                         "near-optimal point between complexity reduction and detail preservation."),
    }
    return SphereState(
        center_node=center,
        nodes=nodes,
        layer1_node_ids=["f1", "m1", "m2"],
        output=SphereOutput(
            partitions=[CorePartition(key="method", node_ids=["m1", "m2"]),
                        CorePartition(key="foundation", node_ids=["f1"])],
            key_hubs=[KeyHub(node_id="f1", pagerank=0.4), KeyHub(node_id="h1", pagerank=0.2)],
        ),
    )


class LiteratureMapToolTests(AgentP2TestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        from app.agents.tools import triage
        from app.workflows import sphere_subgraph

        self.builds: list[dict[str, Any]] = []
        self.next_map = _map()

        async def fake_build(state, run_id=""):
            self.builds.append({"run_id": run_id, "persist": state.get("persist_sphere"),
                                "language": state["language"]})
            return self.next_map

        for patcher in (
            mock.patch.object(triage, "load_paper_ir",
                              return_value=PaperIR(paper_id="paper1", title="SparseMoE Routing")),
            mock.patch.object(sphere_subgraph, "build_literature_map", side_effect=fake_build),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    async def test_the_map_is_numbered_and_every_paper_is_citable(self) -> None:
        from app.agents.tools import build_literature_map
        from app.services import evidence_service

        result = await self._call(build_literature_map)
        self.assertEqual(result["status"], "ok")
        data = result["data"]
        papers = data["papers"]
        # Partitions in the report's order (foundations first), then a hub outside the core.
        self.assertEqual([(p["n"], p["group"]) for p in papers],
                         [(1, "foundation"), (2, "method"), (3, "method"), (4, "hub")])
        self.assertEqual(papers[0]["rank"], "CCF A")
        self.assertEqual(papers[0]["why"], "The model it builds on.")
        # A paper without an abstract is cited by its metadata, and says so.
        self.assertEqual(papers[2]["evidence_id"], papers[2]["metadata_evidence_id"])
        self.assertEqual(papers[2]["evidence_level"], "metadata")
        self.assertIn("abstract_excerpt", papers[1])  # no abstract analysis to show instead
        self.assertEqual(data["clusters"], [{"cluster": 0, "papers": [1, 2]}])
        self.assertEqual(data["hubs"], [1, 4])
        self.assertEqual(data["center"]["venue"], "CVPR")

        stored = await evidence_service.resolve_many(result["evidence_ids"], owner_id=self.owner)
        levels = sorted(e.source_level.value for e in stored)
        # Three abstracts, four papers' metadata, and this paper's own record.
        self.assertEqual(levels, ["abstract"] * 3 + ["metadata"] * 5)
        self.assertIn(data["center"]["evidence_id"], result["evidence_ids"])
        self.assertNotIn("fulltext", levels)
        ddpm = next(e for e in stored if e.source_level.value == "metadata" and "Denoising" in e.quote)
        self.assertIn("NeurIPS (CCF A)", ddpm.quote)
        self.assertIn("cited by 20,000", ddpm.quote)

    async def test_its_papers_join_the_session_as_candidates(self) -> None:
        from app.agents.tools import build_literature_map
        from app.db import agent_repository as repo
        from app.models.agent_models import Availability
        from app.services import paper_catalog

        # The session already holds ResShift's PDF: the map must not reset it.
        held = await paper_catalog.upsert_literature_item(
            title="ResShift: Efficient Diffusion for Super-Resolution", arxiv_id="2307.12348"
        )
        await repo.attach_session_paper(
            session_id=self.session.session_id, literature_id=held,
            availability=Availability.PDF_READY, added_by="agent",
        )
        result = await self._call(build_literature_map)
        self.assertEqual(result["data"]["candidates_added"], 4)
        self.assertEqual(result["data"]["papers"][1]["literature_id"], held)

        papers = {p.literature_id: p for p in await repo.list_session_papers(self.session.session_id)}
        self.assertEqual(papers[held].availability, Availability.PDF_READY)
        self.assertEqual(papers[held].added_by, "agent")
        mapped = [p for p in papers.values() if p.added_by == "sphere"]
        self.assertEqual(len(mapped), 3)
        self.assertTrue(all(p.availability == Availability.CANDIDATE for p in mapped))

    async def test_the_funnel_runs_once_and_the_report_gets_the_map(self) -> None:
        from app.agents.tools import build_literature_map
        from app.services.answer_reports import ReportTurn

        outside = await self._call(build_literature_map)
        self.assertFalse(outside["data"]["reused"])
        self.assertTrue(self.builds[0]["run_id"].startswith("map-"))
        self.assertFalse(self.builds[0]["persist"])

        self.ctx.report = ReportTurn(run_id="r1", mode="sphere", paper_id="paper1")
        again = await self._call(build_literature_map)
        self.assertTrue(again["data"]["reused"])
        self.assertEqual(len(self.builds), 1)
        handed = self.ctx.report.parts["map"]
        self.assertEqual(set(handed["state"]["nodes"]), {"c", "f1", "m1", "m2", "h1"})
        self.assertEqual(set(handed["literature_ids"]), {"f1", "m1", "m2", "h1"})
        self.assertEqual(handed["literature_ids"]["m1"], again["data"]["papers"][1]["literature_id"])

    async def test_inside_a_report_the_steps_report_on_its_row(self) -> None:
        from app.agents.tools import build_literature_map
        from app.services.answer_reports import ReportTurn

        self.ctx.report = ReportTurn(run_id="r1", mode="sphere", paper_id="paper1")
        await self._call(build_literature_map)
        self.assertEqual(self.builds[0], {"run_id": "r1", "persist": True, "language": "zh"})

        # A report of another mode is left alone.
        self.ctx.report = ReportTurn(run_id="r2", mode="lens", paper_id="paper1")
        await self._call(build_literature_map)
        self.assertEqual(self.ctx.report.parts, {})

    async def test_the_funnels_steps_become_tool_progress_events(self) -> None:
        from app.agents.tools import build_literature_map
        from app.db import agent_repository as repo
        from app.workflows.progress import emit_progress

        async def stepping_build(state, run_id=""):
            await emit_progress(run_id, "sphere_expand", "running")
            await emit_progress(run_id, "sphere_expand", "done")
            return self.next_map

        with mock.patch("app.workflows.sphere_subgraph.build_literature_map", side_effect=stepping_build):
            result = await self._call(build_literature_map)

        self.assertEqual(result["status"], "ok")
        events = await repo.list_events(self.session.session_id)
        progress = [e.payload for e in events if e.type.value == "tool.progress"]
        self.assertEqual([(p["tool"], p["step"], p["status"]) for p in progress], [
            ("build_literature_map", "sphere_expand", "running"),
            ("build_literature_map", "sphere_expand", "done"),
        ])

    async def test_an_empty_core_set_is_not_kept(self) -> None:
        from app.agents.tools import build_literature_map

        self.next_map = SphereState(center_node=SphereNode(node_id="c", title="SparseMoE Routing"))
        failed = await self._call(build_literature_map)
        self.assertEqual(failed["status"], "error")
        self.next_map = _map()
        retried = await self._call(build_literature_map)
        self.assertEqual(retried["status"], "ok")
        self.assertEqual(len(self.builds), 2)


class MappedPapersInThePromptTests(unittest.TestCase):
    def test_map_candidates_are_listed_apart(self) -> None:
        from app.agents.prompts import _format_papers
        from app.models.agent_models import Availability, SessionPaper

        held = SessionPaper(session_id="s", literature_id="lit_1", paper_id="p1", title="SinSR",
                            availability=Availability.PARSED, year=2024, year_known=True)
        mapped = SessionPaper(session_id="s", literature_id="lit_2", title="ResShift", year=2023,
                              year_known=True, added_by="sphere")
        fetched = mapped.model_copy(update={"literature_id": "lit_3", "title": "DDPM",
                                            "availability": Availability.UNAVAILABLE})
        text = _format_papers([held, mapped, fetched], "en")
        self.assertIn("- SinSR (2024) — paper_id=p1, parsed", text)
        self.assertIn("- DDPM (2023) — literature_id=lit_3, unavailable", text)
        self.assertIn("Literature-map candidates (1, metadata and abstract only):\n"
                      "- ResShift (2023) — literature_id=lit_2", text)
        # Without map candidates the list reads as it always did.
        self.assertNotIn("Literature-map", _format_papers([held], "en"))


if __name__ == "__main__":
    unittest.main()
