"""`get_paper_elements` (P16 phase 3): a paper's formulas, procedures, tables and figures, citable."""

from __future__ import annotations

import unittest
from unittest import mock

from app.models.paper_ir import Block, PaperIR
from tests.test_agent_p2 import AgentP2TestCase

TABLE_HTML = (
    "Table 1. BLEU on WMT14. <table><tr><td>model</td><td>BLEU</td></tr>"
    "<tr><td>dense</td><td>28.4</td></tr><tr><td>ours</td><td>29.7</td></tr></table>"
)


def _paper_ir() -> PaperIR:
    """Elements placed in the seeded sections: 2 Method holds blocks 10-29, 3 Experiments 30-49."""
    return PaperIR(
        paper_id="paper1",
        title="SparseMoE Routing",
        sections=[],
        blocks=[
            Block(type="text", text="Intro.", page_idx=0, order_idx=2, section_path="1 Introduction"),
            Block(type="equation", text="$$\ng(x) = \\mathrm{softmax}(W_g x)\\tag{1}\n$$",
                  page_idx=2, order_idx=11, section_path="2 Method"),
            Block(type="image", text="Figure 2. Overall framework of the proposed router.",
                  img_path="images/fw.jpg", page_idx=2, order_idx=12, section_path="2 Method",
                  bbox=[10, 20, 300, 200]),
            Block(type="image", text="[image]", img_path="images/x.jpg", page_idx=2, order_idx=13,
                  section_path="2 Method"),
            Block(type="code", sub_type="algorithm",
                  text="Algorithm 1 Top-2 routing\n1: compute g(x)\n2: keep the top two",
                  page_idx=3, order_idx=14, section_path="2 Method"),
            Block(type="table", text=TABLE_HTML, page_idx=6, order_idx=31, section_path="3 Experiments"),
            Block(type="image", text="Figure 3. BLEU against the number of experts.",
                  img_path="images/plot.jpg", page_idx=6, order_idx=32, section_path="3 Experiments"),
        ],
    )


class PaperElementTests(AgentP2TestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        from app.agents.tools import elements

        patcher = mock.patch.object(elements, "load_paper_ir", return_value=_paper_ir())
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_every_kind_is_listed_with_its_own_evidence(self) -> None:
        from app.agents.tools import get_paper_elements
        from app.services import evidence_service

        result = await self._call(get_paper_elements, paper_id="paper1")
        self.assertEqual(result["status"], "ok")
        data = result["data"]
        self.assertEqual(data["counts"], {"equation": 1, "algorithm": 1, "table": 1, "figure": 2})
        by_kind = {}
        for element in data["elements"]:
            by_kind.setdefault(element["kind"], []).append(element)

        [eq] = by_kind["equation"]
        self.assertEqual(eq["label"], "Eq. 1")
        self.assertEqual(eq["latex"], "g(x) = \\mathrm{softmax}(W_g x)\\tag{1}")
        self.assertEqual(eq["page"], 3)

        [table] = by_kind["table"]
        self.assertEqual(table["label"], "Table 1")
        self.assertIn("| ours | 29.7 |", table["table"])

        [algorithm] = by_kind["algorithm"]
        self.assertEqual(algorithm["label"], "Algorithm 1")

        framework, plot = by_kind["figure"]
        self.assertEqual(framework["image_url"], "/api/papers/paper1/images/fw.jpg")
        self.assertTrue(framework["framework"])
        self.assertNotIn("framework", plot)

        # Each element is citable, and what it quotes is the element itself.
        ids = [e["evidence_id"] for e in data["elements"]]
        self.assertEqual(result["evidence_ids"], ids)
        stored = {e.evidence_id: e for e in await evidence_service.resolve_many(ids, owner_id=self.owner)}
        self.assertEqual(stored[eq["evidence_id"]].quote, eq["latex"])
        self.assertEqual(stored[eq["evidence_id"]].source_level.value, "fulltext")
        self.assertEqual(stored[framework["evidence_id"]].locator.page_label, 3)
        self.assertEqual(stored[framework["evidence_id"]].locator.bbox, [10, 20, 300, 200])
        self.assertIn("29.7", stored[table["evidence_id"]].quote)

    async def test_kinds_and_a_section_narrow_the_list(self) -> None:
        from app.agents.tools import get_paper_elements

        result = await self._call(
            get_paper_elements, kinds=["figure", "table"], section_id="paper1:s3"
        )
        data = result["data"]
        self.assertEqual(data["section"], "3 Experiments")
        self.assertEqual(data["counts"], {"table": 1, "figure": 1})
        self.assertEqual([e["kind"] for e in data["elements"]], ["table", "figure"])

        missing = await self._call(get_paper_elements, paper_id="paper1", section_id="paper1:s9")
        self.assertEqual(missing["status"], "error")
        self.assertEqual(missing["error"]["code"], "section_not_found")

        none = await self._call(
            get_paper_elements, paper_id="paper1", kinds=["equation"], section_id="paper1:s1"
        )
        self.assertEqual((none["status"], none["data"]["elements"]), ("ok", []))
        self.assertIn("No such elements", none["note"])

    async def test_a_long_list_is_cut_and_says_so(self) -> None:
        from app.agents.tools import elements, get_paper_elements

        with mock.patch.dict(elements.MAX_PER_KIND, {"figure": 1}):
            result = await self._call(get_paper_elements, paper_id="paper1", kinds=["figure"])
        self.assertEqual(len(result["data"]["elements"]), 1)
        self.assertEqual(result["data"]["counts"], {"figure": 2})
        self.assertIn("1 of 2 figures", result["note"])

    async def test_a_paper_outside_the_session_is_refused(self) -> None:
        from app.agents.tools import get_paper_elements

        result = await self._call(get_paper_elements, paper_id="stranger")
        self.assertNotEqual(result["status"], "ok")


if __name__ == "__main__":
    unittest.main()
