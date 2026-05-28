from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.db import database as db
from app.models.paper_ir import Block, PaperIR, Section
from app.services.paper_ir import build_and_store_paper_ir
from app.services.qa_retrieval import (
    build_paper_nodes,
    retrieve_qa_context,
    retrieve_qa_context_for_paper,
)


def _paper_ir() -> PaperIR:
    title = Block(type="title", page_idx=0, order_idx=0, text="SuperMark")
    abstract = Block(
        type="text",
        page_idx=0,
        order_idx=1,
        text="Abstract. SuperMark is a training-free watermarking method.",
        section_path="Abstract",
    )
    experiments_title = Block(
        type="title",
        page_idx=4,
        order_idx=2,
        text="4 Experiments",
        section_path="4 Experiments",
    )
    metrics_text = Block(
        type="text",
        page_idx=4,
        order_idx=3,
        text="Evaluation Metrics. We report bit accuracy, PSNR, SSIM, and fidelity.",
        section_path="4 Experiments",
    )
    results_table = Block(
        type="table",
        page_idx=4,
        order_idx=4,
        text=(
            "<table><tr><td>Metric</td><td>Value</td></tr>"
            "<tr><td>PSNR</td><td>35.2</td></tr>"
            "<tr><td>SSIM</td><td>0.98</td></tr></table>"
        ),
        section_path="4 Experiments",
    )
    discussion = Block(
        type="text",
        page_idx=5,
        order_idx=5,
        text="Discussion. The adaptive attack accuracy remains high.",
        section_path="5 Discussion",
    )
    return PaperIR(
        paper_id="paper",
        title="SuperMark",
        sections=[
            Section(path="Abstract", title="Abstract", level=1, blocks=[abstract]),
            Section(
                path="4 Experiments",
                title="4 Experiments",
                level=1,
                blocks=[experiments_title, metrics_text, results_table],
            ),
            Section(path="5 Discussion", title="5 Discussion", level=1, blocks=[discussion]),
        ],
        blocks=[title, abstract, experiments_title, metrics_text, results_table, discussion],
    )


class QaRetrievalHierarchyTests(unittest.TestCase):
    def test_build_paper_nodes_preserves_section_parentage_and_original_chunks(self) -> None:
        nodes = build_paper_nodes(_paper_ir())

        root = next(node for node in nodes if node.node_type == "paper")
        experiments = next(
            node for node in nodes
            if node.node_type == "section" and node.title_path == "4 Experiments"
        )
        table = next(
            node for node in nodes
            if node.node_type == "chunk" and node.block_type == "table" and "PSNR" in node.text
        )

        self.assertEqual(experiments.parent_id, root.node_id)
        self.assertEqual(table.parent_id, experiments.node_id)
        self.assertEqual(table.page_start, 4)
        self.assertEqual(table.page_end, 4)
        self.assertTrue(table.text.startswith("<table>"))
        self.assertIn("4 Experiments", table.text_for_search)

    def test_metric_question_expands_from_relevant_leaf_to_section_siblings(self) -> None:
        context, blocks_used = retrieve_qa_context(_paper_ir(), "该方案的关键数据指标与性能分析是什么？")

        self.assertGreaterEqual(blocks_used, 3)
        self.assertIn("[p.5] [4 Experiments] 4 Experiments", context)
        self.assertIn("Evaluation Metrics", context)
        self.assertIn("<table>", context)
        self.assertIn("PSNR", context)


class PaperNodeStorageTests(unittest.IsolatedAsyncioTestCase):
    async def test_build_and_store_paper_ir_persists_hierarchy_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            db.set_db_path(base / "app.db")
            await db.init_db()
            await db.execute(
                "INSERT INTO papers (paper_id, file_path, title) VALUES (?, ?, ?)",
                ("paper", "papers/paper/original.pdf", ""),
            )

            output_dir = base / "mineru"
            output_dir.mkdir()
            (output_dir / "content_list.json").write_text(
                json.dumps([
                    {"type": "text", "text_level": 1, "text": "SuperMark", "page_idx": 0},
                    {"type": "text", "text": "Abstract. Training-free watermarking.", "page_idx": 0},
                    {"type": "text", "text_level": 1, "text": "4 Experiments", "page_idx": 4},
                    {
                        "type": "text",
                        "text": "Evaluation Metrics. We report PSNR and SSIM.",
                        "page_idx": 4,
                    },
                    {
                        "type": "table",
                        "table_body": "<table><tr><td>PSNR</td><td>35.2</td></tr></table>",
                        "page_idx": 4,
                    },
                ]),
                encoding="utf-8",
            )

            paper_ir = await build_and_store_paper_ir(output_dir, "paper")

            rows = await db.fetch_all(
                "SELECT node_type, title_path, text FROM paper_nodes "
                "WHERE paper_id = ? ORDER BY depth, order_idx",
                ("paper",),
            )

            self.assertTrue(any(row["node_type"] == "paper" for row in rows))
            self.assertTrue(
                any(row["node_type"] == "section" and row["title_path"] == "4 Experiments" for row in rows)
            )
            self.assertTrue(
                any(row["node_type"] == "chunk" and "PSNR" in row["text"] for row in rows)
            )

            context, blocks_used = await retrieve_qa_context_for_paper(
                "paper",
                paper_ir,
                "What are the PSNR and SSIM metrics?",
            )
            self.assertGreaterEqual(blocks_used, 3)
            self.assertIn("[p.5] [4 Experiments]", context)
            self.assertIn("PSNR", context)


if __name__ == "__main__":
    unittest.main()
