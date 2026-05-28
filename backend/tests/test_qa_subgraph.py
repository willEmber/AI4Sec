from __future__ import annotations

import unittest

from app.models.paper_ir import Block, PaperIR, Section
from app.workflows.qa_subgraph import _assemble_context


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
        ],
        blocks=[title, abstract, experiments_title, metrics_text, results_table],
    )


class QaContextAssemblyTests(unittest.TestCase):
    def test_chinese_metric_question_retrieves_experiment_blocks(self) -> None:
        context, _ = _assemble_context(_paper_ir(), "该方案的关键数据指标与性能分析是什么？")

        self.assertIn("[p.5] [4 Experiments] 4 Experiments", context)
        self.assertIn("Evaluation Metrics", context)
        self.assertIn("PSNR", context)

    def test_table_blocks_are_searchable_for_metric_questions(self) -> None:
        context, _ = _assemble_context(_paper_ir(), "What are the PSNR and SSIM metrics?")

        self.assertIn("<table>", context)
        self.assertIn("SSIM", context)


if __name__ == "__main__":
    unittest.main()
