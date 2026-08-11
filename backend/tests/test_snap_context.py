from __future__ import annotations

import unittest

from app.models.paper_ir import Block, PaperIR, Section
from app.services.ir_extract import (
    compact_table_text,
    html_table_to_pipe,
    section_text,
)
from app.services.snap_context import build_triage_context

_RESULTS_HTML = (
    "<table>"
    "<tr><th>Model</th><th>BLEU</th><th>Params</th></tr>"
    "<tr><td>Baseline</td><td>26.3</td><td>65M</td></tr>"
    "<tr><td>Ours</td><td><b>28.4</b></td><td>213M</td></tr>"
    "</table>"
)
_NOTATION_HTML = "<table><tr><td>x</td><td>the input sequence</td></tr></table>"


def _block(block_type: str, page: int, text: str, section: str = "") -> Block:
    return Block(type=block_type, page_idx=page, text=text, section_path=section)


def _paper(
    *,
    long_intro: bool = True,
    with_tables: bool = True,
    with_headings: bool = True,
) -> PaperIR:
    """A paper shaped like the failure cases: long intro, tables, late conclusion."""
    if not with_headings:
        blocks = [_block("text", 0, "Untitled body text about diffusion models. " * 50)]
        return PaperIR(
            paper_id="p",
            title="No Headings Detected",
            sections=[Section(path="", title="", level=0, blocks=blocks)],
            blocks=blocks,
        )

    intro_blocks = [
        _block("text", 0, "Problem statement paragraph. " * (400 if long_intro else 5), "1 Introduction"),
        _block("text", 1, "Our contributions are threefold: (1) X, (2) Y, (3) Z.", "1 Introduction"),
    ]
    exp_blocks = [_block("text", 5, "We evaluate on WMT14. " * 200, "4 Experiments")]
    if with_tables:
        exp_blocks += [
            _block("table", 5, "Table 5: Notation and symbols.\n" + _NOTATION_HTML, "4 Experiments"),
            _block("table", 8, "Table 2: Comparison of results on WMT14.\n" + _RESULTS_HTML, "4 Experiments"),
        ]
    sections = [
        Section(path="Abstract", title="Abstract", blocks=[_block("text", 0, "We propose a model.", "Abstract")]),
        Section(path="1 Introduction", title="1 Introduction", blocks=intro_blocks),
        Section(path="4 Experiments", title="4 Experiments", blocks=exp_blocks),
        Section(
            path="6 Conclusion",
            title="6 Conclusion",
            blocks=[_block("text", 9, "We showed a 2.1 BLEU improvement.", "6 Conclusion")],
        ),
    ]
    return PaperIR(
        paper_id="p",
        title="Attention Is All You Need",
        sections=sections,
        blocks=[b for s in sections for b in s.blocks],
    )


class TestHtmlTableConversion(unittest.TestCase):
    def test_pipe_table_keeps_numbers_and_drops_markup(self) -> None:
        pipe = html_table_to_pipe(_RESULTS_HTML)
        self.assertIn("| Model | BLEU | Params |", pipe)
        self.assertIn("28.4", pipe)
        self.assertNotIn("<b>", pipe)
        # Row 2 must be the Markdown header separator.
        self.assertRegex(pipe.splitlines()[1], r"^\|[\s\-|]+\|$")

    def test_caption_is_preserved_ahead_of_the_table(self) -> None:
        out = compact_table_text("Table 2: Comparison on WMT14.\n" + _RESULTS_HTML)
        self.assertTrue(out.startswith("Table 2: Comparison on WMT14."))
        self.assertIn("26.3", out)

    def test_unparseable_html_falls_back_to_stripped_text(self) -> None:
        out = compact_table_text("Caption only, no rows <table></table>")
        self.assertIn("Caption only", out)
        self.assertNotIn("<table>", out)

    def test_output_is_hard_capped(self) -> None:
        huge = "<table>" + "<tr><td>0.123456</td></tr>" * 500 + "</table>"
        self.assertLessEqual(len(compact_table_text(huge, max_chars=300)), 300)

    def test_cell_pipes_are_escaped(self) -> None:
        pipe = html_table_to_pipe("<table><tr><td>a|b</td><td>c</td></tr></table>")
        self.assertIn(r"a\|b", pipe)


class TestSectionText(unittest.TestCase):
    def test_numeric_subsections_inherit_parent_keywords(self) -> None:
        sections = [
            Section(path="5 Training", title="5 Training", blocks=[]),
            Section(
                path="5.3 Optimizer",
                title="5.3 Optimizer",
                blocks=[_block("text", 4, "We used Adam with beta2=0.98.")],
            ),
        ]
        ir = PaperIR(paper_id="p", sections=sections, blocks=[b for s in sections for b in s.blocks])
        # "5.3 Optimizer" matches nothing itself; it must inherit "training".
        self.assertIn("Adam", section_text(ir, {"training"}))


class TestTriageContextBudget(unittest.TestCase):
    def test_conclusion_survives_a_tight_budget(self) -> None:
        """The old prefix-cut dropped the conclusion first; priority funding must not."""
        ctx = build_triage_context(_paper(), total_budget=4_000)
        self.assertIn("2.1 BLEU improvement", ctx.text)
        self.assertGreater(ctx.used["conclusion"], 0)

    def test_results_tables_are_included_with_their_numbers(self) -> None:
        ctx = build_triage_context(_paper())
        self.assertIn("## Results Tables", ctx.text)
        self.assertIn("28.4", ctx.text)
        self.assertIn("26.3", ctx.text)
        self.assertGreaterEqual(ctx.tables_used, 1)

    def test_results_table_outranks_notation_table(self) -> None:
        ctx = build_triage_context(_paper())
        results_at = ctx.text.index("Table 2")
        notation_at = ctx.text.find("Table 5")
        self.assertTrue(notation_at == -1 or results_at < notation_at)

    def test_intro_keeps_its_trailing_contribution_list(self) -> None:
        """Head+tail fitting: a long intro's contributions sit at its end."""
        ctx = build_triage_context(_paper(long_intro=True), total_budget=6_000)
        self.assertIn("contributions are threefold", ctx.text)
        self.assertIn("Problem statement", ctx.text)
        self.assertIn("intro", ctx.truncated)

    def test_budget_bounds_the_assembled_string(self) -> None:
        for budget in (3_000, 8_000, 40_000):
            ctx = build_triage_context(_paper(), total_budget=budget)
            self.assertLessEqual(len(ctx.text), budget, f"budget={budget}")

    def test_unspent_budget_cascades_to_truncated_slots(self) -> None:
        """A short abstract must not leave the budget unused."""
        small = build_triage_context(_paper(), total_budget=6_000)
        large = build_triage_context(_paper(), total_budget=30_000)
        self.assertGreater(large.used["experiments"], small.used["experiments"])

    def test_sections_render_in_reading_order(self) -> None:
        ctx = build_triage_context(_paper())
        order = [
            ctx.text.index("## Abstract"),
            ctx.text.index("## Introduction & Contributions"),
            ctx.text.index("## Results Tables"),
            ctx.text.index("## Conclusion & Discussion"),
        ]
        self.assertEqual(order, sorted(order))

    def test_raw_block_fallback_when_no_heading_matches(self) -> None:
        """Papers whose headings MinerU missed must still produce a context."""
        ctx = build_triage_context(_paper(with_headings=False))
        self.assertIn("diffusion models", ctx.text)
        self.assertGreater(ctx.used["intro"], 0)

    def test_empty_paper_yields_empty_context(self) -> None:
        ctx = build_triage_context(PaperIR(paper_id="p"))
        self.assertEqual(ctx.text, "")
        self.assertEqual(ctx.slots_present, [])


if __name__ == "__main__":
    unittest.main()
