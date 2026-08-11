from __future__ import annotations

import json
import unittest

from app.models.snap_models import SnapClaim, SnapFinding, SnapReport
from app.services.snap_report import parse_snap_report, render_markdown, system_prompt

_FULL = {
    "one_liner": "FooNet reaches 28.4 BLEU on WMT14 with 3x less compute.",
    "contributions": [
        {"text": "A sparse attention variant", "page": 3},
        {"text": "A training schedule that halves steps", "page": 5},
    ],
    "findings": [
        {
            "metric": "BLEU", "dataset": "WMT14 EN-DE", "value": "28.4",
            "baseline": "26.3 (best prior)", "delta": "+2.1", "page": 8,
            "note": "single seed",
        },
        {"metric": "Params", "dataset": "—", "value": "213M", "page": 8},
    ],
    "suitable_for": "Long-sequence translation under a compute budget.",
    "limitations": [{"text": "Only evaluated on translation", "page": 9}],
}


class TestReportParsing(unittest.TestCase):
    def test_full_payload_round_trips(self) -> None:
        report = parse_snap_report(json.dumps(_FULL))
        self.assertFalse(report.degraded)
        self.assertEqual(len(report.contributions), 2)
        self.assertEqual(report.findings[0].delta, "+2.1")
        self.assertEqual(report.findings[0].page, 8)
        self.assertEqual(report.limitations[0].page, 9)

    def test_fenced_and_prose_wrapped_json_are_recovered(self) -> None:
        fenced = parse_snap_report("```json\n" + json.dumps(_FULL) + "\n```")
        wrapped = parse_snap_report("Here you go:\n" + json.dumps(_FULL) + "\nDone.")
        self.assertFalse(fenced.degraded)
        self.assertFalse(wrapped.degraded)

    def test_unquantified_findings_are_dropped(self) -> None:
        """A finding with no number is the vague claim the schema exists to block."""
        report = parse_snap_report(json.dumps({
            "one_liner": "x",
            "findings": [
                {"metric": "quality", "dataset": "various", "note": "significantly better"},
                {"metric": "BLEU", "dataset": "WMT14", "value": "28.4", "page": 8},
            ],
        }))
        self.assertEqual(len(report.findings), 1)
        self.assertEqual(report.findings[0].value, "28.4")

    def test_a_paper_with_no_numbers_yields_no_findings(self) -> None:
        report = parse_snap_report(json.dumps({
            "one_liner": "A position paper.",
            "contributions": [{"text": "An argument", "page": 1}],
            "findings": [],
        }))
        self.assertFalse(report.degraded)
        self.assertEqual(report.findings, [])

    def test_string_contributions_are_accepted(self) -> None:
        report = parse_snap_report(json.dumps({
            "one_liner": "x", "contributions": ["No page given"],
        }))
        self.assertEqual(report.contributions[0].page, 0)

    def test_invalid_pages_become_zero(self) -> None:
        report = parse_snap_report(json.dumps({
            "one_liner": "x",
            "contributions": [{"text": "a", "page": "unknown"}, {"text": "b", "page": -3}],
        }))
        self.assertEqual([c.page for c in report.contributions], [0, 0])

    def test_non_json_degrades_and_keeps_the_text(self) -> None:
        report = parse_snap_report("## Summary\nThis paper is about X.")
        self.assertTrue(report.degraded)
        self.assertIn("This paper is about X.", report.raw_markdown)

    def test_parsed_but_empty_payload_counts_as_degraded(self) -> None:
        self.assertTrue(parse_snap_report('{"suitable_for": "anything"}').degraded)

    def test_empty_input_degrades(self) -> None:
        self.assertTrue(parse_snap_report("").degraded)


class TestMarkdownRendering(unittest.TestCase):
    def test_findings_render_as_a_table(self) -> None:
        md = render_markdown(parse_snap_report(json.dumps(_FULL)))
        self.assertIn("| Metric | Dataset | This paper | Baseline | Δ | Page |", md)
        self.assertIn("| BLEU | WMT14 EN-DE | 28.4 | 26.3 (best prior) | +2.1 | [p.8] |", md)

    def test_missing_optional_columns_render_as_dashes(self) -> None:
        md = render_markdown(parse_snap_report(json.dumps(_FULL)))
        self.assertIn("| Params | — | 213M | — | — | [p.8] |", md)

    def test_notes_are_listed_under_the_table(self) -> None:
        md = render_markdown(parse_snap_report(json.dumps(_FULL)))
        self.assertIn("- BLEU: single seed [p.8]", md)

    def test_no_findings_states_so_explicitly(self) -> None:
        report = parse_snap_report(json.dumps({"one_liner": "x", "contributions": [{"text": "a", "page": 1}]}))
        md = render_markdown(report)
        self.assertIn("no quantified results", md)

    def test_pipes_in_cells_are_escaped(self) -> None:
        report = parse_snap_report(json.dumps({
            "one_liner": "x",
            "findings": [{"metric": "a|b", "dataset": "d", "value": "1"}],
        }))
        self.assertIn(r"a\|b", render_markdown(report))

    def test_chinese_rendering_uses_chinese_headings(self) -> None:
        md = render_markdown(parse_snap_report(json.dumps(_FULL)), language="zh")
        self.assertIn("## 一句话总结", md)
        self.assertIn("## 关键实验发现", md)
        self.assertIn("| 指标 | 数据集 | 本文 | 基线 | 变化 | 页码 |", md)
        # Metric and dataset names stay in English.
        self.assertIn("BLEU", md)
        self.assertIn("WMT14 EN-DE", md)

    def test_both_languages_produce_the_same_section_count(self) -> None:
        """Structure lives in code now, so the two languages cannot drift apart."""
        report = parse_snap_report(json.dumps(_FULL))
        en = render_markdown(report, language="en")
        zh = render_markdown(report, language="zh")
        self.assertEqual(
            len([ln for ln in en.splitlines() if ln.startswith("## ")]),
            len([ln for ln in zh.splitlines() if ln.startswith("## ")]),
        )

    def test_degraded_report_renders_the_raw_text_unchanged(self) -> None:
        raw = "## Summary\nSome prose the model produced."
        self.assertEqual(render_markdown(SnapReport(degraded=True, raw_markdown=raw)), raw)

    def test_page_citations_are_emitted_for_claims(self) -> None:
        md = render_markdown(parse_snap_report(json.dumps(_FULL)))
        self.assertIn("- A sparse attention variant [p.3]", md)
        self.assertIn("- Only evaluated on translation [p.9]", md)

    def test_claim_without_a_page_omits_the_citation(self) -> None:
        report = SnapReport(one_liner="x", contributions=[SnapClaim(text="No page", page=0)])
        self.assertIn("- No page\n", render_markdown(report))


class TestPromptComposition(unittest.TestCase):
    def test_chinese_directive_is_appended_only_for_zh(self) -> None:
        self.assertIn("Simplified Chinese", system_prompt("zh"))
        self.assertNotIn("Simplified Chinese", system_prompt("en"))

    def test_repair_directive_is_opt_in(self) -> None:
        self.assertIn("could not be parsed", system_prompt("en", repair=True))
        self.assertNotIn("could not be parsed", system_prompt("en"))

    def test_prompt_forbids_judging_worth(self) -> None:
        """The verdict belongs to the independent reviewer, not this pass."""
        self.assertIn("Do not judge whether the paper is worth reading", system_prompt("en"))



class TestCitationAudit(unittest.TestCase):
    """Coverage is computed from the fields, not guessed from prose.

    The regex auditor counted the one-liner and the applicability blurb as
    uncited claims — they are whole-paper syntheses with no single page — and so
    reported ~67% for a fully cited report.
    """

    def test_fully_cited_report_scores_one(self) -> None:
        from app.services.snap_report import citation_audit

        audit = citation_audit(parse_snap_report(json.dumps(_FULL)))
        self.assertEqual(audit["coverage"], 1.0)
        self.assertEqual(audit["claims_uncited"], 0)
        # 2 contributions + 1 limitation + 2 findings
        self.assertEqual(audit["claims_total"], 5)

    def test_syntheses_are_not_counted_as_claims(self) -> None:
        from app.services.snap_report import citation_audit

        report = parse_snap_report(json.dumps({
            "one_liner": "A long synthesis sentence with no single page to cite.",
            "suitable_for": "Another synthesis with no page.",
            "contributions": [{"text": "cited", "page": 3}],
        }))
        audit = citation_audit(report)
        self.assertEqual(audit["claims_total"], 1)
        self.assertEqual(audit["coverage"], 1.0)

    def test_uncited_claims_are_counted_and_sampled(self) -> None:
        from app.services.snap_report import citation_audit

        report = parse_snap_report(json.dumps({
            "one_liner": "x",
            "contributions": [{"text": "no page here", "page": 0}, {"text": "cited", "page": 4}],
        }))
        audit = citation_audit(report)
        self.assertEqual(audit["claims_total"], 2)
        self.assertEqual(audit["claims_uncited"], 1)
        self.assertEqual(audit["coverage"], 0.5)
        self.assertEqual(audit["uncited_samples"], ["no page here"])

    def test_empty_report_is_vacuously_covered(self) -> None:
        from app.services.snap_report import citation_audit

        audit = citation_audit(SnapReport(one_liner="x"))
        self.assertEqual(audit["coverage"], 1.0)
        self.assertEqual(audit["claims_total"], 0)

if __name__ == "__main__":
    unittest.main()
