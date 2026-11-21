from __future__ import annotations

import json
import unittest

from app.services.snap_report import parse_snap_report

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


if __name__ == "__main__":
    unittest.main()
