"""The offline half of `scripts.verify_multi_turn`: its checks and the trace reading.

The live script spends tokens; what it concludes from a turn's record and from
Phoenix's spans is plain data processing, and is pinned here.
"""

from __future__ import annotations

import json
import unittest

from scripts import phoenix_traces
from scripts.verify_multi_turn import Turn, _english, check_turn


def _record(**overrides):
    record = {
        "status": "done",
        "error_code": "",
        "answer": "答案 [ev_aaaaaaaa]",
        "tool_calls": [],
        "citations": ["ev_aaaaaaaa"],
        "evidence": {"ev_aaaaaaaa": {"paper_id": "p1"}},
        "grouped_citation_markers": 0,
        "invented_citations": [],
    }
    record.update(overrides)
    return record


def _by_name(results):
    return {r["check"]: r for r in results}


class CheckTurnTests(unittest.TestCase):
    def test_a_turn_without_an_answer_fails_hard(self) -> None:
        checks = _by_name(check_turn(Turn("q"), _record(answer=""), {}))
        self.assertFalse(checks["finished"]["ok"])
        self.assertTrue(checks["finished"]["hard"])

    def test_tool_expectations_are_hard_and_call_counts_soft(self) -> None:
        turn = Turn("q", tools_any=("save_memory",), tools_none=("run_insight_snap",), max_tool_calls=1)
        record = _record(tool_calls=[{"tool": "run_insight_snap"}, {"tool": "search_paper_content"}])
        checks = _by_name(check_turn(turn, record, {}))
        self.assertFalse(checks["tools_any"]["ok"])
        self.assertFalse(checks["tools_none"]["ok"])
        self.assertTrue(checks["tools_none"]["hard"])
        self.assertFalse(checks["max_tool_calls"]["ok"])
        self.assertFalse(checks["max_tool_calls"]["hard"])

    def test_a_comparison_must_cite_every_paper(self) -> None:
        turn = Turn("q", cites_papers=("a", "b"))
        checks = _by_name(check_turn(turn, _record(), {"a": "p1", "b": "p2"}))
        self.assertFalse(checks["cites_papers"]["ok"])
        self.assertIn("b", checks["cites_papers"]["detail"])

    def test_grouped_and_invented_markers_are_flagged(self) -> None:
        record = _record(grouped_citation_markers=2, invented_citations=["ev_zzzzzzzz"])
        checks = _by_name(check_turn(Turn("q"), record, {}))
        self.assertFalse(checks["citation_format"]["ok"])
        self.assertFalse(checks["no_invented_citations"]["ok"])

    def test_english_ignores_citations_formulas_and_numbers(self) -> None:
        self.assertTrue(_english("The AUC is 0.95 [ev_81401df3d9530d84e1b5] with $x_t = \\alpha$."))
        self.assertFalse(_english("主要局限在于 robustness 不足 [ev_81401df3d9530d84e1b5]"))


def _span(trace, span_id, kind, start, end, parent="p", **attributes):
    return {
        "name": kind.lower(),
        "context": {"trace_id": trace, "span_id": span_id},
        "parent_id": parent,
        "span_kind": kind,
        "start_time": f"2026-10-15T00:00:{start:02d}+00:00",
        "end_time": f"2026-10-15T00:00:{end:02d}+00:00",
        "status_code": "OK",
        "attributes": {"openinference.span.kind": kind, **attributes},
    }


def _tool_output(status: str, text: str) -> str:
    envelope = json.dumps({"status": status, "data": {"text": text}})
    # As Phoenix stores it: the ToolMessage's own status says only that the
    # tool node did not raise.
    return json.dumps({"type": "tool", "data": {"content": envelope, "status": "success"}})


class TraceAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        # One turn: a step issuing two reads together, then a step repeating
        # one of them alone, then the answer.
        read = {"tool.name": "read_paper_section", "input.value": '{"section_id": "s1"}'}
        self.spans = [
            _span("t1", "root", "CHAIN", 0, 30, parent=None),
            _span("t1", "l1", "LLM", 1, 4, **{
                "llm.token_count.prompt": 7000, "llm.token_count.completion": 200,
                "llm.tools.0.tool.json_schema": "x" * 50,
                "llm.tools.1.tool.json_schema": "y" * 50,
                "llm.input_messages.0.message.role": "system",
                "llm.input_messages.0.message.content": "s" * 400,
                "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "read_paper_section",
                "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": '{"section_id": "s1"}',
                "llm.output_messages.0.message.tool_calls.1.tool_call.function.name": "get_paper_outline",
                "llm.output_messages.0.message.tool_calls.1.tool_call.function.arguments": "{}",
            }),
            _span("t1", "x1", "TOOL", 4, 5, **read, **{"output.value": _tool_output("ok", "abc")}),
            _span("t1", "x2", "TOOL", 4, 5, **{"tool.name": "get_paper_outline", "input.value": "{}",
                                                  "output.value": _tool_output("ok", "outline")}),
            _span("t1", "l2", "LLM", 6, 8, **{
                "llm.token_count.prompt": 9000, "llm.token_count.completion": 50,
                "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "read_paper_section",
            }),
            _span("t1", "x3", "TOOL", 8, 9, **read, **{"output.value": _tool_output("error", "abc")}),
            _span("t1", "l3", "LLM", 10, 29, **{
                "llm.token_count.prompt": 9500, "llm.token_count.completion": 1500,
            }),
        ]

    def test_steps_record_tokens_and_the_calls_each_issued(self) -> None:
        turn = phoenix_traces.analyse(self.spans)["turns"][0]
        self.assertEqual(turn["model_calls"], 3)
        self.assertEqual(turn["calls_per_step"], [2, 1, 0])
        self.assertEqual(turn["single_call_steps"], 1)
        self.assertEqual(turn["prompt_tokens"], 25_500)
        self.assertEqual(turn["max_prompt_tokens"], 9500)
        self.assertEqual(turn["steps"][0]["tools_offered"], 2)
        self.assertEqual(turn["steps"][0]["tool_schema_chars"], 100)
        self.assertEqual(turn["steps"][0]["input_chars_by_role"], {"system": 400})

    def test_repeated_identical_calls_are_counted(self) -> None:
        turn = phoenix_traces.analyse(self.spans)["turns"][0]
        self.assertEqual(turn["repeated_calls"], [
            {"tool": "read_paper_section", "input": '{"section_id": "s1"}', "times": 2}
        ])

    def test_tool_results_are_measured_inside_the_tool_message(self) -> None:
        turn = phoenix_traces.analyse(self.spans)["turns"][0]
        first = turn["tools"][0]
        self.assertEqual(first["status"], "ok")
        self.assertEqual(turn["tools"][2]["status"], "error")
        self.assertEqual(first["result_chars"], len(json.dumps({"status": "ok", "data": {"text": "abc"}})))

    def test_turns_are_ordered_and_traces_without_a_model_call_dropped(self) -> None:
        later = [dict(s, context={"trace_id": "t2", "span_id": s["context"]["span_id"] + "b"},
                      start_time=s["start_time"].replace("00:00:", "00:01:"),
                      end_time=s["end_time"].replace("00:00:", "00:01:")) for s in self.spans]
        stray = [_span("t3", "only", "CHAIN", 0, 1, parent=None)]
        result = phoenix_traces.analyse(later + stray + self.spans)
        self.assertEqual([t["trace_id"] for t in result["turns"]], ["t1", "t2"])
        self.assertEqual(result["totals"]["model_calls"], 6)
        self.assertEqual(result["per_tool"]["read_paper_section"]["calls"], 4)

    def test_nested_attributes_are_flattened_to_openinference_names(self) -> None:
        nested = {"llm": {"token_count": {"prompt": 3}, "tools": [{"tool": {"json_schema": "s"}}]}}
        flat = phoenix_traces.flatten(nested)
        self.assertEqual(flat, {"llm.token_count.prompt": 3, "llm.tools.0.tool.json_schema": "s"})


if __name__ == "__main__":
    unittest.main()
