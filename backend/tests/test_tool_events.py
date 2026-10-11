"""What the event stream says about a tool call: its arguments and a preview."""

from __future__ import annotations

import json
import unittest

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from app.agents.tool_events import ToolCallTracker, preview_of, summarise_input
from app.services.agent_runner import _tool_completion_payload


def _chunk(index: int, *, name: str | None = None, args: str = "", call_id: str | None = None):
    return AIMessageChunk(
        content="",
        id="m1",
        tool_call_chunks=[
            {"name": name, "args": args, "id": call_id, "index": index, "type": "tool_call_chunk"}
        ],
    )


class ToolCallTrackerTest(unittest.TestCase):
    def test_a_streamed_call_is_announced_once_its_arguments_are_complete(self) -> None:
        tracker = ToolCallTracker()
        self.assertEqual(tracker.feed(_chunk(0, name="search_papers", call_id="c1")), [])
        self.assertEqual(tracker.feed(_chunk(0, args='{"query": "gra')), [])
        self.assertEqual(
            tracker.feed(_chunk(0, args='ph"}')),
            [("search_papers", "c1", {"query": "graph"})],
        )
        # Nothing is announced twice, whether as a late chunk or the full message.
        self.assertEqual(tracker.feed(_chunk(0, args="")), [])
        full = AIMessage(
            content="", id="m1",
            tool_calls=[{"name": "search_papers", "args": {"query": "graph"}, "id": "c1"}],
        )
        self.assertEqual(tracker.feed(full), [])
        self.assertEqual(tracker.drain(), [])

    def test_a_call_without_arguments_is_announced_when_drained(self) -> None:
        tracker = ToolCallTracker()
        self.assertEqual(tracker.feed(_chunk(0, name="list_memories", call_id="c2")), [])
        self.assertEqual(tracker.drain(), [("list_memories", "c2", {})])
        self.assertEqual(tracker.drain(), [])

    def test_finished_tool_calls_are_announced_with_their_arguments(self) -> None:
        tracker = ToolCallTracker()
        message = AIMessage(
            content="",
            tool_calls=[
                {"name": "web_search", "args": {"query": "a"}, "id": "x"},
                {"name": "web_search", "args": {"query": "b"}, "id": "y"},
            ],
        )
        self.assertEqual(
            tracker.feed(message),
            [("web_search", "x", {"query": "a"}), ("web_search", "y", {"query": "b"})],
        )


class SummariesTest(unittest.TestCase):
    def test_input_keeps_what_was_asked_and_drops_defaults(self) -> None:
        summary = summarise_input(
            {
                "query": "q" * 500, "year_from": 0, "year_to": 2025, "venues": ["NeurIPS", ""],
                "platforms": None, "open_access_only": False, "nested": {"a": 1},
            }
        )
        self.assertEqual(set(summary), {"query", "year_to", "venues"})
        self.assertLessEqual(len(summary["query"]), 200)
        self.assertEqual(summary["venues"], ["NeurIPS"])

    def test_preview_is_short_and_bounded(self) -> None:
        hits = [
            {"section": f"3.{i} Method", "page": i + 1, "type": "text", "text": "word " * 200}
            for i in range(9)
        ]
        preview = preview_of({"paper_id": "p", "hits": hits})
        self.assertEqual(len(preview), 5)
        self.assertEqual(preview[0]["title"], "3.0 Method")
        self.assertEqual(preview[0]["page"], 1)
        self.assertLessEqual(len(preview[0]["text"]), 180)

    def test_a_search_across_papers_previews_every_paper(self) -> None:
        preview = preview_of(
            {
                "papers": [
                    {"title": "A", "hits": [{"section": "a1", "text": "x"}, {"section": "a2", "text": "y"}]},
                    {"title": "B", "hits": [{"section": "b1", "text": "z"}]},
                ]
            }
        )
        self.assertEqual([p["title"] for p in preview], ["a1", "b1", "a2"])
        self.assertEqual(preview[1]["meta"], "B")

    def test_the_completion_event_carries_the_call_id_and_preview(self) -> None:
        result = {
            "status": "ok",
            "evidence_ids": ["ev_1"],
            "data": {
                "query": "q",
                "results": [
                    {"title": "P", "year": 2024, "venue": "ICML", "url": "https://x.org/p",
                     "abstract_excerpt": "abs"},
                ],
            },
        }
        payload = _tool_completion_payload(
            ToolMessage(content=json.dumps(result), tool_call_id="c9", name="search_papers")
        )
        self.assertEqual(payload["call_id"], "c9")
        self.assertEqual(
            payload["preview"],
            [{"title": "P", "text": "abs", "url": "https://x.org/p", "meta": "2024 · ICML"}],
        )
        self.assertEqual(payload["summary"]["results_count"], 1)


if __name__ == "__main__":
    unittest.main()
