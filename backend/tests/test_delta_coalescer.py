"""`DeltaCoalescer`: fewer delta events, never out of order."""

from __future__ import annotations

import asyncio
import unittest
from typing import Any
from unittest import mock

from app.models.agent_models import EventType


class DeltaCoalescerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.events: list[tuple[EventType, dict[str, Any]]] = []

        async def fake_emit(*, session_id: str, run_id: str, type: EventType, payload=None):
            self.events.append((type, dict(payload or {})))
            return None

        patcher = mock.patch("app.services.agent_runner.emit", side_effect=fake_emit)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _coalescer(self, *, flush_ms: int = 10_000, flush_chars: int = 1_000):
        from app.services.agent_runner import DeltaCoalescer

        return DeltaCoalescer(
            session_id="as_1", run_id="ar_1", flush_ms=flush_ms, flush_chars=flush_chars
        )

    async def test_fragments_are_joined_into_one_event(self) -> None:
        stream = self._coalescer()
        for part in ("The ", "answer ", "is 42."):
            await stream.add(part)
        self.assertEqual(self.events, [])
        await stream.close()
        self.assertEqual(self.events, [(EventType.MESSAGE_DELTA, {"text": "The answer is 42."})])

    async def test_size_threshold_flushes_immediately(self) -> None:
        stream = self._coalescer(flush_chars=5)
        await stream.add("abc")
        await stream.add("defg")
        self.assertEqual(self.events, [(EventType.MESSAGE_DELTA, {"text": "abcdefg"})])
        await stream.close()
        self.assertEqual(len(self.events), 1)

    async def test_timer_flushes_text_left_waiting(self) -> None:
        stream = self._coalescer(flush_ms=20)
        await stream.add("pause here")
        await asyncio.sleep(0.1)
        self.assertEqual(self.events, [(EventType.MESSAGE_DELTA, {"text": "pause here"})])
        await stream.close()
        self.assertEqual(len(self.events), 1)

    async def test_other_events_never_overtake_buffered_text(self) -> None:
        stream = self._coalescer()
        await stream.add("Let me read the method section.")
        await stream.emit(EventType.TOOL_STARTED, {"tool": "read_paper_section"})
        await stream.add("It uses ")
        await stream.add("contrastive loss.")
        await stream.emit(EventType.RUN_COMPLETED, {})
        self.assertEqual(
            [t for t, _ in self.events],
            [
                EventType.MESSAGE_DELTA,
                EventType.TOOL_STARTED,
                EventType.MESSAGE_DELTA,
                EventType.RUN_COMPLETED,
            ],
        )
        self.assertEqual(self.events[2][1], {"text": "It uses contrastive loss."})

    async def test_zero_window_writes_through(self) -> None:
        stream = self._coalescer(flush_ms=0)
        await stream.add("a")
        await stream.add("b")
        self.assertEqual([p["text"] for _, p in self.events], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
