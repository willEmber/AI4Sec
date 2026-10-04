"""Parsing starts when a paper reaches a conversation, not when a turn asks.

The parse is the same keyed job the agent tool runs, so what matters here is
that starting it early costs one submission, that a turn arriving meanwhile
joins it, and that nothing starts when there is nothing to do.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest import mock

from tests.test_agent_p2 import AgentP2TestCase


class ParseOnAttachTests(AgentP2TestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        from app.config import get_settings

        env = mock.patch.dict(os.environ, {"MINERU_TOKEN": "test-token", "PARSE_ON_ATTACH": "true"})
        env.start()
        self.addCleanup(env.stop)
        get_settings.cache_clear()

    async def _unparsed_paper(self, paper_id: str = "paper2") -> str:
        from app.config import get_settings
        from app.db import database as db

        await db.execute(
            "INSERT INTO papers (paper_id, file_path, title) VALUES (?, ?, ?)",
            (paper_id, f"papers/{paper_id}/original.pdf", "Unparsed"),
        )
        pdf = get_settings().data_dir / "papers" / paper_id / "original.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(b"%PDF-1.4\n")
        return paper_id

    def _mineru(self, parse_pdf: mock.AsyncMock):
        stack = mock.patch.multiple(
            "app.services.mineru_adapter", parse_pdf=parse_pdf, resume_parse=mock.AsyncMock()
        )
        ir = mock.patch(
            "app.services.parse_service.build_and_store_paper_ir",
            new=mock.AsyncMock(return_value=mock.Mock(sections=[1, 2])),
        )
        return stack, ir

    async def _settle(self) -> None:
        from app.services import parse_service

        await asyncio.gather(*list(parse_service._background.values()))

    async def _availability(self, session_id: str, paper_id: str) -> str:
        from app.db import agent_repository as repo

        papers = await repo.list_session_papers(session_id)
        return next(p.availability.value for p in papers if p.paper_id == paper_id)

    async def test_attaching_parses_once_and_marks_every_session(self) -> None:
        from app.api.agent import _attach_papers
        from app.db import agent_repository as repo

        paper_id = await self._unparsed_paper()
        other = await repo.create_session(owner_id=self.owner, language="en")
        parse_pdf = mock.AsyncMock(return_value=Path("/tmp/out"))
        stack, ir = self._mineru(parse_pdf)
        with stack, ir:
            await _attach_papers(self.session.session_id, [paper_id])
            await _attach_papers(other.session_id, [paper_id])
            self.assertEqual(await self._availability(other.session_id, paper_id), "pdf_ready")
            await self._settle()

        parse_pdf.assert_awaited_once()
        self.assertEqual(await self._availability(self.session.session_id, paper_id), "parsed")
        self.assertEqual(await self._availability(other.session_id, paper_id), "parsed")

    async def test_a_turn_asking_meanwhile_joins_the_running_parse(self) -> None:
        from app.api.agent import _attach_papers
        from app.services import parse_service

        paper_id = await self._unparsed_paper()
        entered, finish = asyncio.Event(), asyncio.Event()

        async def slow_parse(*_args, **_kwargs):
            entered.set()
            await finish.wait()
            return Path("/tmp/out")

        parse_pdf = mock.AsyncMock(side_effect=slow_parse)
        stack, ir = self._mineru(parse_pdf)
        with stack, ir, mock.patch("app.services.agent_runner.emit", new=mock.AsyncMock()):
            await _attach_papers(self.session.session_id, [paper_id])
            await entered.wait()
            waiter = asyncio.create_task(
                parse_service.run_agent_parse(self.ctx, paper_id, tool_name="ensure_paper_parsed")
            )
            await asyncio.sleep(0.05)
            self.assertFalse(waiter.done())
            finish.set()
            handle = await waiter
            await self._settle()

        parse_pdf.assert_awaited_once()
        self.assertEqual(handle.status, "done")
        self.assertTrue(handle.reused)

    async def test_nothing_starts_for_a_parsed_paper_or_without_a_token(self) -> None:
        from app.config import get_settings
        from app.services import parse_service

        self.assertFalse(await parse_service.start_background_parse("paper1"))

        paper_id = await self._unparsed_paper()
        with mock.patch.dict(os.environ, {"MINERU_TOKEN": ""}):
            get_settings.cache_clear()
            self.assertFalse(await parse_service.start_background_parse(paper_id))
        with mock.patch.dict(os.environ, {"PARSE_ON_ATTACH": "false"}):
            get_settings.cache_clear()
            self.assertFalse(await parse_service.start_background_parse(paper_id))
        get_settings.cache_clear()

    async def test_shutdown_hands_the_lease_back(self) -> None:
        from app.api.agent import _attach_papers
        from app.services import agent_jobs, parse_service

        paper_id = await self._unparsed_paper()
        entered = asyncio.Event()

        async def never_finishes(*_args, **_kwargs):
            entered.set()
            await asyncio.Event().wait()

        stack, ir = self._mineru(mock.AsyncMock(side_effect=never_finishes))
        with stack, ir:
            await _attach_papers(self.session.session_id, [paper_id])
            await entered.wait()
            await parse_service.stop_background_parses()

        job = await agent_jobs.get_job(parse_service.parse_idempotency_key(paper_id))
        self.assertEqual(job["status"], "pending")
        self.assertTrue(job["remote_id"])
        self.assertEqual(parse_service._background, {})
