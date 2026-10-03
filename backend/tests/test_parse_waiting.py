"""Queue, progress and recovery regressions without a live provider or database."""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from app.agents.context import AgentContext
from app.models.agent_models import ErrorCode
from app.services import agent_jobs, agent_worker, mineru_adapter, parse_service


class PollTests(unittest.TestCase):
    def poll(self, results: list[dict], times: list[float], **kwargs):
        client = Mock()
        client.get_batch_results.side_effect = [{"extract_result": [r]} for r in results]
        clock = iter(times)
        return mineru_adapter._poll_until_done_sync(
            client, "batch", time_fn=lambda: next(clock), sleep_fn=lambda _: None,
            **kwargs,
        )

    def test_persistent_queue_has_a_separate_waiting_limit(self):
        with self.assertRaises(mineru_adapter.MinerUQueueTimeoutError) as caught:
            self.poll([{"state": "pending"}] * 2, [0, 0, 6], queue_timeout_s=5)
        self.assertEqual(caught.exception.last_state_counts, {"pending": 1})
        self.assertEqual(caught.exception.poll_count, 2)

    def test_queue_limit_does_not_cut_off_active_extraction(self):
        results = self.poll(
            [{"state": "pending"}, {"state": "running"}, {"state": "done"}],
            [0, 0, 10, 20], queue_timeout_s=5,
        )
        self.assertEqual(results[0]["state"], "done")

    def test_page_progress_reaches_the_snapshot_callback(self):
        events = []
        self.poll(
            [{"state": "running", "extract_progress": {"extracted_pages": 3, "total_pages": 8}},
             {"state": "done"}], [0, 0, 1], on_poll=events.append,
        )
        self.assertEqual(events[0]["extracted_pages"], 3)
        self.assertEqual(events[0]["total_pages"], 8)
        self.assertEqual(events[1]["total_pages"], 8)
        self.assertEqual(events[0]["phase"], "running")

    def test_zero_disables_the_queue_limit_but_not_the_total_limit(self):
        with self.assertRaises(mineru_adapter.MinerUPollTimeoutError) as caught:
            self.poll([{"state": "pending"}], [0, 11], timeout_s=10, queue_timeout_s=0)
        self.assertNotIsInstance(caught.exception, mineru_adapter.MinerUQueueTimeoutError)


class WaitingTests(unittest.IsolatedAsyncioTestCase):
    def timeout(self):
        return mineru_adapter.MinerUQueueTimeoutError("batch", 300, 51, {"pending": 1}, 300)

    async def test_timeout_keeps_the_parse_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = SimpleNamespace(data_dir=Path(tmp))
            execute = AsyncMock()
            with patch.object(mineru_adapter, "get_settings", return_value=settings), \
                 patch.object(mineru_adapter, "_get_client"), \
                 patch.object(mineru_adapter.asyncio, "to_thread", AsyncMock(side_effect=self.timeout())), \
                 patch.object(mineru_adapter.db, "execute", execute):
                with self.assertRaises(mineru_adapter.MinerUQueueTimeoutError):
                    await mineru_adapter.resume_parse("paper", "parse", "batch")
            self.assertEqual(execute.await_args_list[-1].args[1][0], "pending")
        with patch.object(agent_jobs, "get_job_by_id", AsyncMock(return_value={"remote_id": "parse"})), \
             patch.object(parse_service.db, "fetch_one", AsyncMock(return_value={"status": "pending", "remote_batch_id": "batch"})):
            self.assertEqual(await parse_service.resumable_batch("job"), ("parse", "batch"))

    async def test_owner_and_joined_waiter_get_the_same_timeout_contract(self):
        entered, finish = asyncio.Event(), asyncio.Event()

        async def work(_job_id):
            entered.set()
            await finish.wait()
            raise self.timeout()

        row = {"job_id": "job", "status": "running", "attempts": 1, "remote_id": "parse"}
        with patch.object(agent_jobs, "get_job", AsyncMock(return_value=row)), \
             patch.object(agent_jobs, "_claim", AsyncMock(return_value=("job", 1))) as claim, \
             patch.object(agent_jobs, "_finish", AsyncMock()) as record:
            owner = asyncio.create_task(agent_jobs.run_once(kind="parse", idempotency_key="p", work=work))
            await entered.wait()
            joined = asyncio.create_task(agent_jobs.run_once(kind="parse", idempotency_key="p", work=work))
            await asyncio.sleep(0)
            finish.set()
            outcomes = await asyncio.gather(owner, joined, return_exceptions=True)
        self.assertTrue(all(isinstance(e, agent_jobs.JobFailed) and e.code == ErrorCode.TIMEOUT for e in outcomes))
        claim.assert_awaited_once()
        self.assertEqual(record.await_args.kwargs["status"], "pending")
        self.assertTrue(record.await_args.kwargs["refund_attempt"])

    async def test_a_live_third_attempt_is_not_reported_as_failed(self):
        row = {"job_id": "job", "kind": "parse", "status": "running", "attempts": 3,
               "lease_expires_at": agent_jobs._lease_expiry(), "remote_id": "parse"}
        with patch.object(agent_jobs, "get_job", AsyncMock(return_value=row)), \
             patch.object(agent_jobs, "_claim", AsyncMock(return_value=None)), \
             patch.object(agent_jobs, "fail_exhausted_job", AsyncMock()) as fail:
            handle = await agent_jobs.run_once(kind="parse", idempotency_key="p", work=AsyncMock())
        self.assertEqual(handle.status, "running")
        fail.assert_not_awaited()

    async def test_exhausted_stale_jobs_are_closed_before_any_resume(self):
        row = {"job_id": "job", "kind": "parse", "attempts": 3}
        with patch.object(agent_jobs, "list_stale_jobs", AsyncMock(return_value=[row])), \
             patch.object(agent_jobs, "fail_exhausted_job", AsyncMock(return_value=True)) as fail, \
             patch.object(agent_worker, "_parse_is_resumable", AsyncMock()) as resume:
            await agent_worker.recover_stale_jobs()
        fail.assert_awaited_once_with("job")
        resume.assert_not_awaited()

    async def test_progress_reports_remote_queue_before_completion(self):
        ctx = AgentContext(owner_id="owner", session_id="session", run_id="run")
        seen = []
        observed = asyncio.Event()

        async def emit(**kwargs):
            seen.append(kwargs["payload"])
            if kwargs["payload"].get("phase") == "pending":
                observed.set()

        async def run_once(**kwargs):
            await asyncio.wait_for(observed.wait(), 2)
            return agent_jobs.JobHandle("job", "parse", "done", {})

        row = {"status": "running", "last_state_counts": '{"pending": 1}', "progress_json": '{}'}
        with patch("app.services.agent_runner.emit", side_effect=emit), \
             patch.object(parse_service.db, "fetch_one", AsyncMock(return_value=row)), \
             patch.object(agent_jobs, "run_once", side_effect=run_once):
            await parse_service.run_agent_parse(ctx, "paper", tool_name="ensure_paper_parsed")
        self.assertEqual(seen[0]["phase"], "submitting")
        self.assertTrue(any(e.get("phase") == "pending" for e in seen))
        self.assertEqual(seen[-1]["status"], "done")

    async def test_timeout_blocks_repeated_waits_only_for_the_current_turn(self):
        ctx = AgentContext(owner_id="owner", session_id="session", run_id="run")
        timeout = agent_jobs.JobFailed(ErrorCode.TIMEOUT, "still queued")
        with patch("app.services.agent_runner.emit", AsyncMock()), \
             patch.object(parse_service.db, "fetch_one", AsyncMock(return_value=None)), \
             patch.object(agent_jobs, "run_once", AsyncMock(side_effect=timeout)) as run:
            for tool in ["ensure_paper_parsed", "run_insight_snap"]:
                with self.assertRaises(agent_jobs.JobFailed):
                    await parse_service.run_agent_parse(ctx, "paper", tool_name=tool)
            self.assertEqual(run.await_count, 1)
            next_turn = AgentContext(owner_id="owner", session_id="session", run_id="next")
            with self.assertRaises(agent_jobs.JobFailed):
                await parse_service.run_agent_parse(next_turn, "paper", tool_name="ensure_paper_parsed")
            self.assertEqual(run.await_count, 2)

    async def test_mode_timeout_returns_partial_instead_of_a_fulltext_report(self):
        from app.agents.tools import modes

        ctx = AgentContext(owner_id="owner", session_id="session")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "papers/paper/original.pdf"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"%PDF")
            with patch.object(modes, "get_settings", return_value=SimpleNamespace(data_dir=Path(tmp))), \
                 patch.object(modes, "_parsed_already", AsyncMock(return_value=False)), \
                 patch.object(parse_service, "run_agent_parse", AsyncMock(side_effect=agent_jobs.JobFailed(ErrorCode.TIMEOUT, "queued"))):
                result = await modes._ensure_parsed(ctx, "paper", tool_name="run_insight_snap")
            self.assertEqual(result.status.value, "partial")
            self.assertIn("do not retry", result.note)

    async def test_parse_tool_tells_the_model_the_full_text_is_still_pending(self):
        from app.agents.tools import acquisition

        ctx = AgentContext(owner_id="owner", session_id="session")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "papers/paper/original.pdf"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"%PDF")
            with patch.object(acquisition, "get_settings", return_value=SimpleNamespace(data_dir=Path(tmp))), \
                 patch.object(acquisition.repo, "session_owns_paper", AsyncMock(return_value=True)), \
                 patch.object(acquisition.paper_catalog, "literature_for_paper", AsyncMock(return_value=None)), \
                 patch.object(acquisition, "_parsed_already", AsyncMock(return_value=False)), \
                 patch.object(parse_service, "run_agent_parse", AsyncMock(side_effect=agent_jobs.JobFailed(ErrorCode.TIMEOUT, "queued"))):
                raw = await acquisition.ensure_paper_parsed.coroutine(
                    runtime=SimpleNamespace(context=ctx), paper_id="paper",
                )
        result = json.loads(raw)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["data"]["availability"], "pdf_ready")
        self.assertIn("Do not retry", result["note"])


if __name__ == "__main__":
    unittest.main()
