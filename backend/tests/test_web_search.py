"""P7 step 1: the web search service layer.

No network. Providers answer through `httpx.MockTransport`, because what
matters is the contract: what each provider is sent, how a refused or spent
key is rotated past, when the next provider is asked, and what the caller is
told about coverage — a provider that could not search is not one that found
nothing, and a page with no date is not a page inside the date range.
"""

from __future__ import annotations

import json
import os
import unittest
from datetime import date
from typing import Any, Callable
from unittest import mock

import httpx

from app.services.web_search import chunking, http as web_http
from app.services.web_search.credentials import EXA, FIRECRAWL, TAVILY, parse_key_list
from app.services.web_search.fetch import fetch_page
from app.services.web_search.keypool import KeyPool
from app.services.web_search.models import WebSearchRequest
from app.services.web_search.search import search_web
from app.services.web_search.urls import (
    domain_matches,
    fetchable_reason,
    normalize_url,
    scholarly_identifiers,
)

Handler = Callable[[httpx.Request, dict[str, Any]], httpx.Response]


class Router:
    """Answers by URL substring and records every request it saw."""

    def __init__(self, routes: dict[str, Handler | list[Handler]]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        self.calls.append((str(request.url), dict(request.headers), body))
        for key, handler in self.routes.items():
            if key in str(request.url):
                if isinstance(handler, list):
                    handler = handler.pop(0) if len(handler) > 1 else handler[0]
                return handler(request, body)
        raise AssertionError(f"unexpected request {request.url}")

    def to(self, fragment: str) -> list[tuple[str, dict[str, str], dict[str, Any]]]:
        return [c for c in self.calls if fragment in c[0]]


def reply(status: int = 200, data: Any = None, headers: dict[str, str] | None = None) -> Handler:
    return lambda _req, _body: httpx.Response(status, json=data if data is not None else {}, headers=headers)


def tavily_results(*urls: str, date_field: str | None = None) -> dict[str, Any]:
    return {
        "results": [
            {
                "url": u,
                "title": f"Title {i}",
                "content": f"snippet about {u}",
                "score": 0.9 - i * 0.1,
                **({"published_date": date_field} if date_field else {}),
            }
            for i, u in enumerate(urls)
        ]
    }


def exa_results(*items: tuple[str, str | None]) -> dict[str, Any]:
    return {
        "results": [
            {"url": u, "title": f"Exa {u}", "publishedDate": d, "highlights": [f"highlight {u}"]}
            for u, d in items
        ]
    }


def pools(tavily: tuple[str, ...] = ("tv-1",), exa: tuple[str, ...] = ("ex-1",), firecrawl: tuple[str, ...] = ("fc-1",)):
    return {
        TAVILY: KeyPool(TAVILY, tavily),
        EXA: KeyPool(EXA, exa),
        FIRECRAWL: KeyPool(FIRECRAWL, firecrawl),
    }


LONG_PAGE = "# Guide\n\n" + ("Installation requires Python 3.11 and CUDA 12. " * 20)


class _Base(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        patches = [
            mock.patch.object(web_http, "SERVER_RETRY_DELAY", 0.0),
            mock.patch("app.services.web_search.search.web_search_enabled", return_value=True),
            mock.patch("app.services.web_search.fetch.web_search_enabled", return_value=True),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def client(self, router: Router) -> httpx.AsyncClient:
        client = httpx.AsyncClient(transport=httpx.MockTransport(router))
        self.addAsyncCleanup(client.aclose)
        return client


class CredentialTests(unittest.TestCase):
    def test_key_list_split_dedupes_and_keeps_order(self) -> None:
        self.assertEqual(
            parse_key_list(' "k1", k2;k3\nk1 , \'k4\' '),
            ("k1", "k2", "k3", "k4"),
        )
        self.assertEqual(parse_key_list(""), ())

    def test_singular_spellings_are_read(self) -> None:
        from app.config import AppSettings

        env = {"TAVILY_KEY": "t1,t2", "EXA_API_KEY": "e1", "FIRECRAWL_API_KEY": "f1"}
        with mock.patch.dict(os.environ, env, clear=True):
            s = AppSettings(_env_file=None)
        self.assertEqual(s.tavily_api_keys, "t1,t2")
        self.assertEqual(s.exa_api_keys, "e1")
        self.assertEqual(s.firecrawl_api_keys, "f1")

    def test_plural_name_wins_over_singular(self) -> None:
        from app.config import AppSettings

        with mock.patch.dict(os.environ, {"TAVILY_KEY": "old", "TAVILY_API_KEYS": "new1,new2"}, clear=True):
            s = AppSettings(_env_file=None)
        self.assertEqual(s.tavily_api_keys, "new1,new2")


class KeyPoolTests(unittest.TestCase):
    def test_round_robin_skips_unusable_keys(self) -> None:
        pool = KeyPool(TAVILY, ("a", "b", "c"))
        self.assertEqual([pool.acquire() for _ in range(4)], ["a", "b", "c", "a"])
        pool.mark_refused("b")
        pool.mark_out_of_credit("c")
        self.assertEqual({pool.acquire() for _ in range(3)}, {"a"})

    def test_unavailable_reason_prefers_the_recoverable_cause(self) -> None:
        self.assertEqual(KeyPool(TAVILY, ()).unavailable_status(), "skipped_no_key")
        pool = KeyPool(TAVILY, ("a", "b", "c"))
        pool.mark_refused("a")
        pool.mark_refused("b")
        pool.mark_refused("c")
        self.assertIsNone(pool.acquire())
        self.assertEqual(pool.unavailable_status(), "auth_failed")

        pool = KeyPool(TAVILY, ("a", "b", "c"))
        pool.mark_refused("a")
        pool.mark_out_of_credit("b")
        pool.mark_rate_limited("c", 30)
        self.assertEqual(pool.unavailable_status(), "rate_limited")

        pool = KeyPool(TAVILY, ("a", "b"))
        pool.mark_refused("a")
        pool.mark_out_of_credit("b")
        self.assertEqual(pool.unavailable_status(), "quota_exhausted")

    def test_describe_never_exposes_a_key(self) -> None:
        pool = KeyPool(TAVILY, ("tvly-secret-key-123456",))
        pool.mark_refused("tvly-secret-key-123456")
        text = json.dumps(pool.describe())
        self.assertNotIn("secret-key", text)


class UrlTests(unittest.TestCase):
    def test_normalize_url_folds_variants_of_one_page(self) -> None:
        self.assertEqual(
            normalize_url("https://www.Example.com/docs/?utm_source=x&b=2&a=1#intro"),
            normalize_url("http://example.com/docs?a=1&b=2"),
        )
        self.assertNotEqual(normalize_url("https://example.com/a"), normalize_url("https://example.com/b"))

    def test_domain_matching_includes_subdomains_only(self) -> None:
        self.assertTrue(domain_matches("https://docs.github.com/x", ["github.com"]))
        self.assertTrue(domain_matches("https://www.github.com/x", ["www.github.com"]))
        self.assertFalse(domain_matches("https://notgithub.com/x", ["github.com"]))

    def test_local_and_non_http_urls_are_refused(self) -> None:
        for url in (
            "file:///etc/passwd",
            "http://localhost:8000/api",
            "http://127.0.0.1/",
            "http://10.0.0.5/admin",
            "http://169.254.169.254/latest/meta-data",
            "http://[::1]/",
            "http://intranet/",
            "http://printer.local/",
            "https://user:pw@example.com/",
        ):
            self.assertTrue(fetchable_reason(url), url)
        self.assertEqual(fetchable_reason("https://github.com/org/repo"), "")
        self.assertEqual(fetchable_reason("http://8.8.8.8/"), "")

    def test_paper_urls_are_recognised(self) -> None:
        self.assertEqual(scholarly_identifiers("https://arxiv.org/abs/2401.12345v2"), {"arxiv_id": "2401.12345"})
        self.assertEqual(scholarly_identifiers("https://arxiv.org/pdf/2401.12345.pdf"), {"arxiv_id": "2401.12345"})
        self.assertEqual(scholarly_identifiers("https://arxiv.org/abs/hep-th/9901001"), {"arxiv_id": "hep-th/9901001"})
        self.assertEqual(
            scholarly_identifiers("https://doi.org/10.1038/Nature12373"), {"doi": "10.1038/nature12373"}
        )
        self.assertEqual(
            scholarly_identifiers("https://openreview.net/forum?id=AbC123"), {"openreview_id": "AbC123"}
        )
        self.assertEqual(scholarly_identifiers("https://aclanthology.org/2023.acl-long.1/"), {"acl_id": "2023.acl-long.1"})
        self.assertEqual(scholarly_identifiers("https://github.com/org/repo"), {})
        self.assertEqual(scholarly_identifiers("https://arxiv.org/list/cs.CL/recent"), {})


class SearchTests(_Base):
    async def test_tavily_request_carries_filters_and_bearer_key(self) -> None:
        router = Router({"api.tavily.com/search": reply(data=tavily_results("https://a.org/x"))})
        req = WebSearchRequest(
            query="flash attention benchmark",
            max_results=5,
            topic="news",
            start_date=date(2025, 1, 1),
            include_domains=("a.org",),
        )
        outcome = await search_web(req, client=self.client(router), pools=pools())

        url, headers, body = router.calls[0]
        self.assertEqual(headers["authorization"], "Bearer tv-1")
        self.assertNotIn("api_key", body)
        self.assertEqual(body["topic"], "news")
        self.assertEqual(body["start_date"], "2025-01-01")
        self.assertEqual(body["include_domains"], ["a.org"])
        self.assertEqual(body["search_depth"], "basic")
        self.assertEqual([p.status for p in outcome.providers], ["ok"])
        self.assertEqual(outcome.remote_filters, ["date", "domains"])
        self.assertEqual(len(outcome.results), 1)

    async def test_spent_key_is_rotated_past_within_one_search(self) -> None:
        router = Router(
            {
                "api.tavily.com/search": [
                    reply(432, {"detail": {"error": "plan limit"}}),
                    reply(401, {"detail": {"error": "bad key"}}),
                    reply(data=tavily_results("https://a.org/x")),
                ]
            }
        )
        p = pools(tavily=("tv-1", "tv-2", "tv-3"))
        outcome = await search_web(WebSearchRequest(query="q"), client=self.client(router), pools=p)

        used = [c[1]["authorization"] for c in router.to("tavily")]
        self.assertEqual(used, ["Bearer tv-1", "Bearer tv-2", "Bearer tv-3"])
        self.assertEqual(outcome.providers[0].status, "ok")
        # The refused key is gone for good; the spent one is cooling.
        states = {d["last_error"] for d in p[TAVILY].describe()}
        self.assertEqual(states, {"HTTP 432", "HTTP 401", ""})

    async def test_next_provider_is_asked_only_when_the_first_did_not_search(self) -> None:
        router = Router(
            {
                "api.tavily.com/search": reply(401),
                "api.exa.ai/search": reply(data=exa_results(("https://b.org/y", "2025-03-01T00:00:00.000Z"))),
            }
        )
        outcome = await search_web(WebSearchRequest(query="q"), client=self.client(router), pools=pools())

        self.assertEqual([(p.platform, p.status) for p in outcome.providers], [(TAVILY, "auth_failed"), (EXA, "ok")])
        self.assertEqual(outcome.results[0].url, "https://b.org/y")
        self.assertEqual(outcome.results[0].published_date, "2025-03-01")
        self.assertFalse(outcome.complete)
        self.assertEqual(router.to("firecrawl"), [])

    async def test_an_empty_answer_is_an_answer(self) -> None:
        router = Router({"api.tavily.com/search": reply(data={"results": []})})
        outcome = await search_web(WebSearchRequest(query="q"), client=self.client(router), pools=pools())

        self.assertEqual([p.status for p in outcome.providers], ["empty"])
        self.assertTrue(outcome.searched)
        self.assertEqual(router.to("exa"), [])

    async def test_no_keys_anywhere_is_reported_not_hidden(self) -> None:
        router = Router({})
        outcome = await search_web(
            WebSearchRequest(query="q"), client=self.client(router), pools=pools((), (), ())
        )
        self.assertEqual([p.status for p in outcome.providers], ["skipped_no_key"] * 3)
        self.assertFalse(outcome.searched)
        self.assertEqual(router.calls, [])

    async def test_server_error_is_retried_once(self) -> None:
        router = Router(
            {"api.tavily.com/search": [reply(503), reply(data=tavily_results("https://a.org/x"))]}
        )
        outcome = await search_web(WebSearchRequest(query="q"), client=self.client(router), pools=pools())
        self.assertEqual(len(router.to("tavily")), 2)
        self.assertEqual(outcome.providers[0].status, "ok")

    async def test_bad_request_is_not_blamed_on_the_key(self) -> None:
        router = Router(
            {
                "api.tavily.com/search": reply(400, {"detail": {"error": "Invalid start_date tv-1"}}),
                "api.exa.ai/search": reply(data=exa_results()),
            }
        )
        p = pools(tavily=("tv-1", "tv-2"))
        outcome = await search_web(WebSearchRequest(query="q"), client=self.client(router), pools=p)

        self.assertEqual(len(router.to("tavily")), 1)
        self.assertEqual(outcome.providers[0].status, "failed")
        self.assertNotIn("tv-1", outcome.providers[0].detail)
        self.assertTrue(all(d["usable"] for d in p[TAVILY].describe()))

    async def test_technical_topic_starts_with_exa(self) -> None:
        router = Router({"api.exa.ai/search": reply(data=exa_results(("https://proj.io", None)))})
        outcome = await search_web(
            WebSearchRequest(query="official project page of X", topic="technical"),
            client=self.client(router),
            pools=pools(),
        )
        self.assertEqual(outcome.providers[0].platform, EXA)
        _, headers, body = router.calls[0]
        self.assertEqual(headers["x-api-key"], "ex-1")
        self.assertEqual(body["contents"]["highlights"]["query"], "official project page of X")

    async def test_deep_search_merges_two_providers_by_url(self) -> None:
        router = Router(
            {
                "api.tavily.com/search": reply(
                    data=tavily_results("https://www.shared.org/page/", "https://only-tavily.org")
                ),
                "api.exa.ai/search": reply(
                    data=exa_results(("https://shared.org/page?utm_source=exa", "2024-05-01"), ("https://only-exa.org", None))
                ),
            }
        )
        outcome = await search_web(
            WebSearchRequest(query="q", depth="deep"), client=self.client(router), pools=pools()
        )
        self.assertEqual([p.platform for p in outcome.providers], [TAVILY, EXA])
        self.assertEqual(len(outcome.results), 3)
        top = outcome.results[0]
        self.assertEqual(sorted(top.providers), [EXA, TAVILY])
        # The date comes from whichever provider had one.
        self.assertEqual(top.published_date, "2024-05-01")
        self.assertEqual(router.calls[0][2]["search_depth"], "advanced")

    async def test_date_filter_drops_known_out_of_range_and_sets_aside_unchecked(self) -> None:
        # Tavily is out of credit, so Firecrawl answers a news query, for which
        # it cannot apply a date range: its undated result is neither in nor out.
        router = Router(
            {
                "api.tavily.com/search": reply(432),
                "api.exa.ai/search": reply(402),
                "api.firecrawl.dev/v2/search": reply(
                    data={
                        "success": True,
                        "data": {
                            "news": [
                                {"url": "https://n.org/new", "title": "new", "snippet": "s", "date": "Mar 3, 2025"},
                                {"url": "https://n.org/old", "title": "old", "snippet": "s", "date": "Jan 2, 2020"},
                                {"url": "https://n.org/undated", "title": "u", "snippet": "s", "date": "3 days ago"},
                            ]
                        },
                    }
                ),
            }
        )
        outcome = await search_web(
            WebSearchRequest(query="q", topic="news", start_date=date(2025, 1, 1)),
            client=self.client(router),
            pools=pools(),
        )
        self.assertEqual([r.url for r in outcome.results], ["https://n.org/new"])
        self.assertEqual([r.url for r in outcome.unknown_date], ["https://n.org/undated"])
        self.assertEqual(outcome.excluded, {"date": 1})
        self.assertNotIn("tbs", router.to("firecrawl")[0][2])

    async def test_undated_result_passes_when_the_provider_applied_the_range(self) -> None:
        router = Router({"api.tavily.com/search": reply(data=tavily_results("https://a.org/x"))})
        outcome = await search_web(
            WebSearchRequest(query="q", start_date=date(2025, 1, 1)),
            client=self.client(router),
            pools=pools(),
        )
        self.assertEqual(len(outcome.results), 1)
        self.assertIsNone(outcome.results[0].published_date)
        self.assertEqual(outcome.unknown_date, [])

    async def test_domain_filters_are_enforced_locally_too(self) -> None:
        router = Router(
            {"api.tavily.com/search": reply(data=tavily_results("https://docs.a.org/x", "https://b.org/y"))}
        )
        outcome = await search_web(
            WebSearchRequest(query="q", include_domains=("a.org",)),
            client=self.client(router),
            pools=pools(),
        )
        self.assertEqual([r.url for r in outcome.results], ["https://docs.a.org/x"])
        self.assertEqual(outcome.excluded, {"include_domains": 1})

    async def test_firecrawl_date_range_and_site_operators(self) -> None:
        router = Router(
            {
                "api.tavily.com/search": reply(401),
                "api.exa.ai/search": reply(401),
                "api.firecrawl.dev/v2/search": reply(data={"data": {"web": []}}),
            }
        )
        await search_web(
            WebSearchRequest(
                query="q",
                start_date=date(2024, 12, 1),
                end_date=date(2024, 12, 31),
                include_domains=("a.org", "b.org"),
                exclude_domains=("c.org",),
            ),
            client=self.client(router),
            pools=pools(),
        )
        body = router.to("firecrawl")[0][2]
        self.assertEqual(body["tbs"], "cdr:1,cd_min:12/1/2024,cd_max:12/31/2024")
        self.assertEqual(body["query"], "q (site:a.org OR site:b.org) -site:c.org")

    async def test_disabled_switch_sends_nothing(self) -> None:
        router = Router({})
        with mock.patch("app.services.web_search.search.web_search_enabled", return_value=False):
            outcome = await search_web(WebSearchRequest(query="q"), client=self.client(router), pools=pools())
        self.assertEqual({p.status for p in outcome.providers}, {"unsupported"})
        self.assertEqual(router.calls, [])


class FetchTests(_Base):
    async def test_first_provider_with_real_content_wins(self) -> None:
        router = Router(
            {
                "api.exa.ai/contents": reply(
                    data={"results": [{"url": "https://a.org/guide", "text": LONG_PAGE}], "statuses": []}
                )
            }
        )
        outcome = await fetch_page("https://a.org/guide", client=self.client(router), pools=pools())
        self.assertTrue(outcome.fetched)
        self.assertEqual(outcome.page.provider, EXA)
        self.assertEqual([a.status for a in outcome.attempts], ["ok"])
        self.assertEqual(router.to("exa")[0][2]["urls"], ["https://a.org/guide"])

    async def test_falls_through_failures_and_script_shells(self) -> None:
        router = Router(
            {
                "api.tavily.com/extract": reply(
                    data={"results": [], "failed_results": [{"url": "u", "error": "blocked"}]}
                ),
                "api.exa.ai/contents": reply(
                    data={"results": [{"url": "https://spa.io", "text": "Loading…"}], "statuses": [{"id": "https://spa.io", "status": "success"}]}
                ),
                "api.firecrawl.dev/v2/scrape": reply(
                    data={"success": True, "data": {"markdown": LONG_PAGE, "metadata": {"title": "SPA", "statusCode": 200}}}
                ),
            }
        )
        outcome = await fetch_page("https://spa.io", client=self.client(router), pools=pools())
        self.assertEqual(
            [(a.platform, a.status) for a in outcome.attempts],
            [(EXA, "empty"), (TAVILY, "empty"), (FIRECRAWL, "ok")],
        )
        self.assertEqual(outcome.attempts[1].detail, "blocked")
        self.assertEqual(outcome.page.provider, FIRECRAWL)
        self.assertEqual(outcome.page.title, "SPA")
        scrape = router.to("firecrawl")[0][2]
        self.assertTrue(scrape["onlyMainContent"])
        self.assertEqual(scrape["parsers"][0]["maxPages"], 20)

    async def test_exa_per_url_error_is_reported(self) -> None:
        router = Router(
            {
                "api.tavily.com/extract": reply(401),
                "api.exa.ai/contents": reply(
                    data={"results": [], "statuses": [{"id": "u", "status": "error", "error": {"tag": "CRAWL_NOT_FOUND", "httpStatusCode": 404}}]}
                ),
                "api.firecrawl.dev/v2/scrape": reply(
                    data={"success": True, "data": {"markdown": "", "metadata": {"statusCode": 404}}}
                ),
            }
        )
        outcome = await fetch_page("https://gone.org/x", client=self.client(router), pools=pools())
        self.assertFalse(outcome.fetched)
        self.assertEqual(
            [(a.status, a.detail) for a in outcome.attempts],
            [("empty", "CRAWL_NOT_FOUND"), ("auth_failed", "no usable key (auth_failed)"), ("empty", "the site answered HTTP 404")],
        )

    async def test_short_page_is_kept_when_nothing_longer_exists(self) -> None:
        short = "A short but real page with a single sentence of content."
        router = Router(
            {
                "api.exa.ai/contents": reply(402),
                "api.tavily.com/extract": reply(data={"results": [{"url": "u", "raw_content": short}]}),
                "api.firecrawl.dev/v2/scrape": reply(402),
            }
        )
        outcome = await fetch_page("https://tiny.org", client=self.client(router), pools=pools())
        self.assertEqual(outcome.page.markdown, short)

    async def test_refused_url_never_reaches_a_provider(self) -> None:
        router = Router({})
        with self.assertRaises(ValueError):
            await fetch_page("http://169.254.169.254/latest", client=self.client(router), pools=pools())
        self.assertEqual(router.calls, [])


class ChunkingTests(unittest.IsolatedAsyncioTestCase):
    PAGE = (
        "# FlashAttention\n\nIntro paragraph about attention kernels.\n\n"
        "## Installation\n\nInstall with pip. Requires CUDA 12 and PyTorch 2.2.\n\n"
        "## Benchmarks\n\nOn A100 the forward pass is 2x faster than the baseline.\n\n"
        "### 中文说明\n\n该方法在长序列上显存占用更低。\n"
    )

    def test_chunks_follow_headings_and_keep_offsets(self) -> None:
        chunks = chunking.split_markdown(self.PAGE)
        self.assertEqual(
            [c.heading_path for c in chunks],
            [
                "FlashAttention",
                "FlashAttention › Installation",
                "FlashAttention › Benchmarks",
                "FlashAttention › Benchmarks › 中文说明",
            ],
        )
        for c in chunks:
            self.assertEqual(self.PAGE[c.start : c.end].strip(), c.text)

    def test_long_paragraph_is_split_under_the_ceiling(self) -> None:
        text = "Sentence number one is here. " * 200
        chunks = chunking.split_markdown(text, target_chars=500, max_chars=800)
        self.assertGreater(len(chunks), 5)
        self.assertTrue(all(len(c.text) <= 800 for c in chunks))
        self.assertEqual("".join(text[c.start : c.end] for c in chunks), text.rstrip("\n"))

    async def test_lexical_ranking_finds_the_matching_section(self) -> None:
        chunks = chunking.split_markdown(self.PAGE)
        ranked, ranker = await chunking.rank_chunks(chunks, "what CUDA version is required", limit=1, use_rerank=False)
        self.assertEqual(ranker, "lexical")
        self.assertIn("CUDA 12", ranked[0][0].text)

        ranked, _ = await chunking.rank_chunks(chunks, "长序列显存", limit=1, use_rerank=False)
        self.assertIn("显存", ranked[0][0].text)

    async def test_rerank_order_is_used_and_failure_falls_back(self) -> None:
        chunks = chunking.split_markdown(self.PAGE)

        async def fake_rerank(candidates, question, limit):
            return [(candidates[-1], 0.99)]

        with mock.patch.object(chunking, "_rerank", fake_rerank):
            ranked, ranker = await chunking.rank_chunks(chunks, "CUDA", limit=1)
        self.assertEqual(ranker, "rerank")

        with mock.patch.object(chunking, "_rerank", mock.AsyncMock(return_value=None)):
            ranked, ranker = await chunking.rank_chunks(chunks, "CUDA", limit=1)
        self.assertEqual(ranker, "lexical")
        self.assertIn("CUDA", ranked[0][0].text)

    def test_navigation_is_recognised_and_skipped(self) -> None:
        page = (
            "# Welcome to vLLM[¶](https://docs.vllm.ai/#welcome \"Permanent link\")\n\n"
            "*   - [x] [Home](https://docs.vllm.ai/) *   - [x] [User Guide](https://docs.vllm.ai/usage/) "
            "*   [Models](https://docs.vllm.ai/models/) *   [API](https://docs.vllm.ai/api/)\n\n"
            "## GPUs\n\nvLLM runs on NVIDIA GPUs with compute capability 7.0 or higher, see "
            "[the table](https://docs.vllm.ai/gpus/).\n"
        )
        chunks = chunking.split_markdown(page)
        self.assertEqual(chunks[0].heading_path, "Welcome to vLLM")
        self.assertEqual([c.navigation for c in chunks], [True, False])
        picked, _ = chunking.chunks_from_offset(chunks, 0, 5000)
        self.assertEqual([c.heading_path for c in picked], ["Welcome to vLLM › GPUs"])

    async def test_navigation_never_outranks_content(self) -> None:
        page = (
            "[GPUs](https://a/gpus) [GPUs setup](https://a/s) [GPU list](https://a/l) [More GPUs](https://a/m)\n\n"
            "Supported hardware is listed below; any recent accelerator works.\n"
        )
        chunks = chunking.split_markdown(page, target_chars=10)
        ranked, _ = await chunking.rank_chunks(chunks, "GPUs", limit=5, use_rerank=False)
        self.assertEqual([c.navigation for c, _ in ranked], [False])

    def test_offset_paging_covers_the_page_once(self) -> None:
        chunks = chunking.split_markdown(self.PAGE)
        seen: list[int] = []
        offset: int | None = 0
        while offset is not None:
            picked, offset = chunking.chunks_from_offset(chunks, offset, max_chars=60)
            self.assertTrue(picked)
            seen.extend(c.index for c in picked)
        self.assertEqual(seen, list(range(len(chunks))))


class PublicationRankTavilyTests(_Base):
    async def test_rank_fallback_uses_the_pool(self) -> None:
        from app.services.publication_rank.tavily_search import TavilySearchClient

        router = Router(
            {"api.tavily.com/search": [reply(432), reply(data={"answer": "CCF A", "results": []})]}
        )
        pool = KeyPool(TAVILY, ("tv-1", "tv-2"))
        client = TavilySearchClient(pool=pool)
        with mock.patch("app.services.publication_rank.tavily_search.web_search_enabled", return_value=True), \
            mock.patch("app.services.publication_rank.tavily_search.new_client", lambda _t: self.client(router)):
            data = await client.search("CVPR")
        self.assertEqual(data["answer"], "CCF A")
        self.assertEqual([c[1]["authorization"] for c in router.calls], ["Bearer tv-1", "Bearer tv-2"])

    def test_no_keys_means_not_configured(self) -> None:
        from app.services.publication_rank.tavily_search import TavilySearchClient

        self.assertFalse(TavilySearchClient(pool=KeyPool(TAVILY, ())).configured)
        self.assertFalse(TavilySearchClient(api_key="").configured)


if __name__ == "__main__":
    unittest.main()
