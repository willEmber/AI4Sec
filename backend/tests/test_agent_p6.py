"""P6: the search layer the agent's discovery tools stand on.

No network. The adapters get a fake HTTP client that records what they send,
because what matters here is the contract with each provider — which filters
are pushed down and in what syntax, how a refusal is reported — and what the
agent is told: a platform that could not search is not a platform that found
nothing, a citation count belongs to the index that counted it, and a passage
from a paper we never parsed is abstract-level evidence.
"""

from __future__ import annotations

import json
import os
import time
import unittest
from typing import Any
from unittest import mock

from tests.test_agent_p2 import AgentP2TestCase


class FakeClient:
    """Stands in for paper_search's HTTPClient; answers by URL substring."""

    def __init__(self, routes: dict[str, Any]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def _answer(self, method: str, url: str, payload: dict[str, Any]) -> Any:
        self.calls.append((method, url, payload))
        for key, value in self.routes.items():
            if key in url:
                if isinstance(value, BaseException):
                    raise value
                return value(payload) if callable(value) else value
        raise AssertionError(f"unexpected {method} {url}")

    async def get_json(self, url, *, params=None, headers=None):
        return self._answer("GET", url, dict(params or {}))

    async def get_json_with_headers(self, url, *, params=None, headers=None):
        return self._answer("GET", url, dict(params or {})), {"X-RateLimit-Remaining-USD": "0.9"}

    async def get_text(self, url, *, params=None, headers=None):
        return self._answer("GET", url, dict(params or {}))

    async def post_json(self, url, *, json_body, headers=None):
        return self._answer("POST", url, dict(json_body))


def _http_error(status: int, body: str = "", headers: dict[str, str] | None = None):
    from app.services.paper_search.http_client import HTTPStatusError

    return HTTPStatusError("GET", "https://example.org/x?api_key=SECRET", status, body, headers=headers)


class _Reset(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        from app.services.paper_search import ratelimit

        ratelimit.reset_all()
        # Pacing is tested on its own; elsewhere it would only slow tests down.
        self._pace = mock.patch.object(ratelimit.IntervalLimiter, "wait", new=mock.AsyncMock())
        self._pace.start()

    def tearDown(self) -> None:
        self._pace.stop()
        from app.services.paper_search import ratelimit

        ratelimit.reset_all()


# ── credentials and pacing ──────────────────────────────────────────────────


class CredentialTests(unittest.TestCase):
    def test_s2_key_is_read_under_either_spelling(self) -> None:
        from app.services.paper_search import credentials

        with mock.patch.dict(os.environ, {"S2_API_Key": "mixed"}, clear=False):
            os.environ.pop("S2_API_KEY", None)
            self.assertEqual(credentials.s2_headers(), {"x-api-key": "mixed"})
        with mock.patch.dict(os.environ, {"S2_API_KEY": "canonical", "S2_API_Key": "mixed"}):
            self.assertEqual(credentials.s2_api_key(), "canonical")

    def test_the_retired_s2_variable_is_not_read(self) -> None:
        """A pruned key answers 403; sending it is worse than sending none."""
        from app.services.paper_search import credentials

        env = {"PAPERSEARCH_SEMANTICSCHOLAR_API_KEY": "old"}
        with mock.patch.dict(os.environ, env, clear=False):
            for name in ("S2_API_KEY", "S2_API_Key"):
                os.environ.pop(name, None)
            self.assertEqual(credentials.s2_headers(), {})

    def test_openalex_prefers_the_key_over_mailto(self) -> None:
        from app.services.paper_search import credentials

        with mock.patch.dict(os.environ, {"OPENALEX_KEY": "k"}):
            self.assertEqual(credentials.openalex_auth_params("a@b.c"), {"api_key": "k"})
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENALEX_KEY", None)
            os.environ.pop("OPENALEX_API_KEY", None)
            self.assertEqual(credentials.openalex_auth_params("a@b.c"), {"mailto": "a@b.c"})

    def test_keys_never_reach_error_text(self) -> None:
        err = _http_error(500)
        self.assertNotIn("SECRET", str(err))
        self.assertIn("api_key=***", str(err))


class PacingTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_callers_are_spaced_in_arrival_order(self) -> None:
        """Asserted on the reserved slots: wall-clock gaps jitter under load."""
        import asyncio

        from app.services.paper_search import ratelimit

        limiter = ratelimit.IntervalLimiter(0.5)
        slept: list[float] = []

        async def fake_sleep(delay: float) -> None:
            slept.append(delay)

        with mock.patch.object(ratelimit.time, "monotonic", return_value=100.0), \
             mock.patch.object(ratelimit.asyncio, "sleep", new=fake_sleep):
            await asyncio.gather(*(limiter.wait() for _ in range(4)))
        self.assertEqual(slept, [0.5, 1.0, 1.5], "the first caller goes at once")

    def test_a_spent_budget_disables_the_provider_until_reset(self) -> None:
        from app.services.paper_search.ratelimit import DailyBudget, is_quota_response

        budget = DailyBudget("OpenAlex")
        headers = {"X-RateLimit-Remaining-USD": "0", "X-RateLimit-Reset": "120"}
        self.assertTrue(is_quota_response(429, headers))
        self.assertFalse(is_quota_response(429, {"X-RateLimit-Remaining-USD": "0.5"}))
        budget.mark_exhausted(headers)
        self.assertTrue(budget.exhausted())
        budget.observe({"X-RateLimit-Remaining-USD": "0.8"})
        self.assertFalse(budget.exhausted())


class RerankEndpointTests(unittest.TestCase):
    def test_the_maas_gateway_reranks_on_its_compatible_api(self) -> None:
        from app.services.paper_search.llm import rerank_endpoint

        base = "https://llm-x.cn-beijing.maas.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1"
        self.assertEqual(
            rerank_endpoint(base),
            "https://llm-x.cn-beijing.maas.aliyuncs.com/compatible-api/v1/reranks",
        )
        self.assertEqual(rerank_endpoint("https://api.example.com/v1"), "https://api.example.com/v1/rerank")
        self.assertEqual(rerank_endpoint(base, "https://override/r"), "https://override/r")


# ── adapters ────────────────────────────────────────────────────────────────


def _s2_record(**kw: Any) -> dict[str, Any]:
    record = {
        "paperId": "s2abc",
        "title": "Drop-Upcycling",
        "abstract": "We reinitialise experts.",
        "year": 2025,
        "venue": "International Conference on Learning Representations",
        "publicationVenue": {"name": "International Conference on Learning Representations", "type": "conference", "alternate_names": ["ICLR"]},
        "externalIds": {"ArXiv": "2502.19261", "DOI": "10.48550/arXiv.2502.19261", "DBLP": "conf/iclr/X25"},
        "citationCount": 21,
        "influentialCitationCount": 3,
        "publicationDate": "2025-02-26",
        "openAccessPdf": {"url": ""},
        "isOpenAccess": False,
        "fieldsOfStudy": ["Computer Science"],
        "publicationTypes": ["Conference"],
        "tldr": {"text": "Partial re-initialisation helps."},
        "authors": [{"name": "T. Nakamura"}],
        "url": "https://www.semanticscholar.org/paper/s2abc",
    }
    record.update(kw)
    return record


class SemanticScholarAdapterTests(_Reset):
    async def test_filters_are_pushed_down_in_s2_syntax(self) -> None:
        from app.services.paper_search import SearchFilters, Settings
        from app.services.paper_search.platforms.semanticscholar import search_semanticscholar

        client = FakeClient({"/paper/search": {"data": [_s2_record()]}})
        filters = SearchFilters(
            year_from=2023, min_citations=10, venues=("ICLR",), fields_of_study=("Computer Science",),
            open_access_only=True, publication_types=("conference",),
        )
        papers = await search_semanticscholar(client, query="moe", limit=5, settings=Settings(), filters=filters)
        params = client.calls[0][2]
        self.assertEqual(params["year"], "2023-")
        self.assertEqual(params["minCitationCount"], "10")
        self.assertEqual(params["venue"], "ICLR")
        self.assertEqual(params["fieldsOfStudy"], "Computer Science")
        self.assertIn("openAccessPdf", params)
        self.assertEqual(params["publicationTypes"], "Conference")

        paper = papers[0]
        self.assertEqual(paper.arxiv_id, "2502.19261")
        self.assertEqual(paper.citation_counts, {"SemanticScholar": 21})
        self.assertEqual(paper.venue_type, "conference")
        self.assertIn("ICLR", paper.venue_aliases)
        self.assertEqual(paper.tldr, "Partial re-initialisation helps.")

    async def test_citation_order_uses_bulk_then_fetches_the_kept_rows(self) -> None:
        from app.services.paper_search import SearchFilters, Settings
        from app.services.paper_search.platforms.semanticscholar import search_semanticscholar

        client = FakeClient(
            {
                "/paper/search/bulk": {"data": [{"paperId": "a"}, {"paperId": "b"}, {"paperId": "c"}]},
                "/paper/batch": lambda body: [_s2_record(paperId=i, title=f"T{i}") for i in body["ids"]],
            }
        )
        papers = await search_semanticscholar(
            client, query="moe", limit=2, settings=Settings(), filters=SearchFilters(sort="citations")
        )
        self.assertEqual(client.calls[0][2]["sort"], "citationCount:desc")
        self.assertEqual(client.calls[1][2]["ids"], ["a", "b"])
        self.assertEqual([p.s2_paper_id for p in papers], ["a", "b"])

    async def test_429_is_retried_then_succeeds(self) -> None:
        from app.services.paper_search import Settings
        from app.services.paper_search.platforms import semanticscholar

        attempts = {"n": 0}

        def flaky(_params):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise _http_error(429)
            return {"data": [_s2_record()]}

        client = FakeClient({"/paper/search": flaky})
        with mock.patch.object(semanticscholar.asyncio, "sleep", new=mock.AsyncMock()):
            papers = await semanticscholar.search_semanticscholar(client, query="q", limit=1, settings=Settings())
        self.assertEqual(attempts["n"], 2)
        self.assertEqual(len(papers), 1)


class OpenAlexAdapterTests(_Reset):
    def _work(self, **kw: Any) -> dict[str, Any]:
        work = {
            "id": "https://openalex.org/W1",
            "doi": "https://doi.org/10.48550/arxiv.2502.19261",
            "title": "Drop-Upcycling",
            "publication_year": 2025,
            "publication_date": "2025-02-26",
            "type": "preprint",
            "cited_by_count": 4,
            "is_retracted": False,
            "open_access": {"is_oa": True, "oa_url": "https://arxiv.org/pdf/2502.19261"},
            "best_oa_location": {"pdf_url": "https://arxiv.org/pdf/2502.19261"},
            "primary_location": {"landing_page_url": "https://arxiv.org/abs/2502.19261", "source": {"display_name": "arXiv", "type": "repository"}},
            "locations": [],
            "authorships": [{"author": {"display_name": "T. Nakamura"}}],
        }
        work.update(kw)
        return work

    async def test_every_order_matches_title_and_abstract_with_the_key(self) -> None:
        from app.services.paper_search import SearchFilters, Settings
        from app.services.paper_search.platforms.openalex import search_openalex

        client = FakeClient({"api.openalex.org/works": {"results": [self._work()]}})
        with mock.patch.dict(os.environ, {"OPENALEX_KEY": "k"}):
            papers = await search_openalex(
                client, query="drop, upcycling: moe", limit=5, settings=Settings(),
                filters=SearchFilters(year_from=2024, year_to=2025, min_citations=5, sort="citations"),
            )
        params = client.calls[0][2]
        self.assertEqual(params["api_key"], "k")
        self.assertNotIn("search", params)
        self.assertTrue(params["filter"].startswith("title_and_abstract.search:drop upcycling moe"))
        self.assertIn("publication_year:2024-2025", params["filter"])
        self.assertIn("cited_by_count:>4", params["filter"])
        self.assertEqual(params["sort"], "cited_by_count:desc")

        paper = papers[0]
        self.assertEqual(paper.arxiv_id, "2502.19261")
        self.assertEqual(paper.oa_pdf_url, "https://arxiv.org/pdf/2502.19261")
        self.assertEqual(paper.venue_type, "preprint")
        self.assertEqual(paper.citation_counts, {"OpenAlex": 4})

    async def test_a_spent_budget_raises_quota_exhausted_and_sticks(self) -> None:
        from app.services.paper_search import Settings
        from app.services.paper_search.platforms.openalex import search_openalex
        from app.services.paper_search.ratelimit import OPENALEX_BUDGET, QuotaExhaustedError

        spent = _http_error(429, headers={"X-RateLimit-Remaining-USD": "0", "X-RateLimit-Reset": "600"})
        client = FakeClient({"api.openalex.org/works": spent})
        with self.assertRaises(QuotaExhaustedError):
            await search_openalex(client, query="q", limit=5, settings=Settings())
        self.assertTrue(OPENALEX_BUDGET.exhausted())
        # The next call does not even reach the network.
        with self.assertRaises(QuotaExhaustedError):
            await search_openalex(client, query="q", limit=5, settings=Settings())
        self.assertEqual(len(client.calls), 1)


class ArxivAdapterTests(_Reset):
    FEED = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
    <entry><id>http://arxiv.org/abs/2502.19261v2</id><title>Drop-Upcycling</title><summary>abs</summary>
    <published>2025-02-26T00:00:00Z</published><author><name>T. Nakamura</name></author>
    <arxiv:primary_category term="cs.LG"/></entry></feed>"""

    def test_query_ands_terms_drops_stopwords_and_adds_dates(self) -> None:
        from app.services.paper_search import SearchFilters
        from app.services.paper_search.platforms.arxiv import build_search_query

        q = build_search_query("mixture of experts upcycling", SearchFilters(year_from=2024))
        self.assertEqual(
            q, "(all:mixture AND all:experts AND all:upcycling) AND submittedDate:[202401010000 TO 209912312359]"
        )

    async def test_entries_carry_the_arxiv_id_and_pdf_link(self) -> None:
        from app.services.paper_search import Settings
        from app.services.paper_search.platforms.arxiv import search_arxiv

        client = FakeClient({"export.arxiv.org": self.FEED})
        papers = await search_arxiv(client, query="upcycling", limit=5, settings=Settings())
        self.assertEqual(papers[0].arxiv_id, "2502.19261")
        self.assertEqual(papers[0].oa_pdf_url, "https://arxiv.org/pdf/2502.19261")
        self.assertTrue(papers[0].is_open_access)

    async def test_a_406_with_a_feed_is_still_a_result(self) -> None:
        from app.services.paper_search import Settings
        from app.services.paper_search.platforms.arxiv import search_arxiv

        client = FakeClient({"export.arxiv.org": _http_error(406, body=self.FEED)})
        papers = await search_arxiv(client, query="upcycling", limit=5, settings=Settings())
        self.assertEqual(len(papers), 1)
        self.assertEqual(len(client.calls), 1, "the plain-HTTP mirror is not retried")


class DblpAdapterTests(unittest.TestCase):
    def test_sparql_uses_the_text_index_and_venue_labels(self) -> None:
        from app.services.paper_search import SearchFilters
        from app.services.paper_search.platforms.dblp import build_sparql

        q = build_sparql(
            "sparse mixture of experts", limit=5, offset=0,
            filters=SearchFilters(year_from=2024, venues=("ICLR",), publication_types=("conference",)),
        )
        self.assertIn('ql:contains-word "mixture"', q)
        self.assertNotIn('"of"', q)
        self.assertIn('?venue = "ICLR" || STRSTARTS(?venue, "ICLR (")', q)
        self.assertIn('"2024"^^xsd:gYear', q)
        self.assertIn("dblp:Inproceedings", q)

    def test_an_unbounded_scan_defaults_to_recent_years(self) -> None:
        from app.services.paper_search.platforms.dblp import DEFAULT_YEARS, build_sparql

        q = build_sparql("graph neural networks", limit=5, offset=0, filters=None)
        floor = time.gmtime().tm_year - (DEFAULT_YEARS - 1)
        self.assertIn(f'"{floor}"^^xsd:gYear', q)

    def test_details_yield_arxiv_and_openreview_ids_and_ordered_authors(self) -> None:
        from app.services.paper_search.platforms.dblp import apply_details, binding_to_paper

        paper = binding_to_paper(
            {
                "pub": {"value": "https://dblp.org/rec/conf/iclr/X25"},
                "title": {"value": "Drop-Upcycling."},
                "year": {"value": "2025"},
                "venue": {"value": "ICLR"},
                "type": {"value": "https://dblp.org/rdf/schema#Inproceedings"},
            }
        )
        apply_details(
            paper,
            {
                "doi": {"value": ""},
                "ees": {"value": "https://openreview.net/forum?id=gx1wHnf5Vp https://arxiv.org/abs/2502.19261v2"},
                "authors": {"value": "2|Takuya Akiba;;1|Taishi Nakamura;;3|Jun Suzuki 0001"},
            },
        )
        self.assertEqual(paper.title, "Drop-Upcycling")
        self.assertEqual(paper.venue_type, "conference")
        self.assertEqual(paper.openreview_id, "gx1wHnf5Vp")
        self.assertEqual(paper.arxiv_id, "2502.19261")
        self.assertEqual(paper.authors, "Taishi Nakamura; Takuya Akiba; Jun Suzuki")


class OpenReviewTests(unittest.TestCase):
    def test_venue_lines_classify_into_outcomes(self) -> None:
        from app.services.paper_search.platforms.openreview import classify_acceptance

        self.assertEqual(classify_acceptance("ICLR 2025 Poster"), ("accepted", "poster"))
        self.assertEqual(classify_acceptance("ICLR 2024 Oral"), ("accepted", "oral"))
        self.assertEqual(classify_acceptance("Submitted to ICLR 2025")[0], "not_accepted")
        self.assertEqual(classify_acceptance("ACL ARR 2025 February Submission")[0], "not_accepted")
        self.assertEqual(classify_acceptance("NeurIPS 2025 LLM Evaluation Workshop Poster"), ("accepted", "poster"))
        self.assertEqual(classify_acceptance("ICLR 2025 Conference Withdrawn Submission")[0], "withdrawn")
        self.assertEqual(classify_acceptance("")[0], "unknown")

    def test_a_rejected_submission_has_no_venue(self) -> None:
        """So a venue filter cannot count a rejection as publication there."""
        from app.services.paper_search.platforms.openreview import note_to_paper

        note = {
            "id": "abc", "forum": "abc",
            "invitations": ["ICLR.cc/2025/Conference/-/Submission"],
            "content": {"title": {"value": "T"}, "venue": {"value": "Submitted to ICLR 2025"}, "pdf": {"value": "/pdf/x"}},
        }
        paper = note_to_paper(note)
        self.assertEqual(paper.venue, "")
        self.assertEqual(paper.acceptance, "Submitted to ICLR 2025")
        self.assertEqual(paper.year, 2025)
        note["content"]["venue"] = {"value": "ICLR 2025 Spotlight"}
        self.assertEqual(note_to_paper(note).venue, "ICLR 2025")

    def test_forum_replies_split_into_decision_meta_and_reviews(self) -> None:
        from app.services.paper_search.platforms.openreview import parse_forum_notes

        notes = [
            {"id": "r1", "invitations": ["ICLR.cc/2025/Conference/Submission1/-/Official_Review"],
             "content": {"rating": {"value": "8: accept"}, "confidence": {"value": 4}, "weaknesses": {"value": "Small scale."}}},
            {"id": "d1", "invitations": ["ICLR.cc/2025/Conference/Submission1/-/Decision"],
             "content": {"decision": {"value": "Accept (Poster)"}}},
            {"id": "m1", "invitations": ["ICLR.cc/2025/Conference/Submission1/-/Meta_Review"],
             "content": {"metareview": {"value": "Solid."}, "recommendation": {"value": "Accept"}}},
        ]
        forum = parse_forum_notes(notes)
        self.assertEqual(forum["decision"], "Accept (Poster)")
        self.assertEqual(forum["meta_review"], "Accept Solid.")
        self.assertEqual(forum["reviews"][0]["rating"], "8: accept")
        self.assertEqual(forum["reviews"][0]["weaknesses"], "Small scale.")

    def test_queries_drop_stopwords_and_titles_go_as_phrases(self) -> None:
        """OpenReview ANDs terms and does not index stopwords: one "of" finds nothing."""
        from app.services.paper_search.platforms.openreview import search_term, title_phrase

        self.assertEqual(search_term("Attention Is All You Need"), "Attention Need")
        self.assertEqual(search_term("mixture of experts"), "mixture experts")
        self.assertEqual(title_phrase('A "quoted" title'), '"A quoted title"')

    def test_venue_groups(self) -> None:
        from app.services.paper_search.platforms.openreview import known_venue, venue_group

        self.assertEqual(venue_group("NeurIPS", 2024), "NeurIPS.cc/2024/Conference")
        self.assertEqual(venue_group("TMLR", 2024), "TMLR")
        self.assertFalse(known_venue("CVPR"))


class OpenReviewAuthTests(_Reset):
    def setUp(self) -> None:
        super().setUp()
        from app.services.paper_search.platforms import openreview

        openreview.reset_auth()
        self.addCleanup(openreview.reset_auth)
        env = mock.patch.dict(os.environ, {"OPENREVIEW_USERNAME": "me@example.org", "OPENREVIEW_PASSWORD": "pw"})
        env.start()
        self.addCleanup(env.stop)

    async def test_a_refused_login_falls_back_to_anonymous_search(self) -> None:
        """Search works without an account, so a wrong password must only cost the reviews."""
        from app.services.paper_search.platforms import openreview

        refused = _http_error(400, body='{"name":"Error","message":"Invalid username or password (x)"}')
        client = FakeClient(
            {"/login": refused, "/notes/search": {"notes": [PeerReviewToolTests.SUBMISSION]}}
        )
        first = await openreview.search_submissions(client, "Drop-Upcycling", limit=5)
        second = await openreview.search_submissions(client, "Drop-Upcycling", limit=5)

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertEqual([c[1].rsplit("/", 1)[-1] for c in client.calls].count("login"), 1,
                         "a refused login is not retried on every call")
        self.assertIn("Invalid username or password", openreview.login_error())

    async def test_an_accepted_login_is_cached_and_sent(self) -> None:
        from app.services.paper_search.platforms import openreview

        seen: list[dict[str, str]] = []

        class Recording(FakeClient):
            async def get_json(self, url, *, params=None, headers=None):
                seen.append(dict(headers or {}))
                return await super().get_json(url, params=params, headers=headers)

        client = Recording({"/login": {"token": "tok"}, "/notes/search": {"notes": []}})
        await openreview.search_submissions(client, "x", limit=1)
        await openreview.search_submissions(client, "y", limit=1)
        self.assertEqual([h.get("Authorization") for h in seen], ["Bearer tok", "Bearer tok"])
        self.assertEqual(openreview.login_error(), "")


class IeeeAdapterTests(_Reset):
    async def test_citations_and_access_are_read_from_the_record(self) -> None:
        from app.services.paper_search import SearchFilters, Settings
        from app.services.paper_search.platforms.ieeexplore import search_ieeexplore

        def article(n, access):
            return {"title": f"T{n}", "publication_year": "2024", "content_type": "Conferences",
                    "publication_title": "ICASSP", "citing_paper_count": "12", "access_type": access,
                    "doi": f"10.1109/x.{n}", "pdf_url": "https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1"}

        client = FakeClient({"ieeexploreapi": {"articles": [article(1, "LOCKED"), article(2, "OPEN_ACCESS"), article(3, "EPHEMERA")]}})
        papers = await search_ieeexplore(
            client, query="q", limit=5, settings=Settings(ieee_api_key="k"),
            filters=SearchFilters(year_from=2023, publication_types=("conference",)),
        )
        params = client.calls[0][2]
        self.assertEqual(params["start_year"], "2023")
        self.assertEqual(params["content_type"], "Conferences")
        self.assertEqual([p.is_open_access for p in papers], [False, True, None])
        self.assertEqual(papers[0].citation_counts, {"IEEE Xplore": 12})
        self.assertEqual(papers[0].venue_type, "conference")
        self.assertEqual(papers[0].oa_pdf_url, "", "stamp.jsp is an HTML frame, not a PDF")


# ── orchestration ───────────────────────────────────────────────────────────


def _paper(**kw: Any):
    from app.services.paper_search import Paper

    base = {"title": "T", "abstract": "", "url": "", "doi": "", "authors": "", "source_platform": "X"}
    base.update(kw)
    return Paper(**base)


def _spec(name: str, result: Any, filters=frozenset(), sorts=frozenset({"relevance"})):
    from app.services.paper_search.platforms import PlatformSpec

    async def search(client, **_kw):
        if isinstance(result, BaseException):
            raise result
        return [_paper(**r) if isinstance(r, dict) else r for r in result]

    return PlatformSpec(name, search, frozenset(filters), frozenset(sorts))


class OrchestratorTests(_Reset):
    async def _run(self, specs, **kw):
        from app.services.paper_search import Settings, search
        from app.services.paper_search.platforms import PLATFORMS

        registry = {s.name: s for s in specs}
        with mock.patch.object(search, "resolve_spec", side_effect=lambda n: registry.get(n) or PLATFORMS.get(n)), \
             mock.patch.object(search, "_enrich_missing_dois", new=mock.AsyncMock()):
            return await search.search_papers_detailed(
                "moe", [s.name for s in specs], settings=Settings(), rerank_mode="off", **kw
            )

    async def test_duplicates_merge_field_by_field_across_platforms(self) -> None:
        outcome = await self._run(
            [
                _spec("arXiv", [dict(title="Drop-Upcycling: Training MoE", arxiv_id="2502.19261", source_platform="arXiv", venue="arXiv [cs.LG]", venue_type="preprint")]),
                _spec("SemanticScholar", [dict(title="Drop-Upcycling: training MoE", source_platform="SemanticScholar", doi="10.1/x", citation_counts={"SemanticScholar": 21}, venue="ICLR", venue_type="conference")]),
                _spec("OpenAlex", [dict(title="Drop Upcycling Training MoE", source_platform="OpenAlex", doi="10.1/x", citation_counts={"OpenAlex": 4}, is_retracted=False)]),
            ]
        )
        self.assertEqual(len(outcome.papers), 1)
        paper = outcome.papers[0]
        self.assertEqual(paper.arxiv_id, "2502.19261")
        self.assertEqual(paper.doi, "10.1/x")
        self.assertEqual(paper.citation_counts, {"SemanticScholar": 21, "OpenAlex": 4})
        self.assertEqual(paper.citation_count, 21)
        self.assertEqual(paper.venue, "ICLR", "a real venue replaces the preprint server")
        self.assertEqual(paper.venue_type, "conference")
        self.assertEqual(paper.sources, ["arXiv", "SemanticScholar", "OpenAlex"])

    async def test_each_platform_reports_what_happened(self) -> None:
        from app.services.paper_search.platforms.openreview import OpenReviewLoginRequired
        from app.services.paper_search.ratelimit import MissingCredentialError

        outcome = await self._run(
            [
                _spec("SemanticScholar", _http_error(403)),
                _spec("OpenAlex", [dict(title="A")]),
                _spec("arXiv", _http_error(406)),
                _spec("IEEE Xplore", MissingCredentialError("no key")),
                _spec("OpenReview", OpenReviewLoginRequired("login")),
                _spec("Crossref", []),
            ]
        )
        status = {s.platform: s.status for s in outcome.platforms}
        self.assertEqual(
            status,
            {
                "SemanticScholar": "auth_failed",
                "OpenAlex": "ok",
                "arXiv": "rate_limited",
                "IEEE Xplore": "skipped_no_key",
                "OpenReview": "auth_failed",
                "Crossref": "empty",
            },
        )

    async def test_unverifiable_values_fail_the_filter_and_are_counted(self) -> None:
        from app.services.paper_search import SearchFilters

        outcome = await self._run(
            [
                _spec(
                    "SemanticScholar",
                    [
                        dict(title="Cited enough", citation_counts={"SemanticScholar": 50}, year=2024),
                        dict(title="Too few", citation_counts={"SemanticScholar": 2}, year=2024),
                        dict(title="Count unknown", year=2024),
                        dict(title="Undated", citation_counts={"SemanticScholar": 99}),
                    ],
                    filters={"min_citations", "year"},
                )
            ],
            filters=SearchFilters(year_from=2023, min_citations=10),
        )
        self.assertEqual({p.title for p in outcome.papers}, {"Cited enough", "Undated"})
        self.assertEqual(outcome.excluded, {"min_citations": 1, "min_citations_unknown": 1})
        status = outcome.platforms[0]
        self.assertEqual(status.local_filters, [])

    async def test_citation_order_puts_unknown_counts_last(self) -> None:
        from app.services.paper_search import SearchFilters

        outcome = await self._run(
            [
                _spec(
                    "OpenAlex",
                    [dict(title="Low", citation_counts={"OpenAlex": 1}), dict(title="None"), dict(title="High", citation_counts={"OpenAlex": 90})],
                    sorts={"relevance", "citations"},
                )
            ],
            filters=SearchFilters(sort="citations"),
        )
        self.assertEqual([p.title for p in outcome.papers], ["High", "Low", "None"])

    async def test_a_spent_openalex_is_replaced_by_semantic_scholar(self) -> None:
        from app.services.paper_search.ratelimit import OPENALEX_BUDGET

        OPENALEX_BUDGET.mark_exhausted({"X-RateLimit-Reset": "600"})
        seen: dict[str, int] = {}

        from app.services.paper_search.platforms import PlatformSpec

        async def s2(client, *, limit, **_kw):
            seen["limit"] = limit
            return [_paper(title="From S2", source_platform="SemanticScholar")]

        outcome = await self._run(
            [_spec("OpenAlex", []), PlatformSpec("SemanticScholar", s2, frozenset(), frozenset({"relevance"}))],
            final_limit=10,
        )
        status = {s.platform: s.status for s in outcome.platforms}
        self.assertEqual(status["OpenAlex"], "quota_exhausted")
        self.assertEqual(status["SemanticScholar"], "ok")
        self.assertEqual(seen["limit"], 20, "S2 takes over OpenAlex's share")

    async def test_auto_rerank_only_for_a_large_pool(self) -> None:
        from app.services.paper_search import Settings, search

        small = [dict(title=f"P{i}") for i in range(5)]
        large = [dict(title=f"Paper number {i}") for i in range(search.RERANK_AUTO_THRESHOLD + 5)]
        rerank = mock.AsyncMock(side_effect=lambda client, *, query, papers, limit, settings: (papers[:limit], True, {}))
        for rows, expected in ((small, 0), (large, 1)):
            rerank.reset_mock()
            spec = _spec("OpenAlex", rows)
            with mock.patch.object(search, "resolve_spec", return_value=spec), \
                 mock.patch.object(search, "_enrich_missing_dois", new=mock.AsyncMock()), \
                 mock.patch.object(search, "_rerank_rank", new=rerank):
                await search.search_papers_detailed("moe", ["OpenAlex"], settings=Settings(), rerank_mode="auto")
            self.assertEqual(rerank.await_count, expected)


# ── agent tools ─────────────────────────────────────────────────────────────


def _outcome(papers, statuses=None):
    from app.services.paper_search import PlatformStatus, SearchOutcome

    return SearchOutcome(
        papers=papers,
        platforms=statuses or [PlatformStatus(platform="SemanticScholar", status="ok", count=len(papers))],
    )


class SearchToolP6Tests(AgentP2TestCase):
    async def test_identifiers_reach_the_catalog_so_download_can_use_them(self) -> None:
        from app.agents.tools import search_papers
        from app.services import paper_catalog

        paper = _paper(
            title="Drop-Upcycling", abstract="We reinitialise.", year=2025, source_platform="arXiv",
            arxiv_id="2502.19261", oa_pdf_url="https://arxiv.org/pdf/2502.19261", s2_paper_id="s2abc",
            citation_counts={"SemanticScholar": 21, "OpenAlex": 4}, sources=["arXiv", "SemanticScholar"],
        )
        with mock.patch("app.agents.tools.discovery.run_search", new=mock.AsyncMock(return_value=_outcome([paper]))):
            result = await self._call(search_papers, query="upcycling")

        row = result["data"]["results"][0]
        item = await paper_catalog.get_literature_item(row["literature_id"])
        self.assertEqual(item["arxiv_id"], "2502.19261")
        self.assertEqual(item["oa_pdf_url"], "https://arxiv.org/pdf/2502.19261")
        self.assertEqual(item["s2_paper_id"], "s2abc")
        self.assertEqual(row["citation_counts"], {"SemanticScholar": 21, "OpenAlex": 4})
        self.assertTrue(row["full_text_route"])

    async def test_citation_counts_are_metadata_evidence_not_abstract(self) -> None:
        from app.agents.tools import search_papers
        from app.services import evidence_service

        paper = _paper(title="P", abstract="Abstract.", year=2024, citation_counts={"OpenAlex": 7}, source_platform="OpenAlex")
        with mock.patch("app.agents.tools.discovery.run_search", new=mock.AsyncMock(return_value=_outcome([paper]))):
            result = await self._call(search_papers, query="q")

        row = result["data"]["results"][0]
        facts = await evidence_service.resolve(row["metadata_evidence_id"], owner_id=self.owner)
        self.assertEqual(facts.source_level.value, "metadata")
        self.assertIn("7 (OpenAlex)", facts.quote)
        abstract = await evidence_service.resolve(row["evidence_id"], owner_id=self.owner)
        self.assertEqual(abstract.source_level.value, "abstract")

    async def test_a_platform_that_could_not_search_makes_the_result_partial(self) -> None:
        from app.agents.tools import search_papers
        from app.services.paper_search import PlatformStatus

        statuses = [
            PlatformStatus(platform="SemanticScholar", status="ok", count=1),
            PlatformStatus(platform="OpenAlex", status="quota_exhausted", detail="OpenAlex daily quota exhausted"),
        ]
        paper = _paper(title="P", abstract="a", year=2024)
        with mock.patch("app.agents.tools.discovery.run_search", new=mock.AsyncMock(return_value=_outcome([paper], statuses))):
            result = await self._call(search_papers, query="q")

        self.assertEqual(result["status"], "partial")
        self.assertIn("OpenAlex", result["note"])
        self.assertIn("quota_exhausted", result["note"])

    async def test_no_platform_searched_is_a_retryable_rate_limit(self) -> None:
        from app.agents.tools import search_papers
        from app.services.paper_search import PlatformStatus

        statuses = [
            PlatformStatus(platform="SemanticScholar", status="rate_limited"),
            PlatformStatus(platform="OpenAlex", status="quota_exhausted"),
        ]
        with mock.patch("app.agents.tools.discovery.run_search", new=mock.AsyncMock(return_value=_outcome([], statuses))):
            result = await self._call(search_papers, query="q")

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "rate_limited")
        self.assertTrue(result["error"]["retryable"])

    async def test_venues_bring_in_dblp_and_openreview_and_filters_reach_the_search(self) -> None:
        from app.agents.tools import search_papers

        run = mock.AsyncMock(return_value=_outcome([]))
        with mock.patch("app.agents.tools.discovery.run_search", new=run):
            await self._call(
                search_papers, query="q", venues=["ICLR"], year_from=2024, min_citations=5, sort="citations"
            )
        args, kwargs = run.call_args
        self.assertEqual(args[1], ["SemanticScholar", "OpenAlex", "DBLP", "OpenReview"])
        self.assertEqual(kwargs["filters"].venues, ("ICLR",))
        self.assertEqual(kwargs["filters"].min_citations, 5)
        self.assertEqual(kwargs["filters"].sort, "citations")

    async def test_bad_sort_and_types_are_argument_errors(self) -> None:
        from app.agents.tools import search_papers

        run = mock.AsyncMock()
        with mock.patch("app.agents.tools.discovery.run_search", new=run):
            bad_sort = await self._call(search_papers, query="q", sort="hype")
            bad_type = await self._call(search_papers, query="q", publication_types=["thesis"])
        self.assertEqual(bad_sort["error"]["code"], "invalid_argument")
        self.assertEqual(bad_type["error"]["code"], "invalid_argument")
        run.assert_not_awaited()


class MetadataToolTests(AgentP2TestCase):
    async def test_counts_are_kept_per_index_and_recorded_as_metadata(self) -> None:
        from app.agents.tools import get_paper_metadata
        from app.services import evidence_service
        from app.services.citation_graph import PaperMetadata

        s2 = mock.AsyncMock(return_value=[_s2_record(), None])
        oa = mock.AsyncMock(return_value={"10.48550/arxiv.2502.19261": PaperMetadata(title="Drop-Upcycling", doi="10.48550/arxiv.2502.19261", cited_by_count=4, is_retracted=False)})
        with mock.patch("app.agents.tools.discovery.s2_batch", new=s2), \
             mock.patch("app.services.citation_graph.openalex_batch_fetch_by_doi", new=oa):
            result = await self._call(get_paper_metadata, arxiv_ids=["2502.19261", "9999.99999"])

        self.assertEqual(result["status"], "partial")
        paper = result["data"]["papers"][0]
        self.assertEqual(paper["citations"], {"semanticscholar": 21, "openalex": 4})
        self.assertEqual(paper["influential_citation_count"], 3)
        self.assertIn("ICLR", paper["venue_aliases"])
        self.assertEqual(paper["ids"]["arxiv_id"], "2502.19261")
        self.assertEqual(result["data"]["not_found"][0]["query"], "9999.99999")
        evidence = await evidence_service.resolve(paper["evidence_id"], owner_id=self.owner)
        self.assertEqual(evidence.source_level.value, "metadata")
        self.assertIn("21 (Semantic Scholar)", evidence.quote)

    async def test_nothing_to_look_up_is_an_argument_error(self) -> None:
        from app.agents.tools import get_paper_metadata

        result = await self._call(get_paper_metadata)
        self.assertEqual(result["error"]["code"], "invalid_argument")


class SnippetToolTests(AgentP2TestCase):
    async def test_passages_are_abstract_level_with_their_section(self) -> None:
        from app.agents.tools import search_paper_snippets
        from app.services import evidence_service, paper_catalog

        hits = {
            "data": [
                {
                    "score": 0.4,
                    "paper": {"corpusId": "276617861", "title": "Drop-Upcycling", "authors": ["T. Nakamura"]},
                    "snippet": {"text": "A ratio of 0.5 worked best.", "snippetKind": "body", "section": "Analysis 1"},
                }
            ]
        }
        get = mock.AsyncMock(return_value=hits)
        batch = mock.AsyncMock(return_value=[_s2_record()])
        with mock.patch("app.agents.tools.discovery.s2_get", new=get), \
             mock.patch("app.agents.tools.discovery.s2_batch", new=batch):
            result = await self._call(search_paper_snippets, query="reinitialisation ratio", year_from=2024)

        self.assertEqual(get.call_args.args[2]["year"], "2024-")
        snippet = result["data"]["snippets"][0]
        self.assertEqual(snippet["evidence_level"], "abstract")
        self.assertEqual(snippet["section"], "Analysis 1")
        evidence = await evidence_service.resolve(snippet["evidence_id"], owner_id=self.owner)
        self.assertEqual(evidence.locator.section_path, "Analysis 1")
        self.assertIsNone(evidence.locator.page_label)
        item = await paper_catalog.get_literature_item(snippet["literature_id"])
        self.assertEqual(item["arxiv_id"], "2502.19261")

    async def test_scoping_to_papers_passes_their_ids(self) -> None:
        from app.agents.tools import search_paper_snippets
        from app.services import paper_catalog

        lit = await paper_catalog.upsert_literature_item(title="X", arxiv_id="2502.19261")
        get = mock.AsyncMock(return_value={"data": []})
        with mock.patch("app.agents.tools.discovery.s2_get", new=get):
            await self._call(search_paper_snippets, query="lr", literature_ids=[lit])
        self.assertEqual(get.call_args.args[2]["paperIds"], "ARXIV:2502.19261")


class PeerReviewToolTests(AgentP2TestCase):
    SUBMISSION = {
        "id": "gx1", "forum": "gx1",
        "invitations": ["ICLR.cc/2025/Conference/-/Submission"],
        "content": {"title": {"value": "Drop-Upcycling: Training Sparse Mixture of Experts"},
                    "venue": {"value": "ICLR 2025 Poster"}, "authors": {"value": ["T. Nakamura"]}},
    }

    async def test_without_an_account_the_decision_is_reported_and_reviews_are_not(self) -> None:
        from app.agents.tools import get_peer_reviews
        from app.services.paper_search.platforms import openreview

        with mock.patch.object(openreview, "search_submissions", new=mock.AsyncMock(return_value=[self.SUBMISSION])), \
             mock.patch.object(openreview, "get_forum_notes", new=mock.AsyncMock(side_effect=openreview.OpenReviewLoginRequired("x"))):
            result = await self._call(get_peer_reviews, title="Drop-Upcycling: Training Sparse Mixture of Experts")

        self.assertEqual(result["status"], "partial")
        self.assertIn("OPENREVIEW_USERNAME", result["note"])
        self.assertEqual(result["data"]["status"]["state"], "accepted")
        self.assertEqual(result["data"]["status"]["tier"], "poster")
        self.assertEqual(result["data"]["reviews"], [])

    async def test_a_refused_account_is_named_in_the_note(self) -> None:
        from app.agents.tools import get_peer_reviews
        from app.services.paper_search.platforms import openreview

        with mock.patch.object(openreview, "search_submissions", new=mock.AsyncMock(return_value=[self.SUBMISSION])), \
             mock.patch.object(openreview, "get_forum_notes", new=mock.AsyncMock(side_effect=openreview.OpenReviewLoginRequired("x"))), \
             mock.patch.object(openreview, "has_account", return_value=True), \
             mock.patch.object(openreview, "login_error", return_value="HTTP 400: Invalid username or password"):
            result = await self._call(get_peer_reviews, title="Drop-Upcycling: Training Sparse Mixture of Experts")

        self.assertEqual(result["status"], "partial")
        self.assertIn("refused the configured account", result["note"])
        self.assertIn("Invalid username or password", result["note"])

    async def test_reviews_are_web_evidence_attributed_to_reviewers(self) -> None:
        from app.agents.tools import get_peer_reviews
        from app.services import evidence_service
        from app.services.paper_search.platforms import openreview

        replies = [
            self.SUBMISSION,
            {"id": "r1", "invitations": ["ICLR.cc/2025/Conference/Submission1/-/Official_Review"],
             "content": {"rating": {"value": "8: accept"}, "weaknesses": {"value": "Only one scale."}}},
            {"id": "r2", "invitations": ["ICLR.cc/2025/Conference/Submission1/-/Official_Review"],
             "content": {"rating": {"value": "5: marginally below"}}},
        ]
        with mock.patch.object(openreview, "search_submissions", new=mock.AsyncMock(return_value=[self.SUBMISSION])), \
             mock.patch.object(openreview, "get_forum_notes", new=mock.AsyncMock(return_value=replies)):
            result = await self._call(get_peer_reviews, title="Drop-Upcycling: Training Sparse Mixture of Experts")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["data"]["rating_summary"], {"count": 2, "mean": 6.5, "min": 5.0, "max": 8.0})
        evidence = await evidence_service.resolve(result["data"]["reviews"][0]["evidence_id"], owner_id=self.owner)
        self.assertEqual(evidence.source_level.value, "external_web")
        self.assertIn("Only one scale.", evidence.quote)

    async def test_a_paper_not_on_openreview_is_unavailable(self) -> None:
        from app.agents.tools import get_peer_reviews
        from app.services.paper_search.platforms import openreview

        with mock.patch.object(openreview, "search_submissions", new=mock.AsyncMock(return_value=[])):
            result = await self._call(get_peer_reviews, title="A journal-only paper about fluids")
        self.assertEqual(result["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
