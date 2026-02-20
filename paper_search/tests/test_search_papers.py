from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


# Make `paper_search_standalone/` importable as `paper_search`.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from paper_search.models import Paper
from paper_search.search import search_papers


class SearchPapersFlowTest(unittest.IsolatedAsyncioTestCase):
    async def test_search_papers_offline_flow(self) -> None:
        # Keep environment deterministic and network-free.
        safe_env = {
            "PAPERSEARCH_DOI_ENRICH_ENABLED": "0",
            "PAPERSEARCH_LLM_BASEURL": "",
            "PAPERSEARCH_LLM_APIKEY": "",
            "PAPERSEARCH_RERANK_MODELNAME": "",
        }

        raw = [
            Paper(
                title="Paper One",
                abstract="Short snippet...",
                url="https://example.com/a",
                doi="10.1/DOI",
                authors="Alice",
                source_platform="OpenAlex",
            ),
            # Same DOI: should merge into the first item and upgrade abstract/url/authors if needed.
            Paper(
                title="Paper One",
                abstract="This is a longer abstract without ellipsis.",
                url="",
                doi="10.1/doi",
                authors="",
                source_platform="Crossref",
            ),
            # No DOI: should still be returned.
            Paper(
                title="Paper Two",
                abstract="Some abstract",
                url="https://example.com/b",
                doi="",
                authors="Bob",
                source_platform="arXiv",
            ),
        ]

        with patch.dict(os.environ, safe_env, clear=False), patch(
            "paper_search.search.load_env_file", return_value=None
        ), patch(
            "paper_search.search._run_searchers", new=AsyncMock(return_value=list(raw))
        ):
            out = await search_papers(query="test query", platforms=["OpenAlex", "Crossref", "arXiv"], final_limit=10)

        data = json.loads(out)
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 2)

        # Ensure required keys exist and deprecated fields are no longer returned.
        for item in data:
            for key in (
                "title",
                "abstract",
                "url",
                "doi",
                "authors",
                "source_platform",
            ):
                self.assertIn(key, item)
            self.assertNotIn("pdf_url", item)
            self.assertNotIn("oa_paper_url", item)
            self.assertNotIn("agent_remark", item)

        by_title = {it["title"]: it for it in data}
        one = by_title["Paper One"]
        self.assertEqual(one["doi"], "10.1/doi")
        self.assertEqual(one["abstract"], "This is a longer abstract without ellipsis.")

        two = by_title["Paper Two"]
        self.assertEqual(two["doi"], "")

    async def test_query_max_chars_guard(self) -> None:
        with patch.dict(os.environ, {"PAPERSEARCH_QUERY_MAX_CHARS": "3"}, clear=False), patch(
            "paper_search.search.load_env_file", return_value=None
        ):
            with self.assertRaises(ValueError):
                await search_papers(query="abcd", platforms=["OpenAlex"])

    async def test_unknown_platforms_are_ignored(self) -> None:
        with patch.dict(os.environ, {"PAPERSEARCH_DOI_ENRICH_ENABLED": "0"}, clear=False), patch(
            "paper_search.search.load_env_file", return_value=None
        ):
            out = await search_papers(query="x", platforms=["GoogleScholar"], final_limit=5, summary_enabled=False)
        data = json.loads(out)
        self.assertEqual(data, [])

    async def test_output_fields_filter(self) -> None:
        safe_env = {
            "PAPERSEARCH_DOI_ENRICH_ENABLED": "0",
        }
        raw = [
            Paper(
                title="T",
                abstract="A",
                url="U",
                doi="10.1/x",
                authors="Au",
                source_platform="OpenAlex",
            )
        ]

        with patch.dict(os.environ, safe_env, clear=False), patch(
            "paper_search.search.load_env_file", return_value=None
        ), patch(
            "paper_search.search._run_searchers", new=AsyncMock(return_value=list(raw))
        ):
            out = await search_papers(query="x", platforms=["OpenAlex"], fields=["title", "doi"])

        data = json.loads(out)
        self.assertEqual(len(data), 1)
        self.assertEqual(list(data[0].keys()), ["title", "doi"])
        self.assertEqual(data[0]["title"], "T")
        self.assertEqual(data[0]["doi"], "10.1/x")

    async def test_output_fields_unknown_raises(self) -> None:
        with patch.dict(os.environ, {"PAPERSEARCH_DOI_ENRICH_ENABLED": "0"}, clear=False), patch(
            "paper_search.search.load_env_file", return_value=None
        ):
            with self.assertRaises(ValueError):
                await search_papers(query="x", platforms=["OpenAlex"], fields=["not_a_field"])
