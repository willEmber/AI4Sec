from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch


# Make `paper_search_standalone/` importable as `paper_search`.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from paper_search.config import Settings
from paper_search.models import Paper
from paper_search.platforms import resolve_platform
from paper_search.platforms.arxiv import search_arxiv
from paper_search.platforms.crossref import search_crossref
from paper_search.platforms.infoxmed import search_infoxmed
from paper_search.platforms.ieeexplore import search_ieeexplore
from paper_search.platforms.openalex import search_openalex
from paper_search.platforms.pubmed import search_pubmed
from paper_search.platforms.semanticscholar import search_semanticscholar


class FakeHTTPClient:
    def __init__(
        self,
        *,
        json_by_url_prefix: dict[str, Any] | None = None,
        text_by_url_prefix: dict[str, str] | None = None,
        status_json_by_url_prefix: dict[str, tuple[int, Any]] | None = None,
        post_json_by_url_prefix: dict[str, Any] | None = None,
    ) -> None:
        self.json_by_url_prefix = json_by_url_prefix or {}
        self.text_by_url_prefix = text_by_url_prefix or {}
        self.status_json_by_url_prefix = status_json_by_url_prefix or {}
        self.post_json_by_url_prefix = post_json_by_url_prefix or {}

        self.calls: list[tuple[str, str]] = []

    def _match(self, mapping: dict[str, Any], url: str) -> Any:
        if url in mapping:
            return mapping[url]
        for prefix, value in mapping.items():
            if url.startswith(prefix):
                return value
        raise KeyError(url)

    async def get_json(self, url: str, *, params=None, headers=None) -> Any:
        _ = (params, headers)
        self.calls.append(("get_json", url))
        return self._match(self.json_by_url_prefix, url)

    async def get_text(self, url: str, *, params=None, headers=None) -> str:
        _ = (params, headers)
        self.calls.append(("get_text", url))
        return str(self._match(self.text_by_url_prefix, url))

    async def post_json(self, url: str, *, json_body, headers=None) -> Any:
        _ = (json_body, headers)
        self.calls.append(("post_json", url))
        return self._match(self.post_json_by_url_prefix, url)

    async def get_status_and_json(self, url: str, *, params=None, headers=None) -> tuple[int, Any]:
        _ = (params, headers)
        self.calls.append(("get_status_and_json", url))
        return self._match(self.status_json_by_url_prefix, url)


class PlatformsTest(unittest.IsolatedAsyncioTestCase):
    def test_resolve_platforms(self) -> None:
        self.assertIsNotNone(resolve_platform("OpenAlex"))
        self.assertIsNotNone(resolve_platform("SemanticScholar"))
        self.assertIsNotNone(resolve_platform("arXiv"))
        self.assertIsNotNone(resolve_platform("PubMed"))
        self.assertIsNotNone(resolve_platform("Crossref"))
        self.assertIsNotNone(resolve_platform("InfoXMed"))
        self.assertIsNotNone(resolve_platform("IEEE Xplore"))
        # GoogleScholar is intentionally disabled/hidden in standalone.
        self.assertIsNone(resolve_platform("GoogleScholar"))

    async def test_openalex_parse(self) -> None:
        client = FakeHTTPClient(
            json_by_url_prefix={
                "https://api.openalex.org/works": {
                    "results": [
                        {
                            "title": "  Test Paper ",
                            "abstract_inverted_index": {"hello": [0], "world": [1]},
                            "doi": "https://doi.org/10.1234/ABC",
                            "primary_location": {"landing_page_url": "https://example.com/paper"},
                            "authorships": [
                                {"author": {"display_name": "Alice"}},
                                {"author": {"display_name": "Bob"}},
                            ],
                        }
                    ]
                }
            }
        )
        settings = Settings()
        papers = await search_openalex(client, query="x", limit=5, settings=settings)
        self.assertEqual(len(papers), 1)
        p = papers[0]
        self.assertIsInstance(p, Paper)
        self.assertEqual(p.title, "Test Paper")
        self.assertEqual(p.abstract, "hello world")
        self.assertEqual(p.doi, "10.1234/abc")
        self.assertEqual(p.url, "https://example.com/paper")
        self.assertEqual(p.authors, "Alice; Bob")
        self.assertEqual(p.source_platform, "OpenAlex")

    async def test_semanticscholar_parse(self) -> None:
        client = FakeHTTPClient(
            json_by_url_prefix={
                "https://api.semanticscholar.org/graph/v1/paper/search": {
                    "data": [
                        {
                            "title": "S2 Paper",
                            "abstract": "This is abstract",
                            "authors": [{"name": "A"}, {"name": "B"}],
                            "url": "https://www.semanticscholar.org/paper/x",
                            "externalIds": {"DOI": "10.5555/XYZ"},
                            "openAccessPdf": {"url": "https://oa.example/paper.pdf"},
                        }
                    ]
                }
            }
        )
        settings = Settings()
        papers = await search_semanticscholar(client, query="x", limit=5, settings=settings)
        self.assertEqual(len(papers), 1)
        p = papers[0]
        self.assertEqual(p.title, "S2 Paper")
        self.assertEqual(p.abstract, "This is abstract")
        self.assertEqual(p.url, "https://www.semanticscholar.org/paper/x")
        self.assertEqual(p.doi, "10.5555/xyz")
        self.assertEqual(p.authors, "A; B")
        self.assertEqual(p.oa_paper_url, "https://oa.example/paper.pdf")
        self.assertEqual(p.source_platform, "SemanticScholar")

    async def test_ieeexplore_parse(self) -> None:
        client = FakeHTTPClient(
            json_by_url_prefix={
                "https://ieeexploreapi.ieee.org/api/v1/search/articles": {
                    "articles": [
                        {
                            "article_title": "  IEEE Paper  ",
                            "abstract": "A <b>good</b> abstract",
                            "doi": "10.1109/TEST.2025.1",
                            "html_url": "https://ieeexplore.ieee.org/document/1/",
                            "authors": {
                                "authors": [{"full_name": "Alice"}, {"full_name": "Bob"}],
                            },
                        }
                    ]
                }
            }
        )
        settings = replace(Settings(), ieee_api_key="test-key")
        with patch(
            "paper_search.platforms.ieeexplore._IEEE_RATE_LIMITER.acquire",
            new=AsyncMock(),
        ) as mocked_acquire:
            papers = await search_ieeexplore(client, query="x", limit=5, settings=settings)
        mocked_acquire.assert_awaited_once()

        self.assertEqual(len(papers), 1)
        p = papers[0]
        self.assertEqual(p.title, "IEEE Paper")
        self.assertEqual(p.abstract, "A good abstract")
        self.assertEqual(p.doi, "10.1109/test.2025.1")
        self.assertEqual(p.url, "https://ieeexplore.ieee.org/document/1/")
        self.assertEqual(p.authors, "Alice; Bob")
        self.assertEqual(p.source_platform, "IEEE Xplore")

    async def test_ieeexplore_missing_api_key_returns_empty(self) -> None:
        client = FakeHTTPClient()
        settings = Settings()
        papers = await search_ieeexplore(client, query="x", limit=5, settings=settings)
        self.assertEqual(papers, [])

    async def test_arxiv_parse(self) -> None:
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1234.5678</id>
    <title>  My Title  </title>
    <summary>  My Summary  </summary>
    <arxiv:doi>10.1000/XYZ</arxiv:doi>
    <author><name>John Doe</name></author>
    <author><name>Jane Roe</name></author>
  </entry>
</feed>
"""
        client = FakeHTTPClient(
            text_by_url_prefix={
                "https://export.arxiv.org/api/query?": xml,
            }
        )
        settings = Settings()
        papers = await search_arxiv(client, query="x", limit=5, settings=settings)
        self.assertEqual(len(papers), 1)
        p = papers[0]
        self.assertEqual(p.title, "My Title")
        self.assertEqual(p.abstract, "My Summary")
        self.assertEqual(p.url, "http://arxiv.org/abs/1234.5678")
        self.assertEqual(p.doi, "10.1000/xyz")
        self.assertEqual(p.authors, "John Doe; Jane Roe")
        self.assertEqual(p.oa_paper_url, "https://arxiv.org/pdf/1234.5678.pdf")
        self.assertEqual(p.source_platform, "arXiv")

    async def test_pubmed_parse(self) -> None:
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345</PMID>
      <Article>
        <ArticleTitle> Pub Title </ArticleTitle>
        <Abstract>
          <AbstractText> Pub Abstract </AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <ForeName>A</ForeName>
            <LastName>B</LastName>
          </Author>
        </AuthorList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="doi">10.1/DOI</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""
        client = FakeHTTPClient(
            json_by_url_prefix={
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi": {
                    "esearchresult": {"idlist": ["12345"]}
                }
            },
            text_by_url_prefix={
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi": xml,
            },
        )

        settings = Settings()
        papers = await search_pubmed(client, query="x", limit=5, settings=settings)
        self.assertEqual(len(papers), 1)
        p = papers[0]
        self.assertEqual(p.title, "Pub Title")
        self.assertEqual(p.abstract, "Pub Abstract")
        self.assertEqual(p.doi, "10.1/doi")
        self.assertEqual(p.url, "https://pubmed.ncbi.nlm.nih.gov/12345/")
        self.assertEqual(p.authors, "A B")
        self.assertEqual(p.source_platform, "PubMed")

    async def test_crossref_parse(self) -> None:
        client = FakeHTTPClient(
            json_by_url_prefix={
                "https://api.crossref.org/works": {
                    "message": {
                        "items": [
                            {
                                "title": ["CR Title"],
                                "abstract": "<jats:p>CR Abstract</jats:p>",
                                "DOI": "10.2/DOI",
                                "URL": "https://doi.org/10.2/doi",
                                "author": [{"given": "X", "family": "Y"}],
                            }
                        ]
                    }
                }
            }
        )
        settings = Settings()
        papers = await search_crossref(client, query="x", limit=5, settings=settings)
        self.assertEqual(len(papers), 1)
        p = papers[0]
        self.assertEqual(p.title, "CR Title")
        self.assertEqual(p.abstract, "CR Abstract")
        self.assertEqual(p.doi, "10.2/doi")
        self.assertEqual(p.url, "https://doi.org/10.2/doi")
        self.assertEqual(p.authors, "X Y")
        self.assertEqual(p.source_platform, "Crossref")

    async def test_infoxmed_parse(self) -> None:
        client = FakeHTTPClient(
            post_json_by_url_prefix={
                "https://api.infox-med.com/search/home/keywords": {
                    "code": "0",
                    "data": {
                        "records": [
                            {
                                "docTitleZh": "中文标题",
                                "docTitle": "English Title",
                                "docAbstractZh": "中文摘要",
                                "docAbstract": "English abstract",
                                "docDoi": "10.3/DOI",
                                "pmid": "",
                                "docAuthor": "A;B",
                            }
                        ]
                    },
                }
            }
        )
        settings = replace(Settings(), infoxmed_category=0)
        papers = await search_infoxmed(client, query="x", limit=5, settings=settings)
        self.assertEqual(len(papers), 1)
        p = papers[0]
        self.assertEqual(p.title, "中文标题")
        self.assertEqual(p.abstract, "中文摘要")
        self.assertEqual(p.doi, "10.3/doi")
        self.assertEqual(p.url, "https://doi.org/10.3/doi")
        self.assertEqual(p.authors, "A;B")
        self.assertEqual(p.source_platform, "InfoXMed")
