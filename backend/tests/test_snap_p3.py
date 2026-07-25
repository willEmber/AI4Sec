from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from app.models.paper_ir import Block, PaperIR
from app.models.snap_models import CitationIntents, TriageSignals
from app.services.grobid_client import GrobidHeader, _parse_tei_header, extract_header
from app.services.triage_signals import (
    _s2_citation_intents,
    compute_external_score,
    get_or_collect_triage_signals,
    load_cached_signals,
    store_signals,
)
from app.services.triage_verdict import (
    parse_content_review,
    render_verdict_markdown,
    synthesize_verdict,
)

_TEI = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title level="a" type="main">Attention Is All You Need</title></titleStmt>
      <publicationStmt><date type="published" when="2017-06-12" /></publicationStmt>
      <sourceDesc><biblStruct>
        <idno type="DOI">https://doi.org/10.5555/3295222</idno>
        <idno type="arXiv">arXiv:1706.03762</idno>
        <analytic>
          <author><persName><forename type="first">Ashish</forename><surname>Vaswani</surname></persName></author>
          <author><persName><forename type="first">Noam</forename><surname>Shazeer</surname></persName></author>
        </analytic>
        <monogr>
          <title level="j">Advances in Neural Information Processing Systems</title>
          <imprint><date type="published" when="2017" /></imprint>
        </monogr>
      </biblStruct></sourceDesc>
    </fileDesc>
    <profileDesc><abstract><p>We propose the Transformer.</p></abstract></profileDesc>
  </teiHeader>
</TEI>"""


class TestGrobidHeaderParsing(unittest.TestCase):
    def test_tei_header_fields_are_extracted(self) -> None:
        header = _parse_tei_header(_TEI)
        self.assertTrue(header.available)
        self.assertEqual(header.title, "Attention Is All You Need")
        self.assertEqual(header.doi, "10.5555/3295222")  # URL prefix stripped
        self.assertEqual(header.arxiv_id, "1706.03762")
        self.assertEqual(header.year, 2017)
        self.assertEqual(header.venue, "Advances in Neural Information Processing Systems")
        self.assertEqual(header.authors, ["Ashish Vaswani", "Noam Shazeer"])
        self.assertEqual(header.authors_str, "Ashish Vaswani, Noam Shazeer")
        self.assertIn("Transformer", header.abstract)

    def test_malformed_xml_reports_unavailable(self) -> None:
        self.assertFalse(_parse_tei_header("<TEI><unclosed>").available)

    def test_empty_tei_yields_no_fields_but_stays_available(self) -> None:
        header = _parse_tei_header(
            '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader/></TEI>'
        )
        self.assertTrue(header.available)
        self.assertEqual(header.doi, "")


class TestGrobidDisabledByDefault(unittest.IsolatedAsyncioTestCase):
    async def test_no_url_configured_skips_the_request(self) -> None:
        """GROBID is opt-in; without a URL the pipeline must not change behaviour."""
        with TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "paper.pdf"
            pdf.write_bytes(b"%PDF-1.4")
            header = await extract_header(pdf)
        self.assertFalse(header.available)

    async def test_missing_file_is_handled(self) -> None:
        with patch("app.services.grobid_client.get_settings") as settings:
            settings.return_value.grobid_url = "http://localhost:8070"
            settings.return_value.grobid_timeout_seconds = 5
            header = await extract_header("/nonexistent/paper.pdf")
        self.assertFalse(header.available)


class TestCitationIntents(unittest.IsolatedAsyncioTestCase):
    async def test_intents_are_tallied_from_citation_edges(self) -> None:
        payload = {
            "data": [
                {"intents": ["background"], "isInfluential": False},
                {"intents": ["methodology"], "isInfluential": True},
                {"intents": ["methodology", "result"], "isInfluential": True},
                {"intents": [], "isInfluential": False},
            ]
        }
        with patch("app.services.citation_graph._get_json", AsyncMock(return_value=payload)):
            intents = await _s2_citation_intents(None, "abc")

        self.assertTrue(intents.available)
        self.assertEqual(intents.sampled, 4)
        self.assertEqual(intents.background, 1)
        self.assertEqual(intents.methodology, 2)
        self.assertEqual(intents.result, 1)
        self.assertEqual(intents.influential, 2)

    async def test_no_paper_id_short_circuits(self) -> None:
        intents = await _s2_citation_intents(None, "")
        self.assertFalse(intents.available)

    async def test_failed_request_yields_an_empty_tally(self) -> None:
        with patch("app.services.citation_graph._get_json", AsyncMock(return_value=None)):
            intents = await _s2_citation_intents(None, "abc")
        self.assertFalse(intents.available)

    def test_intents_render_as_an_evidence_row(self) -> None:
        signals = compute_external_score(TriageSignals(
            venue="ICML", ccf_rank="A", year=2021, cited_by_count=800, resolved=True,
            intents=CitationIntents(background=40, methodology=52, result=9, sampled=98),
            provenance={"intents": "Semantic Scholar"},
        ))
        review = parse_content_review(json.dumps({"contribution_strength": 3}))
        md = render_verdict_markdown(synthesize_verdict(review, signals), review, signals)
        self.assertIn("methodology 52", md)
        self.assertIn("of 98 sampled", md)

    def test_intents_row_is_omitted_when_unsampled(self) -> None:
        signals = compute_external_score(TriageSignals(venue="ICML", resolved=True))
        review = parse_content_review(json.dumps({"contribution_strength": 3}))
        md = render_verdict_markdown(synthesize_verdict(review, signals), review, signals)
        self.assertNotIn("Cited as", md)


class TestSignalsCache(unittest.IsolatedAsyncioTestCase):
    """The cache is keyed by paper, so re-running the same PDF costs no API calls."""

    async def asyncSetUp(self) -> None:
        from app.db import database as db

        self._tmp = TemporaryDirectory()
        db.set_db_path(Path(self._tmp.name) / "app.db")
        await db.init_db()
        await db.execute(
            "INSERT OR REPLACE INTO papers (paper_id, file_path) VALUES (?, ?)",
            ("p1", "papers/p1/original.pdf"),
        )

    async def asyncTearDown(self) -> None:
        self._tmp.cleanup()

    def _ir(self) -> PaperIR:
        return PaperIR(
            paper_id="p1",
            title="FooNet",
            blocks=[Block(type="text", page_idx=0, text="Our code: https://github.com/acme/foonet", section_path="Abstract")],
        )

    async def test_store_then_load_round_trips(self) -> None:
        signals = compute_external_score(
            TriageSignals(venue="ICML", ccf_rank="A", cited_by_count=120, resolved=True)
        )
        await store_signals("p1", signals)
        loaded = await load_cached_signals("p1", ttl_hours=168)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.cited_by_count, 120)
        self.assertEqual(loaded.ccf_rank, "A")

    async def test_expired_entry_is_a_miss(self) -> None:
        await store_signals("p1", TriageSignals(resolved=True, cited_by_count=5))
        self.assertIsNone(await load_cached_signals("p1", ttl_hours=0))

    async def test_unknown_paper_is_a_miss(self) -> None:
        self.assertIsNone(await load_cached_signals("nope", ttl_hours=168))

    async def test_cache_hit_skips_the_network(self) -> None:
        await store_signals(
            "p1", compute_external_score(TriageSignals(resolved=True, cited_by_count=77, venue="ICML"))
        )
        with patch(
            "app.services.triage_signals.collect_triage_signals",
            AsyncMock(side_effect=AssertionError("network must not be touched")),
        ):
            signals = await get_or_collect_triage_signals(self._ir(), paper_id="p1", ttl_hours=168)
        self.assertEqual(signals.cited_by_count, 77)

    async def test_cache_hit_is_rescored_not_replayed(self) -> None:
        """Stored scores are recomputed so a weight change takes effect at once."""
        stale = TriageSignals(resolved=True, cited_by_count=500, year=2020, venue="ICML", ccf_rank="A")
        stale.external_score = 0.0  # as if written by an older scoring rule
        await store_signals("p1", stale)
        signals = await get_or_collect_triage_signals(self._ir(), paper_id="p1", ttl_hours=168)
        self.assertGreater(signals.external_score, 0.5)

    async def test_miss_collects_and_persists(self) -> None:
        fresh = compute_external_score(
            TriageSignals(resolved=True, cited_by_count=42, venue="NeurIPS", ccf_rank="A")
        )
        with patch(
            "app.services.triage_signals.collect_triage_signals", AsyncMock(return_value=fresh)
        ):
            await get_or_collect_triage_signals(self._ir(), paper_id="p1", ttl_hours=168)
        cached = await load_cached_signals("p1", ttl_hours=168)
        self.assertIsNotNone(cached)
        assert cached is not None
        self.assertEqual(cached.cited_by_count, 42)

    async def test_unresolved_result_is_not_cached(self) -> None:
        """Caching a total miss would pin 'unknown' on the paper for a week."""
        unresolved = compute_external_score(
            TriageSignals(resolved=False, unavailable=["citations", "open_access", "retraction"])
        )
        with patch(
            "app.services.triage_signals.collect_triage_signals", AsyncMock(return_value=unresolved)
        ):
            await get_or_collect_triage_signals(self._ir(), paper_id="p1", ttl_hours=168)
        self.assertIsNone(await load_cached_signals("p1", ttl_hours=168))

    async def test_ttl_zero_disables_the_cache_entirely(self) -> None:
        fresh = compute_external_score(TriageSignals(resolved=True, cited_by_count=9, venue="ICML"))
        with patch(
            "app.services.triage_signals.collect_triage_signals", AsyncMock(return_value=fresh)
        ):
            await get_or_collect_triage_signals(self._ir(), paper_id="p1", ttl_hours=0)
        self.assertIsNone(await load_cached_signals("p1", ttl_hours=168))

    async def test_denormalized_columns_are_written(self) -> None:
        from app.db import database as db

        await store_signals(
            "p1", TriageSignals(resolved=True, cited_by_count=333, is_retracted=True)
        )
        row = await db.fetch_one(
            "SELECT cited_by_count, is_retracted FROM paper_signals WHERE paper_id = ?", ("p1",)
        )
        self.assertEqual(row["cited_by_count"], 333)
        self.assertEqual(row["is_retracted"], 1)


if __name__ == "__main__":
    unittest.main()
