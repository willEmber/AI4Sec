"""Where and when a paper was published: its own identifiers, its venue's year.

Semantic Scholar's `year` is when a paper first appeared, so SinSR (CVPR 2024,
arXiv November 2023) was stored as "Computer Vision and Pattern Recognition,
2023", and its citations per year were counted over four years instead of
three. S2 also lists the published version's DOI; the year is that version's.
"""

from __future__ import annotations

import unittest
from typing import Any, Self
from unittest import mock

from app.models.paper_ir import Block, PaperIR
from app.services import paper_metadata

# What S2 returns for ARXIV:2311.14760 (trimmed, 2026-11-21).
S2_SINSR = {
    "externalIds": {
        "DBLP": "conf/cvpr/WangYCWGCLQKW24",
        "ArXiv": "2311.14760",
        "DOI": "10.1109/CVPR52733.2024.02437",
    },
    "publicationVenue": {"name": "Computer Vision and Pattern Recognition", "type": "conference"},
    "venue": "Computer Vision and Pattern Recognition",
    "year": 2023,
    "publicationDate": "2023-11-23",
}
CVPR_DOI = "10.1109/CVPR52733.2024.02437"


def _ir() -> PaperIR:
    return PaperIR(
        paper_id="p1",
        title="SinSR: Diffusion-Based Image Super-Resolution in a Single Step",
        blocks=[Block(type="text", text="arXiv:2311.14760v2 [cs.CV] 28 Nov 2023")],
    )


class S2RecordTests(unittest.TestCase):
    def test_the_published_doi_is_read_beside_the_preprint_year(self) -> None:
        self.assertEqual(
            paper_metadata._parse_s2_response(S2_SINSR),
            {"venue": "Computer Vision and Pattern Recognition", "year": 2023, "doi": CVPR_DOI},
        )

    def test_arxivs_own_doi_is_not_a_published_version(self) -> None:
        record = {**S2_SINSR, "externalIds": {"DOI": "10.48550/arXiv.2311.14760"}}
        self.assertEqual(paper_metadata._parse_s2_response(record)["doi"], "")
        self.assertEqual(paper_metadata._parse_s2_response({"venue": "X"})["year"], 0)


class S2TitleMatchGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_s2_title_match_rejects_unrelated_fuzzy_hit_and_accepts_dehyphenated_title(self) -> None:
        class _Resp:
            status_code = 200

            def __init__(self, payload: dict[str, Any]) -> None:
                self._payload = payload

            def json(self) -> dict[str, Any]:
                return self._payload

        sepmark = {
            "data": [{
                "title": "SepMark: Deep Separable Watermarking for Unified Source Tracing and Deepfake Detection",
                "venue": "Proceedings of the 31st ACM International Conference on Multimedia",
                "year": 2023,
                "externalIds": {"DOI": "10.1145/3581783.3612471"},
                "publicationVenue": {"name": "ACM Multimedia"},
            }]
        }
        supermark_exact = {
            "data": [{
                "title": "SuperMark: Robust and Training-free Image Watermarking via Diffusion-based Super-Resolution",
                "venue": "ICLR",
                "year": 2025,
                "externalIds": {},
                "publicationVenue": {"name": "ICLR"},
            }]
        }

        client = mock.AsyncMock()
        client.__aenter__.return_value = client
        client.get = mock.AsyncMock(return_value=_Resp(sepmark))
        with (
            mock.patch("app.services.paper_metadata.httpx.AsyncClient", return_value=client),
            mock.patch("app.services.paper_search.ratelimit.S2_LIMITER.wait", mock.AsyncMock()),
        ):
            rejected = await paper_metadata._s2_lookup(
                title="SUPERMARK: ROBUST AND TRAINING-FREE IM-AGE WATERMARKING VIA DIFFUSION-BASED SUPER-RESOLUTION"
            )
            self.assertEqual(rejected, {})

            client.get = mock.AsyncMock(return_value=_Resp(supermark_exact))
            accepted = await paper_metadata._s2_lookup(
                title="SUPERMARK: ROBUST AND TRAINING-FREE IM-AGE WATERMARKING VIA DIFFUSION-BASED SUPER-RESOLUTION"
            )
            self.assertEqual(accepted["venue"], "ICLR")


class _Rank:
    success, sci, ccf, error = True, "", "A", ""


class _RankClient:
    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def query(self, venue: str) -> _Rank:
        return _Rank()


class EnrichTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.written: list[tuple] = []
        self.crossref = mock.AsyncMock(return_value={
            "venue": "2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)",
            "year": 2024,
        })
        self.s2 = mock.AsyncMock(return_value=paper_metadata._parse_s2_response(S2_SINSR))

        async def execute(sql: str, params: tuple = ()) -> int:
            self.written.append(params)
            return 1

        for patcher in (
            mock.patch.object(paper_metadata.db, "fetch_one", mock.AsyncMock(return_value=None)),
            mock.patch.object(paper_metadata.db, "execute", side_effect=execute),
            mock.patch.object(paper_metadata, "_crossref_lookup", self.crossref),
            mock.patch.object(paper_metadata, "_s2_lookup", self.s2),
            mock.patch("app.services.publication_rank.UnifiedRankClient", _RankClient),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    async def test_a_preprint_first_paper_gets_its_venues_year(self) -> None:
        found = await paper_metadata.enrich_metadata("p1", paper_ir_json=_ir().model_dump_json())

        # S2 names the venue (the name the rank lists know); Crossref dates it.
        self.assertEqual(found["venue"], "Computer Vision and Pattern Recognition")
        self.assertEqual(found["year"], 2024)
        self.assertEqual(found["doi"], CVPR_DOI)
        self.assertEqual(found["ccf"], "A")
        self.crossref.assert_awaited_once_with(CVPR_DOI)
        self.assertEqual(self.written, [(CVPR_DOI, "Computer Vision and Pattern Recognition", 2024, "", "A", "p1")])

    async def test_without_a_crossref_record_the_s2_year_stands(self) -> None:
        self.crossref.return_value = {}
        found = await paper_metadata.enrich_metadata("p1", paper_ir_json=_ir().model_dump_json())
        self.assertEqual((found["year"], found["doi"]), (2023, CVPR_DOI))

    async def test_without_a_published_doi_nothing_more_is_asked(self) -> None:
        self.s2.return_value = {"venue": "Computer Vision and Pattern Recognition", "year": 2023, "doi": ""}
        found = await paper_metadata.enrich_metadata("p1", paper_ir_json=_ir().model_dump_json())
        self.assertEqual((found["year"], found["doi"]), (2023, ""))
        self.crossref.assert_not_awaited()

    async def test_a_doi_in_the_paper_keeps_the_crossref_first_path(self) -> None:
        ir = _ir()
        ir.blocks.append(Block(type="text", text="DOI: 10.1145/3292500.3330701"))
        found = await paper_metadata.enrich_metadata("p1", paper_ir_json=ir.model_dump_json())
        self.assertEqual(found["doi"], "10.1145/3292500.3330701")
        self.crossref.assert_awaited_once_with("10.1145/3292500.3330701")
        self.s2.assert_not_awaited()


class OwnIdentifierTests(unittest.IsolatedAsyncioTestCase):
    """A paper's DOI and arXiv id are its own, never one of its references'."""

    def _ir(self, *blocks: Block) -> str:
        return PaperIR(paper_id="p", blocks=list(blocks)).model_dump_json()

    async def test_the_first_page_stamp_is_read(self) -> None:
        ir = self._ir(Block(type="aside_text", text="arXiv:2404.04956v3 [cs.CV] 6 May 2024"))
        self.assertEqual(await paper_metadata._extract_arxiv_id_from_ir(ir), "2404.04956v3")

    async def test_a_references_identifier_is_not_the_papers(self) -> None:
        # The CVF copy of Gaussian Shading: no stamp, and reference [2] is an ICLR 2018 paper.
        ir = self._ir(
            Block(type="text", page_idx=0, text="Gaussian Shading", section_path="Gaussian Shading"),
            Block(type="ref_text", page_idx=8, section_path="Gaussian Shading/References",
                  text="[2] J. Balle et al. Variational image compression. arXiv:1802.01436, 2018."),
            Block(type="text", page_idx=9, section_path="Gaussian Shading/7. References",
                  text="[30] doi:10.1109/TIP.2004.833105"),
        )
        self.assertEqual(await paper_metadata._extract_arxiv_id_from_ir(ir), "")
        self.assertEqual(await paper_metadata._extract_doi_from_ir(ir), "")

    async def test_a_section_merely_ending_in_references_still_counts(self) -> None:
        ir = self._ir(Block(type="text", page_idx=5, section_path="P/3. Preferences",
                            text="See doi:10.1000/own.doi for the dataset."))
        self.assertEqual(await paper_metadata._extract_doi_from_ir(ir), "10.1000/own.doi")


class RefreshStoredRecordTests(unittest.IsolatedAsyncioTestCase):
    """Rows stored without a DOI are reused, never looked up again; this does it once."""

    async def asyncSetUp(self) -> None:
        from tests.pg_support import open_fresh_database

        await open_fresh_database(self)
        from app.db import database

        self.db = database
        await database.execute(
            "INSERT INTO papers (paper_id, file_path, title, doi, venue, year, sci_rank, ccf_rank) "
            "VALUES ('p1', 'x.pdf', 'SinSR', '', 'Computer Vision and Pattern Recognition', 2023, '', 'A'), "
            "('p2', 'y.pdf', 'Other', '10.1000/xyz', 'Some Journal', 2020, '', ''), "
            "('p3', 'z.pdf', 'Gaussian Shading', '', 'International Conference on Learning Representations', "
            "2018, '', 'A')"
        )
        await database.execute(
            "INSERT INTO paper_signals (paper_id, signals_json, cited_by_count, is_retracted) "
            "VALUES ('p1', '{}', 481, 0)"
        )
        self.ir = _ir()
        self.s2 = mock.AsyncMock(return_value=paper_metadata._parse_s2_response(S2_SINSR))
        for patcher in (
            mock.patch("app.services.paper_ir_loader.load_paper_ir",
                       mock.AsyncMock(side_effect=lambda _pid: self.ir)),
            mock.patch.object(paper_metadata, "_s2_lookup", self.s2),
            mock.patch.object(paper_metadata, "_crossref_lookup", mock.AsyncMock(return_value={"year": 2024})),
            mock.patch("app.services.publication_rank.UnifiedRankClient", _RankClient),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    async def _row(self, paper_id: str) -> dict[str, Any]:
        return await self.db.fetch_one("SELECT doi, venue, year FROM papers WHERE paper_id = ?", (paper_id,))

    async def test_a_dry_run_reports_and_writes_nothing(self) -> None:
        found = await paper_metadata.refresh_metadata("p1")
        self.assertEqual(found, {"doi": ("", CVPR_DOI), "year": (2023, 2024)})
        self.assertEqual((await self._row("p1"))["year"], 2023)

    async def test_applied_it_writes_the_record_and_drops_the_stale_signals(self) -> None:
        await paper_metadata.refresh_metadata("p1", apply=True)
        self.assertEqual(await self._row("p1"),
                         {"doi": CVPR_DOI, "venue": "Computer Vision and Pattern Recognition", "year": 2024})
        self.assertIsNone(await self.db.fetch_one("SELECT 1 FROM paper_signals WHERE paper_id = 'p1'"))

    async def test_the_files_work_takes_the_published_year(self) -> None:
        # Works keep the first year a source gave; the prompt lists the paper with it.
        from app.services import paper_catalog

        work = await paper_catalog.upsert_literature_item(
            title="SinSR", doi=CVPR_DOI.lower(), year=2023, source="search")
        await paper_catalog.link_paper_file(literature_id=work, paper_id="p1", origin="upload")
        other = await paper_catalog.upsert_literature_item(
            title="Other", doi="10.1000/another-version", year=2020, source="search")
        await paper_catalog.link_paper_file(literature_id=other, paper_id="p3", origin="upload")

        await paper_metadata.refresh_metadata("p1", apply=True)
        self.assertEqual((await paper_catalog.get_literature_item(work))["year"], 2024)

        # A work under another DOI is another version: its year is its own.
        await paper_catalog.record_published_year("p3", doi=CVPR_DOI, year=2024)
        self.assertEqual((await paper_catalog.get_literature_item(other))["year"], 2020)

    async def test_a_venue_read_from_a_reference_is_replaced(self) -> None:
        self.ir = PaperIR(paper_id="p3", title="Gaussian Shading", blocks=[
            Block(type="ref_text", page_idx=8, text="[2] Balle et al. arXiv:1802.01436"),
        ])
        found = await paper_metadata.refresh_metadata("p3", apply=True)
        self.assertEqual(found["venue"],
                         ("International Conference on Learning Representations",
                          "Computer Vision and Pattern Recognition"))
        # Asked by title: the reference's arXiv id is not the paper's.
        self.assertEqual(self.s2.await_args.kwargs["arxiv_id"], "")
        self.assertEqual(self.s2.await_args.kwargs["title"], "Gaussian Shading")

    async def test_an_unchanged_record_is_told_apart_from_a_failed_lookup(self) -> None:
        await self.db.execute("UPDATE papers SET doi = '', year = 2024 WHERE paper_id = 'p1'")
        self.assertEqual(await paper_metadata.refresh_metadata("p1", apply=True), {"doi": ("", CVPR_DOI)})
        self.ir = PaperIR(paper_id="p3", title="Gaussian Shading")
        await self.db.execute("UPDATE papers SET venue = 'Computer Vision and Pattern Recognition', "
                              "year = 2024 WHERE paper_id = 'p3'")
        self.s2.return_value = {"venue": "Computer Vision and Pattern Recognition", "year": 2024, "doi": ""}
        self.assertEqual(await paper_metadata.refresh_metadata("p3"), {})

    async def test_a_lookup_that_finds_nothing_leaves_the_row(self) -> None:
        self.s2.return_value = {}
        self.assertIsNone(await paper_metadata.refresh_metadata("p1", apply=True))
        self.assertEqual((await self._row("p1"))["year"], 2023)

    async def test_a_row_with_a_doi_is_left_alone(self) -> None:
        self.assertIsNone(await paper_metadata.refresh_metadata("p2", apply=True))
        self.assertIsNone(await paper_metadata.refresh_metadata("missing"))
        self.s2.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
