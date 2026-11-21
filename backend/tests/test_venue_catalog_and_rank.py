"""Unit tests for P17 authoritative venue catalog, normalizer guards, and UnifiedRankClient."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import httpx

from app.services.publication_rank import (
    EasyScholarClient,
    PublicationRankResult,
    RankCache,
    UnifiedRankClient,
    VenueLookupInput,
    classify_subtrack,
    clean_venue_text,
    extract_dblp_prefix,
    extract_issns,
    get_venue_catalog,
    is_preprint_venue,
    is_umbrella_series,
    normalize_venue_display,
    parse_openreview_status,
    resolve_venue_profiles_batch,
)
from app.services.publication_rank.llm_rank import _parse_llm_response
from app.services.sphere_scorer import compute_venue_component


class NormalizerAndGuardTests(unittest.TestCase):
    def test_html_entities_and_decorations_cleaned(self) -> None:
        self.assertEqual(
            clean_venue_text("Optics &amp; Laser Technology"),
            "Optics & Laser Technology",
        )
        self.assertEqual(
            normalize_venue_display("2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)"),
            "CVPR",
        )
        self.assertEqual(
            normalize_venue_display("Proceedings of the 34th ACM International Conference on Multimedia"),
            "ACM MM",
        )
        self.assertEqual(normalize_venue_display("ACL (1)"), "ACL")

    def test_workshop_and_companion_never_collapse_to_main_conference(self) -> None:
        for raw in (
            "IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)",
            "IEEE International Conference on Computer Vision Workshops",
            "CVPRW",
            "ECCV Workshops",
        ):
            sub, _ = classify_subtrack(raw)
            self.assertEqual(sub, "workshop", msg=raw)
            self.assertNotEqual(normalize_venue_display(raw), "CVPR")

        # Real conferences ending in W must NEVER be misclassified as workshops!
        for main_conf in ("WWW", "The Web Conference", "CSCW", "ICWSM"):
            sub, _ = classify_subtrack(main_conf)
            self.assertEqual(sub, "main", msg=main_conf)

        sub_comp, _ = classify_subtrack("Companion Proceedings of the ACM Web Conference 2024")
        self.assertEqual(sub_comp, "companion")

    def test_findings_isolated_from_main_conference(self) -> None:
        for raw in (
            "ACL (Findings)",
            "EMNLP 2024 (Findings)",
            "Findings of the Association for Computational Linguistics: ACL 2024",
            "Findings of EMNLP",
        ):
            sub, host = classify_subtrack(raw)
            self.assertEqual(sub, "findings", msg=raw)
            self.assertIn(host, {"ACL", "EMNLP"})

    def test_umbrella_series_and_issns_detected(self) -> None:
        self.assertTrue(is_umbrella_series("Lecture Notes in Computer Science"))
        self.assertTrue(is_umbrella_series("Proceedings of Machine Learning Research"))
        self.assertTrue(is_umbrella_series("CEUR Workshop Proceedings"))
        self.assertTrue(is_umbrella_series("AAAI Publications (The Association for the Advancement of AI)"))
        self.assertFalse(is_umbrella_series("IEEE Transactions on Pattern Analysis and Machine Intelligence"))
        # LNCS print/eISSN must be stripped so they never match as standalone journals
        self.assertEqual(extract_issns(["0302-9743", "1611-3349", "0162-8828"]), ["0162-8828"])

    def test_dblp_prefix_ignores_corr_preprint(self) -> None:
        self.assertEqual(extract_dblp_prefix("conf/eccv/HeZRS16"), "conf/eccv")
        self.assertEqual(extract_dblp_prefix("https://dblp.org/rec/journals/pami/He20"), "journals/pami")
        self.assertEqual(extract_dblp_prefix("journals/corr/abs-2401-00001"), "")

    def test_openreview_status_distinguishes_accepted_from_submitted(self) -> None:
        venue, sub, tier = parse_openreview_status("ICLR 2025 Oral")
        self.assertEqual((venue, sub, tier), ("ICLR", "main", "oral"))

        venue2, sub2, _ = parse_openreview_status("Submitted to ICLR 2025")
        self.assertEqual((venue2, sub2), ("", "preprint"))
        self.assertTrue(is_preprint_venue("ACL ARR 2025 February Submission"))

        venue3, sub3, _ = parse_openreview_status("NeurIPS 2024 Datasets and Benchmarks Track")
        self.assertEqual((venue3, sub3), ("NeurIPS", "main"))

        _, sub_tiny, _ = parse_openreview_status("ICLR 2024 Tiny Papers")
        self.assertEqual(sub_tiny, "workshop")

    def test_zero_width_and_soft_hyphen_pdf_artifacts_stripped(self) -> None:
        self.assertEqual(clean_venue_text("CV\u200bPR\ufeff 2024"), "CVPR 2024")
        self.assertEqual(
            clean_venue_text("IEEE Trans\u00adactions on Software Engineering"),
            "IEEE Transactions on Software Engineering",
        )
        self.assertEqual(extract_dblp_prefix("http://dblp2.uni-trier.de/db/conf/vrst/"), "conf/vrst")


class AuthoritativeCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = get_venue_catalog()

    def test_catalog_loaded_comprehensive_dataset(self) -> None:
        self.assertGreater(self.catalog.size, 20_000)

    def test_exact_distinction_between_similar_journal_titles(self) -> None:
        pr = self.catalog.resolve("Pattern Recognition")
        prl = self.catalog.resolve("Pattern Recognition Letters")

        self.assertEqual(pr.ccf_rank, "B")
        self.assertEqual(pr.sci_rank, "Q1")
        self.assertEqual(prl.ccf_rank, "C")
        self.assertNotEqual(pr.canonical_name, prl.canonical_name)

    def test_springer_lncs_unmasked_via_dblp_key(self) -> None:
        prof = self.catalog.resolve(
            VenueLookupInput(
                venue="Lecture Notes in Computer Science",
                dblp_key="conf/eccv/SomePaper24",
                issns=["0302-9743"],
            )
        )
        self.assertEqual(prof.acronym, "ECCV")
        self.assertEqual(prof.ccf_rank, "B")
        self.assertEqual(prof.core_rank, "A*")
        self.assertEqual(prof.verification_status, "verified")

    def test_accepted_arxiv_paper_surfaces_formal_conference_tier_and_keeps_file_flag(self) -> None:
        prof = self.catalog.resolve(
            VenueLookupInput(
                venue="arXiv",
                openreview_status="ICLR 2025 Spotlight",
                is_preprint_source=True,
            )
        )
        self.assertEqual(prof.acronym, "ICLR")
        self.assertEqual(prof.ccf_rank, "A")
        self.assertEqual(prof.core_rank, "A*")
        self.assertEqual(prof.openreview_tier, "spotlight")
        self.assertTrue(prof.is_preprint_file)
        self.assertIn("ICLR", prof.prompt_badge("zh"))
        self.assertIn("预印本文件", prof.prompt_badge("zh"))

    def test_unaccepted_openreview_stays_preprint(self) -> None:
        prof = self.catalog.resolve(
            VenueLookupInput(
                venue="arXiv",
                openreview_status="Submitted to ICLR 2025",
                is_preprint_source=True,
            )
        )
        self.assertEqual(prof.subtrack, "preprint")
        self.assertEqual(prof.rankings, [])
        self.assertFalse(prof.peer_reviewed)

    def test_workshop_and_findings_have_no_inherited_ccf_rank(self) -> None:
        ws = self.catalog.resolve("IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)")
        self.assertEqual(ws.subtrack, "workshop")
        self.assertEqual(ws.rankings, [])

        fd = self.catalog.resolve("ACL (Findings)")
        self.assertEqual(fd.subtrack, "findings")
        self.assertEqual(fd.rankings, [])
        self.assertTrue(fd.peer_reviewed)

    def test_html_escaped_journal_and_chinese_ccf_t1_and_tmlr_and_warning(self) -> None:
        opt = self.catalog.resolve("Optics &amp; Laser Technology")
        self.assertEqual(opt.sci_rank, "Q1")
        self.assertEqual(opt.verification_status, "verified")

        jos = self.catalog.resolve("软件学报")
        self.assertEqual(jos.compact_ranks().get("CCF_CN"), "T1 (2025)")

        tmlr = self.catalog.resolve("Transactions on machine learning research")
        self.assertEqual(tmlr.acronym, "TMLR")
        self.assertIn("OpenReview", tmlr.note)

        warn_j = self.catalog.resolve("Computers & Electrical Engineering")
        self.assertIn("预警", warn_j.warning)
        # Warning journal gets heavily penalized in venue quality score even without explicit cas_warning kwarg
        self.assertLessEqual(compute_venue_component("Computers & Electrical Engineering"), 0.15)

    def test_standalone_workshop_named_conferences_keep_ccf_and_core_ranks(self) -> None:
        for venue, want_ccf in (
            ("IEEE/ACM International Workshop on Quality of Service", "B"),
            ("USENIX Workshop on Hot Topics in Operating Systems", "B"),
            ("International Workshop on Practice and Theory in Public Key Cryptography", "B"),
            ("ACM The Workshop on Hot Topics in Networks", "C"),
        ):
            prof = self.catalog.resolve(venue)
            self.assertEqual(prof.subtrack, "main", msg=venue)
            self.assertEqual(prof.ccf_rank, want_ccf, msg=venue)

        # JMLR W&CP umbrella series must NOT block conf/icml unmasking despite "Workshop" in title
        icml_2015 = self.catalog.resolve(
            VenueLookupInput(venue="JMLR Workshop and Conference Proceedings", dblp_key="conf/icml/2015")
        )
        self.assertEqual(icml_2015.subtrack, "main")
        self.assertEqual(icml_2015.ccf_rank, "A")
        self.assertEqual(icml_2015.acronym, "ICML")

    def test_regional_and_modified_conferences_never_hijack_flagships(self) -> None:
        cases = (
            ("AsiaCCS '24: Proceedings of the 19th ACM Asia Conference on Computer and Communications Security", "C"),
            ("CCS '24: Proceedings of the 2024 ACM SIGSAC Conference on Computer and Communications Security", "A"),
            ("EuroS&P 2024: IEEE European Symposium on Security and Privacy", "C"),
            ("PAKDD 2024: Pacific-Asia Conference on Knowledge Discovery and Data Mining", "C"),
            ("ICMR '24: Proceedings of the 2024 International Conference on Multimedia Retrieval", "B"),
            ("IEEE International Symposium on Circuits and Systems", "B"),
            ("The International ACM Symposium on High-Performance Parallel and Distributed Computing", "A"),
            ("International Joint Conference on Artificial Intelligence", "B"),  # CCF 2026 7th edition
        )
        for query, want_ccf in cases:
            prof = self.catalog.resolve(query)
            self.assertEqual(prof.ccf_rank, want_ccf, msg=f"{query} -> {prof.canonical_name} ({prof.compact_ranks()})")

        iswc = self.catalog.resolve("International Semantic Web Conference")
        self.assertNotEqual(iswc.acronym, "WWW")
        self.assertEqual(iswc.ccf_rank, "B")

    def test_iso4_journal_abbreviations_and_pacm_issues_and_short_acronyms(self) -> None:
        iso4_cases = (
            ("IEEE Trans. Knowl. Data Eng.", "CCF", "A"),
            ("IEEE Trans. Softw. Eng.", "CCF", "A"),
            ("IEEE Trans. Dependable Secure Comput.", "CCF", "A"),
            ("IEEE Trans. Inf. Forensics Secur.", "CCF", "A"),
            ("IEEE/ACM Trans. Netw.", "CCF", "A"),
            ("IEEE/ACM Trans. Audio Speech Lang. Process.", "CCF", "B"),
            ("ACM Trans. Graph.", "CCF", "A"),
            ("ACM Trans. Softw. Eng. Methodol.", "CCF", "A"),
            ("Nat. Mach. Intell.", "JCR", "Q1"),
            ("Proc. Natl. Acad. Sci. U.S.A.", "JCR", "Q1"),
            ("Proceedings of the ACM on Programming Languages (POPL)", "CCF", "A"),
            ("Proceedings of the ACM on Programming Languages (OOPSLA)", "CCF", "A"),
            ("SC24: International Conference for High Performance Computing, Networking, Storage and Analysis", "CCF", "A"),
            ("ACM SIGGRAPH 2024 Conference Papers", "CCF", "A"),
            ("AI", "CCF", "A"),
            ("IOT", "CCF", "C"),
            ("KR", "CCF", "B"),
            ("SIGSPATIAL", "CCF", "C"),
        )
        for query, sys_name, want_rank in iso4_cases:
            prof = self.catalog.resolve(query)
            ranks = {r.ranking_system: r.rank for r in prof.rankings}
            self.assertEqual(ranks.get(sys_name), want_rank, msg=f"{query} -> {ranks}")

    def test_sphere_scorer_avoids_workshop_and_substring_false_positives(self) -> None:
        self.assertAlmostEqual(compute_venue_component("CVPRW 2024"), 0.20)
        self.assertAlmostEqual(compute_venue_component("ACL (Findings)"), 0.65)
        self.assertAlmostEqual(compute_venue_component("CVPR 2024"), 1.00)
        self.assertAlmostEqual(compute_venue_component("Robotics: Science and Systems"), 1.00)
        self.assertAlmostEqual(compute_venue_component("软件学报"), 0.85)
        # Contains 'acl' substring ("Oracle") — must NOT match ACL (0.90)!
        self.assertLessEqual(compute_venue_component("Oracle Bone Inscriptions Journal"), 0.55)


class CachePriorityAndAsyncClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_cache_source_priority_prevents_llm_overwriting_easyscholar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = RankCache(db_path=Path(tmp) / "test.sqlite3")
            await cache.init()
            try:
                await cache.put(
                    PublicationRankResult(name="CustomJ", sci="Q1", ccf="A", success=True),
                    source="easyscholar",
                )
                # Lower-priority llm_websearch or failure must NOT overwrite verified easyscholar
                await cache.put(
                    PublicationRankResult(name="CustomJ", sci="Q3", ccf=None, success=True),
                    source="llm_websearch",
                )
                await cache.put(
                    PublicationRankResult(name="CustomJ", success=False, error="timeout"),
                    source="none",
                )
                got = await cache.get("CustomJ")
                assert got is not None
                self.assertEqual(got.sci, "Q1")
                self.assertEqual(got.ccf, "A")
                self.assertEqual(got.source, "easyscholar")
            finally:
                await cache.close()

    async def test_async_easyscholar_prefers_issn_and_deduplicates_inflight(self) -> None:
        calls: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            import asyncio

            pub = request.url.params.get("publicationName", "")
            calls.append(pub)
            await asyncio.sleep(0.02)
            return httpx.Response(
                200,
                json={
                    "code": 200,
                    "msg": "success",
                    "data": {"officialRank": {"all": {"sci": "Q1", "sciif": "9.8", "sciUp": "计算机科学1区"}}},
                },
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http:
            es = EasyScholarClient(secret_key="test-key", api_url="https://es.test/rank")
            with mock.patch("app.services.publication_rank.publication_rank._GLOBAL_ASYNC_ES_LIMITER.wait", new=mock.AsyncMock()):
                res1, res2 = await self._gather2(
                    es.aquery("Abbrev J. Name", issns=["1234-567X"], client=http),
                    es.aquery("Abbrev J. Name", issns=["1234-567X"], client=http),
                )
            es.close()

        self.assertTrue(res1.success)
        self.assertEqual(res1.sci, "Q1")
        self.assertEqual(res2.sci, "Q1")
        # Queried by ISSN first and deduplicated concurrent inflight calls into 1 HTTP request
        self.assertEqual(calls, ["1234-567X"])

    @staticmethod
    async def _gather2(a: Any, b: Any) -> Any:
        import asyncio

        return await asyncio.gather(a, b)

    def test_llm_parser_keeps_jcr_cas_ccf_core_separate(self) -> None:
        res = _parse_llm_response(
            '```json\n{"jcr": "Q2", "cas": "计算机3区", "ccf": "C", "core": "B"}\n```',
            "Journal X",
        )
        prof = res.to_venue_profile()
        systems = {r.ranking_system: r.rank for r in prof.rankings}
        self.assertEqual(systems, {"JCR": "Q2", "CAS": "3区", "CCF": "C", "CORE": "B"})
        self.assertEqual(prof.verification_status, "unverified")

        res_null = _parse_llm_response('{"jcr": null, "cas": null, "ccf": null, "core": null}', "Unknown Outlet")
        self.assertFalse(res_null.success)

    async def test_batch_profile_resolution_is_fast_and_offline_by_default(self) -> None:
        profiles = await resolve_venue_profiles_batch(
            ["CVPR", "Pattern Recognition", "ACL (Findings)", "arXiv"],
            allow_network=False,
        )
        self.assertEqual([p.subtrack for p in profiles], ["main", "main", "findings", "preprint"])
        self.assertEqual(profiles[0].ccf_rank, "A")
        self.assertEqual(profiles[1].ccf_rank, "B")


class PostgresRankCacheIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_postgres_l2_cache_monotonicity_issn_and_batch(self) -> None:
        try:
            from tests.pg_support import open_fresh_database
        except ImportError:
            from pg_support import open_fresh_database

        await open_fresh_database(self)
        cache = RankCache()
        await cache.init()
        self.assertTrue(cache._use_pg)

        # 1) Store verified EasyScholar entry with ISSN
        await cache.put(
            PublicationRankResult(
                name="Journal of Custom Systems",
                sci="Q1",
                ccf="B",
                success=True,
                extra={"sciif": "7.4"},
                source="easyscholar",
            ),
            source="easyscholar",
            issn="2099-123X",
        )
        # 2) Attempt lower-priority LLM overwrite and transient failure overwrite -> must be rejected by PG WHERE guard
        await cache.put(
            PublicationRankResult(
                name="Journal of Custom Systems",
                sci="Q4",
                ccf="C",
                success=True,
                source="llm_websearch",
            ),
            source="llm_websearch",
            issn="2099-123X",
        )
        await cache.put(
            PublicationRankResult(name="Dead Outlet", success=False, error="not indexed"),
            source="none",
        )

        by_issn = await cache.get("Abbreviated JCS", issn="2099123x")
        assert by_issn is not None
        self.assertEqual(by_issn.sci, "Q1")
        self.assertEqual(by_issn.ccf, "B")
        self.assertEqual(by_issn.source, "easyscholar")
        self.assertEqual(by_issn.extra.get("sciif"), "7.4")

        batch = await cache.get_batch(["Journal of Custom Systems", "Missing Venue"])
        assert batch["Journal of Custom Systems"] is not None
        self.assertEqual(batch["Journal of Custom Systems"].ccf, "B")
        self.assertIsNone(batch["Missing Venue"])

        self.assertEqual(await cache.count(only_failures=True), 1)
        deleted_failures = await cache.clear(only_failures=True)
        self.assertEqual(deleted_failures, 1)
        self.assertIsNotNone(await cache.get("Journal of Custom Systems"))


if __name__ == "__main__":
    unittest.main()
