from __future__ import annotations

import json
import unittest
import unittest.mock
from unittest.mock import AsyncMock, patch

from app.models.paper_ir import Block, PaperIR
from app.models.snap_models import CodeRepo, RepoHost, TriageSignals
from app.services.triage_signals import (
    collect_triage_signals,
    compute_external_score,
    extract_code_repos,
    score_code_availability,
)
from app.services.triage_verdict import (
    TIER_MUST_READ,
    TIER_SELECTIVE,
    TIER_SKIP,
    ContentReview,
    parse_content_review,
    render_verdict_markdown,
    synthesize_verdict,
)


def _blocks(*specs: tuple[int, str, str]) -> PaperIR:
    blocks = [
        Block(type="text", page_idx=page, text=text, section_path=section)
        for page, text, section in specs
    ]
    return PaperIR(paper_id="p", title="FooNet", blocks=blocks)


def _good_review(**overrides) -> ContentReview:
    payload = {
        "contribution_strength": 2,
        "evidence_strength": 3,
        "novelty": 1,
        "reproducibility": 2,
        "limitation_severity": 1,
        "reasons": ["Solid ablation [p.7]"],
    }
    payload.update(overrides)
    return parse_content_review(json.dumps(payload))


def _strong_signals(**overrides) -> TriageSignals:
    base = dict(
        venue="Advances in Neural Information Processing Systems",
        ccf_rank="A",
        year=2023,
        cited_by_count=342,
        influential_citation_count=28,
        is_open_access=True,
        resolved=True,
        repos=[
            CodeRepo(
                url="https://github.com/acme/foonet",
                host=RepoHost.GITHUB,
                slug="acme/foonet",
                is_official=True,
                stars=1200,
                probe_ok=True,
                evidence_page=1,
            )
        ],
        provenance={"ccf_rank": "EasyScholar", "citations": "OpenAlex"},
    )
    base.update(overrides)
    return compute_external_score(TriageSignals(**base))


class TestCodeRepoExtraction(unittest.TestCase):
    def test_ownership_cue_marks_repo_official(self) -> None:
        ir = _blocks((3, "Our code is available at https://github.com/acme/foonet.", "5 Conclusion"))
        repos = extract_code_repos(ir)
        self.assertEqual(len(repos), 1)
        self.assertTrue(repos[0].is_official)
        self.assertEqual(repos[0].slug, "acme/foonet")
        self.assertEqual(repos[0].evidence_page, 4)

    def test_first_page_link_is_official_without_a_cue(self) -> None:
        ir = _blocks((0, "FooNet. https://github.com/acme/foonet", "Abstract"))
        self.assertTrue(extract_code_repos(ir)[0].is_official)

    def test_baseline_link_in_related_work_is_not_official(self) -> None:
        ir = _blocks((2, "We compare with BERT (https://github.com/google-research/bert).", "2 Related Work"))
        repos = extract_code_repos(ir)
        self.assertEqual(len(repos), 1)
        self.assertFalse(repos[0].is_official)

    def test_bibliography_links_are_never_official(self) -> None:
        """An ownership cue inside References is still somebody else's repo."""
        ir = _blocks((9, "[12] Devlin et al. Source code available at https://github.com/g/bert", "References"))
        self.assertFalse(extract_code_repos(ir)[0].is_official)

    def test_deep_links_collapse_to_owner_repo_and_dedupe(self) -> None:
        ir = _blocks(
            (0, "Code: https://github.com/acme/foonet/tree/main/src", "Abstract"),
            (4, "See also https://github.com/acme/foonet.git for details.", "4 Experiments"),
        )
        repos = extract_code_repos(ir)
        self.assertEqual([r.url for r in repos], ["https://github.com/acme/foonet"])

    def test_github_non_repo_paths_are_rejected(self) -> None:
        ir = _blocks((3, "Built with https://github.com/features/actions for CI.", "4 Setup"))
        self.assertEqual(extract_code_repos(ir), [])

    def test_bare_domain_without_a_repo_path_is_rejected(self) -> None:
        ir = _blocks((0, "Hosted on github.com and elsewhere.", "Abstract"))
        self.assertEqual(extract_code_repos(ir), [])

    def test_artifact_hosts_are_recognized(self) -> None:
        ir = _blocks((
            4,
            "Weights released at https://huggingface.co/acme/foonet and data at https://zenodo.org/record/123456.",
            "4 Experiments",
        ))
        hosts = {r.host for r in extract_code_repos(ir)}
        self.assertEqual(hosts, {RepoHost.HUGGINGFACE, RepoHost.ZENODO})

    def test_project_page_is_captured(self) -> None:
        ir = _blocks((1, "Project page: https://acme.github.io/foonet/", "1 Introduction"))
        repos = extract_code_repos(ir)
        self.assertEqual(repos[0].host, RepoHost.PROJECT_PAGE)
        self.assertTrue(repos[0].is_official)

    def test_official_evidence_upgrades_an_earlier_third_party_sighting(self) -> None:
        ir = _blocks(
            (2, "Prior work used https://github.com/acme/foonet as a baseline.", "2 Related Work"),
            (8, "We release our code at https://github.com/acme/foonet.", "6 Conclusion"),
        )
        repos = extract_code_repos(ir)
        self.assertEqual(len(repos), 1)
        self.assertTrue(repos[0].is_official)
        self.assertEqual(repos[0].evidence_page, 9)


class TestExternalScoring(unittest.TestCase):
    def test_unknown_citations_are_not_scored_as_zero(self) -> None:
        """The whole point: 'unknown' must not read as 'never cited'."""
        unknown = compute_external_score(
            TriageSignals(venue="Nature Methods", sci_rank="Q1", year=2024, unavailable=["citations"])
        )
        genuinely_zero = compute_external_score(
            TriageSignals(venue="Nature Methods", sci_rank="Q1", year=2024, cited_by_count=0)
        )
        self.assertGreater(unknown.external_score, genuinely_zero.external_score)
        self.assertFalse(unknown.citations_known)

    def test_official_repo_outscores_third_party_links(self) -> None:
        official = TriageSignals(repos=[CodeRepo(url="u", is_official=True)])
        third_party = TriageSignals(repos=[CodeRepo(url="u", is_official=False)])
        self.assertGreater(score_code_availability(official), score_code_availability(third_party))
        self.assertEqual(score_code_availability(TriageSignals()), 0.0)

    def test_popular_probed_repo_scores_above_an_unprobed_one(self) -> None:
        unprobed = TriageSignals(repos=[CodeRepo(url="u", is_official=True)])
        popular = TriageSignals(
            repos=[CodeRepo(url="u", is_official=True, stars=1500, probe_ok=True)]
        )
        self.assertGreater(score_code_availability(popular), score_code_availability(unprobed))

    def test_conference_name_is_normalized_for_display(self) -> None:
        signals = _strong_signals()
        self.assertEqual(signals.venue_normalized, "NeurIPS")

    def test_preprint_venue_is_flagged(self) -> None:
        signals = compute_external_score(TriageSignals(venue="arXiv preprint", year=2026))
        self.assertTrue(signals.is_preprint)
        self.assertLess(signals.venue_score, 0.3)


class TestContentReviewParsing(unittest.TestCase):
    def test_fenced_json_is_parsed(self) -> None:
        review = parse_content_review('```json\n{"contribution_strength": 3}\n```')
        self.assertTrue(review.available)
        self.assertEqual(review.contribution_strength, 3)

    def test_json_wrapped_in_prose_is_salvaged(self) -> None:
        review = parse_content_review('Here is my review: {"novelty": 2} Hope that helps.')
        self.assertTrue(review.available)
        self.assertEqual(review.novelty, 2)

    def test_non_json_yields_an_unavailable_review(self) -> None:
        review = parse_content_review("I think this paper is quite good.")
        self.assertFalse(review.available)

    def test_out_of_range_scores_are_clamped(self) -> None:
        review = parse_content_review('{"contribution_strength": 9, "novelty": -4, "evidence_strength": "x"}')
        self.assertEqual(review.contribution_strength, 3)
        self.assertEqual(review.novelty, 0)
        self.assertEqual(review.evidence_strength, 1)  # unparseable → neutral

    def test_string_shaped_must_read_sections_are_accepted(self) -> None:
        review = parse_content_review('{"must_read_sections": ["§4.3 Ablation [p.7]"]}')
        self.assertEqual(review.must_read_sections[0]["where"], "§4.3 Ablation [p.7]")

    def test_severe_limitations_lower_the_content_score(self) -> None:
        mild = _good_review(limitation_severity=0)
        severe = _good_review(limitation_severity=3)
        self.assertGreater(mild.content_score, severe.content_score)


class TestVerdictSynthesis(unittest.TestCase):
    def test_strong_paper_reaches_must_read(self) -> None:
        verdict = synthesize_verdict(_good_review(), _strong_signals())
        self.assertEqual(verdict.tier, TIER_MUST_READ)
        self.assertIn("venue_rank", verdict.drivers)
        self.assertIn("citation_impact", verdict.drivers)
        self.assertIn("official_code", verdict.drivers)

    def test_retraction_forces_skip_regardless_of_everything_else(self) -> None:
        verdict = synthesize_verdict(_good_review(), _strong_signals(is_retracted=True))
        self.assertEqual(verdict.tier, TIER_SKIP)
        self.assertIn("retracted", verdict.overrides)

    def test_missing_content_review_cannot_produce_must_read(self) -> None:
        """Venue + citations alone must never claim a paper is a must-read."""
        verdict = synthesize_verdict(ContentReview(), _strong_signals())
        self.assertEqual(verdict.tier, TIER_SELECTIVE)
        self.assertIn("no_content_review", verdict.overrides)
        self.assertFalse(verdict.content_available)

    def test_strong_content_survives_an_empty_external_record(self) -> None:
        """A good paper published last month has no citations yet."""
        thin = compute_external_score(
            TriageSignals(venue="arXiv", year=2026, unavailable=["citations", "open_access", "retraction"])
        )
        strong = _good_review(
            contribution_strength=3, evidence_strength=3, novelty=3, limitation_severity=0
        )
        self.assertNotEqual(synthesize_verdict(strong, thin).tier, TIER_SKIP)

    def test_weak_paper_in_a_weak_venue_is_skippable(self) -> None:
        weak = _good_review(
            contribution_strength=0, evidence_strength=0, novelty=0,
            reproducibility=0, limitation_severity=3,
        )
        thin = compute_external_score(TriageSignals(venue="arXiv", year=2025, cited_by_count=0))
        verdict = synthesize_verdict(weak, thin)
        self.assertEqual(verdict.tier, TIER_SKIP)
        self.assertIn("weak_evidence", verdict.drivers)


class TestVerdictRendering(unittest.TestCase):
    def test_evidence_table_cites_a_source_for_each_signal(self) -> None:
        signals = _strong_signals()
        md = render_verdict_markdown(synthesize_verdict(_good_review(), signals), _good_review(), signals)
        self.assertIn("| Citation impact | 342 citations (85.5/yr) | OpenAlex |", md)
        self.assertIn("EasyScholar", md)
        self.assertIn("github.com/acme/foonet", md)
        self.assertIn("1.2k★", md)

    def test_unresolved_paper_reports_unknown_not_zero(self) -> None:
        signals = compute_external_score(
            TriageSignals(unavailable=["citations", "open_access", "retraction"])
        )
        md = render_verdict_markdown(synthesize_verdict(_good_review(), signals), _good_review(), signals)
        self.assertIn("unknown", md)
        self.assertIn("could not be matched", md)
        self.assertNotIn("0 citations", md)

    def test_retraction_is_surfaced_as_a_banner(self) -> None:
        signals = _strong_signals(is_retracted=True)
        md = render_verdict_markdown(synthesize_verdict(_good_review(), signals), _good_review(), signals)
        self.assertIn("RETRACTED", md)
        self.assertIn("Can skip", md)

    def test_missing_review_is_disclosed_in_the_output(self) -> None:
        signals = _strong_signals()
        empty = ContentReview()
        md = render_verdict_markdown(synthesize_verdict(empty, signals), empty, signals)
        self.assertIn("content review did not complete", md)

    def test_chinese_rendering_uses_chinese_labels(self) -> None:
        signals = _strong_signals()
        review = _good_review()
        md = render_verdict_markdown(synthesize_verdict(review, signals), review, signals, language="zh")
        self.assertIn("## 分诊结论:必读", md)
        self.assertIn("引用影响", md)
        self.assertIn("官方代码", md)
        # Page citations and venue names must stay in their original form.
        self.assertIn("NeurIPS", md)

    def test_selective_verdict_names_the_section_to_read(self) -> None:
        signals = compute_external_score(TriageSignals(venue="arXiv", year=2025, cited_by_count=4))
        review = _good_review(
            must_read_sections=[{"where": "§4.3 Ablation [p.7]", "why": "isolates A vs B"}]
        )
        verdict = synthesize_verdict(review, signals)
        md = render_verdict_markdown(verdict, review, signals)
        self.assertEqual(verdict.tier, TIER_SELECTIVE)
        self.assertIn("§4.3 Ablation [p.7]", md.splitlines()[0])


class TestSignalCollection(unittest.IsolatedAsyncioTestCase):
    async def test_signals_merge_both_indexes_and_reuse_pub_rank(self) -> None:
        openalex = {
            "id": "https://openalex.org/W42",
            "cited_by_count": 300,
            "publication_year": 2023,
            "is_retracted": False,
            "open_access": {"is_oa": True, "oa_url": "https://x/pdf"},
            "referenced_works": ["W1", "W2"],
        }
        s2 = {
            "paperId": "abc",
            "citationCount": 342,  # higher — must win
            "influentialCitationCount": 28,
            "publicationTypes": ["JournalArticle"],
            "externalIds": {"DOI": "10.1/x"},
        }
        ir = _blocks((0, "Our code: https://github.com/acme/foonet", "Abstract"))
        with patch(
            "app.services.triage_signals._openalex_triage_lookup",
            AsyncMock(return_value=openalex),
        ), patch(
            "app.services.triage_signals._s2_triage_lookup", AsyncMock(return_value=s2)
        ):
            signals = await collect_triage_signals(
                ir, pub_rank={"venue": "NeurIPS", "year": 2023, "ccf": "A"}, doi="10.1/x"
            )

        self.assertTrue(signals.resolved)
        self.assertEqual(signals.cited_by_count, 342)
        self.assertEqual(signals.provenance["citations"], "Semantic Scholar")
        self.assertEqual(signals.influential_citation_count, 28)
        self.assertEqual(signals.ccf_rank, "A")
        self.assertEqual(signals.provenance["ccf_rank"], "EasyScholar")
        self.assertTrue(signals.is_open_access)
        self.assertTrue(signals.has_official_code)
        self.assertGreater(signals.external_score, 0.5)

    async def test_total_lookup_failure_marks_fields_unknown(self) -> None:
        ir = _blocks((0, "No links here.", "Abstract"))
        with patch(
            "app.services.triage_signals._openalex_triage_lookup",
            AsyncMock(side_effect=RuntimeError("network down")),
        ), patch(
            "app.services.triage_signals._s2_triage_lookup", AsyncMock(return_value=None)
        ):
            signals = await collect_triage_signals(ir, pub_rank={"venue": "ICML", "ccf": "A"})

        self.assertFalse(signals.resolved)
        self.assertFalse(signals.citations_known)
        self.assertIn("open_access", signals.unavailable)
        # Venue rank from pub_rank still survives.
        self.assertEqual(signals.ccf_rank, "A")
        self.assertGreater(signals.venue_score, 0.9)

    async def test_network_can_be_disabled_entirely(self) -> None:
        ir = _blocks((0, "Our code: https://github.com/acme/foonet", "Abstract"))
        with patch(
            "app.services.triage_signals._openalex_triage_lookup",
            AsyncMock(side_effect=AssertionError("must not be called")),
        ):
            signals = await collect_triage_signals(ir, pub_rank={}, enable_network=False)
        self.assertFalse(signals.citations_known)
        self.assertTrue(signals.has_official_code)  # PDF-derived, no network needed

    async def test_retraction_from_openalex_is_recorded(self) -> None:
        ir = _blocks((0, "Abstract text.", "Abstract"))
        with patch(
            "app.services.triage_signals._openalex_triage_lookup",
            AsyncMock(return_value={"id": "W1", "is_retracted": True, "cited_by_count": 90}),
        ), patch(
            "app.services.triage_signals._s2_triage_lookup", AsyncMock(return_value=None)
        ):
            signals = await collect_triage_signals(ir, pub_rank={"venue": "Some Journal"})
        self.assertTrue(signals.is_retracted)
        self.assertEqual(signals.provenance["retraction"], "OpenAlex")


if __name__ == "__main__":
    unittest.main()


class TestSnapOrchestration(unittest.IsolatedAsyncioTestCase):
    """The report call, the blind content review, and the signal lookup must all
    run, and the verdict must be appended by us rather than written by the model."""

    def _state(self, **overrides) -> dict:
        ir = PaperIR(
            paper_id="p",
            title="FooNet",
            sections=[],
            blocks=[
                Block(type="text", page_idx=0, text="We propose FooNet.", section_path="Abstract"),
                Block(
                    type="table",
                    page_idx=7,
                    text="Table 2: Results.\n<table><tr><td>Ours</td><td>28.4</td></tr></table>",
                    section_path="4 Experiments",
                ),
            ],
        )
        state = {
            "paper_id": "p",
            "run_id": "r",
            "paper_ir_json": ir.model_dump_json(),
            "pub_rank_json": json.dumps({"venue": "NeurIPS", "year": 2023, "ccf": "A", "doi": "10.1/x"}),
            "language": "en",
            "llm_model": "",
        }
        state.update(overrides)
        return state

    async def test_verdict_is_appended_and_signals_are_persisted(self) -> None:
        from app.workflows import snap_subgraph

        calls: list[str] = []

        async def fake_chat(messages, **kwargs):
            system = messages[0]["content"]
            if "demanding peer reviewer" in system:
                calls.append("review")
                # The review must never be shown the external evidence.
                user = messages[1]["content"]
                assert "CCF" not in user and "Published in" not in user, user[:200]
                return json.dumps({
                    "contribution_strength": 2, "evidence_strength": 3, "novelty": 2,
                    "reproducibility": 2, "limitation_severity": 1,
                    "reasons": ["Thorough ablation [p.7]"],
                })
            calls.append("report")
            return json.dumps({
                "one_liner": "FooNet improves BLEU by 2.1 on WMT14.",
                "contributions": [{"text": "A new attention variant", "page": 3}],
                "findings": [{
                    "metric": "BLEU", "dataset": "WMT14 EN-DE", "value": "28.4",
                    "baseline": "26.3", "delta": "+2.1", "page": 7, "note": "single seed",
                }],
                "suitable_for": "Sequence-to-sequence tasks with long inputs.",
                "limitations": [{"text": "Only tested on translation", "page": 9}],
            })

        llm = unittest.mock.MagicMock()
        llm.chat = AsyncMock(side_effect=fake_chat)

        with patch.object(snap_subgraph, "get_llm_service", return_value=llm), patch.object(
            snap_subgraph,
            "get_or_collect_triage_signals",
            AsyncMock(return_value=_strong_signals()),
        ):
            result = await snap_subgraph.run_insight_snap(self._state())

        self.assertEqual(sorted(calls), ["report", "review"])
        markdown = result["final_markdown"]
        self.assertIn("One-Sentence Summary", markdown)
        # Findings render as a table with the metric, the value and the delta.
        self.assertIn("| BLEU | WMT14 EN-DE | 28.4 | 26.3 | +2.1 | [p.7] |", markdown)
        self.assertIn("## Triage Verdict", markdown)
        self.assertIn("OpenAlex", markdown)

        payload = json.loads(result["final_json"])
        self.assertEqual(payload["verdict"]["tier"], TIER_MUST_READ)
        self.assertEqual(payload["signals"]["cited_by_count"], 342)
        self.assertTrue(payload["content_review"]["available"])
        self.assertFalse(payload["report"]["degraded"])
        self.assertEqual(payload["report"]["findings"][0]["metric"], "BLEU")
        # P0: the results table must have reached the context.
        self.assertGreaterEqual(payload["context_stats"]["tables"], 1)
        # P0-2: the evidence-pool call is gone.
        self.assertNotIn("evidence_pool", payload)

    async def test_report_failure_propagates_but_signal_failure_does_not(self) -> None:
        """Signals are corroboration; losing them must not lose the run."""
        from app.workflows import snap_subgraph

        llm = unittest.mock.MagicMock()
        llm.chat = AsyncMock(return_value=json.dumps({
            "one_liner": "A paper.", "contribution_strength": 2,
            "contributions": [{"text": "Something", "page": 2}],
        }))

        with patch.object(snap_subgraph, "get_llm_service", return_value=llm), patch.object(
            snap_subgraph,
            "get_or_collect_triage_signals",
            AsyncMock(side_effect=RuntimeError("all indexes down")),
        ):
            result = await snap_subgraph.run_insight_snap(self._state())

        payload = json.loads(result["final_json"])
        self.assertIn("## Triage Verdict", result["final_markdown"])
        # Venue rank from pub_rank survives the fallback path.
        self.assertEqual(payload["signals"]["ccf_rank"], "A")
        self.assertFalse(payload["signals"]["resolved"])

    async def test_reader_question_is_passed_to_the_review(self) -> None:
        from app.workflows import snap_subgraph

        seen: list[str] = []

        async def fake_chat(messages, **kwargs):
            seen.append(messages[0]["content"])
            return json.dumps({
                "one_liner": "A paper.",
                "contributions": [{"text": "Something", "page": 2}],
                "contribution_strength": 2,
                "question_relevance": 3,
                "question_note": "Answers it [p.4]",
            })

        llm = unittest.mock.MagicMock()
        llm.chat = AsyncMock(side_effect=fake_chat)

        with patch.object(snap_subgraph, "get_llm_service", return_value=llm), patch.object(
            snap_subgraph, "get_or_collect_triage_signals", AsyncMock(return_value=_strong_signals())
        ):
            result = await snap_subgraph.run_insight_snap(
                self._state(user_question="Does it scale to long sequences?")
            )

        self.assertTrue(any("Does it scale to long sequences?" in s for s in seen))
        self.assertIn("Answers it [p.4]", result["final_markdown"])


class TestContentReviewPrompt(unittest.TestCase):
    """The review's prose is rendered verbatim, so it must match the report's language."""

    def test_chinese_directive_is_appended_only_for_zh(self) -> None:
        from app.services.triage_verdict import content_review_prompt

        self.assertIn("Simplified Chinese", content_review_prompt("zh"))
        self.assertNotIn("Simplified Chinese", content_review_prompt("en"))

    def test_page_citations_stay_english_in_the_zh_directive(self) -> None:
        from app.services.triage_verdict import content_review_prompt

        self.assertIn("[p.X]", content_review_prompt("zh"))

    def test_question_is_embedded_when_present(self) -> None:
        from app.services.triage_verdict import content_review_prompt

        prompt = content_review_prompt("en", question="Does it scale?")
        self.assertIn("Does it scale?", prompt)
        self.assertIn("question_relevance", prompt)

    def test_no_question_omits_the_extra_fields(self) -> None:
        from app.services.triage_verdict import content_review_prompt

        self.assertNotIn("question_relevance", content_review_prompt("en"))

    def test_review_prompt_never_leaks_external_signals(self) -> None:
        prompt = __import__(
            "app.services.triage_verdict", fromlist=["content_review_prompt"]
        ).content_review_prompt("zh", question="x")
        self.assertIn("NOT told where it was published", prompt)
        self.assertIn("Never mention venues, citation counts, or code availability.", prompt)
