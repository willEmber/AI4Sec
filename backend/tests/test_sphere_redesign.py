from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, patch

from app.models.sphere_models import (
    CandidateSource,
    EdgeType,
    RelationType,
    SphereConfig,
    SphereEdge,
    SphereNode,
    SphereState,
    tier_for_sources,
)
from app.services.citation_graph import _oa_extract_venue
from app.services.sphere_scorer import (
    build_similarity_edges,
    combined_score,
    compute_pagerank,
    compute_quality_score,
    is_preprint_venue,
    is_survey_title,
    label_propagation,
    normalize_venue,
)
from app.workflows.sphere_subgraph import (
    _merge_duplicate_nodes,
    _parse_references_llm,
    step_relevance_gate,
    step_score_and_rank,
    step_select_core_set,
)

YEAR_NOW = 2026


def _node(nid: str, **kwargs) -> SphereNode:
    return SphereNode(node_id=nid, title=kwargs.pop("title", f"Paper {nid}"), **kwargs)


class TierTest(unittest.TestCase):
    def test_seed_ref_is_t1(self):
        self.assertEqual(tier_for_sources([CandidateSource.SEED_REF]), 1)

    def test_query_search_only_is_t3(self):
        self.assertEqual(tier_for_sources([CandidateSource.QUERY_SEARCH]), 3)

    def test_mixed_takes_strongest(self):
        self.assertEqual(
            tier_for_sources([CandidateSource.QUERY_SEARCH, CandidateSource.S2_CITED_BY]),
            1,
        )

    def test_library_is_t2(self):
        self.assertEqual(tier_for_sources([CandidateSource.LIBRARY]), 2)


class QualityScoreTest(unittest.TestCase):
    """The golden regression from the old pipeline: a 1997 highly-cited
    classic must never lose to a fresh zero-citation paper."""

    def setUp(self):
        self.config = SphereConfig()

    def test_classic_beats_uncited_recent(self):
        lstm = _node("a", year=1997, cited_by_count=90000, venue="Neural Computation")
        fresh = _node("b", year=YEAR_NOW, cited_by_count=0, venue="Some Journal")
        q_lstm = compute_quality_score(lstm, self.config, YEAR_NOW)
        q_fresh = compute_quality_score(fresh, self.config, YEAR_NOW)
        self.assertGreater(q_lstm, q_fresh)

    def test_venue_rank_beats_keyword_heuristic(self):
        ranked = _node("a", year=2024, cited_by_count=10, venue="Obscure J", sci_rank="Q1")
        unranked = _node("b", year=2024, cited_by_count=10, venue="Obscure J")
        self.assertGreater(
            compute_quality_score(ranked, self.config, YEAR_NOW),
            compute_quality_score(unranked, self.config, YEAR_NOW),
        )

    def test_zero_citation_no_venue_scores_low(self):
        junk = _node("a", year=YEAR_NOW, cited_by_count=0, venue="")
        q = compute_quality_score(junk, self.config, YEAR_NOW)
        # Only the recency component contributes
        self.assertLessEqual(q, self.config.w_recency + 1e-9)


class CombinedScoreTest(unittest.TestCase):
    def test_relevance_dominates_quality(self):
        essential = _node("a", relevance=3, quality_score=0.2)
        marginal = _node("b", relevance=1, quality_score=0.6)
        self.assertGreater(combined_score(essential), combined_score(marginal))

    def test_unjudged_counts_as_marginal(self):
        unjudged = _node("a", relevance=-1, quality_score=0.5)
        judged_same = _node("b", relevance=1, quality_score=0.5)
        self.assertAlmostEqual(combined_score(unjudged), combined_score(judged_same))

    def test_influential_nudge(self):
        plain = _node("a", relevance=2, quality_score=0.5)
        boosted = _node("b", relevance=2, quality_score=0.5, influential=True)
        self.assertGreater(combined_score(boosted), combined_score(plain))


class SimilarityEdgesTest(unittest.TestCase):
    def test_direct_cite_and_coupling(self):
        nodes = {
            "a": _node("a", openalex_id="W1", referenced_ids=["W2", "W10", "W11", "W12"]),
            "b": _node("b", openalex_id="W2", referenced_ids=["W10", "W11", "W13"]),
            "c": _node("c", openalex_id="W3", referenced_ids=["W99"]),
        }
        edges = build_similarity_edges(nodes, min_coupling=2)
        cites = [e for e in edges if e.edge_type == EdgeType.CITES]
        coupling = [e for e in edges if e.edge_type == EdgeType.COUPLING]

        # a references W2 == b
        self.assertTrue(any(e.source_node_id == "a" and e.target_node_id == "b" for e in cites))
        # a and b share W10, W11 (>= 2) → coupling edge; c shares nothing
        self.assertEqual(len(coupling), 1)
        pair = {coupling[0].source_node_id, coupling[0].target_node_id}
        self.assertEqual(pair, {"a", "b"})

    def test_no_refs_no_edges(self):
        nodes = {"a": _node("a"), "b": _node("b")}
        self.assertEqual(build_similarity_edges(nodes), [])


class PageRankTest(unittest.TestCase):
    def test_hub_outranks_leaf(self):
        # b is cited by a, c, d → b should rank highest
        nodes = {nid: _node(nid) for nid in ("a", "b", "c", "d")}
        edges = [
            SphereEdge(source_node_id="a", target_node_id="b"),
            SphereEdge(source_node_id="c", target_node_id="b"),
            SphereEdge(source_node_id="d", target_node_id="b"),
            SphereEdge(source_node_id="a", target_node_id="c"),
        ]
        rank = compute_pagerank(nodes, edges)
        self.assertEqual(max(rank, key=rank.get), "b")


class LabelPropagationTest(unittest.TestCase):
    def test_two_communities(self):
        ids = ["a1", "a2", "a3", "b1", "b2", "b3"]
        edges = [
            SphereEdge(source_node_id="a1", target_node_id="a2", edge_type=EdgeType.COUPLING),
            SphereEdge(source_node_id="a2", target_node_id="a3", edge_type=EdgeType.COUPLING),
            SphereEdge(source_node_id="a1", target_node_id="a3", edge_type=EdgeType.COUPLING),
            SphereEdge(source_node_id="b1", target_node_id="b2", edge_type=EdgeType.COUPLING),
            SphereEdge(source_node_id="b2", target_node_id="b3", edge_type=EdgeType.COUPLING),
            SphereEdge(source_node_id="b1", target_node_id="b3", edge_type=EdgeType.COUPLING),
        ]
        communities = label_propagation(ids, edges)
        self.assertEqual(communities["a1"], communities["a2"])
        self.assertEqual(communities["a2"], communities["a3"])
        self.assertEqual(communities["b1"], communities["b2"])
        self.assertNotEqual(communities["a1"], communities["b1"])


class MergeDuplicatesTest(unittest.TestCase):
    def test_title_node_merges_into_doi_node(self):
        sphere = SphereState()
        center = _node("center", title="Center Paper")
        sphere.center_node = center
        sphere.nodes["center"] = center

        # Same paper twice: once keyed by title (PDF ref), once by DOI (API).
        pdf_ref = _node(
            "n1", title="Attention Is All You Need",
            doi="10.5555/3295222", sources=[CandidateSource.SEED_REF],
        )
        api_hit = _node(
            "n2", title="Attention is all you need",
            doi="10.5555/3295222", cited_by_count=100000,
            sources=[CandidateSource.S2_CITED_BY],
        )
        sphere.nodes["n1"] = pdf_ref
        sphere.nodes["n2"] = api_hit
        sphere.edges = [
            SphereEdge(source_node_id="center", target_node_id="n1"),
            SphereEdge(source_node_id="n2", target_node_id="center"),
        ]

        merged = _merge_duplicate_nodes(sphere)
        self.assertEqual(merged, 1)
        self.assertEqual(len(sphere.nodes), 2)  # center + survivor
        survivor = next(n for nid, n in sphere.nodes.items() if nid != "center")
        self.assertEqual(survivor.cited_by_count, 100000)
        self.assertIn(CandidateSource.SEED_REF, survivor.sources)
        self.assertIn(CandidateSource.S2_CITED_BY, survivor.sources)
        self.assertEqual(survivor.tier, 1)
        # Edges remapped, no dangling references
        for e in sphere.edges:
            self.assertIn(e.source_node_id, sphere.nodes)
            self.assertIn(e.target_node_id, sphere.nodes)


class ScoreAndRankGateTest(unittest.IsolatedAsyncioTestCase):
    async def test_t3_hard_gate_drops_uncited_search_hits(self):
        config = SphereConfig(venue_rank_enabled=False, t3_min_citations=5)
        sphere = SphereState(config=config)
        center = _node("center")
        sphere.center_node = center
        sphere.nodes["center"] = center

        junk = _node(
            "junk", year=2026, cited_by_count=0,
            sources=[CandidateSource.QUERY_SEARCH], tier=3,
        )
        cited_search = _node(
            "ok_search", year=2024, cited_by_count=50,
            sources=[CandidateSource.QUERY_SEARCH], tier=3,
        )
        seed = _node(
            "seed", year=1997, cited_by_count=0,
            sources=[CandidateSource.SEED_REF], tier=1,
        )
        for n in (junk, cited_search, seed):
            sphere.nodes[n.node_id] = n

        state = {"paper_id": "p1", "language": "en"}
        await step_score_and_rank(sphere, state, run_id="")

        self.assertNotIn("junk", sphere.nodes)       # T3 + uncited → dropped
        self.assertIn("ok_search", sphere.nodes)     # T3 but well-cited → kept
        self.assertIn("seed", sphere.nodes)          # T1 never hard-gated


class RelevanceGateTest(unittest.IsolatedAsyncioTestCase):
    async def test_gate_labels_and_drops_unrelated(self):
        config = SphereConfig(min_relevance=1)
        sphere = SphereState(config=config)
        center = _node("center", title="Attention Is All You Need",
                       abstract_text="We propose the Transformer.")
        sphere.center_node = center
        sphere.nodes["center"] = center

        # 12 relevant nodes + 2 noise nodes (keeps survivors above the
        # safety-valve floor of 10 so the drop actually happens)
        for i in range(12):
            sphere.nodes[f"rel{i}"] = _node(f"rel{i}", abstract_text="related work")
        sphere.nodes["noise0"] = _node("noise0", abstract_text="weld defect detection")
        sphere.nodes["noise1"] = _node("noise1", abstract_text="fraud detection")

        cands_order = [n.node_id for n in sphere.nodes.values() if n.node_id != "center"]

        def _fake_response(idx_of):
            entries = []
            for i, nid in enumerate(idx_of):
                if nid.startswith("noise"):
                    entries.append({"idx": i, "relation": "unrelated", "relevance": 0,
                                    "reason": "keyword overlap only"})
                else:
                    entries.append({"idx": i, "relation": "follow_up", "relevance": 2,
                                    "reason": "extends the Transformer"})
            return json.dumps(entries)

        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value=_fake_response(cands_order))

        with patch("app.workflows.sphere_subgraph.get_llm_service", return_value=mock_llm):
            state = {"paper_id": "p1", "language": "en", "llm_model": "m"}
            await step_relevance_gate(sphere, state, run_id="")

        self.assertNotIn("noise0", sphere.nodes)
        self.assertNotIn("noise1", sphere.nodes)
        self.assertEqual(sphere.nodes["rel0"].relation_type, RelationType.FOLLOW_UP)
        self.assertEqual(sphere.nodes["rel0"].relevance, 2)

    async def test_safety_valve_keeps_nodes_when_gate_nukes_everything(self):
        config = SphereConfig(min_relevance=1)
        sphere = SphereState(config=config)
        center = _node("center", title="X")
        sphere.center_node = center
        sphere.nodes["center"] = center
        for i in range(5):
            sphere.nodes[f"n{i}"] = _node(f"n{i}", abstract_text="text")

        all_zero = json.dumps([
            {"idx": i, "relation": "unrelated", "relevance": 0, "reason": "no"}
            for i in range(5)
        ])
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value=all_zero)

        with patch("app.workflows.sphere_subgraph.get_llm_service", return_value=mock_llm):
            state = {"paper_id": "p1", "language": "en", "llm_model": "m"}
            await step_relevance_gate(sphere, state, run_id="")

        # Fewer than 10 survivors → drop is skipped entirely
        self.assertEqual(len(sphere.nodes), 6)


class CoreSetSelectionTest(unittest.IsolatedAsyncioTestCase):
    async def test_quota_partitioning(self):
        config = SphereConfig(
            core_cap=10, quota_foundation=2, quota_method=3,
            quota_follow_up=2, quota_frontier=2, quota_library=1,
        )
        sphere = SphereState(config=config)
        center = _node("center")
        sphere.center_node = center
        sphere.nodes["center"] = center

        def _add(nid: str, relation: RelationType, relevance: int = 2, **kw):
            n = _node(nid, relation_type=relation, relevance=relevance, **kw)
            sphere.nodes[nid] = n
            return n

        for i in range(4):
            _add(f"f{i}", RelationType.FOUNDATION, quality_score=0.5 + i * 0.1)
        for i in range(5):
            _add(f"m{i}", RelationType.COMPETITOR, quality_score=0.5)
        for i in range(3):
            _add(f"u{i}", RelationType.FOLLOW_UP, quality_score=0.5)
        _add("app0", RelationType.APPLICATION, relevance=2, year=2026, quality_score=0.4)
        lib = _add("lib0", RelationType.METHOD_NEIGHBOR, quality_score=0.3)
        lib.sources = [CandidateSource.LIBRARY]

        with patch("app.workflows.sphere_subgraph._persist_sphere", new=AsyncMock()):
            state = {"paper_id": "p1", "language": "en"}
            await step_select_core_set(sphere, state, run_id="")

        parts = {p.key: p.node_ids for p in sphere.output.partitions}
        # Quotas fill first (backfill may add extras to best-fitting partitions
        # when other quotas run dry, so >= not ==)
        self.assertGreaterEqual(len(parts["foundation"]), 2)
        # Best-quality foundations picked first
        self.assertIn("f3", parts["foundation"])
        self.assertIn("f2", parts["foundation"])
        self.assertGreaterEqual(len(parts["method"]), 3)
        self.assertEqual(len(parts["follow_up"]), 2)
        # Application papers land in their own partition, never "method"
        self.assertIn("app0", parts.get("application", []))
        self.assertNotIn("app0", parts["method"])
        # Library node was already eligible for method quota, but the library
        # quota only takes unselected nodes — either way it must be in core.
        self.assertIn("lib0", sphere.layer1_node_ids)
        self.assertLessEqual(len(sphere.layer1_node_ids), config.core_cap)
        # Every core node is marked layer 1
        for nid in sphere.layer1_node_ids:
            self.assertGreaterEqual(sphere.nodes[nid].layer, 1)

    async def test_unrelated_never_backfills(self):
        config = SphereConfig(core_cap=10)
        sphere = SphereState(config=config)
        center = _node("center")
        sphere.center_node = center
        sphere.nodes["center"] = center
        sphere.nodes["bad"] = _node(
            "bad", relation_type=RelationType.UNRELATED, relevance=0,
            quality_score=0.9,
        )
        sphere.nodes["good"] = _node(
            "good", relation_type=RelationType.FOLLOW_UP, relevance=2,
            quality_score=0.2,
        )

        with patch("app.workflows.sphere_subgraph._persist_sphere", new=AsyncMock()):
            state = {"paper_id": "p1", "language": "en"}
            await step_select_core_set(sphere, state, run_id="")

        self.assertIn("good", sphere.layer1_node_ids)
        self.assertNotIn("bad", sphere.layer1_node_ids)


class VersionMergeTest(unittest.TestCase):
    """P1: arXiv preprint and published version carry different DOIs but the
    same title — they must collapse into one node with the published venue."""

    def _sphere_with(self, *nodes: SphereNode) -> SphereState:
        sphere = SphereState()
        center = _node("center", title="Center Paper")
        sphere.center_node = center
        sphere.nodes["center"] = center
        for n in nodes:
            sphere.nodes[n.node_id] = n
        return sphere

    def test_preprint_and_published_versions_merge(self):
        arxiv = _node(
            "va", title="An Image is Worth 16x16 Words",
            doi="10.48550/arxiv.2010.11929", year=2020,
            venue="arXiv (Cornell University)", cited_by_count=21710,
            sources=[CandidateSource.S2_CITED_BY],
        )
        iclr = _node(
            "vb", title="An Image is Worth 16x16 Words",
            doi="10.5555/iclr.2021", year=2021,
            venue="International Conference on Learning Representations",
            cited_by_count=558,
            sources=[CandidateSource.OPENALEX_CITED_BY],
        )
        sphere = self._sphere_with(arxiv, iclr)
        merged = _merge_duplicate_nodes(sphere)

        self.assertEqual(merged, 1)
        self.assertEqual(len(sphere.nodes), 2)  # center + survivor
        survivor = next(n for nid, n in sphere.nodes.items() if nid != "center")
        # Version merge: citations sum (the record is split between versions)
        self.assertEqual(survivor.cited_by_count, 21710 + 558)
        # Published venue (and its year) wins over the preprint server
        self.assertFalse(is_preprint_venue(survivor.venue))
        self.assertEqual(survivor.year, 2021)
        self.assertIn(CandidateSource.S2_CITED_BY, survivor.sources)
        self.assertIn(CandidateSource.OPENALEX_CITED_BY, survivor.sources)

    def test_same_title_distant_years_not_merged(self):
        old = _node("na", title="A Generic Title About Attention", doi="10.1/old", year=1998)
        new = _node("nb", title="A Generic Title About Attention", doi="10.1/new", year=2020)
        sphere = self._sphere_with(old, new)
        merged = _merge_duplicate_nodes(sphere)
        self.assertEqual(merged, 0)
        self.assertEqual(len(sphere.nodes), 3)


class VenueNormalizeTest(unittest.TestCase):
    """P3: sprawling OpenAlex conference names → canonical acronyms."""

    def test_conference_names_map_to_acronyms(self):
        cases = {
            "2021 IEEE/CVF International Conference on Computer Vision (ICCV)": "ICCV",
            "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition": "CVPR",
            "Advances in Neural Information Processing Systems": "NeurIPS",
            "International Conference on Learning Representations": "ICLR",
            "2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)": "WACV",
        }
        for raw, expected in cases.items():
            self.assertEqual(normalize_venue(raw), expected, raw)

    def test_journals_and_workshops_pass_through(self):
        self.assertEqual(normalize_venue("Neural Computation"), "Neural Computation")
        workshop = "CVPR 2021 Workshop on Computer Vision and Pattern Recognition Applications"
        self.assertEqual(normalize_venue(workshop), workshop)

    def test_preprint_detection(self):
        self.assertTrue(is_preprint_venue("arXiv (Cornell University)"))
        self.assertFalse(is_preprint_venue("Nature"))


class OaVenueExtractTest(unittest.TestCase):
    """P3: prefer the published location over the arXiv primary_location."""

    @staticmethod
    def _loc(name: str, stype: str = "") -> dict:
        source = {"display_name": name}
        if stype:
            source["type"] = stype
        return {"source": source}

    def test_published_location_wins_over_arxiv_primary(self):
        work = {
            "primary_location": self._loc("arXiv (Cornell University)"),
            "locations": [
                self._loc("arXiv (Cornell University)"),
                self._loc("International Conference on Learning Representations"),
            ],
        }
        self.assertEqual(
            _oa_extract_venue(work),
            "International Conference on Learning Representations",
        )

    def test_non_preprint_primary_kept(self):
        work = {
            "primary_location": self._loc("Neural Computation"),
            "locations": [self._loc("arXiv (Cornell University)")],
        }
        self.assertEqual(_oa_extract_venue(work), "Neural Computation")

    def test_arxiv_only_paper_keeps_arxiv(self):
        work = {
            "primary_location": self._loc("arXiv (Cornell University)"),
            "locations": [self._loc("arXiv (Cornell University)")],
        }
        self.assertEqual(_oa_extract_venue(work), "arXiv (Cornell University)")

    def test_institutional_repository_skipped_by_type(self):
        """P3 regression: HAL/LA Referencia/Apollo/UvA-DARE are typed
        'repository' in OpenAlex but don't match the arXiv-ish name regex —
        the real conference/journal location must still win."""
        work = {
            "primary_location": self._loc(
                "HAL (Le Centre pour la Communication Scientifique Directe)",
                stype="repository",
            ),
            "locations": [
                self._loc(
                    "HAL (Le Centre pour la Communication Scientifique Directe)",
                    stype="repository",
                ),
                self._loc(
                    "Empirical Methods in Natural Language Processing",
                    stype="conference",
                ),
            ],
        }
        self.assertEqual(
            _oa_extract_venue(work),
            "Empirical Methods in Natural Language Processing",
        )

    def test_repository_only_paper_keeps_repository_name(self):
        work = {
            "primary_location": self._loc("LA Referencia", stype="repository"),
            "locations": [self._loc("LA Referencia", stype="repository")],
        }
        self.assertEqual(_oa_extract_venue(work), "LA Referencia")


class FrontierRelativeTest(unittest.IsolatedAsyncioTestCase):
    """P4: frontier = newest relevant papers after the center paper,
    not an absolute year window (which stays empty for older papers)."""

    async def test_newest_post_center_papers_fill_frontier(self):
        config = SphereConfig(core_cap=10, quota_frontier=1)
        sphere = SphereState(config=config)
        center = _node("center", year=2017)
        sphere.center_node = center
        sphere.nodes["center"] = center

        # Unjudged-relation but relevant papers from different years, each
        # clearing the frontier impact floor (citations or a ranked venue)
        for nid, year in (("y2023", 2023), ("y2019", 2019)):
            sphere.nodes[nid] = _node(
                nid, year=year, relevance=1, quality_score=0.5,
                cited_by_count=config.frontier_min_citations,
            )
        # Pre-center paper: never frontier
        sphere.nodes["y2015"] = _node(
            "y2015", year=2015, relevance=1, quality_score=0.9,
            cited_by_count=config.frontier_min_citations,
        )

        with patch("app.workflows.sphere_subgraph._persist_sphere", new=AsyncMock()):
            state = {"paper_id": "p1", "language": "en"}
            await step_select_core_set(sphere, state, run_id="")

        parts = {p.key: p.node_ids for p in sphere.output.partitions}
        # Quota slot goes to the newest post-center paper despite lower quality
        self.assertEqual(parts["frontier"][0], "y2023")
        self.assertNotIn("y2015", parts.get("frontier", []))

    async def test_low_citation_recent_paper_fails_impact_floor(self):
        """P3: a merely-newest, low-impact paper must not claim the frontier
        seat over one that actually clears the citation/venue floor."""
        config = SphereConfig(core_cap=10, quota_frontier=1)
        sphere = SphereState(config=config)
        center = _node("center", year=2017)
        sphere.center_node = center
        sphere.nodes["center"] = center

        sphere.nodes["low_impact"] = _node(
            "low_impact", year=2025, relevance=1, quality_score=0.5,
            cited_by_count=2,
        )
        sphere.nodes["qualified"] = _node(
            "qualified", year=2022, relevance=1, quality_score=0.5,
            cited_by_count=config.frontier_min_citations,
        )

        with patch("app.workflows.sphere_subgraph._persist_sphere", new=AsyncMock()):
            state = {"paper_id": "p1", "language": "en"}
            await step_select_core_set(sphere, state, run_id="")

        parts = {p.key: p.node_ids for p in sphere.output.partitions}
        self.assertIn("qualified", parts.get("frontier", []))
        self.assertNotIn("low_impact", parts.get("frontier", []))


class TieBreakTest(unittest.IsolatedAsyncioTestCase):
    """P2: equal combined scores must fall back to raw citation counts,
    not dict insertion order."""

    async def test_higher_cited_wins_tie(self):
        config = SphereConfig(core_cap=1)
        sphere = SphereState(config=config)
        center = _node("center")
        sphere.center_node = center
        sphere.nodes["center"] = center

        # Insert the low-cited one FIRST so insertion order would pick it
        sphere.nodes["low"] = _node(
            "low", relation_type=RelationType.COMPETITOR, relevance=2,
            quality_score=0.5, cited_by_count=10,
        )
        sphere.nodes["high"] = _node(
            "high", relation_type=RelationType.COMPETITOR, relevance=2,
            quality_score=0.5, cited_by_count=100000,
        )

        with patch("app.workflows.sphere_subgraph._persist_sphere", new=AsyncMock()):
            state = {"paper_id": "p1", "language": "en"}
            await step_select_core_set(sphere, state, run_id="")

        self.assertEqual(sphere.layer1_node_ids, ["high"])


class SurveyTitleTest(unittest.TestCase):
    """P4: cheap title-convention detector for review/survey papers."""

    def test_detects_common_survey_titles(self):
        titles = [
            "Attention mechanisms in computer vision: A survey",
            "Pre-trained models for natural language processing: A survey",
            "An introduction to Deep Learning in Natural Language Processing",
            "Pre-trained models: Past, present and future",
            "A Review of Deep Learning Methods for Semantic Segmentation",
        ]
        for title in titles:
            self.assertTrue(is_survey_title(title), title)

    def test_regular_paper_titles_not_flagged(self):
        titles = [
            "Attention Is All You Need",
            "Deep Residual Learning for Image Recognition",
            "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
        ]
        for title in titles:
            self.assertFalse(is_survey_title(title), title)


class SurveyPartitionTest(unittest.IsolatedAsyncioTestCase):
    """P4: surveys never occupy method/foundation/follow_up seats, even
    though the LLM gate may still label them method_neighbor etc."""

    async def test_survey_goes_to_its_own_partition(self):
        config = SphereConfig(core_cap=10)
        sphere = SphereState(config=config)
        center = _node("center")
        sphere.center_node = center
        sphere.nodes["center"] = center

        sphere.nodes["survey0"] = _node(
            "survey0", title="Attention mechanisms in computer vision: A survey",
            relation_type=RelationType.METHOD_NEIGHBOR, relevance=2,
            quality_score=0.9, cited_by_count=2000,
        )
        sphere.nodes["method0"] = _node(
            "method0", title="Graph Attention Networks",
            relation_type=RelationType.METHOD_NEIGHBOR, relevance=2,
            quality_score=0.5,
        )

        with patch("app.workflows.sphere_subgraph._persist_sphere", new=AsyncMock()):
            state = {"paper_id": "p1", "language": "en"}
            await step_select_core_set(sphere, state, run_id="")

        parts = {p.key: p.node_ids for p in sphere.output.partitions}
        self.assertIn("survey0", parts.get("survey", []))
        self.assertNotIn("survey0", parts.get("method", []))
        self.assertIn("method0", parts.get("method", []))

    async def test_low_relevance_survey_excluded_not_backfilled_as_method(self):
        config = SphereConfig(core_cap=10)
        sphere = SphereState(config=config)
        center = _node("center")
        sphere.center_node = center
        sphere.nodes["center"] = center

        sphere.nodes["survey_marginal"] = _node(
            "survey_marginal", title="A Survey on Attention Mechanisms",
            relation_type=RelationType.METHOD_NEIGHBOR, relevance=1,
            quality_score=0.9, cited_by_count=5000,
        )

        with patch("app.workflows.sphere_subgraph._persist_sphere", new=AsyncMock()):
            state = {"paper_id": "p1", "language": "en"}
            await step_select_core_set(sphere, state, run_id="")

        parts = {p.key: p.node_ids for p in sphere.output.partitions}
        self.assertNotIn("survey_marginal", parts.get("method", []))


class FunnelInstrumentationTest(unittest.IsolatedAsyncioTestCase):
    """P2 observability: select_core_set records a funnel entry with the
    strongest near-misses, so a missing landmark paper is traceable."""

    async def test_select_records_near_misses(self):
        config = SphereConfig(core_cap=1, quota_method=1)
        sphere = SphereState(config=config)
        center = _node("center")
        sphere.center_node = center
        sphere.nodes["center"] = center

        sphere.nodes["winner"] = _node(
            "winner", relation_type=RelationType.COMPETITOR, relevance=3,
            quality_score=0.9, cited_by_count=50000,
        )
        sphere.nodes["bert_like"] = _node(
            "bert_like", relation_type=RelationType.FOLLOW_UP, relevance=3,
            quality_score=0.85, cited_by_count=90000,
        )

        with patch("app.workflows.sphere_subgraph._persist_sphere", new=AsyncMock()):
            state = {"paper_id": "p1", "language": "en"}
            await step_select_core_set(sphere, state, run_id="")

        select_entries = [e for e in sphere.funnel if e["stage"] == "select"]
        self.assertEqual(len(select_entries), 1)
        entry = select_entries[0]
        self.assertEqual(entry["core"], 1)
        self.assertGreaterEqual(entry["near_misses"], 1)
        # The dropped high-impact paper is named in the near-miss preview
        self.assertTrue(any("Paper bert_like" in s for s in entry["near_misses_top"]))


class RefParseLLMTest(unittest.IsolatedAsyncioTestCase):
    """P2: LLM bibliography parsing with regex fallback."""

    async def test_parses_titles_years_dois(self):
        raw_refs = [
            "[13] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8):1735–1780, 1997.",
            "[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. CoRR, abs/1409.0473, 2014.",
        ]
        response = json.dumps([
            {"idx": 0, "title": "Long short-term memory", "year": 1997, "doi": ""},
            {"idx": 1, "title": "Neural machine translation by jointly learning to align and translate", "year": 2014, "doi": ""},
        ])
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value=response)

        with patch("app.workflows.sphere_subgraph.get_llm_service", return_value=mock_llm):
            parsed = await _parse_references_llm(raw_refs, model="m", paper_id="p1")

        self.assertEqual(parsed[0]["title"], "Long short-term memory")
        self.assertEqual(parsed[0]["year"], 1997)
        self.assertIn("align and translate", parsed[1]["title"])

    async def test_failed_batch_returns_empty(self):
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value="not json at all")

        with patch("app.workflows.sphere_subgraph.get_llm_service", return_value=mock_llm):
            parsed = await _parse_references_llm(["[1] Some ref."], model="m", paper_id="p1")

        self.assertEqual(parsed, {})


if __name__ == "__main__":
    unittest.main()
