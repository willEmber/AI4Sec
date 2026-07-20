from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EdgeType(str, Enum):
    CITES = "cites"
    CITED_BY = "cited_by"
    RELATED = "related"
    COUPLING = "coupling"  # bibliographic coupling between candidates


class CandidateSource(str, Enum):
    SEED_REF = "seed_ref"
    OPENALEX_REF = "openalex_ref"
    OPENALEX_CITED_BY = "openalex_cited_by"
    OPENALEX_RELATED = "openalex_related"
    S2_REF = "s2_ref"
    S2_CITED_BY = "s2_cited_by"
    S2_RECO = "s2_reco"
    QUERY_SEARCH = "query_search"
    LIBRARY = "library"  # matched from the user's own Dify knowledge base


# Provenance trust tiers: how much we trust that a candidate is genuinely
# related to the center paper, based on where it came from.
#   T1 — citation-graph evidence (the paper cites / is cited by / is
#        recommended for the center paper)
#   T2 — indirect evidence (OpenAlex related works, user's own library)
#   T3 — keyword search only (weakest; must clear extra quality gates)
_SOURCE_TIER: dict[CandidateSource, int] = {
    CandidateSource.SEED_REF: 1,
    CandidateSource.OPENALEX_REF: 1,
    CandidateSource.S2_REF: 1,
    CandidateSource.OPENALEX_CITED_BY: 1,
    CandidateSource.S2_CITED_BY: 1,
    CandidateSource.S2_RECO: 1,
    CandidateSource.OPENALEX_RELATED: 2,
    CandidateSource.LIBRARY: 2,
    CandidateSource.QUERY_SEARCH: 3,
}


def tier_for_sources(sources: list[CandidateSource]) -> int:
    """Best (lowest) trust tier across all sources that surfaced a node."""
    if not sources:
        return 3
    return min(_SOURCE_TIER.get(s, 3) for s in sources)


class RelationType(str, Enum):
    """How a candidate relates to the center paper (LLM relevance gate)."""

    FOUNDATION = "foundation"          # prior work the center paper builds on
    METHOD_NEIGHBOR = "method_neighbor"  # same methodology family
    COMPETITOR = "competitor"          # solves the same problem, comparable
    FOLLOW_UP = "follow_up"            # extends/builds on the center paper
    APPLICATION = "application"        # applies the center paper's ideas elsewhere
    UNRELATED = "unrelated"            # keyword overlap only — discard
    UNKNOWN = "unknown"                # not yet judged / gate failed


def make_node_id(doi: str = "", title: str = "") -> str:
    """Generate a deterministic 16-char node ID from DOI or title."""
    if doi:
        return hashlib.sha1(doi.strip().lower().encode()).hexdigest()[:16]
    # Inline title fingerprint (same logic as paper_search utils)
    import re
    raw = re.sub(r"\s+", " ", (title or "").strip()).lower()
    ascii_norm = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", raw)).strip()
    if ascii_norm:
        return hashlib.sha1(ascii_norm.encode()).hexdigest()[:16]
    unicode_norm = re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", raw, flags=re.UNICODE)).replace("_", " ").strip()
    return hashlib.sha1(unicode_norm.encode()).hexdigest()[:16]


class SphereNode(BaseModel):
    node_id: str = ""
    doi: str = ""
    arxiv_id: str = ""
    openalex_id: str = ""
    s2_paper_id: str = ""
    title: str = ""
    year: int = 0
    venue: str = ""
    authors: str = ""
    abstract_text: str = ""
    cited_by_count: int = 0
    pdf_path: str = ""
    mineru_parsed: bool = False
    source: CandidateSource = CandidateSource.SEED_REF
    sources: list[CandidateSource] = Field(default_factory=list)
    # Dify document id when this node also exists in the user's knowledge base.
    library_document_id: str = ""

    # Provenance trust tier (1 strongest .. 3 weakest), see tier_for_sources.
    tier: int = 3
    # OpenAlex work IDs referenced by this paper (for bibliographic coupling).
    referenced_ids: list[str] = Field(default_factory=list)

    # Relevance gate results (LLM-judged relation to the center paper)
    relation_type: RelationType = RelationType.UNKNOWN
    relevance: int = -1  # 0-3; -1 = not judged
    relation_reason: str = ""

    # Quality signals
    quality_score: float = 0.0  # [0,1] citations/venue/recency composite
    sci_rank: str = ""          # e.g. "Q1" (EasyScholar)
    ccf_rank: str = ""          # e.g. "A" (EasyScholar)
    influential: bool = False   # S2 isInfluential citation flag

    # Graph + final ranking scores
    score_graph: float = 0.0    # PageRank on the similarity graph
    score_total: float = 0.0    # combined relevance × quality (selection key)

    layer: int = 0  # 0=all, 1=core set (abstract-analyzed), 2=full-text-parsed
    cluster_id: int = -1

    # LLM extraction results (populated in layer1_abstract_snap)
    method_summary: str = ""
    contribution: str = ""
    conclusion: str = ""
    task: str = ""
    dataset: str = ""
    metric_keywords: str = ""

    # Human-readable provenance/quality summary (rendered in reports)
    reason: str = ""


class SphereEdge(BaseModel):
    source_node_id: str
    target_node_id: str
    edge_type: EdgeType = EdgeType.CITES
    weight: float = 1.0


class SphereConfig(BaseModel):
    radius: int = 1
    candidate_cap: int = 200     # max candidates fetched from all channels
    gate_cap: int = 120          # max candidates entering the LLM relevance gate
    core_cap: int = 40           # size of the final core set
    pdf_parse_cap: int = 0       # Phase A: skip downloads
    comparison_top_k: int = 10

    # Relevance gate
    min_relevance: int = 1       # candidates below this are discarded
    # T3 (keyword-search-only) hard gate: need citations or a ranked venue
    t3_min_citations: int = 5
    # Frontier quality floor: a merely-newest paper can't claim a frontier
    # seat unless it has real impact (citations) or a ranked venue
    frontier_min_citations: int = 10

    # Quality score weights (citation impact / venue rank / recency)
    w_citation: float = 0.5
    w_venue: float = 0.3
    w_recency: float = 0.2
    venue_rank_enabled: bool = True

    # Core-set quotas per relation partition (sum ≈ core_cap).
    # "frontier" is relative: the newest still-unselected relevant papers
    # published after the center paper — no absolute year window, so it
    # works for a 2017 center paper as well as a 2025 one.
    # "survey" quarantines review papers: useful entry points, but they are
    # not methods/competitors and must not eat those partitions' seats.
    quota_foundation: int = 8
    quota_method: int = 8        # method_neighbor + competitor
    quota_follow_up: int = 8
    quota_application: int = 5   # cross-domain applications of the center's ideas
    quota_frontier: int = 5
    quota_survey: int = 2
    quota_library: int = 4


class ThemeCluster(BaseModel):
    name: str = ""
    definition: str = ""
    representative_node_ids: list[str] = Field(default_factory=list)


class TimelineEntry(BaseModel):
    year: int = 0
    node_ids: list[str] = Field(default_factory=list)
    summary: str = ""


class KeyHub(BaseModel):
    node_id: str = ""
    title: str = ""
    pagerank: float = 0.0
    cited_by_count: int = 0
    reason: str = ""


class ComparisonRow(BaseModel):
    node_id: str = ""
    title: str = ""
    problem: str = ""
    assumption: str = ""
    method: str = ""
    dataset: str = ""
    metric: str = ""
    strength: str = ""
    weakness: str = ""


class GapIdea(BaseModel):
    title: str = ""
    description: str = ""
    evidence_node_ids: list[str] = Field(default_factory=list)


class ReadingPath(BaseModel):
    name: str = ""  # fast / deep / frontier
    description: str = ""
    node_ids: list[str] = Field(default_factory=list)


class LibraryMatch(BaseModel):
    """A paper from the user's own Dify knowledge base related to the center paper.

    Captured directly in step 4 and rendered as its own section, independent of
    the citation-graph scoring/cap pipeline (so it always survives to the output).
    """

    document_id: str = ""
    title: str = ""
    score: float = 0.0
    snippet: str = ""


class CorePartition(BaseModel):
    """One relation-based partition of the core set (rendered as a section)."""

    key: str = ""   # foundation / method / follow_up / application / frontier / survey / library
    node_ids: list[str] = Field(default_factory=list)


class SphereOutput(BaseModel):
    sphere_overview: str = ""
    partitions: list[CorePartition] = Field(default_factory=list)
    themes: list[ThemeCluster] = Field(default_factory=list)
    timeline: list[TimelineEntry] = Field(default_factory=list)
    key_hubs: list[KeyHub] = Field(default_factory=list)
    comparison_table: list[ComparisonRow] = Field(default_factory=list)
    gaps_and_ideas: list[GapIdea] = Field(default_factory=list)
    reading_paths: list[ReadingPath] = Field(default_factory=list)


class SphereState(BaseModel):
    config: SphereConfig = Field(default_factory=SphereConfig)
    center_node: SphereNode | None = None
    nodes: dict[str, SphereNode] = Field(default_factory=dict)
    edges: list[SphereEdge] = Field(default_factory=list)
    pdf_refs: list[dict[str, str]] = Field(default_factory=list)
    layer1_node_ids: list[str] = Field(default_factory=list)
    layer2_node_ids: list[str] = Field(default_factory=list)
    library_matches: list[LibraryMatch] = Field(default_factory=list)
    output: SphereOutput = Field(default_factory=SphereOutput)
    # Pipeline funnel instrumentation: one record per stage with candidate
    # counts and the highest-cited papers dropped there. Surfaced in
    # final_json.debug so "why is BERT missing" is answerable from the output.
    funnel: list[dict[str, Any]] = Field(default_factory=list)
