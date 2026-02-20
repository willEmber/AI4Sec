from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EdgeType(str, Enum):
    CITES = "cites"
    CITED_BY = "cited_by"
    RELATED = "related"


class CandidateSource(str, Enum):
    SEED_REF = "seed_ref"
    OPENALEX_REF = "openalex_ref"
    OPENALEX_CITED_BY = "openalex_cited_by"
    OPENALEX_RELATED = "openalex_related"
    S2_REF = "s2_ref"
    S2_CITED_BY = "s2_cited_by"
    S2_RECO = "s2_reco"
    QUERY_SEARCH = "query_search"


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

    # Scoring
    score_text: float = 0.0
    score_graph: float = 0.0
    score_time: float = 0.0
    score_venue: float = 0.0
    score_novelty: float = 0.0
    score_total: float = 0.0

    layer: int = 0  # 0=all, 1=abstract-analyzed, 2=full-text-parsed
    cluster_id: int = -1

    # LLM extraction results (populated in layer1_abstract_snap)
    method_summary: str = ""
    contribution: str = ""
    conclusion: str = ""
    task: str = ""
    dataset: str = ""
    metric_keywords: str = ""

    # Metadata reason string (populated in layer0_summarize_metadata)
    reason: str = ""


class SphereEdge(BaseModel):
    source_node_id: str
    target_node_id: str
    edge_type: EdgeType = EdgeType.CITES
    weight: float = 1.0


class SphereConfig(BaseModel):
    radius: int = 1
    candidate_cap: int = 200
    layer1_cap: int = 40
    pdf_parse_cap: int = 0  # Phase A: skip downloads
    num_clusters: int = 5
    comparison_top_k: int = 10

    # Scoring weights
    w_text: float = 0.30
    w_graph: float = 0.25
    w_time: float = 0.15
    w_venue: float = 0.10
    w_novelty: float = 0.20


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


class SphereOutput(BaseModel):
    sphere_overview: str = ""
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
    output: SphereOutput = Field(default_factory=SphereOutput)
