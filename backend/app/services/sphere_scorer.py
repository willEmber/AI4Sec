from __future__ import annotations

import math
import random
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.sphere_models import SphereConfig, SphereEdge, SphereNode


# ──────────────────────────────────────────────────────────────────────
# Quality score: citation impact + venue rank + recency
# ──────────────────────────────────────────────────────────────────────

# citations/year at which the citation component saturates to 1.0.
# Deliberately high: at 30 every well-known paper ties at 1.0 and selection
# inside a quota degenerates to insertion order (BERT once lost a coin flip
# to DistilBERT this way). At 200, mid-range papers still differentiate.
_CITATION_SATURATION_CPY = 200.0

# Venue rank → score. SCI quartiles and CCF tiers from EasyScholar.
_SCI_RANK_SCORES = {"Q1": 1.0, "Q2": 0.75, "Q3": 0.5, "Q4": 0.35}
_CCF_RANK_SCORES = {"A": 1.0, "B": 0.75, "C": 0.5}

# Keyword fallback when no EasyScholar rank is available
_TOP_VENUES = {
    "nature", "science", "cell", "lancet", "nejm", "bmj",
    "neurips", "nips", "icml", "iclr", "cvpr", "iccv", "eccv",
    "aaai", "ijcai", "acl", "emnlp", "naacl",
    "sigmod", "vldb", "kdd", "www", "sigir",
    "ieee transactions", "acm transactions",
    "pnas", "physical review", "jama",
}

_PREPRINT_VENUE_RE = re.compile(
    r"arxiv|preprint|biorxiv|medrxiv|ssrn|research square|repec", re.IGNORECASE
)


def is_preprint_venue(venue: str) -> bool:
    """True when a venue string denotes a preprint server, not a real outlet."""
    return bool(venue) and bool(_PREPRINT_VENUE_RE.search(venue))


# Survey/review papers cite everything and compare nothing of their own, so
# they poison the comparison matrix and crowd out real method neighbors.
# Detected by title convention ("A Survey on…", "…: A Review", "An
# Introduction to…", "Past, Present and Future") — imperfect but cheap.
_SURVEY_TITLE_RE = re.compile(
    r"\b(?:surveys?|reviews?|overview|tutorial|an introduction to|"
    r"past,? present,? and future)\b",
    re.IGNORECASE,
)


def is_survey_title(title: str) -> bool:
    """Heuristic: does this title read like a survey/review paper?"""
    return bool(title) and bool(_SURVEY_TITLE_RE.search(title))


# OpenAlex reports conference proceedings under sprawling names like
# "2021 IEEE/CVF International Conference on Computer Vision (ICCV)".
# EasyScholar's CCF tables key on canonical acronyms, so normalize before
# rank lookup (and for compact display). Order matters: more specific
# patterns (WACV, workshops) must be checked before broader ones (ICCV).
_VENUE_ACRONYM_RULES: list[tuple[str, str]] = [
    (r"winter conference on applications of computer vision", "WACV"),
    (r"computer vision and pattern recognition", "CVPR"),
    (r"international conference on computer vision", "ICCV"),
    (r"european conference on computer vision", "ECCV"),
    (r"neural information processing systems", "NeurIPS"),
    (r"international conference on machine learning", "ICML"),
    (r"international conference on learning representations", "ICLR"),
    (r"empirical methods in natural language processing", "EMNLP"),
    (r"north american chapter of the association for computational linguistics", "NAACL"),
    (r"annual meeting of the association for computational linguistics", "ACL"),
    (r"aaai conference on artificial intelligence", "AAAI"),
    (r"international joint conference on artificial intelligence", "IJCAI"),
    (r"knowledge discovery and data mining", "KDD"),
    (r"research and development in information retrieval", "SIGIR"),
    (r"(?:the )?web conference|world wide web conference", "WWW"),
    (r"international conference on acoustics,? speech,? and signal processing", "ICASSP"),
    (r"conference of the international speech communication association|interspeech", "INTERSPEECH"),
]


def normalize_venue(venue: str) -> str:
    """Map long conference-proceeding names to canonical acronyms.

    Journals and unrecognized names pass through unchanged. Workshop
    proceedings are never mapped to the main conference's acronym.
    """
    if not venue:
        return ""
    v = venue.strip()
    low = v.lower()
    if "workshop" in low:
        return v
    for pattern, acronym in _VENUE_ACRONYM_RULES:
        if re.search(pattern, low):
            return acronym
    return v


def compute_citation_component(cited_by_count: int, year: int, current_year: int) -> float:
    """Age-normalized citation impact in [0,1]: log-scaled citations per year."""
    if cited_by_count <= 0:
        return 0.0
    age = max(1, current_year - year + 1) if year > 0 else 5
    cpy = cited_by_count / age
    return min(1.0, math.log1p(cpy) / math.log1p(_CITATION_SATURATION_CPY))


def compute_venue_component(venue: str, sci_rank: str = "", ccf_rank: str = "") -> float:
    """Venue quality in [0,1]. EasyScholar ranks win; keyword heuristic as fallback."""
    scores = []
    if sci_rank:
        scores.append(_SCI_RANK_SCORES.get(sci_rank.strip().upper(), 0.3))
    if ccf_rank:
        scores.append(_CCF_RANK_SCORES.get(ccf_rank.strip().upper(), 0.3))
    if scores:
        return max(scores)

    if not venue:
        return 0.0
    v = normalize_venue(venue).lower()
    if is_preprint_venue(v) or "workshop" in v:
        return 0.2
    for tv in _TOP_VENUES:
        if tv in v:
            return 0.9
    if any(kw in v for kw in ("conference", "symposium", "journal", "transactions", "review")):
        return 0.5
    return 0.3


def compute_recency_component(year: int, current_year: int) -> float:
    """Small recency bonus in [0,1] (only meaningful for the frontier quota)."""
    if year <= 0:
        return 0.0
    diff = max(0, current_year - year)
    return math.exp(-0.3 * diff)


def compute_quality_score(
    node: SphereNode,
    config: SphereConfig,
    current_year: int,
) -> float:
    """Composite paper quality in [0,1]. Deliberately has NO 'novelty' reward:
    an uncited recent paper must earn its place via venue rank or the
    relevance gate, never via absence of citations."""
    s_citation = compute_citation_component(node.cited_by_count, node.year, current_year)
    s_venue = compute_venue_component(node.venue, node.sci_rank, node.ccf_rank)
    s_recency = compute_recency_component(node.year, current_year)

    quality = (
        config.w_citation * s_citation
        + config.w_venue * s_venue
        + config.w_recency * s_recency
    )
    node.quality_score = quality
    return quality


def combined_score(node: SphereNode) -> float:
    """Selection key: LLM-judged relevance × paper quality.

    Unjudged nodes (relevance == -1, e.g. gate failure) count as relevance 1
    so they can still backfill, but never outrank judged relevant papers.
    """
    rel = node.relevance if node.relevance >= 0 else 1
    rel_norm = min(3, max(0, rel)) / 3.0
    score = 0.55 * rel_norm + 0.45 * node.quality_score
    if node.influential:
        score += 0.05  # S2 "highly influential citation" nudge
    node.score_total = score
    return score


# ──────────────────────────────────────────────────────────────────────
# Similarity graph: direct citations + bibliographic coupling
# ──────────────────────────────────────────────────────────────────────

def build_similarity_edges(
    nodes: dict[str, SphereNode],
    min_coupling: int = 2,
    max_edges_per_node: int = 10,
) -> list[SphereEdge]:
    """Build candidate↔candidate edges (Connected Papers-style).

    Two edge kinds:
      - CITES: node A's referenced_ids contain node B's openalex_id
      - COUPLING: A and B share >= min_coupling references
        (weight = overlap / size of the smaller reference list)

    Only nodes with a non-empty referenced_ids participate; the center node
    is treated like any other node.
    """
    from app.models.sphere_models import EdgeType, SphereEdge

    entries = [
        (nid, n, set(n.referenced_ids))
        for nid, n in nodes.items()
        if n.referenced_ids
    ]
    oa_to_nid = {
        n.openalex_id: nid for nid, n in nodes.items() if n.openalex_id
    }

    edges: list[SphereEdge] = []
    seen_pairs: set[tuple[str, str]] = set()
    coupling_candidates: list[tuple[float, str, str]] = []

    # Direct citations among candidates
    for nid, node, refs in entries:
        for ref_oa in refs:
            tgt = oa_to_nid.get(ref_oa)
            if tgt and tgt != nid:
                pair = (nid, tgt)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    edges.append(SphereEdge(
                        source_node_id=nid,
                        target_node_id=tgt,
                        edge_type=EdgeType.CITES,
                    ))

    # Bibliographic coupling (symmetric)
    for i in range(len(entries)):
        nid_a, _, refs_a = entries[i]
        for j in range(i + 1, len(entries)):
            nid_b, _, refs_b = entries[j]
            overlap = len(refs_a & refs_b)
            if overlap >= min_coupling:
                weight = overlap / max(1, min(len(refs_a), len(refs_b)))
                coupling_candidates.append((min(1.0, weight), nid_a, nid_b))

    # Keep the strongest couplings per node to bound graph density
    coupling_candidates.sort(key=lambda t: t[0], reverse=True)
    degree: dict[str, int] = {}
    for weight, nid_a, nid_b in coupling_candidates:
        if degree.get(nid_a, 0) >= max_edges_per_node and degree.get(nid_b, 0) >= max_edges_per_node:
            continue
        edges.append(SphereEdge(
            source_node_id=nid_a,
            target_node_id=nid_b,
            edge_type=EdgeType.COUPLING,
            weight=weight,
        ))
        degree[nid_a] = degree.get(nid_a, 0) + 1
        degree[nid_b] = degree.get(nid_b, 0) + 1

    return edges


def compute_pagerank(
    nodes: dict[str, SphereNode],
    edges: list[SphereEdge],
    damping: float = 0.85,
    iterations: int = 20,
) -> dict[str, float]:
    """Weighted PageRank; undirected edge types (COUPLING/RELATED) count both ways."""
    from app.models.sphere_models import EdgeType

    node_ids = list(nodes.keys())
    n = len(node_ids)
    if n == 0:
        return {}

    out_weight: dict[str, float] = {nid: 0.0 for nid in node_ids}
    incoming: dict[str, list[tuple[str, float]]] = {nid: [] for nid in node_ids}

    def _add(src: str, tgt: str, w: float) -> None:
        if src in out_weight and tgt in incoming:
            out_weight[src] += w
            incoming[tgt].append((src, w))

    for edge in edges:
        w = max(0.0, edge.weight) or 1.0
        _add(edge.source_node_id, edge.target_node_id, w)
        if edge.edge_type in (EdgeType.COUPLING, EdgeType.RELATED):
            _add(edge.target_node_id, edge.source_node_id, w)

    rank: dict[str, float] = {nid: 1.0 / n for nid in node_ids}
    base = (1 - damping) / n
    for _ in range(iterations):
        new_rank: dict[str, float] = {}
        for nid in node_ids:
            s = 0.0
            for src, w in incoming[nid]:
                total_out = out_weight[src]
                if total_out > 0:
                    s += rank[src] * (w / total_out)
            new_rank[nid] = base + damping * s
        rank = new_rank

    max_rank = max(rank.values()) if rank else 1.0
    if max_rank > 0:
        rank = {k: v / max_rank for k, v in rank.items()}
    return rank


# ──────────────────────────────────────────────────────────────────────
# Community detection: label propagation
# ──────────────────────────────────────────────────────────────────────

def label_propagation(
    node_ids: list[str],
    edges: list[SphereEdge],
    iterations: int = 10,
    seed: int = 42,
) -> dict[str, int]:
    """Weighted label propagation over an undirected view of the graph.

    Returns {node_id: community_id} with community ids renumbered by
    descending community size. Deterministic for a fixed seed.
    """
    rng = random.Random(seed)
    id_set = set(node_ids)

    neighbors: dict[str, list[tuple[str, float]]] = {nid: [] for nid in node_ids}
    for e in edges:
        a, b = e.source_node_id, e.target_node_id
        if a in id_set and b in id_set and a != b:
            w = max(0.0, e.weight) or 1.0
            neighbors[a].append((b, w))
            neighbors[b].append((a, w))

    labels: dict[str, int] = {nid: i for i, nid in enumerate(node_ids)}

    order = list(node_ids)
    for _ in range(iterations):
        rng.shuffle(order)
        changed = False
        for nid in order:
            if not neighbors[nid]:
                continue
            weight_by_label: dict[int, float] = {}
            for nbr, w in neighbors[nid]:
                lbl = labels[nbr]
                weight_by_label[lbl] = weight_by_label.get(lbl, 0.0) + w
            best_label = max(
                weight_by_label.items(),
                key=lambda kv: (kv[1], -kv[0]),  # tie-break deterministically
            )[0]
            if best_label != labels[nid]:
                labels[nid] = best_label
                changed = True
        if not changed:
            break

    # Renumber communities by size (0 = largest)
    by_label: dict[int, list[str]] = {}
    for nid, lbl in labels.items():
        by_label.setdefault(lbl, []).append(nid)
    ordered = sorted(by_label.values(), key=len, reverse=True)
    result: dict[str, int] = {}
    for new_id, members in enumerate(ordered):
        for nid in members:
            result[nid] = new_id
    return result
