from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.sphere_models import SphereConfig, SphereEdge, SphereNode

# Ensure paper_search utils are importable
_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from paper_search.paper_search.utils import jaccard_similarity


def compute_text_similarity(
    center_title: str,
    center_abstract: str,
    cand_title: str,
    cand_abstract: str,
) -> float:
    """Weighted Jaccard: 3x weight on title tokens, 1x on abstract tokens."""
    title_sim = jaccard_similarity(center_title, cand_title)
    abstract_sim = jaccard_similarity(center_abstract, cand_abstract)
    # Weight title 3x more than abstract
    if abstract_sim > 0:
        return (3 * title_sim + abstract_sim) / 4
    return title_sim


def compute_pagerank(
    nodes: dict[str, SphereNode],
    edges: list[SphereEdge],
    damping: float = 0.85,
    iterations: int = 20,
) -> dict[str, float]:
    """Pure Python iterative PageRank for a small graph (<500 nodes)."""
    node_ids = list(nodes.keys())
    n = len(node_ids)
    if n == 0:
        return {}

    # Build adjacency: target -> list of sources (who links TO target)
    outgoing_count: dict[str, int] = {nid: 0 for nid in node_ids}
    incoming: dict[str, list[str]] = {nid: [] for nid in node_ids}

    for edge in edges:
        src = edge.source_node_id
        tgt = edge.target_node_id
        if src in outgoing_count and tgt in incoming:
            outgoing_count[src] += 1
            incoming[tgt].append(src)

    # Initialize ranks
    rank: dict[str, float] = {nid: 1.0 / n for nid in node_ids}

    base = (1 - damping) / n
    for _ in range(iterations):
        new_rank: dict[str, float] = {}
        for nid in node_ids:
            s = 0.0
            for src in incoming[nid]:
                out = outgoing_count[src]
                if out > 0:
                    s += rank[src] / out
            new_rank[nid] = base + damping * s
        rank = new_rank

    # Normalize to [0, 1]
    max_rank = max(rank.values()) if rank else 1.0
    if max_rank > 0:
        rank = {k: v / max_rank for k, v in rank.items()}

    return rank


def compute_time_score(year: int, current_year: int = 2026) -> float:
    """Exponential decay: recent papers score higher. Clamped to [0, 1]."""
    if year <= 0:
        return 0.0
    diff = current_year - year
    if diff < 0:
        diff = 0
    return max(0.0, min(1.0, math.exp(-0.1 * diff)))


def compute_venue_score(venue: str) -> float:
    """Simple heuristic venue scoring. Returns [0, 1]."""
    if not venue:
        return 0.0
    v = venue.lower()
    # Top-tier venues get higher scores
    top_venues = {
        "nature", "science", "cell", "lancet", "nejm", "bmj",
        "neurips", "nips", "icml", "iclr", "cvpr", "iccv", "eccv",
        "aaai", "ijcai", "acl", "emnlp", "naacl",
        "sigmod", "vldb", "kdd", "www", "sigir",
        "ieee transactions", "acm transactions",
        "pnas", "physical review", "jama",
    }
    for tv in top_venues:
        if tv in v:
            return 0.9
    # Conference/journal heuristic
    if any(kw in v for kw in ("conference", "symposium", "workshop", "journal", "transactions", "review")):
        return 0.5
    if "arxiv" in v or "preprint" in v:
        return 0.2
    return 0.3


def multi_objective_score(
    node: SphereNode,
    pagerank_scores: dict[str, float],
    center_title: str,
    center_abstract: str,
    config: SphereConfig,
) -> float:
    """Compute weighted multi-objective score for a candidate node."""
    s_text = compute_text_similarity(
        center_title, center_abstract, node.title, node.abstract_text
    )
    s_graph = pagerank_scores.get(node.node_id, 0.0)
    s_time = compute_time_score(node.year)
    s_venue = compute_venue_score(node.venue)

    # Novelty: inverse of citation count (normalized)
    # Very highly cited = low novelty, low cited = high novelty
    if node.cited_by_count > 0:
        s_novelty = 1.0 / (1.0 + math.log(node.cited_by_count))
    else:
        s_novelty = 1.0

    node.score_text = s_text
    node.score_graph = s_graph
    node.score_time = s_time
    node.score_venue = s_venue
    node.score_novelty = s_novelty

    total = (
        config.w_text * s_text
        + config.w_graph * s_graph
        + config.w_time * s_time
        + config.w_venue * s_venue
        + config.w_novelty * s_novelty
    )
    node.score_total = total
    return total


def mmr_select(
    candidates: list[SphereNode],
    k: int,
    lambda_param: float = 0.7,
) -> list[str]:
    """Greedy MMR (Maximal Marginal Relevance) diversity selection.

    Returns up to k node IDs balancing relevance and diversity.
    """
    if not candidates:
        return []
    if len(candidates) <= k:
        return [c.node_id for c in candidates]

    # Sort by total score descending
    sorted_cands = sorted(candidates, key=lambda c: c.score_total, reverse=True)

    selected: list[SphereNode] = [sorted_cands[0]]
    selected_ids: set[str] = {sorted_cands[0].node_id}
    remaining = sorted_cands[1:]

    while len(selected) < k and remaining:
        best_score = -1.0
        best_idx = 0

        for i, cand in enumerate(remaining):
            relevance = cand.score_total

            # Max similarity to already selected
            max_sim = 0.0
            for sel in selected:
                sim = jaccard_similarity(cand.title + " " + cand.abstract_text,
                                         sel.title + " " + sel.abstract_text)
                if sim > max_sim:
                    max_sim = sim

            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_idx = i

        selected.append(remaining[best_idx])
        selected_ids.add(remaining[best_idx].node_id)
        remaining.pop(best_idx)

    return [s.node_id for s in selected]
