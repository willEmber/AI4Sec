"""The literature map a Research Sphere written as the answer stands on (P16 §4.1).

`build_literature_map` runs the deterministic part of Research Sphere — the
reference extraction, citation-graph expansion, enrichment, quality scoring,
relevance gate, coupling graph and quota-based core set, then a short analysis
of each core paper's abstract — and hands the conversation the result. The
landscape itself (themes, comparison, gaps, reading paths) is the report the
conversation then writes.

The funnel takes minutes and does not depend on the conversation, so it is a
`sphere_map:{paper}:{language}` job: a second report on the paper reuses it.
What does depend on the reader is registered on every call:

* each core paper (and each hub outside the core) becomes a work in the
  catalogue, its abstract `abstract` evidence and its venue and citation
  count `metadata` evidence — nothing here is `fulltext`;
* each is attached to the session as a candidate (`added_by='sphere'`), so a
  follow-up can `download_paper` it by `literature_id` instead of searching
  again. A paper the session already holds keeps what it had.

The open Sphere report receives the map itself (`ReportTurn.parts["map"]`),
from which its structured view is built.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from langchain.tools import ToolRuntime, tool

from app.agents.context import AgentContext
from app.agents.tools.triage import _paper
from app.db import agent_repository as repo
from app.models.agent_models import ErrorCode, EventType, ToolResult
from app.models.sphere_models import SphereNode, SphereState
from app.services import agent_jobs, evidence_service, paper_catalog
from app.services.sphere_scorer import normalize_venue
from app.workflows import progress
from app.workflows.sphere_subgraph import map_papers

logger = logging.getLogger("scholar.agents.tools.literature_map")


async def _emit(ctx: AgentContext, type: EventType, payload: dict[str, Any]) -> None:
    """Publish an event for this turn. Lazy import: the runner imports the tools."""
    from app.services import agent_runner

    with contextlib.suppress(Exception):
        await agent_runner.emit(
            session_id=ctx.session_id, run_id=ctx.run_id, type=type, payload=payload
        )


async def _forward_progress(
    ctx: AgentContext, tool_name: str, queue: asyncio.Queue[Any]
) -> None:
    """Turn the run's step progress into `tool.progress` events."""
    while True:
        item = await queue.get()
        if item is None:
            return
        await _emit(ctx, EventType.TOOL_PROGRESS, {"tool": tool_name, **item})

_TEXT_LIMIT = 240
_ABSTRACT_EXCERPT = 400


def map_key(*, paper_id: str, language: str) -> str:
    return f"sphere_map:{paper_id}:{language}"


def _url(node: SphereNode) -> str:
    if node.doi:
        doi = node.doi.removeprefix("https://doi.org/").removeprefix("http://dx.doi.org/")
        return f"https://doi.org/{doi}"
    if node.arxiv_id:
        return f"https://arxiv.org/abs/{node.arxiv_id}"
    return ""


def _rank(node: SphereNode) -> str:
    return " · ".join(
        part for part in (f"CCF {node.ccf_rank}" if node.ccf_rank else "",
                          f"JCR {node.sci_rank}" if node.sci_rank else "") if part
    )


def _metadata_quote(node: SphereNode, *, as_of: str) -> str:
    parts = [node.title + (f" ({node.year})" if node.year else "")]
    if node.venue:
        venue = normalize_venue(node.venue) or node.venue
        rank = _rank(node)
        parts.append(f"published at {venue}" + (f" ({rank})" if rank else ""))
    if node.cited_by_count:
        parts.append(f"cited by {node.cited_by_count:,} (OpenAlex / Semantic Scholar, as of {as_of})")
    return " — ".join(parts)


async def _register(
    ctx: AgentContext, entries: Iterable[tuple[str, SphereNode]], *, as_of: str
) -> list[dict[str, Any]]:
    """Each paper as a work with its evidence: `[{group, node, literature_id, ids}]`."""
    out = []
    for group, node in entries:
        literature_id = await paper_catalog.upsert_literature_item(
            title=node.title,
            doi=node.doi,
            arxiv_id=node.arxiv_id,
            openalex_id=node.openalex_id,
            s2_paper_id=node.s2_paper_id,
            authors=[a.strip() for a in (node.authors or "").split(";") if a.strip()],
            year=node.year or None,
            venue=node.venue,
            abstract=node.abstract_text,
            url=_url(node),
            source="research_sphere",
        )
        abstract = (node.abstract_text or "").strip()
        abstract_id = ""
        if len(abstract) > 30:
            abstract_id = (await evidence_service.record_abstract_evidence(
                literature_id=literature_id,
                quote=abstract,
                owner_id=ctx.owner_id,
                session_id=ctx.session_id,
                provider="research_sphere",
                source_url=_url(node),
            )).evidence_id
        metadata_id = (await evidence_service.record_metadata_evidence(
            literature_id=literature_id,
            quote=_metadata_quote(node, as_of=as_of),
            owner_id=ctx.owner_id,
            session_id=ctx.session_id,
            provider="research_sphere",
            source_url=_url(node),
        )).evidence_id
        out.append({
            "group": group, "node": node, "literature_id": literature_id,
            "abstract_id": abstract_id, "metadata_id": metadata_id,
        })
    return out


def _paper_entry(n: int, item: dict[str, Any]) -> dict[str, Any]:
    node: SphereNode = item["node"]
    entry: dict[str, Any] = {
        "n": n,
        "literature_id": item["literature_id"],
        "group": item["group"],
        "relation": node.relation_type.value,
        "title": node.title,
        "year": node.year or None,
        "venue": normalize_venue(node.venue) if node.venue else "",
        "rank": _rank(node),
        "cited_by": node.cited_by_count,
        "why": (node.relation_reason or node.reason)[:_TEXT_LIMIT],
        "method": node.method_summary[:_TEXT_LIMIT],
        "contribution": node.contribution[:_TEXT_LIMIT],
        "task": node.task[:120],
        "dataset": node.dataset[:120],
        "evidence_id": item["abstract_id"] or item["metadata_id"],
        "metadata_evidence_id": item["metadata_id"],
    }
    if not node.method_summary and item["abstract_id"]:
        entry["abstract_excerpt"] = node.abstract_text.strip()[:_ABSTRACT_EXCERPT]
    if not item["abstract_id"]:
        entry["evidence_level"] = "metadata"
    return {k: v for k, v in entry.items() if v not in ("", None) or k in ("n", "evidence_id")}


def _hand_to_report(ctx: AgentContext, paper_id: str, result: dict[str, Any]) -> None:
    report = ctx.report
    if report is not None and report.mode == "sphere" and report.paper_id == paper_id:
        report.parts["map"] = result


@tool(parse_docstring=False)
async def build_literature_map(runtime: ToolRuntime[AgentContext], paper_id: str = "") -> str:
    """Map the literature around a parsed paper: its core related work, grouped and ranked.

    Starts from the paper's references, expands through the citation graph and
    a keyword search, scores candidates by citations, venue tier and recency,
    has each judged for its relation to this paper, and keeps a core set by
    quota per relation — foundations, method neighbours and competitors,
    follow-ups, applications, the frontier, surveys. Each core paper comes with
    why it was kept, its venue and citation count, and what its abstract says
    it does, each citable; clusters group papers by shared references and hubs
    are the most central in that graph. The papers are added to this
    conversation as candidates, so any of them can be downloaded by its
    `literature_id`. Takes several minutes the first time; asking again returns
    the same map. Omit paper_id when the session holds one paper.
    """
    ctx = runtime.context
    paper_id, paper_ir, denial = await _paper(ctx, paper_id)
    if denial is not None:
        return denial.to_json()
    language = "zh" if ctx.language == "zh" else "en"

    report = ctx.report
    in_report = report is not None and report.mode == "sphere" and report.paper_id == paper_id
    # The steps report progress on a run id. Inside a report it is the report's
    # own row, which then also holds the map's nodes; outside one, an id only
    # this call listens on.
    progress_id = report.run_id if in_report else f"map-{uuid.uuid4().hex[:12]}"

    async def _work(_job_id: str) -> dict[str, Any]:
        from app.workflows.sphere_subgraph import build_literature_map as build
        from app.workflows.sphere_subgraph import trim_to_map

        state: dict[str, Any] = {
            "paper_id": paper_id,
            "llm_model": ctx.llm_model,
            "language": language,
            "paper_ir_json": paper_ir.model_dump_json(),
            "persist_sphere": in_report,
        }
        sphere = await build(state, progress_id)
        if not sphere.layer1_node_ids:
            # Not cached: an outage of the indexes looks like this too.
            raise RuntimeError("No related papers survived the funnel.")
        return {
            "state": trim_to_map(sphere).model_dump(mode="json"),
            "num_nodes": len(sphere.nodes),
            "num_edges": len(sphere.edges),
        }

    queue = progress.subscribe(progress_id)
    forwarder = asyncio.create_task(
        _forward_progress(ctx, "build_literature_map", queue)
    )
    try:
        handle = await agent_jobs.run_once(
            kind="sphere_map",
            idempotency_key=map_key(paper_id=paper_id, language=language),
            work=_work,
            session_id=ctx.session_id,
            run_id=ctx.run_id,
            request={"paper_id": paper_id, "language": language},
        )
    except agent_jobs.JobFailed as exc:
        return ToolResult.failed(
            ErrorCode.UPSTREAM_ERROR, f"The literature map failed: {exc.message}"[:400], retryable=False
        ).to_json()
    finally:
        progress.unsubscribe(progress_id, queue)
        queue.put_nowait(None)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(forwarder, timeout=5)

    if handle.status == "running":
        return ToolResult.partial(
            {"paper_id": paper_id},
            note="Another turn is building this map. Do not start it again; say the "
            "literature map is still being built and answer from what you have.",
        ).to_json()

    sphere = SphereState.model_validate(handle.result["state"])
    as_of = datetime.now(UTC).date().isoformat()
    registered = await _register(ctx, map_papers(sphere), as_of=as_of)
    await repo.add_session_candidates(
        session_id=ctx.session_id,
        literature_ids=[item["literature_id"] for item in registered],
        added_by="sphere",
        note=f"literature map of {paper_ir.title[:120]}",
    )
    _hand_to_report(ctx, paper_id, {
        **handle.result,
        "literature_ids": {item["node"].node_id: item["literature_id"] for item in registered},
    })

    number = {item["node"].node_id: n for n, item in enumerate(registered, start=1)}
    clusters: dict[int, list[int]] = {}
    for item in registered:
        node = item["node"]
        if item["group"] != "hub" and node.cluster_id >= 0:
            clusters.setdefault(node.cluster_id, []).append(number[node.node_id])
    center = sphere.center_node or SphereNode(title=paper_ir.title)
    # This paper's own record, as the indexes the map asked know it: the report
    # opens by placing it, and would otherwise look it up again.
    center_id = ""
    if center.venue or center.cited_by_count:
        center_id = (await evidence_service.record_metadata_evidence(
            literature_id=await paper_catalog.ensure_literature_for_local_paper(paper_id),
            quote=_metadata_quote(center.model_copy(update={"title": center.title or paper_ir.title}),
                                  as_of=as_of),
            owner_id=ctx.owner_id,
            session_id=ctx.session_id,
            provider="research_sphere",
            source_url=_url(center),
        )).evidence_id
    data: dict[str, Any] = {
        "paper_id": paper_id,
        "title": paper_ir.title,
        "center": {
            k: v for k, v in {
                "title": center.title or paper_ir.title,
                "year": center.year or None,
                "venue": normalize_venue(center.venue) if center.venue else "",
                "cited_by": center.cited_by_count or None,
                "evidence_id": center_id,
            }.items() if v
        },
        "counts": {
            "candidates_weighed": handle.result.get("num_nodes", 0),
            "core": len(sphere.layer1_node_ids),
            "hubs": len(sphere.output.key_hubs),
        },
        "papers": [_paper_entry(n, item) for n, item in enumerate(registered, start=1)],
        "clusters": [
            {"cluster": cid, "papers": members}
            for cid, members in sorted(clusters.items())
            if len(members) >= 2
        ],
        "hubs": [number[h.node_id] for h in sphere.output.key_hubs if h.node_id in number],
        "candidates_added": len(registered),
        "reused": handle.reused,
    }
    if sphere.library_matches:
        data["library_matches"] = [
            {"title": m.title, "score": round(m.score, 3)} for m in sphere.library_matches[:10]
        ]
    evidence_ids = [
        i for item in registered for i in (item["abstract_id"], item["metadata_id"]) if i
    ] + ([center_id] if center_id else [])
    return ToolResult.ok(
        data,
        evidence_ids=evidence_ids,
        note="center is this paper's own venue and citation count. "
        "evidence_id is a paper's abstract (abstract level, not full text; "
        "metadata when it has none) and metadata_evidence_id its venue and citation "
        "count: cite them for what a paper does and how it is placed. group is the "
        "paper's partition (hub: central in the coupling graph but outside the core "
        "set); clusters and hubs refer to papers by n and are unnamed — name the "
        "themes yourself. Every paper is now a candidate in this conversation; "
        "download_paper(literature_id) reads one in full.",
    ).to_json()


LITERATURE_MAP_TOOLS = [build_literature_map]
