"""The structured part of a report written as an answer (P16 §5).

A report mode's page shows cards beside the prose: Lens its digest and
framework figures, Snap its scorecard, Sphere its map. When the report was the
answer of a turn, those are built here at snapshot time, from the finished
text and the paper's parse, into the same `json_data` fields the report tools
wrote, so the report page and its views read both kinds alike.

Each builder is best-effort: the snapshot keeps the report without its
structured view when one raises. Model calls made here are utility calls
(thinking off) and count on the turn's meter.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from app.config import get_settings
from app.models.lens_models import LensDigest, LensFigure
from app.models.snap_models import SnapReport, TriageSignals
from app.models.sphere_models import SphereNode, SphereState
from app.services import report_citations
from app.services.ir_extract import (
    extract_algorithms,
    extract_equations,
    extract_figures,
    extract_tables,
    figure_embed_url,
    select_framework_figures,
)
from app.services.lens_digest import generate_lens_digest
from app.services.llm_gateway import get_registry
from app.services.llm_service import get_llm_service
from app.services.paper_ir_loader import load_paper_ir
from app.services.snap_digest import generate_snap_digest
from app.services.sphere_digest import apply_digest, generate_sphere_digest
from app.services.triage_verdict import ContentReview, synthesize_verdict

if TYPE_CHECKING:
    from app.services.answer_reports import ReportTurn

logger = logging.getLogger(__name__)

# (report, frozen text, its citation table, language) -> fields of json_data
Builder = Callable[
    ["ReportTurn", str, dict[str, dict[str, Any]], str], Awaitable[dict[str, Any]]
]


async def lens_view(
    report: ReportTurn, text: str, citations: dict[str, dict[str, Any]], language: str
) -> dict[str, Any]:
    """Counts, framework figures and the digest the Lens cards render.

    Counts and figures come from the parse, as the elements tool lists them,
    so a figure card shows an image this paper has rather than a URL the
    model wrote. The digest is extracted from the report, as before; it copies
    page citations into the cards, so it reads the report with this paper's
    passages cited by page.
    """
    paper_ir = await load_paper_ir(report.paper_id)
    figures = extract_figures(paper_ir)
    key_figures, _rest = select_framework_figures(figures) if figures else ([], [])
    data: dict[str, Any] = {
        "num_equations": len(extract_equations(paper_ir)),
        "num_algorithms": len(extract_algorithms(paper_ir)),
        "num_tables": len(extract_tables(paper_ir)),
        "num_figures": len(figures),
        "key_figures": [
            LensFigure(
                page=f["page"],
                caption=f["text"][:300],
                url=figure_embed_url(report.paper_id, f.get("img_path", "")),
            ).model_dump(mode="json")
            for f in key_figures
            if f.get("img_path")
        ],
        "digest": LensDigest().model_dump(mode="json"),
    }
    if get_settings().lens_digest_enabled:
        digest = await generate_lens_digest(
            get_llm_service(),
            markdown=report_citations.to_pages(text, citations, paper_id=report.paper_id),
            model=get_registry().utility_model,
            log_label=f"[run:{report.run_id}] lens",
        )
        data["digest"] = digest.model_dump(mode="json")
    return data


async def snap_view(
    report: ReportTurn, text: str, citations: dict[str, dict[str, Any]], language: str
) -> dict[str, Any]:
    """The scorecard: the report's fields from the text, the rest from the tools.

    Signals, blind review and verdict are what `get_triage_signals` and
    `review_paper_blind` returned (`report.parts`), never the report's
    retelling of them. Without a review the verdict rests on the signals
    alone and says so (`content_available` false); without either there is no
    verdict, and the page shows the report's text instead of the scorecard.
    """
    data: dict[str, Any] = {}
    parts = report.parts
    signals = parts.get("signals")
    review = parts.get("content_review")
    if signals:
        data["signals"] = signals
        data["content_review"] = review or ContentReview().model_dump(mode="json")
        data["verdict"] = parts.get("verdict") or synthesize_verdict(
            ContentReview.model_validate(review or {}), TriageSignals.model_validate(signals)
        ).model_dump(mode="json")
    if parts.get("context_stats"):
        data["context_stats"] = parts["context_stats"]
    digest: SnapReport = await generate_snap_digest(
        get_llm_service(),
        markdown=report_citations.to_pages(text, citations, paper_id=report.paper_id),
        model=get_registry().utility_model,
        log_label=f"[run:{report.run_id}] snap",
    )
    data["report"] = digest.model_dump(mode="json")
    return data


async def sphere_view(
    report: ReportTurn, text: str, citations: dict[str, dict[str, Any]], language: str
) -> dict[str, Any]:
    """The map view: the literature map the tool built, the report's reading of it.

    Nodes, partitions, hubs, the timeline's years, the funnel and the counts
    are what `build_literature_map` handed the report (`report.parts["map"]`);
    the overview, themes, comparison, gaps, reading paths and what the
    timeline says of each year are extracted from the text. Without a map
    there is no view, and the page shows the report's text.
    """
    from app.workflows.sphere_subgraph import map_papers, sphere_json_fields

    found = report.parts.get("map")
    if not found:
        return {}
    sphere = SphereState.model_validate(found["state"])
    papers = [sphere.center_node or SphereNode()] + [node for _group, node in map_papers(sphere)]
    literature_ids = found.get("literature_ids") or {}
    numbers = {
        literature_ids[node.node_id]: n
        for n, node in enumerate(papers)
        if n and node.node_id in literature_ids
    }
    digest = await generate_sphere_digest(
        get_llm_service(),
        markdown=report_citations.to_works(
            text, citations, paper_id=report.paper_id, numbers=numbers
        ),
        papers=papers,
        language=language,
        model=get_registry().utility_model,
        log_label=f"[run:{report.run_id}] sphere",
    )
    if digest is not None:
        apply_digest(sphere.output, digest, papers)
    fields = sphere_json_fields(
        sphere,
        paper_id=report.paper_id,
        language=language,
        num_nodes=int(found.get("num_nodes") or len(sphere.nodes)),
        num_edges=int(found.get("num_edges") or 0),
    )
    # The snapshot's own identity fields stand.
    for key in ("mode", "paper_id", "title", "language"):
        fields.pop(key, None)
    return fields


BUILDERS: dict[str, Builder] = {"lens": lens_view, "snap": snap_view, "sphere": sphere_view}
