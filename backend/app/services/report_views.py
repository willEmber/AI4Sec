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


BUILDERS: dict[str, Builder] = {"lens": lens_view}
