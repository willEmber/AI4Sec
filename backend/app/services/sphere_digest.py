"""The Sphere map view's prose fields, extracted from a Sphere written as an answer (P16 §5.3).

The map view renders `json_data.sphere_output`: an overview, named themes, a
timeline, hubs, a comparison table, research gaps and reading paths, each
pointing at papers by node id. A Sphere written as the answer has the prose
first. What the literature map already fixed — the partitions, the years, the
hubs — comes from the tool; what the report decided — how it names the themes,
what it compares, which gaps it sees, which paths it recommends — is extracted
from the text here, the way `lens_digest` indexes a Lens report, so the view
cannot say something the report does not.

The model refers to papers by their number in a list it is given (the center
is 0); numbers it makes up are dropped. Best effort: a digest that will not
parse returns `None`, and the view shows the tool's parts only.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.models.sphere_models import (
    ComparisonRow,
    GapIdea,
    ReadingPath,
    SphereNode,
    SphereOutput,
    ThemeCluster,
    TimelineEntry,
)

logger = logging.getLogger(__name__)

SPHERE_DIGEST_SYSTEM = """You convert a finished "Research Sphere" report — a literature landscape around ONE center paper — into the JSON its map view renders.

The report is the source of truth. Extract only what it states and reuse its wording; add nothing of your own. Papers are referred to by their number in the PAPERS list, where 0 is the center paper. In the report, a `[#n]` after a statement cites paper n: use those marks, and the titles, to tell which papers a theme, row, gap or path is about. Leave out a paper you cannot match; never guess a number.

Return ONLY a JSON object — no markdown fences, no commentary:

{
  "overview": "The report's landscape overview, as plain prose (1-3 paragraphs).",
  "themes": [{"name": "theme name", "definition": "1-2 sentences", "papers": [3, 7]}],
  "timeline": [{"year": 2021, "summary": "what the report says happened that year, at most 2 sentences"}],
  "comparison": [{"paper": 0, "problem": "", "assumption": "", "method": "", "dataset": "", "metric": "", "strength": "", "weakness": ""}],
  "gaps": [{"title": "gap title", "description": "the report's explanation", "papers": [2, 5]}],
  "reading_paths": [{"name": "fast", "description": "what the path is for", "papers": [0, 1, 3]}]
}

Rules:
1. Themes, comparison rows, gaps and reading paths as the report gives them — no padding. An empty list when the report has none.
2. A comparison row per paper of the report's comparison table, in its order, cells as written ("" for a cell it leaves empty; "assumption" is usually "").
3. A reading path's name is "fast", "deep" or "frontier" — the closest of the three for each path the report recommends.
4. One timeline entry per year the report's timeline discusses.
5. Drop citation marks such as [#3] from the strings.
6. Write every string in the language the user message names, as the report is written. JSON keys stay in English exactly as specified.
"""

_REPAIR_DIRECTIVE = """
Your previous response could not be parsed as JSON. Return the same digest as a
single valid JSON object and nothing else: no prose before or after it, no
markdown fences, no trailing commas, all strings double-quoted.
"""

_MAX_REPORT_CHARS = 40_000
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_MARK_RE = re.compile(r"\s*\[(?:#\d+|p\.[^\]]*)\]")
_PATH_NAMES = ("fast", "deep", "frontier")


def _paper_list(papers: list[SphereNode]) -> str:
    return "\n".join(
        f"[{i}] {node.title}" + (f" ({node.year})" if node.year else "")
        for i, node in enumerate(papers)
    )


def _parse(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = _FENCE_RE.sub("", text).strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _text(value: Any, limit: int) -> str:
    return _MARK_RE.sub("", value).strip()[:limit] if isinstance(value, str) else ""


def _ids(value: Any, papers: list[SphereNode], *, center: bool = True) -> list[str]:
    out: list[str] = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, bool) or not isinstance(item, int):
            continue
        if 0 <= item < len(papers) and (center or item > 0):
            nid = papers[item].node_id
            if nid not in out:
                out.append(nid)
    return out


def apply_digest(output: SphereOutput, data: dict[str, Any], papers: list[SphereNode]) -> None:
    """Write the report's parts into `output`, mapping paper numbers to node ids.

    The timeline keeps the map's years and papers; the report adds what it
    says about each year. Partitions and hubs are left as the map made them.
    """
    output.sphere_overview = _text(data.get("overview"), 4000)

    output.themes = [
        ThemeCluster(
            name=_text(t.get("name"), 120),
            definition=_text(t.get("definition"), 600),
            representative_node_ids=_ids(t.get("papers"), papers, center=False),
        )
        for t in (data.get("themes") or [])[:10]
        if isinstance(t, dict) and _text(t.get("name"), 120)
    ]

    rows: list[ComparisonRow] = []
    seen: set[str] = set()
    for row in (data.get("comparison") or [])[:15]:
        if not isinstance(row, dict):
            continue
        ids = _ids([row.get("paper")], papers)
        if not ids or ids[0] in seen:
            continue
        seen.add(ids[0])
        node = next(n for n in papers if n.node_id == ids[0])
        rows.append(ComparisonRow(
            node_id=ids[0],
            title=node.title,
            **{k: _text(row.get(k), 400) for k in
               ("problem", "assumption", "method", "dataset", "metric", "strength", "weakness")},
        ))
    output.comparison_table = rows

    output.gaps_and_ideas = [
        GapIdea(
            title=_text(g.get("title"), 200),
            description=_text(g.get("description"), 1200),
            evidence_node_ids=_ids(g.get("papers"), papers),
        )
        for g in (data.get("gaps") or [])[:8]
        if isinstance(g, dict) and _text(g.get("title"), 200)
    ]

    paths: list[ReadingPath] = []
    for path in (data.get("reading_paths") or [])[:6]:
        if not isinstance(path, dict):
            continue
        name = _text(path.get("name"), 20).casefold()
        ids = _ids(path.get("papers"), papers)
        if not ids:
            continue
        paths.append(ReadingPath(
            name=name if name in _PATH_NAMES else "deep",
            description=_text(path.get("description"), 400),
            node_ids=ids,
        ))
    output.reading_paths = paths

    summaries: dict[int, str] = {}
    for entry in data.get("timeline") or []:
        if isinstance(entry, dict) and isinstance(entry.get("year"), int):
            summary = _text(entry.get("summary"), 400)
            if summary:
                summaries[entry["year"]] = summary
    timeline = {e.year: e for e in output.timeline}
    for year, summary in summaries.items():
        timeline.setdefault(year, TimelineEntry(year=year)).summary = summary
    output.timeline = [timeline[y] for y in sorted(timeline)]


async def generate_sphere_digest(
    llm: Any,
    *,
    markdown: str,
    papers: list[SphereNode],
    language: str = "en",
    model: str = "",
    max_tokens: int = 8192,
    log_label: str = "sphere",
) -> dict[str, Any] | None:
    """The report's map-view fields as JSON with paper numbers; never raises."""
    report = (markdown or "").strip()
    if not report or not papers:
        return None
    if len(report) > _MAX_REPORT_CHARS:
        report = report[:_MAX_REPORT_CHARS] + "\n[...truncated...]"

    # Named outright: an English instruction and English titles drew English
    # strings out of a Chinese report.
    written_in = "Simplified Chinese" if language == "zh" else "English"
    user = (
        f"Write the strings in {written_in}.\n\nPAPERS:\n{_paper_list(papers)}\n\n"
        f"Research Sphere report:\n\n{report}"
    )
    for attempt in (1, 2):
        try:
            raw = await llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": SPHERE_DIGEST_SYSTEM + (_REPAIR_DIRECTIVE if attempt == 2 else ""),
                    },
                    {"role": "user", "content": user},
                ],
                model=model,
                temperature=0.0,
                max_tokens=max_tokens if attempt == 1 else int(max_tokens * 1.5),
                # Utility call: thinking off (see `llm_gateway.registry`).
                enable_thinking=False,
            )
        except Exception as exc:  # noqa: BLE001 — the report stands without its map view
            logger.warning("%s: sphere digest attempt %d failed: %s", log_label, attempt, exc)
            continue
        data = _parse(raw)
        if data is not None:
            return data
        if attempt == 1:
            logger.info("%s: sphere digest JSON unusable — retrying with repair prompt", log_label)

    logger.warning("%s: sphere digest failed twice — the view shows the map only", log_label)
    return None
