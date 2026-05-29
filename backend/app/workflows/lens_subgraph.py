from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from app.models.paper_ir import PaperIR
from app.services.citation_validator import (
    format_coverage_summary,
    validate_citation_coverage,
)
from app.services.evidence_extractor import extract_evidence_pool
from app.services.llm_service import get_llm_service
from app.workflows.state import MainGraphState

logger = logging.getLogger("scholar.graph")

# Slot names exposed to the evidence extractor. Aligned with the four parts of
# the reader-oriented Lens prompt (overview/motivation → method → experiments →
# critical assessment) so the second LLM call can map quotes back to a section.
LENS_SLOTS: list[str] = [
    "motivation",
    "contribution",
    "method",
    "equation",
    "algorithm",
    "figure",
    "dataset_metric",
    "results",
    "limitation",
]

_SECTION_NUM_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)")


def _section_match_keys(paper_ir: PaperIR) -> dict[str, str]:
    """Map each ``section.title`` to a lowercased match string that also includes
    the titles of its numeric ancestors.

    MinerU sometimes flattens heading levels, collapsing ``section_path`` to the
    leaf (e.g. ``5.3 Optimizer`` loses its ``5 Training`` parent). We rebuild the
    ancestor chain from section numbering so a sub-section inherits its parent's
    keyword matches — e.g. ``5.3 Optimizer`` then matches the ``training`` keyword
    and its optimizer/batch/hardware details are no longer dropped.
    """
    num_to_title: dict[str, str] = {}
    for section in paper_ir.sections:
        m = _SECTION_NUM_RE.match(section.title)
        if m:
            num_to_title[m.group(1)] = section.title

    keys: dict[str, str] = {}
    for section in paper_ir.sections:
        titles = [section.title]
        m = _SECTION_NUM_RE.match(section.title)
        if m:
            parts = m.group(1).split(".")
            for i in range(1, len(parts)):
                anc_title = num_to_title.get(".".join(parts[:i]))
                if anc_title:
                    titles.append(anc_title)
        keys[section.title] = " ".join(titles).lower()
    return keys


def _extract_equations(paper_ir: PaperIR) -> list[dict[str, Any]]:
    """Extract equation blocks with context."""
    equations = []
    for block in paper_ir.blocks:
        if block.type in ("equation", "isolate_formula") or "formula" in block.sub_type.lower():
            equations.append({
                "text": block.text,
                "page": block.page_idx + 1,
                "section": block.section_path,
                "bbox": block.bbox,
            })
    return equations


def _extract_algorithms(paper_ir: PaperIR) -> list[dict[str, Any]]:
    """Extract algorithm/code blocks."""
    algos = []
    for block in paper_ir.blocks:
        if block.type in ("code", "algorithm") or "algorithm" in block.sub_type.lower():
            algos.append({
                "text": block.text,
                "page": block.page_idx + 1,
                "section": block.section_path,
                "bbox": block.bbox,
            })
    return algos


def _extract_tables(paper_ir: PaperIR) -> list[dict[str, Any]]:
    """Extract table blocks."""
    tables = []
    for block in paper_ir.blocks:
        if block.type == "table":
            tables.append({
                "text": block.text,
                "page": block.page_idx + 1,
                "section": block.section_path,
                "bbox": block.bbox,
            })
    return tables


def _extract_method_section(paper_ir: PaperIR) -> str:
    """Extract method-related text.

    Matches each section against an ancestor-aware key so nested sub-sections
    (e.g. ``3.2.1 Scaled Dot-Product Attention``) inherit the match from their
    parent (``3 Model Architecture``).
    """
    method_keywords = {
        "method", "approach", "model", "framework", "architecture",
        "proposed", "algorithm", "implementation", "design",
    }
    match_keys = _section_match_keys(paper_ir)
    parts: list[str] = []
    for section in paper_ir.sections:
        key = match_keys.get(section.title, section.title.lower())
        if not any(kw in key for kw in method_keywords):
            continue
        for block in section.blocks:
            if block.type in ("text", "title", "list", "equation"):
                text = block.text.strip()
                if text:
                    parts.append(f"{text} [p.{block.page_idx + 1}]")
    return "\n".join(parts)


def _extract_experiment_section(paper_ir: PaperIR) -> str:
    """Extract experiment / training / results text.

    Includes ``training`` and ``dataset`` keywords and matches against an
    ancestor-aware section key, so reproduction details (optimizer, batch size,
    hardware, schedule) that papers nest under a "Training" section are captured
    instead of surfacing as "not reported".
    """
    exp_keywords = {
        "experiment", "evaluation", "result", "empirical", "ablation",
        "setup", "training", "implementation", "dataset", "benchmark",
    }
    match_keys = _section_match_keys(paper_ir)
    parts: list[str] = []
    for section in paper_ir.sections:
        key = match_keys.get(section.title, section.title.lower())
        if not any(kw in key for kw in exp_keywords):
            continue
        for block in section.blocks:
            if block.type in ("text", "title", "list", "table"):
                text = block.text.strip()
                if text:
                    parts.append(f"{text} [p.{block.page_idx + 1}]")
    return "\n".join(parts)


def _extract_framing_section(paper_ir: PaperIR) -> str:
    """Extract abstract / introduction / related-work / conclusion text.

    Feeds the *Overview & Motivation* part of the report (background, the gap in
    prior work, and the paper's contributions) — material the method/experiment
    extractors do not cover.
    """
    framing_keywords = {
        "abstract", "introduction", "related work", "related", "background",
        "motivation", "conclusion", "conclusions", "summary", "discussion",
    }
    match_keys = _section_match_keys(paper_ir)
    parts: list[str] = []
    for section in paper_ir.sections:
        key = match_keys.get(section.title, section.title.lower())
        if not any(kw in key for kw in framing_keywords):
            continue
        for block in section.blocks:
            if block.type in ("text", "title", "list"):
                text = block.text.strip()
                if text:
                    parts.append(f"{text} [p.{block.page_idx + 1}]")
    return "\n".join(parts)


def _extract_figures(paper_ir: PaperIR) -> list[dict[str, Any]]:
    """Extract figure/image captions with page + section context.

    MinerU stores each figure's caption as the image block's ``text``; empty or
    placeholder captions are skipped so the LLM only sees figures it can actually
    describe to the reader.
    """
    figures: list[dict[str, Any]] = []
    for block in paper_ir.blocks:
        if block.type != "image":
            continue
        caption = block.text.strip()
        if not caption or caption == "[image]":
            continue
        figures.append({
            "text": caption,
            "page": block.page_idx + 1,
            "section": block.section_path,
            "bbox": block.bbox,
            "img_path": block.img_path,
        })
    return figures


# Caption cues that a figure depicts the overall method rather than a result plot.
_FRAMEWORK_FIG_KEYWORDS = (
    "architecture", "framework", "overview", "pipeline", "structure",
    "our method", "our approach", "our model", "proposed method",
    "proposed framework", "proposed model", "overall", "workflow",
    "schematic", "illustration of", "system",
)


def _figure_embed_url(paper_id: str, img_path: str) -> str:
    """Relative URL the frontend resolves (via the Next.js /api rewrite) to the
    backend image route. Empty when the figure has no extracted image file."""
    name = img_path.rsplit("/", 1)[-1] if img_path else ""
    return f"/api/papers/{paper_id}/images/{name}" if name else ""


def _select_framework_figures(
    figures: list[dict[str, Any]], max_n: int = 3
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split figures into (architecture/framework candidates, the rest).

    Candidates are ranked by caption cues and a bonus for "Figure 1" (commonly
    the overview). Only figures that actually have an image file can be
    embedded; if none score, the earliest embeddable figure is used as a
    fallback so the report still shows the paper's lead diagram.
    """
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for i, fig in enumerate(figures):
        cl = fig["text"].lower()
        score = sum(2 for kw in _FRAMEWORK_FIG_KEYWORDS if kw in cl)
        if re.search(r"\b(?:figure|fig\.?)\s*1\b", cl):
            score += 3
        scored.append((score, i, fig))

    embeddable = [s for s in scored if s[2].get("img_path")]
    candidates = sorted((s for s in embeddable if s[0] > 0), key=lambda s: (-s[0], s[1]))
    key = [s[2] for s in candidates[:max_n]]
    if not key and embeddable:
        key = [min(embeddable, key=lambda s: s[1])[2]]

    key_ids = {id(f) for f in key}
    others = [f for f in figures if id(f) not in key_ids]
    return key, others


LENS_SYSTEM_PROMPT = """You are an expert research-paper analyst. Produce a "Logic Lens": a deep, single-paper read-through that makes a researcher truly understand HOW the work operates and WHY it works — not a surface summary. Explain mechanisms, interpret results, and think critically.

Organize the analysis into the four parts below. Keep the four top-level headings, but ADAPT the sub-points to THIS paper: expand what is central, condense what is auxiliary, and drop sub-points that do not apply rather than forcing them. A theory paper, an empirical study, and a systems paper should not read identically.

## 1. Overview & Motivation
- **Problem & why it matters**: the concrete problem, why it is important, and the specific gap or limitation in prior work that this paper targets.
- **Core contributions**: the main contributions / key findings, and the specific problem each one addresses.
- If the context provides venue, year, or ranking metadata, state it briefly here.

## 2. Method Deep-Dive
This is the heart of the analysis — be thorough and concrete here.
- **Core idea & intuition**: state the central insight in plain language first — *why* the approach should work — before the formalism.
- **Pipeline & modules**: the end-to-end data flow and what each major component is responsible for.
- **Key formulas**: for the important equations only, give the formula in LaTeX, the meaning of its variables (add a short symbol table if notation is heavy), the derivation logic / intuition, and how it differs from prior approaches. Skip trivial or boilerplate equations.
- **Key algorithm / procedure**: a step-by-step walkthrough with per-step annotation; discuss complexity when it is relevant.
- **Architecture / framework diagram**: embed the paper's main architecture or framework figure inline, right where you explain the pipeline, by copying its ready-made Markdown image (the `embed:` line) from the `## Key Architecture Figures` context block verbatim. Immediately after the image, walk the reader through the diagram — name each component and trace how data flows through it. This lets the reader see the approach, not just read about it.
- **Other figures**: when you first refer to any other figure, briefly explain in prose what it depicts (from its caption) so the reader grasps it without seeing it.

## 3. Experiments & Results
- **Datasets & metrics**: which datasets and evaluation metrics are used, what each metric actually measures, and why they are appropriate for the task.
- **Setup**: the training / experimental details that ARE reported (optimizer, schedule, key hyperparameters, hardware). Report what is present — do not enumerate every missing field.
- **Results interpretation**: do NOT merely restate numbers. Interpret them in light of the datasets' characteristics, explain what they demonstrate, and draw out the takeaways and what they imply.

## 4. Critical Assessment
- **Why it works**: the likely sources of the method's effectiveness.
- **Limitations & risks**: assumptions, potential confounds, and generalization concerns.
- **Reproducibility**: whether the paper gives enough to reproduce the core results; flag only the genuinely important missing details.
- **Takeaways & open questions**: what this work enables and the promising directions it suggests.

RULES:
1. Every factual claim MUST carry a page citation in the form [p.X].
2. Use LaTeX for ALL mathematical content: `$...$` inline and `$$...$$` for display formulas. A display `$$` must start at the left margin (no indentation), with the LaTeX directly inside the delimiters and no extra blank lines.
3. Prioritize substance and depth on the core method and key results — capture the main points fully; auxiliary details may be condensed. Do not omit valuable content merely to keep the report short.
4. Do NOT fabricate; state only what the provided context supports. When an important detail is genuinely absent, note it briefly in prose — but do NOT pad the report with "not reported" bullets for every missing field. Information density matters more than checklist completeness.
5. Prefer the dedicated `## Paper Framing`, `## Method Section`, `## Experiment Section`, and figure excerpts over `## Full Paper Text`; the targeted excerpts are the most relevant.
6. To display a figure, embed it with the exact Markdown image given in the `## Key Architecture Figures` block — copy the `embed:` line verbatim (both alt text and URL). NEVER invent, guess, or alter an image URL, and do not embed a figure that has no provided URL. Embedding the framework diagram is expected in Part 2; do not embed result/plot figures.
"""


async def run_logic_lens(state: MainGraphState) -> dict[str, Any]:
    """Run Logic Lens deep analysis."""
    paper_id = state["paper_id"]
    t0 = time.perf_counter()

    paper_ir = PaperIR.model_validate_json(state["paper_ir_json"])
    logger.info(f"[{paper_id}] lens: Parsed PaperIR — {len(paper_ir.sections)} sections, {len(paper_ir.blocks)} blocks")

    framing_text = _extract_framing_section(paper_ir)
    equations = _extract_equations(paper_ir)
    algorithms = _extract_algorithms(paper_ir)
    tables = _extract_tables(paper_ir)
    figures = _extract_figures(paper_ir)
    method_text = _extract_method_section(paper_ir)
    experiment_text = _extract_experiment_section(paper_ir)
    logger.info(
        f"[{paper_id}] lens: Extracted — framing={len(framing_text)} chars "
        f"equations={len(equations)} algorithms={len(algorithms)} tables={len(tables)} "
        f"figures={len(figures)} method={len(method_text)} chars experiment={len(experiment_text)} chars"
    )

    # Build context for LLM.
    # Targeted excerpts (framing / method / experiment / figures) lead the
    # context because they drive the four report parts; the generic
    # ``Full Paper Text`` is a smaller catch-all at the end for any section the
    # keyword-based extractors missed.
    context_parts: list[str] = []

    if paper_ir.title:
        context_parts.append(f"# Paper: {paper_ir.title}")

    # Publication metadata (venue / year / SCI / CCF) for the Overview part.
    pub_rank_json = state.get("pub_rank_json", "")
    if pub_rank_json:
        try:
            pr = json.loads(pub_rank_json)
            meta_parts: list[str] = []
            if pr.get("venue"):
                meta_parts.append(f"Published in: {pr['venue']}")
            if pr.get("year"):
                meta_parts.append(f"Year: {pr['year']}")
            if pr.get("sci"):
                meta_parts.append(f"SCI Tier: {pr['sci']}")
            if pr.get("ccf"):
                meta_parts.append(f"CCF Rating: {pr['ccf']}")
            if meta_parts:
                context_parts.append("[Publication Info: " + " | ".join(meta_parts) + "]")
        except (json.JSONDecodeError, TypeError):
            pass

    framing_cap = 7000
    method_cap = 10000
    experiment_cap = 10000
    figures_cap = 3000
    has_targeted = bool(framing_text.strip() or method_text.strip() or experiment_text.strip())
    full_text_cap = 5000 if has_targeted else 14000

    if framing_text.strip():
        ft = framing_text
        if len(ft) > framing_cap:
            ft = ft[:framing_cap] + "\n[...truncated...]"
        context_parts.append(
            f"## Paper Framing (abstract / intro / related work / conclusion)\n{ft}"
        )

    if method_text.strip():
        mt = method_text
        if len(mt) > method_cap:
            mt = mt[:method_cap] + "\n[...truncated...]"
        context_parts.append(f"## Method Section\n{mt}")

    if experiment_text.strip():
        et = experiment_text
        if len(et) > experiment_cap:
            et = et[:experiment_cap] + "\n[...truncated...]"
        context_parts.append(f"## Experiment Section\n{et}")

    if figures:
        key_figs, other_figs = _select_framework_figures(figures)
        if key_figs:
            key_lines: list[str] = []
            for f in key_figs:
                cap = f["text"]
                if len(cap) > 300:
                    cap = cap[:300] + "…"
                url = _figure_embed_url(paper_id, f.get("img_path", ""))
                alt = cap[:80]
                if url:
                    key_lines.append(f"- [p.{f['page']}] {cap}\n  embed: ![{alt}]({url})")
                else:
                    key_lines.append(f"- [p.{f['page']}] {cap}  (no image file to embed)")
            context_parts.append(
                "## Key Architecture Figures (embed these in the Method section)\n"
                + "\n".join(key_lines)
            )
        if other_figs:
            o_lines = []
            for f in other_figs:
                cap = f["text"]
                if len(cap) > 300:
                    cap = cap[:300] + "…"
                o_lines.append(f"- [p.{f['page']}] {cap}")
            o_text = "\n".join(o_lines)
            if len(o_text) > figures_cap:
                o_text = o_text[:figures_cap] + "\n[...truncated...]"
            context_parts.append(f"## Other Figures (captions, for reference)\n{o_text}")

    if equations:
        eq_text = "\n".join(f"- {e['text']} [p.{e['page']}]" for e in equations)
        context_parts.append(f"## Extracted Equations\n{eq_text}")

    if algorithms:
        algo_text = "\n".join(f"- {a['text']} [p.{a['page']}]" for a in algorithms)
        context_parts.append(f"## Extracted Algorithms\n{algo_text}")

    if tables:
        tbl_text = "\n".join(f"- [p.{t['page']}] {t['text'][:500]}" for t in tables)
        context_parts.append(f"## Extracted Tables\n{tbl_text}")

    all_text = []
    for block in paper_ir.blocks:
        if block.type in ("text", "title", "list"):
            all_text.append(f"{block.text.strip()} [p.{block.page_idx + 1}]")
    full_text = "\n".join(all_text)
    if len(full_text) > full_text_cap:
        full_text = full_text[:full_text_cap] + "\n[...truncated...]"
    context_parts.append(f"## Full Paper Text\n{full_text}")

    context = "\n\n".join(context_parts)
    logger.info(
        f"[{paper_id}] lens: Built LLM context — {len(context)} chars total "
        f"(framing={len(framing_text)} method={len(method_text)} experiment={len(experiment_text)} "
        f"figures={len(figures)}, full_text_cap={full_text_cap})"
    )

    llm = get_llm_service()
    messages = [
        {"role": "system", "content": LENS_SYSTEM_PROMPT},
        {"role": "user", "content": f"Perform a deep Logic Lens analysis:\n\n{context}"},
    ]

    model = state.get("llm_model", "")
    logger.info(f"[{paper_id}] lens: Calling LLM (model={model or '(default)'})...")
    t_llm = time.perf_counter()
    markdown = await llm.chat(messages, model=model, temperature=0.3, max_tokens=16384)
    logger.info(f"[{paper_id}] lens: LLM returned in {time.perf_counter()-t_llm:.1f}s — {len(markdown)} chars response")

    audit = validate_citation_coverage(markdown)
    logger.info(f"[{paper_id}] lens: {format_coverage_summary(audit)}")
    if audit["claims_uncited"] > 0 and audit["uncited_samples"]:
        logger.warning(
            f"[{paper_id}] lens: {audit['claims_uncited']} uncited claims — samples: {audit['uncited_samples'][:3]}"
        )

    t_evidence = time.perf_counter()
    evidence_pool = await extract_evidence_pool(
        llm,
        context=context,
        slots=LENS_SLOTS,
        model=model,
        log_label=f"[{paper_id}] lens",
    )
    logger.info(
        f"[{paper_id}] lens: evidence extraction in {time.perf_counter()-t_evidence:.1f}s "
        f"— {len(evidence_pool)} cards"
    )

    logger.info(f"[{paper_id}] lens: TOTAL {time.perf_counter()-t0:.1f}s")
    return {
        "final_markdown": markdown,
        "final_json": json.dumps({
            "mode": "lens",
            "paper_id": state["paper_id"],
            "title": paper_ir.title,
            "num_equations": len(equations),
            "num_algorithms": len(algorithms),
            "num_tables": len(tables),
            "num_figures": len(figures),
            "citation_audit": audit,
            "evidence_pool": evidence_pool,
        }),
        "progress": state.get("progress", []) + [{"step": "run_lens", "status": "done"}],
    }
