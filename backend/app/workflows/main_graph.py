from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph

from app.config import get_settings
from app.db import database as db
from app.services import mineru_adapter
from app.services.paper_ir import build_and_store_paper_ir
from app.workflows.state import MainGraphState

logger = logging.getLogger("scholar.graph")


async def ingest_pdf(state: MainGraphState) -> dict[str, Any]:
    """Verify PDF exists and set path."""
    t0 = time.perf_counter()
    paper_id = state["paper_id"]
    settings = get_settings()
    pdf_path = settings.data_dir / "papers" / paper_id / "original.pdf"

    if not pdf_path.exists():
        logger.error(f"[{paper_id}] ingest_pdf: PDF not found at {pdf_path}")
        return {"error": f"PDF not found: {pdf_path}"}

    size_mb = pdf_path.stat().st_size / 1024 / 1024
    logger.info(f"[{paper_id}] ingest_pdf: OK ({size_mb:.1f} MB) in {time.perf_counter()-t0:.2f}s")
    return {
        "pdf_path": str(pdf_path),
        "progress": state.get("progress", []) + [{"step": "ingest_pdf", "status": "done"}],
    }


async def mineru_parse(state: MainGraphState) -> dict[str, Any]:
    """Run MinerU parsing on the PDF."""
    if state.get("error"):
        return {}

    t0 = time.perf_counter()
    paper_id = state["paper_id"]

    # Check if already parsed
    existing = await db.fetch_one(
        "SELECT parse_id, status, output_dir FROM mineru_parses WHERE paper_id = ? AND status = 'done' ORDER BY created_at DESC LIMIT 1",
        (paper_id,),
    )
    if existing and existing["output_dir"]:
        output_dir = Path(existing["output_dir"])
        if output_dir.exists():
            logger.info(f"[{paper_id}] mineru_parse: CACHED (parse_id={existing['parse_id']}) in {time.perf_counter()-t0:.2f}s")
            return {
                "parse_id": existing["parse_id"],
                "output_dir": str(output_dir),
                "progress": state.get("progress", []) + [{"step": "mineru_parse", "status": "done", "cached": True}],
            }

    parse_id = uuid.uuid4().hex[:16]
    await db.execute(
        "INSERT INTO mineru_parses (parse_id, paper_id, status) VALUES (?, ?, 'pending')",
        (parse_id, paper_id),
    )

    logger.info(f"[{paper_id}] mineru_parse: Starting MinerU API parse (parse_id={parse_id})...")
    try:
        output_dir = await mineru_adapter.parse_pdf(paper_id, parse_id)
        elapsed = time.perf_counter() - t0
        logger.info(f"[{paper_id}] mineru_parse: DONE in {elapsed:.1f}s -> {output_dir}")
        return {
            "parse_id": parse_id,
            "output_dir": str(output_dir),
            "progress": state.get("progress", []) + [{"step": "mineru_parse", "status": "done"}],
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.error(f"[{paper_id}] mineru_parse: FAILED in {elapsed:.1f}s — {e}")
        return {
            "error": f"MinerU parse failed: {e}",
            "progress": state.get("progress", []) + [{"step": "mineru_parse", "status": "failed", "error": str(e)}],
        }


async def build_paper_ir(state: MainGraphState) -> dict[str, Any]:
    """Build PaperIR from MinerU output."""
    if state.get("error"):
        return {}

    t0 = time.perf_counter()
    output_dir = Path(state["output_dir"])
    paper_id = state["paper_id"]

    logger.info(f"[{paper_id}] build_paper_ir: Parsing content_list from {output_dir}...")
    try:
        paper_ir = await build_and_store_paper_ir(output_dir, paper_id)
        elapsed = time.perf_counter() - t0
        logger.info(
            f"[{paper_id}] build_paper_ir: DONE in {elapsed:.2f}s — "
            f"title='{paper_ir.title[:60]}' blocks={len(paper_ir.blocks)} sections={len(paper_ir.sections)}"
        )
        return {
            "paper_ir_json": paper_ir.model_dump_json(),
            "progress": state.get("progress", []) + [{"step": "build_paper_ir", "status": "done"}],
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.error(f"[{paper_id}] build_paper_ir: FAILED in {elapsed:.2f}s — {e}")
        return {
            "error": f"PaperIR build failed: {e}",
            "progress": state.get("progress", []) + [{"step": "build_paper_ir", "status": "failed", "error": str(e)}],
        }


def route_by_mode(state: MainGraphState) -> str:
    """Route to appropriate subgraph based on mode."""
    if state.get("error"):
        return "persist_output"

    mode = state.get("mode", "snap")
    logger.info(f"[{state['paper_id']}] route_by_mode: -> {mode}")
    if mode == "lens":
        return "run_lens"
    elif mode == "sphere":
        return "run_sphere"
    return "run_snap"


async def run_snap(state: MainGraphState) -> dict[str, Any]:
    """Run Insight Snap analysis."""
    t0 = time.perf_counter()
    logger.info(f"[{state['paper_id']}] run_snap: Starting Insight Snap...")
    from app.workflows.snap_subgraph import run_insight_snap
    result = await run_insight_snap(state)
    elapsed = time.perf_counter() - t0
    md_len = len(result.get("final_markdown", ""))
    logger.info(f"[{state['paper_id']}] run_snap: DONE in {elapsed:.1f}s — markdown={md_len} chars")
    return result


async def run_lens(state: MainGraphState) -> dict[str, Any]:
    """Run Logic Lens analysis."""
    t0 = time.perf_counter()
    logger.info(f"[{state['paper_id']}] run_lens: Starting Logic Lens...")
    from app.workflows.lens_subgraph import run_logic_lens
    result = await run_logic_lens(state)
    elapsed = time.perf_counter() - t0
    md_len = len(result.get("final_markdown", ""))
    logger.info(f"[{state['paper_id']}] run_lens: DONE in {elapsed:.1f}s — markdown={md_len} chars")
    return result


async def run_sphere(state: MainGraphState) -> dict[str, Any]:
    """Run Research Sphere analysis."""
    t0 = time.perf_counter()
    logger.info(f"[{state['paper_id']}] run_sphere: Starting Research Sphere...")
    from app.workflows.sphere_subgraph import run_research_sphere
    result = await run_research_sphere(state)
    elapsed = time.perf_counter() - t0
    md_len = len(result.get("final_markdown", ""))
    logger.info(f"[{state['paper_id']}] run_sphere: DONE in {elapsed:.1f}s — markdown={md_len} chars")
    return result


async def persist_output(state: MainGraphState) -> dict[str, Any]:
    """Save results to DB."""
    t0 = time.perf_counter()
    run_id = state["run_id"]
    paper_id = state["paper_id"]

    if state.get("error"):
        await db.execute(
            "UPDATE runs SET status = 'failed', error_msg = ?, finished_at = datetime('now') WHERE run_id = ?",
            (state["error"], run_id),
        )
        logger.info(f"[{paper_id}] persist_output: Saved FAILED status in {time.perf_counter()-t0:.2f}s")
        return {"progress": state.get("progress", []) + [{"step": "persist_output", "status": "failed"}]}

    markdown = state.get("final_markdown", "")
    json_data = state.get("final_json", "{}")

    await db.execute(
        "INSERT OR REPLACE INTO run_outputs (run_id, markdown, json_data) VALUES (?, ?, ?)",
        (run_id, markdown, json_data),
    )
    await db.execute(
        "UPDATE runs SET status = 'done', finished_at = datetime('now') WHERE run_id = ?",
        (run_id,),
    )

    logger.info(f"[{paper_id}] persist_output: Saved run_id={run_id} markdown={len(markdown)} chars in {time.perf_counter()-t0:.2f}s")
    return {"progress": state.get("progress", []) + [{"step": "persist_output", "status": "done"}]}


def build_main_graph() -> StateGraph:
    """Build the main LangGraph workflow."""
    graph = StateGraph(MainGraphState)

    graph.add_node("ingest_pdf", ingest_pdf)
    graph.add_node("mineru_parse", mineru_parse)
    graph.add_node("build_paper_ir", build_paper_ir)
    graph.add_node("run_snap", run_snap)
    graph.add_node("run_lens", run_lens)
    graph.add_node("run_sphere", run_sphere)
    graph.add_node("persist_output", persist_output)

    graph.set_entry_point("ingest_pdf")
    graph.add_edge("ingest_pdf", "mineru_parse")
    graph.add_edge("mineru_parse", "build_paper_ir")
    graph.add_conditional_edges("build_paper_ir", route_by_mode, {
        "run_snap": "run_snap",
        "run_lens": "run_lens",
        "run_sphere": "run_sphere",
        "persist_output": "persist_output",
    })
    graph.add_edge("run_snap", "persist_output")
    graph.add_edge("run_lens", "persist_output")
    graph.add_edge("run_sphere", "persist_output")
    graph.add_edge("persist_output", END)

    return graph
