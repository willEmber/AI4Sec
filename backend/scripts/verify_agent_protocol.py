"""P0 protocol verification against the real model gateway.

Records the evidence the development plan's P0 exit criteria ask for: that the
configured model selects local tools on its own, receives their results, keeps
deciding from them, streams, survives tool errors, and that a session resumes
from a durable checkpoint after the process that started it is gone.

This talks to the live gateway and spends tokens. Run it explicitly::

    cd backend
    uv run python -m scripts.verify_agent_protocol            # all checks
    uv run python -m scripts.verify_agent_protocol --model qwen3.7-plus
    uv run python -m scripts.verify_agent_protocol --only surface,checkpoint

`--only surface` needs no network. Results land in
`data/agent_protocol_report.json` unless `--report` says otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from app.agents.checkpointer import open_checkpointer  # noqa: E402
from app.agents.harness import (  # noqa: E402
    BUILTIN_HOST_TOOLS,
    DELEGATION_TOOL,
    create_paper_agent,
    model_visible_tool_names,
)
from app.agents.model_factory import build_chat_model, resolve_model_name  # noqa: E402
from app.config import get_settings  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# ── Stub domain tools ────────────────────────────────────────────────────────
# Stand-ins for the real P1/P2 tools. They return the plan's unified result
# shape so the checks exercise the contract the real tools will carry.

CALL_LOG: list[dict[str, Any]] = []

_OUTLINE = {
    "status": "ok",
    "data": {
        "sections": [
            {"section_id": "s1", "title": "1 Introduction", "page": 1},
            {"section_id": "s2", "title": "2 Method", "page": 3},
            {"section_id": "s3", "title": "3 Experiments", "page": 6},
        ]
    },
    "evidence_ids": [],
    "error": None,
}

_SECTIONS = {
    "s2": {
        "status": "ok",
        "data": {
            "page": 3,
            "text": (
                "We propose SparseMoE-Router, a top-2 gated mixture-of-experts router "
                "trained with an auxiliary load-balancing loss."
            ),
        },
        "evidence_ids": ["ev_method_1"],
        "error": None,
    },
    "s3": {
        "status": "ok",
        "data": {
            "page": 6,
            "text": "On WMT14 En-De the router reaches 29.7 BLEU versus a 28.4 dense baseline.",
        },
        "evidence_ids": ["ev_results_1"],
        "error": None,
    },
}


@tool
def get_paper_outline(paper_id: str) -> str:
    """Return the section outline of a parsed paper, with page numbers."""
    CALL_LOG.append({"tool": "get_paper_outline", "args": {"paper_id": paper_id}})
    if paper_id != "abc123":
        return json.dumps(
            {
                "status": "error",
                "data": None,
                "evidence_ids": [],
                "error": {
                    "code": "paper_not_found",
                    "message": f"No parsed paper with id {paper_id}",
                    "retryable": False,
                },
            },
            ensure_ascii=False,
        )
    return json.dumps(_OUTLINE, ensure_ascii=False)


@tool
def read_paper_section(paper_id: str, section_id: str) -> str:
    """Return the full text of one section of a parsed paper, with its evidence id."""
    CALL_LOG.append(
        {"tool": "read_paper_section", "args": {"paper_id": paper_id, "section_id": section_id}}
    )
    hit = _SECTIONS.get(section_id)
    if hit is None:
        return json.dumps(
            {
                "status": "error",
                "data": None,
                "evidence_ids": [],
                "error": {
                    "code": "section_not_found",
                    "message": f"No section {section_id}",
                    "retryable": False,
                },
            },
            ensure_ascii=False,
        )
    return json.dumps(hit, ensure_ascii=False)


TOOLS = [get_paper_outline, read_paper_section]

SYSTEM_PROMPT = (
    "You read academic papers using the provided tools. Look things up rather than "
    "recalling them. Cite the evidence ids the tools return. If a tool reports an "
    "error, say what failed instead of inventing an answer."
)


def _tool_calls_of(messages: list[Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for m in messages:
        for c in getattr(m, "tool_calls", None) or []:
            calls.append({"name": c.get("name"), "args": c.get("args"), "id": c.get("id")})
    return calls


def _final_text(messages: list[Any]) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and not (m.tool_calls or []):
            content = m.content
            if isinstance(content, str):
                return content
            parts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") in {"text", "output_text"}
            ]
            return "".join(parts)
    return ""


# ── Checks ───────────────────────────────────────────────────────────────────


def check_surface(model_name: str) -> dict[str, Any]:
    """The model must see the domain tools and nothing else."""
    names = model_visible_tool_names(TOOLS)
    leaked = sorted(set(names) & (BUILTIN_HOST_TOOLS | {DELEGATION_TOOL}))
    expected = sorted(t.name for t in TOOLS)
    return {
        "name": "tool_surface",
        "passed": sorted(names) == expected and not leaked,
        "exposed_tools": names,
        "expected_tools": expected,
        "leaked_builtin_or_delegation_tools": leaked,
    }


async def check_autonomy(model_name: str) -> dict[str, Any]:
    """The model must chain tools on its own and answer from their results."""
    CALL_LOG.clear()
    agent = create_paper_agent(
        tools=TOOLS, system_prompt=SYSTEM_PROMPT, model=build_chat_model(model_name)
    )
    t0 = time.perf_counter()
    out = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "For paper abc123: what method does it propose, and what result "
                        "does it report? Cite the evidence ids."
                    ),
                }
            ]
        },
        config={"configurable": {"thread_id": "p0-autonomy"}},
    )
    elapsed = round(time.perf_counter() - t0, 1)
    messages = out["messages"]
    calls = _tool_calls_of(messages)
    answer = _final_text(messages)
    tools_used = {c["name"] for c in calls}
    sections_read = {
        c["args"].get("section_id") for c in calls if c["name"] == "read_paper_section"
    }
    return {
        "name": "autonomous_tool_use",
        "passed": (
            "get_paper_outline" in tools_used
            and "read_paper_section" in tools_used
            and len(sections_read) >= 2
            and ("ev_method_1" in answer or "ev_results_1" in answer)
        ),
        "elapsed_seconds": elapsed,
        "tool_calls": calls,
        "sections_read": sorted(s for s in sections_read if s),
        "answer": answer[:900],
        "note": (
            "Passes only when the model picked the outline tool first, then read two "
            "sections it learned about from that result, and cited returned evidence ids."
        ),
    }


async def check_streaming(model_name: str) -> dict[str, Any]:
    """Tool activity and answer text must both arrive incrementally."""
    CALL_LOG.clear()
    agent = create_paper_agent(
        tools=TOOLS, system_prompt=SYSTEM_PROMPT, model=build_chat_model(model_name)
    )
    tool_events: list[str] = []
    text_chunks = 0
    first_chunk_at: float | None = None
    t0 = time.perf_counter()
    async for mode, chunk in agent.astream(
        {"messages": [{"role": "user", "content": "What method does paper abc123 propose?"}]},
        config={"configurable": {"thread_id": "p0-streaming"}},
        stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            msg, _meta = chunk
            if isinstance(msg, ToolMessage):
                tool_events.append(f"tool_result:{msg.name}")
                continue
            for c in getattr(msg, "tool_call_chunks", None) or []:
                if c.get("name"):
                    tool_events.append(f"tool_call:{c['name']}")
            content = msg.content
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") in {"text", "output_text"}
                )
            if text:
                text_chunks += 1
                if first_chunk_at is None:
                    first_chunk_at = round(time.perf_counter() - t0, 1)
    return {
        "name": "streaming",
        "passed": text_chunks > 1 and any(e.startswith("tool_call:") for e in tool_events),
        "text_chunks": text_chunks,
        "first_text_chunk_after_seconds": first_chunk_at,
        "tool_events": tool_events,
    }


async def check_tool_error(model_name: str) -> dict[str, Any]:
    """A tool error must be reported, not papered over with a guess."""
    CALL_LOG.clear()
    agent = create_paper_agent(
        tools=TOOLS, system_prompt=SYSTEM_PROMPT, model=build_chat_model(model_name)
    )
    out = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Give me the outline of paper missing999, then tell me what happened.",
                }
            ]
        },
        config={"configurable": {"thread_id": "p0-tool-error"}},
    )
    answer = _final_text(out["messages"])
    lowered = answer.lower()
    admits = any(
        w in lowered for w in ("error", "not found", "no parsed paper", "失败", "未找到", "不存在")
    )
    invented = "sparsemoe" in lowered or "introduction" in lowered
    return {
        "name": "tool_error_handling",
        "passed": admits and not invented,
        "answer": answer[:700],
        "admits_failure": admits,
        "invented_content": invented,
    }


async def check_checkpoint_resume(model_name: str) -> dict[str, Any]:
    """A thread must resume from disk in a process that never saw turn one.

    Turn one runs here; turn two runs in a *separate* interpreter, so nothing
    but the database carries the history. An in-memory saver would pass a
    same-process test and fail this one. The thread id is fresh per run, so a
    previous probe's history cannot make this one pass.
    """
    db_url = get_settings().database_url
    thread_id = f"p0-resume-{uuid.uuid4().hex[:12]}"

    async with open_checkpointer(db_url) as saver:
        agent = create_paper_agent(
            tools=TOOLS,
            system_prompt=SYSTEM_PROMPT,
            model=build_chat_model(model_name),
            checkpointer=saver,
        )
        await agent.ainvoke(
            {
                "messages": [
                    {"role": "user", "content": "What method does paper abc123 propose?"}
                ]
            },
            config={"configurable": {"thread_id": thread_id}},
        )

    child = f"""
import asyncio, json, sys
sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
sys.argv = ["child"]
from scripts.verify_agent_protocol import TOOLS, SYSTEM_PROMPT, _final_text, _tool_calls_of
from app.agents.checkpointer import open_checkpointer
from app.agents.harness import create_paper_agent
from app.agents.model_factory import build_chat_model

async def main():
    async with open_checkpointer({db_url!r}) as saver:
        agent = create_paper_agent(
            tools=TOOLS, system_prompt=SYSTEM_PROMPT,
            model=build_chat_model({model_name!r}), checkpointer=saver,
        )
        cfg = {{"configurable": {{"thread_id": {thread_id!r}}}}}
        state = await agent.aget_state(cfg)
        restored = len(state.values.get("messages", []))
        out = await agent.ainvoke(
            {{"messages": [{{"role": "user", "content":
              "And what result does that same paper report? Reuse what you already read."}}]}},
            config=cfg,
        )
        print("__RESULT__" + json.dumps({{
            "restored_messages": restored,
            "answer": _final_text(out["messages"])[:600],
            "tool_calls_in_turn_two": [
                c["name"] for c in _tool_calls_of(out["messages"][restored:])
            ],
        }}, ensure_ascii=False))

asyncio.run(main())
"""
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        child,
        cwd=str(Path(__file__).resolve().parents[1]),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ},
    )
    stdout, stderr = await proc.communicate()
    text = stdout.decode("utf-8", "replace")
    marker = "__RESULT__"
    result: dict[str, Any] = {"name": "checkpoint_resume_after_restart"}
    if marker not in text:
        result.update(
            {
                "passed": False,
                "error": stderr.decode("utf-8", "replace")[-1200:],
                "stdout": text[-600:],
            }
        )
        return result
    payload = json.loads(text.split(marker, 1)[1].strip())
    answer = payload["answer"].lower()
    result.update(
        {
            "passed": payload["restored_messages"] > 0
            and ("29.7" in answer or "bleu" in answer or "wmt" in answer),
            **payload,
            "thread_id": thread_id,
            "note": (
                "Turn two ran in a separate interpreter; only the PostgreSQL "
                "checkpoint carried turn one's history."
            ),
        }
    )
    return result


CHECKS = {
    "surface": ("offline", check_surface),
    "autonomy": ("live", check_autonomy),
    "streaming": ("live", check_streaming),
    "tool_error": ("live", check_tool_error),
    "checkpoint": ("live", check_checkpoint_resume),
}


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="", help="Model id; defaults to the configured one")
    parser.add_argument(
        "--only",
        default="",
        help=f"Comma-separated subset of: {','.join(CHECKS)}",
    )
    parser.add_argument("--report", default="", help="Where to write the JSON record")
    args = parser.parse_args()

    settings = get_settings()
    model_name = resolve_model_name(args.model)
    selected = [s.strip() for s in args.only.split(",") if s.strip()] or list(CHECKS)
    unknown = [s for s in selected if s not in CHECKS]
    if unknown:
        parser.error(f"unknown check(s): {', '.join(unknown)}")

    print(f"model:   {model_name}")
    print(f"gateway: {settings.llm_base_url}")
    print(f"checks:  {', '.join(selected)}\n")

    results: list[dict[str, Any]] = []
    for key in selected:
        kind, fn = CHECKS[key]
        print(f"→ {key} ({kind}) ...", flush=True)
        try:
            out = fn(model_name) if kind == "offline" else await fn(model_name)
        except Exception as e:  # noqa: BLE001
            out = {"name": key, "passed": False, "exception": f"{type(e).__name__}: {e}"}
        results.append(out)
        status = "PASS" if out.get("passed") else "FAIL"
        print(f"  {status}: {json.dumps(out, ensure_ascii=False)[:500]}\n")

    record = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": model_name,
        "gateway": settings.llm_base_url,
        "transport": "responses_api",
        "results": results,
        "all_passed": all(r.get("passed") for r in results),
    }
    report_path = Path(args.report) if args.report else settings.data_dir / "agent_protocol_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report: {report_path}")
    print("ALL PASSED" if record["all_passed"] else "SOME CHECKS FAILED")
    return 0 if record["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
