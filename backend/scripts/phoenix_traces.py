"""Read a conversation's traces back from Phoenix and say what each turn did.

The event stream records which tools ran; only the trace records what each
model call cost and what each tool returned to the model. Per turn (one
LangGraph trace) this extracts:

* model calls, their prompt / completion tokens, and how much of the prompt
  is tool schemas the model was sent;
* each step's tool calls — how many were issued together, and which ones
  were issued alone although nothing they needed came from the step before;
* each tool call's duration and the size of the result the model received;
* identical calls (same tool, same arguments) repeated within the turn.

Spans come from Phoenix's REST API (`GET /v1/projects/{project}/spans`),
selected by the `session.id` attribute, which OpenInference fills from the
LangGraph `thread_id`. Standalone use, for any conversation::

    uv run python -m scripts.phoenix_traces --thread <thread_id>
    uv run python -m scripts.phoenix_traces --thread <thread_id> --json out.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PAGE = 1000


def flatten(attributes: Any, prefix: str = "") -> dict[str, Any]:
    """`{"llm": {"token_count": {"prompt": 3}}}` → `{"llm.token_count.prompt": 3}`.

    Phoenix returns attributes nested; the OpenInference names are dotted.
    Lists of dicts (messages, tool calls) are indexed the way OTLP flattens them.
    """
    flat: dict[str, Any] = {}
    if isinstance(attributes, dict):
        for key, value in attributes.items():
            flat.update(flatten(value, f"{prefix}{key}."))
    elif isinstance(attributes, list) and attributes and all(isinstance(v, dict) for v in attributes):
        for index, value in enumerate(attributes):
            flat.update(flatten(value, f"{prefix}{index}."))
    else:
        flat[prefix[:-1]] = attributes
    return flat


async def fetch_session_spans(
    base_url: str, project: str, session_id: str, *, timeout: float = 30.0
) -> list[dict[str, Any]]:
    """Every span of one Phoenix session, attributes flattened."""
    spans: list[dict[str, Any]] = []
    cursor = ""
    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout) as client:
        # Spans arrive through a batch exporter; give the last batch a moment.
        for attempt in range(10):
            spans.clear()
            cursor = ""
            while True:
                params: list[tuple[str, str]] = [
                    ("limit", str(PAGE)), ("attribute", f"session.id:{session_id}")
                ]
                if cursor:
                    params.append(("cursor", cursor))
                response = await client.get(f"/v1/projects/{project}/spans", params=params)
                response.raise_for_status()
                body = response.json()
                spans.extend(body.get("data") or [])
                cursor = body.get("next_cursor") or ""
                if not cursor:
                    break
            if spans and not _has_open_traces(spans):
                break
            await asyncio.sleep(1.5)
    for span in spans:
        span["attributes"] = flatten(span.get("attributes") or {})
    return spans


def _has_open_traces(spans: list[dict[str, Any]]) -> bool:
    """A trace whose root has not arrived yet is still being exported."""
    roots = {_trace_id(s) for s in spans if not s.get("parent_id")}
    return any(_trace_id(s) not in roots for s in spans)


def _trace_id(span: dict[str, Any]) -> str:
    return (span.get("context") or {}).get("trace_id", "")


def _span_id(span: dict[str, Any]) -> str:
    return (span.get("context") or {}).get("span_id", "")


def _kind(span: dict[str, Any]) -> str:
    return (
        span["attributes"].get("openinference.span.kind")
        or span.get("span_kind")
        or ""
    ).upper()


def _ts(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _duration(span: dict[str, Any]) -> float:
    return round(_ts(span["end_time"]) - _ts(span["start_time"]), 2)


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _tool_calls_of(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    """Tool calls in an LLM span's output message, in order."""
    calls: dict[int, dict[str, Any]] = {}
    prefix = "llm.output_messages.0.message.tool_calls."
    for key, value in attrs.items():
        if not key.startswith(prefix):
            continue
        index, _, rest = key[len(prefix):].partition(".")
        if not index.isdigit():
            continue
        call = calls.setdefault(int(index), {})
        if rest.endswith("function.name"):
            call["name"] = value
        elif rest.endswith("function.arguments"):
            call["arguments"] = value
    return [calls[i] for i in sorted(calls)]


def _schema_chars(attrs: dict[str, Any]) -> int:
    return sum(len(str(v)) for k, v in attrs.items() if k.startswith("llm.tools."))


def _input_chars(attrs: dict[str, Any]) -> dict[str, int]:
    """Characters of the prompt by message role (system / user / assistant / tool)."""
    roles: dict[int, str] = {}
    sizes: dict[int, int] = defaultdict(int)
    prefix = "llm.input_messages."
    for key, value in attrs.items():
        if not key.startswith(prefix):
            continue
        index, _, rest = key[len(prefix):].partition(".")
        if not index.isdigit():
            continue
        if rest == "message.role":
            roles[int(index)] = str(value)
        elif rest in ("message.content", "message.contents.0.message_content.text") or rest.endswith(
            "function.arguments"
        ):
            sizes[int(index)] += len(str(value or ""))
    by_role: dict[str, int] = defaultdict(int)
    for index, size in sizes.items():
        by_role[roles.get(index, "?")] += size
    return dict(by_role)


def _tool_result_size(span: dict[str, Any]) -> tuple[int, str]:
    """Characters the model received from a tool, and the result's status.

    The output is the serialised ToolMessage — `{"type": "tool", "data":
    {"content": "<envelope JSON>", "status": "success" | "error", ...}}` — whose
    own `status` only says whether the tool node raised; the envelope's says
    what the tool answered.
    """
    output = span["attributes"].get("output.value") or ""
    text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return len(text), ""
    message = parsed.get("data") if isinstance(parsed, dict) else None
    if not (isinstance(message, dict) and "content" in message):
        status = parsed.get("status") if isinstance(parsed, dict) else ""
        return len(text), str(status or "")
    content = message["content"]
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    try:
        envelope = json.loads(text) if isinstance(content, str) else content
    except (TypeError, ValueError):
        envelope = None
    status = envelope.get("status") if isinstance(envelope, dict) else ""
    if not status and message.get("status") == "error":
        status = "error"
    return len(text), str(status or "")


def analyse_trace(spans: list[dict[str, Any]]) -> dict[str, Any]:
    """One turn: its model calls in order, the tool calls each issued, costs."""
    llm = sorted((s for s in spans if _kind(s) == "LLM"), key=lambda s: s["start_time"])
    tools = sorted((s for s in spans if _kind(s) == "TOOL"), key=lambda s: s["start_time"])
    root = next((s for s in spans if not s.get("parent_id")), None)

    steps = []
    for index, span in enumerate(llm, start=1):
        attrs = span["attributes"]
        calls = _tool_calls_of(attrs)
        steps.append({
            "step": index,
            "seconds": _duration(span),
            "prompt_tokens": _int(attrs.get("llm.token_count.prompt")),
            "completion_tokens": _int(attrs.get("llm.token_count.completion")),
            "reasoning_tokens": _int(attrs.get("llm.token_count.completion_details.reasoning")),
            "cached_tokens": _int(attrs.get("llm.token_count.prompt_details.cache_read")),
            "tool_schema_chars": _schema_chars(attrs),
            "input_chars_by_role": _input_chars(attrs),
            "tools_offered": len({k.split(".")[2] for k in attrs if k.startswith("llm.tools.")}),
            "tool_calls": [c.get("name", "?") for c in calls],
            "tool_args": [c.get("arguments", "") for c in calls],
            "status": span.get("status_code", ""),
        })

    tool_rows = []
    for span in tools:
        attrs = span["attributes"]
        size, status = _tool_result_size(span)
        tool_rows.append({
            "tool": attrs.get("tool.name") or span.get("name", ""),
            "seconds": _duration(span),
            "result_chars": size,
            "status": status or span.get("status_code", ""),
            "input": str(attrs.get("input.value") or "")[:300],
        })

    keys = Counter((r["tool"], r["input"]) for r in tool_rows)
    model_seconds = sum(s["seconds"] for s in steps)
    return {
        "trace_id": _trace_id(root) if root else (_trace_id(spans[0]) if spans else ""),
        "start_time": root["start_time"] if root else (spans[0]["start_time"] if spans else ""),
        "seconds": _duration(root) if root else 0.0,
        "model_calls": len(steps),
        "model_seconds": round(model_seconds, 1),
        "tool_seconds": round(sum(r["seconds"] for r in tool_rows), 1),
        "prompt_tokens": sum(s["prompt_tokens"] for s in steps),
        "completion_tokens": sum(s["completion_tokens"] for s in steps),
        "max_prompt_tokens": max((s["prompt_tokens"] for s in steps), default=0),
        "tool_calls": len(tool_rows),
        "calls_per_step": [len(s["tool_calls"]) for s in steps],
        "single_call_steps": sum(1 for s in steps if len(s["tool_calls"]) == 1),
        "repeated_calls": [
            {"tool": tool, "input": inp, "times": n} for (tool, inp), n in keys.items() if n > 1
        ],
        "tool_result_chars": sum(r["result_chars"] for r in tool_rows),
        "steps": steps,
        "tools": tool_rows,
    }


def analyse(spans: list[dict[str, Any]]) -> dict[str, Any]:
    """Every turn of a session, in order, plus totals."""
    by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for span in spans:
        by_trace[_trace_id(span)].append(span)
    turns = sorted((analyse_trace(group) for group in by_trace.values()), key=lambda t: t["start_time"])
    # Drop traces with no model call: a stray span from the graph's own
    # bookkeeping is not a turn.
    turns = [t for t in turns if t["model_calls"]]
    per_tool: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0, "seconds": 0.0, "result_chars": 0})
    for turn in turns:
        for row in turn["tools"]:
            entry = per_tool[row["tool"]]
            entry["calls"] += 1
            entry["seconds"] = round(entry["seconds"] + row["seconds"], 2)
            entry["result_chars"] += row["result_chars"]
    return {
        "spans": len(spans),
        "turns": turns,
        "totals": {
            "model_calls": sum(t["model_calls"] for t in turns),
            "prompt_tokens": sum(t["prompt_tokens"] for t in turns),
            "completion_tokens": sum(t["completion_tokens"] for t in turns),
            "tool_calls": sum(t["tool_calls"] for t in turns),
            "model_seconds": round(sum(t["model_seconds"] for t in turns), 1),
            "tool_seconds": round(sum(t["tool_seconds"] for t in turns), 1),
        },
        "per_tool": dict(per_tool),
    }


def format_summary(scenario: dict[str, Any]) -> str:
    """A few lines per scenario for the terminal."""
    lines = [f"[{scenario['name']}] session {scenario['session_id']}"]
    trace = scenario.get("trace") or {}
    traced = trace.get("turns") or []
    for index, turn in enumerate(scenario["turns"], start=1):
        line = (f"  {index}. {turn['status']:6s} {turn['elapsed_seconds']:6.1f}s "
                f"tools={len(turn['tool_calls']):2d} cites={len(turn['citations']):2d}")
        if index <= len(traced):
            t = traced[index - 1]
            line += (f" | llm={t['model_calls']} prompt={t['prompt_tokens']:>7,} "
                     f"max_prompt={t['max_prompt_tokens']:>6,} per_step={t['calls_per_step']} "
                     f"model={t['model_seconds']}s tool={t['tool_seconds']}s")
        lines.append(line)
    if trace.get("error"):
        lines.append(f"  trace: {trace['error']}")
    if scenario.get("repeated_calls"):
        lines.append(f"  repeated across turns: {len(scenario['repeated_calls'])}")
    return "\n".join(lines)


async def _main() -> int:
    from app.config import get_settings

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--thread", required=True, help="LangGraph thread id (= Phoenix session id)")
    parser.add_argument("--url", default="", help="Phoenix base URL; defaults to TRACING_UI_URL")
    parser.add_argument("--project", default="", help="Phoenix project; defaults to TRACING_PROJECT")
    parser.add_argument("--json", default="", help="Write the full analysis here")
    args = parser.parse_args()

    settings = get_settings()
    base = args.url or settings.tracing_ui_url or "http://localhost:6006"
    spans = await fetch_session_spans(base, args.project or settings.tracing_project or "scholar", args.thread)
    result = analyse(spans)
    for index, turn in enumerate(result["turns"], start=1):
        print(f"turn {index}: {turn['seconds']}s, {turn['model_calls']} model calls "
              f"({turn['model_seconds']}s), {turn['tool_calls']} tool calls ({turn['tool_seconds']}s), "
              f"prompt {turn['prompt_tokens']:,} tokens")
        for step in turn["steps"]:
            print(f"   step {step['step']}: {step['seconds']}s prompt={step['prompt_tokens']:,} "
                  f"out={step['completion_tokens']:,} → {step['tool_calls'] or 'answer'}")
    print(json.dumps(result["totals"], indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
