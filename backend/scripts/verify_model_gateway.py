"""Verify that a model can be used through the gateway its registry profile names.

Run this before listing a new model in `NEWAPI_MODEL_NAME` (report modes) and
again before adding it to `NEWAPI_AGENT_MODELNAME` (conversations). The checks
go through the same code a run uses — `LLMService`, `build_chat_model`,
`create_paper_agent` — so a pass means the application's own path works, not
just that the gateway answers curl.

This talks to the live gateway and spends tokens. Run it explicitly::

    cd backend
    uv run python -m scripts.verify_model_gateway --model gemini-3.8-flash
    uv run python -m scripts.verify_model_gateway --only surface      # offline

Checks:

* ``surface``  the registry as configured: gateways, protocols, who is offered
* ``chat``     a plain answer through `LLMService`
* ``budget``   an output ceiling too small for the answer is reported as
               truncated or empty, never returned as a normal answer
* ``json``     a JSON object comes back parseable from a prompt alone
* ``tools``    the agent's real tool schemas are accepted and one is called
* ``agent``    a multi-step tool turn through the harness, with streamed tool
               calls and token usage counted once per model call
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessageChunk, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from app.agents.harness import create_paper_agent  # noqa: E402
from app.agents.model_factory import build_chat_model  # noqa: E402
from app.services.agent_runner import _new_usage  # noqa: E402
from app.services.llm_gateway import get_registry  # noqa: E402
from app.services.llm_service import LLMEmptyResponseError, LLMService  # noqa: E402

CHECKS = ("surface", "chat", "budget", "json", "tools", "agent")
REPORT_CHECKS = ("chat", "budget", "json")
AGENT_CHECKS = ("tools", "agent")


@tool
def get_page_count(paper_id: str) -> str:
    """Return the number of pages of a paper."""
    return json.dumps({"paper_id": paper_id, "pages": {"p1": 12, "p2": 31}.get(paper_id, 0)})


@tool
def get_first_author(paper_id: str) -> str:
    """Return the first author of a paper."""
    return json.dumps({"paper_id": paper_id, "author": {"p1": "Vaswani", "p2": "Devlin"}.get(paper_id, "?")})


async def check_surface(model: str) -> dict[str, Any]:
    registry = get_registry()
    profile = registry.profile(model)
    return {
        "ok": profile.provider.configured,
        "model": profile.name,
        "gateway": profile.provider.name,
        "protocol": profile.provider.protocol,
        "offered_for_reports": profile.name in registry.selectable_models(),
        "offered_for_conversations": profile.name in registry.agent_models(),
        "utility_model": registry.utility_model,
        "note": "" if profile.provider.configured else "gateway has no base URL or API key",
    }


async def check_chat(model: str) -> dict[str, Any]:
    text = await LLMService().chat(
        [
            {"role": "system", "content": "你是论文阅读助手。"},
            {"role": "user", "content": "用一句话说明什么是注意力机制。"},
        ],
        model=model,
        temperature=0.3,
        max_tokens=2000,
    )
    return {"ok": bool(text.strip()), "answer": text.strip()[:120]}


async def check_budget(model: str) -> dict[str, Any]:
    """Reasoning shares the output ceiling; a starved call must say so."""
    service = LLMService()
    with _CaptureWarnings("scholar.llm") as warnings:
        try:
            text = await service.chat(
                [{"role": "user", "content": "仔细推导并写一篇800字的文章，解释自注意力的复杂度为何是 O(n^2)。"}],
                model=model,
                max_tokens=200,
            )
        except LLMEmptyResponseError as exc:
            return {"ok": True, "outcome": "empty answer raised", "reason": exc.reason}
    truncated = any("truncated" in message for message in warnings.messages)
    return {
        "ok": truncated,
        "outcome": "truncated answer flagged" if truncated else "800 字 fit in 200 tokens?",
        "chars": len(text),
    }


async def check_json(model: str) -> dict[str, Any]:
    text = await LLMService().chat(
        [
            {"role": "system", "content": '只返回 JSON，不要其他文字。格式：{"sci": "Q1" 或 null, "ccf": "A" 或 null}'},
            {"role": "user", "content": "出版物：CVPR。已知它是 CCF A 类会议，没有 SCI 分区。"},
        ],
        model=model,
        temperature=0.1,
        max_tokens=2000,
    )
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return {"ok": False, "raw": text[:200]}
    return {"ok": isinstance(data, dict) and data.get("ccf") == "A", "parsed": data}


async def check_tools(model: str) -> dict[str, Any]:
    from app.agents.tools import ALL_AGENT_TOOLS

    bound = build_chat_model(model).bind_tools(ALL_AGENT_TOOLS)
    reply = await bound.ainvoke(
        "帮我搜索 2024 年以后关于 retrieval-augmented generation 评测的论文，至少 100 次引用。"
    )
    calls = [{"name": c["name"], "args": c["args"]} for c in reply.tool_calls]
    return {
        "ok": any(c["name"] == "search_papers" for c in calls),
        "tools_offered": len(ALL_AGENT_TOOLS),
        "calls": calls,
    }


async def check_agent(model: str) -> dict[str, Any]:
    agent = create_paper_agent(
        tools=[get_page_count, get_first_author],
        system_prompt="You answer questions about papers. Always use the tools; never guess.",
        model=build_chat_model(model),
    )
    counted: dict[str, int] = {}
    tokens = raw_tokens = llm_calls = tool_results = named_chunks = 0
    text = ""
    async for mode, chunk in agent.astream(
        {"messages": [{"role": "user", "content": "p1 和 p2 哪篇页数更多？更长那篇的第一作者是谁？先查页数，再查作者。"}]},
        stream_mode=["messages"],
    ):
        message, _meta = chunk
        if isinstance(message, ToolMessage):
            tool_results += 1
        elif isinstance(message, AIMessageChunk):
            named_chunks += sum(1 for c in message.tool_call_chunks or [] if c.get("name"))
            raw_tokens += int((message.usage_metadata or {}).get("total_tokens") or 0)
            added, new_call = _new_usage(message, counted)
            tokens += added
            llm_calls += int(new_call)
            if isinstance(message.content, str):
                text += message.content
            else:
                text += "".join(b.get("text", "") for b in message.content if isinstance(b, dict))
    return {
        "ok": tool_results >= 3 and "Devlin" in text and tokens > 0 and llm_calls >= 2,
        "tool_results": tool_results,
        "streamed_tool_calls": named_chunks,
        "llm_calls": llm_calls,
        "tokens_counted": tokens,
        "tokens_if_every_chunk_were_summed": raw_tokens,
        "answer_tail": text.strip()[-80:],
    }


class _CaptureWarnings(logging.Handler):
    def __init__(self, logger_name: str) -> None:
        super().__init__(level=logging.WARNING)
        self._logger = logging.getLogger(logger_name)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())

    def __enter__(self) -> "_CaptureWarnings":
        self._logger.addHandler(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        self._logger.removeHandler(self)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="", help="Model id; defaults to the configured one")
    parser.add_argument("--only", default="", help=f"Comma-separated subset of: {', '.join(CHECKS)}")
    parser.add_argument("--report", default="", help="Where to write the JSON record")
    args = parser.parse_args()

    wanted = [c.strip() for c in args.only.split(",") if c.strip()] or list(CHECKS)
    unknown = [c for c in wanted if c not in CHECKS]
    if unknown:
        parser.error(f"unknown checks: {', '.join(unknown)}")

    logging.basicConfig(level=logging.WARNING)
    model = get_registry().profile(args.model).name
    print(f"model: {model}\n")

    results: dict[str, dict[str, Any]] = {}
    for name in wanted:
        started = time.perf_counter()
        try:
            outcome = await globals()[f"check_{name}"](model)
        except Exception as exc:  # noqa: BLE001 — a failed check is a finding, not a crash
            outcome = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
        outcome["seconds"] = round(time.perf_counter() - started, 1)
        results[name] = outcome
        detail = {k: v for k, v in outcome.items() if k not in {"ok", "seconds"}}
        print(f"[{'PASS' if outcome['ok'] else 'FAIL'}] {name} ({outcome['seconds']}s)")
        print(f"       {json.dumps(detail, ensure_ascii=False)}")

    def passed(group: tuple[str, ...]) -> bool | None:
        ran = [results[c]["ok"] for c in group if c in results]
        return all(ran) if len(ran) == len(group) else None

    verdict = {"report_modes": passed(REPORT_CHECKS), "conversations": passed(AGENT_CHECKS)}
    print("\nready for report modes:   ", _word(verdict["report_modes"]))
    print("ready for conversations:  ", _word(verdict["conversations"]))

    if args.report:
        Path(args.report).write_text(
            json.dumps({"model": model, "checks": results, "verdict": verdict}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0 if all(r["ok"] for r in results.values()) else 1


def _word(value: bool | None) -> str:
    return "not checked" if value is None else ("yes" if value else "no")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
