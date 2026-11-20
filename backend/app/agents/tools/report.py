"""`begin_report`: this turn's answer is a whole-paper report (P16).

A report used to be a tool that wrote it in a fixed subgraph and handed the
model the finished text to summarise. Here the conversation writes it, with
the tools and evidence it reads with anyway, so a report cites passages the
way an answer does and sees what the conversation already knows.

The tool only *declares* the report: it checks the paper, opens the `runs`
row, and returns the mode's method (`agents/skills`). The report text is never
a tool argument — a small model writing ten thousand characters of Markdown
and LaTeX into a JSON string breaks it — it is the turn's final answer, which
the runner saves when the turn ends (`services/answer_reports.snapshot`).

When the reader picked the mode in the composer, the runner does the same
thing before the first model call (:func:`start_report`, then
:func:`composer_note`), so the report does not depend on the model deciding to
call this.
"""

from __future__ import annotations

from typing import Literal

from langchain.tools import ToolRuntime, tool

from app.agents import skills
from app.agents.context import AgentContext
from app.agents.tools.modes import MODE_LABELS, _ensure_parsed
from app.agents.tools.reading import _authorize
from app.models.agent_models import ErrorCode, ToolResult, ToolStatus
from app.services import answer_reports

# The report tool a mode keeps while it is not written as an answer.
LEGACY_TOOLS = {"snap": "run_insight_snap", "lens": "run_logic_lens", "sphere": "run_research_sphere"}

_NOTE = (
    "This turn's final answer IS the {label} report: read with the tools the method "
    "names, then write the report once, at the end, under the method's headings. "
    "Cite every factual statement with the evidence numbers your reads return, as in "
    "any answer. Do not narrate between tool calls; do not write the report in pieces. "
    "It is saved and shown to the reader as a report card when the turn ends."
)


async def start_report(
    ctx: AgentContext, *, mode: str, paper_id: str = "", focus: str = ""
) -> ToolResult:
    """Open the report this turn writes. The tool and the composer path share it."""
    enabled = skills.enabled_modes()
    if mode not in enabled:
        legacy = LEGACY_TOOLS.get(mode)
        return ToolResult.failed(
            ErrorCode.INVALID_ARGUMENT,
            f"{mode} reports are not written as answers here"
            + (f"; call `{legacy}` instead." if legacy else "."),
            retryable=False,
        )
    if ctx.report is not None:
        return ToolResult.failed(
            ErrorCode.INVALID_ARGUMENT,
            f"This turn is already writing the {ctx.report.mode} report. One report per "
            "turn: finish this one; to cover another paper or mode, answer a comparison "
            "question or offer to write it next turn.",
            retryable=False,
        )

    paper_id, denial = await _authorize(ctx, paper_id)
    if denial is not None:
        return denial
    denial = await _ensure_parsed(ctx, paper_id, tool_name="begin_report")
    if denial is not None:
        return denial

    method = skills.method(mode, ctx.language)
    if method is None:  # enabled_modes() checked it; a file removed since
        return ToolResult.failed(
            ErrorCode.INTERNAL_ERROR, f"The {mode} method is missing.", retryable=False
        )
    focus = " ".join((focus or "").split())
    run_id = await answer_reports.open_run(
        mode=mode,
        paper_id=paper_id,
        focus=focus,
        llm_model=ctx.llm_model,
        language=ctx.language,
        owner_id=ctx.owner_id,
        owner_token=ctx.owner_token,
        session_id=ctx.session_id,
        agent_run_id=ctx.run_id,
    )
    ctx.report = answer_reports.ReportTurn(
        run_id=run_id,
        mode=mode,
        paper_id=paper_id,
        focus=focus,
        required=method.required,
        tokens_at_start=ctx.usage.tokens or 0,
    )
    label = MODE_LABELS[mode]["zh" if ctx.language == "zh" else "en"]
    return ToolResult.ok(
        {
            "run_id": run_id,
            "mode": mode,
            "mode_label": label,
            "paper_id": paper_id,
            "focus": focus,
            "report_url": f"/paper/{paper_id}/run/{run_id}",
            "required_sections": list(method.required),
            "method": method.text,
        },
        note=_NOTE.format(label=label),
    )


def composer_note(mode: str, result: ToolResult, language: str) -> str:
    """What follows the reader's message when they picked an answer-written mode.

    With the report open, the method itself; otherwise why it could not be
    opened, and how the model may still open it — several papers in the
    session leave the choice of paper to it.
    """
    zh = language == "zh"
    label = MODE_LABELS[mode]["zh" if zh else "en"]
    data = result.data if isinstance(result.data, dict) else {}
    if result.status == ToolStatus.OK and data.get("method"):
        head = (
            f"【用户在界面中选择了「{label}」模式，报告已开始（paper_id={data['paper_id']}）。"
            "本轮的最终回答就是这份报告：按下面的方法阅读和写作，每条事实性结论都引用证据编号。】"
            if zh
            else f"[The reader selected the {label} mode; the report has begun "
            f"(paper_id={data['paper_id']}). This turn's final answer is that report: read "
            "and write by the method below, citing evidence numbers for every factual "
            "statement.]"
        )
        return f"{head}\n\n{data['method']}"
    reason = result.error.message if result.error is not None else (result.note or "")
    if zh:
        return (
            f"【用户在界面中选择了「{label}」模式，但报告没能直接开始：{reason}"
            f"如果能确定用户指的是哪篇论文，调用 `begin_report(mode=\"{mode}\", paper_id=...)` 开始；"
            "否则向用户说明原因。】"
        )
    return (
        f"[The reader selected the {label} mode, but the report could not begin: {reason} "
        f"If you can tell which paper they mean, call `begin_report(mode=\"{mode}\", "
        "paper_id=...)`; otherwise tell them why.]"
    )


@tool(parse_docstring=False)
async def begin_report(
    runtime: ToolRuntime[AgentContext],
    mode: Literal["snap", "lens", "sphere"],
    paper_id: str = "",
    focus: str = "",
) -> str:
    """Start a whole-paper report that this turn's final answer will be.

    Modes: `snap` (Insight Snap — is it worth reading: claims, how well the
    results hold, external signals), `lens` (Logic Lens — the method in depth:
    idea, pipeline, equations, algorithms, experiments), `sphere` (Research
    Sphere — where it sits in the field: related work, clusters, gaps). Only
    the modes the system prompt lists as written with `begin_report` are
    accepted.

    Returns the method for the mode: the sections the report must have and the
    tools to read with. Call it once, before reading for the report; then read,
    then write the report as your answer. Pass `focus` when the reader named
    what they most want covered. Omit paper_id when the session holds one paper.
    """
    result = await start_report(runtime.context, mode=mode, paper_id=paper_id, focus=focus)
    return result.to_json()


REPORT_TOOLS = [begin_report]
