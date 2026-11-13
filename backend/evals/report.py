"""The run's summary as Markdown: the table a person reads, and pastes."""

from __future__ import annotations

from typing import Any


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:,.{digits}f}"


def _result_cell(case: dict[str, Any], score: dict[str, Any], record: dict[str, Any]) -> str:
    parts = []
    if case["answerable"]:
        if score.get("correctness") is not None:
            text = _pct(score["correctness"])
            if score["contradicted"]:
                text += f" ({score['contradicted']} contradicted)"
            parts.append(text)
    else:
        verdict = score["abstention"] or {}
        parts.append("abstained" if verdict.get("correct") else (
            "fabricated" if verdict.get("fabricated") else "did not abstain"
        ))
    if score.get("report_correctness") is not None:
        parts.append(f"report {_pct(score['report_correctness'])}")
    tools = score.get("tools")
    if tools and not tools["ok"]:
        problems = [f"missing {', '.join(tools['missing'])}"] if tools["missing"] else []
        if not tools["any_met"]:
            problems.append("none of the expected readers")
        if tools["forbidden"]:
            problems.append(f"called {', '.join(tools['forbidden'])}")
        parts.append("tools: " + "; ".join(problems))
    result = " · ".join(parts) or "—"
    if not record.get("completed"):
        result += f" · {record.get('status')}/{record.get('completion') or 'no answer'}"
        if record.get("stop_reason"):
            result += f" ({record['stop_reason']})"
    return result


def render_markdown(
    meta: dict[str, Any],
    summary: dict[str, Any],
    results: list[dict[str, Any]],
    regressions: list[dict[str, Any]] | None = None,
    baseline: dict[str, Any] | None = None,
    case_flips: list[dict[str, Any]] | None = None,
) -> str:
    def versus(name: str) -> str:
        before = (baseline or {}).get(name)
        now = summary.get(name)
        if before is None or now is None:
            return ""
        return f" ({(now - before) * 100:+.1f})"

    def cost_versus(name: str) -> str:
        before = ((baseline or {}).get("cost") or {}).get(name)
        now = summary["cost"].get(name)
        if not before or now is None:
            return ""
        return f" ({(now - before) / before * 100:+.0f}%)"

    cost = summary["cost"]
    attempts = summary.get("attempts", summary["cases"])
    lines = [
        f"# Eval: {meta['dataset']} v{meta['dataset_version']}",
        "",
        f"- model: `{meta['model']}` · judge: `{meta['judge_model']}` "
        f"({meta['judge_prompt_version']}) · prompt: `{meta['prompt_version']}`",
        f"- commit: `{meta.get('commit') or 'unknown'}` · {meta['generated_at']}",
        f"- cases: {summary['cases']} scored ({attempts} attempts, "
        f"{meta.get('repeats', 1)} per case), {summary['skipped']} skipped",
        "",
        "| metric | value |",
        "|---|---|",
        f"| attempts passed | {_pct(summary.get('pass_rate'))}{versus('pass_rate')} |",
        f"| answer correctness ({summary['answerable_cases']} answerable) "
        f"| {_pct(summary['correctness'])}{versus('correctness')} |",
        f"| abstention accuracy ({summary['unanswerable_cases']} unanswerable) "
        f"| {_pct(summary['abstention_accuracy'])}{versus('abstention_accuracy')} |",
        f"| report correctness ({summary.get('report_attempts', 0)} reports, "
        f"{summary.get('reports_missing', 0)} missing) "
        f"| {_pct(summary.get('report_correctness'))}{versus('report_correctness')} |",
        f"| tool selection ({summary.get('tool_checked_attempts', 0)} checked) "
        f"| {_pct(summary.get('tool_selection_accuracy'))}{versus('tool_selection_accuracy')} |",
        f"| citation precision ({summary['citations_judged']} citations) "
        f"| {_pct(summary['citation_precision'])}{versus('citation_precision')} |",
        f"| citation recall ({summary['statements_needing_support']} sentences) "
        f"| {_pct(summary['citation_recall'])}{versus('citation_recall')} |",
        f"| turns completed | {_pct(summary['completion_rate'])}{versus('completion_rate')} |",
        f"| cases contradicting a reference fact | {summary['cases_with_contradiction']} |",
        f"| flaky cases (passed some attempts) | {len(summary.get('flaky_cases') or [])} |",
        f"| uncited factual sentences | {summary['uncited_claims']} |",
        f"| invented citation ids | {summary['citations_invented']} |",
        f"| grouped markers `[ev_a, ev_b]` (not rendered as badges) "
        f"| {summary.get('grouped_citation_markers', 0)} |",
        f"| tool calls / turn | {_num(cost['tool_calls_mean'])} |",
        f"| model calls / turn | {_num(cost['llm_calls_mean'])} |",
        f"| tokens / turn (mean / p95) "
        f"| {_num(cost['tokens_mean'], 0)}{cost_versus('tokens_mean')} / {_num(cost.get('tokens_p95'), 0)} |",
        f"| wall seconds / turn (median / p95) "
        f"| {_num(cost['wall_seconds_median'])} / {_num(cost.get('wall_seconds_p95'))}"
        f"{cost_versus('wall_seconds_p95')} |",
        f"| judge errors | {summary['judge_errors']} |",
        "",
        "## By category",
        "",
        "| category | attempts | passed | correctness | citation precision | citation recall |",
        "|---|---|---|---|---|---|",
    ]
    for name, row in summary["by_category"].items():
        lines.append(
            f"| {name} | {row['cases']} | {_pct(row.get('pass_rate'))} | {_pct(row['correctness'])} "
            f"| {_pct(row['citation_precision'])} | {_pct(row['citation_recall'])} |"
        )

    if regressions:
        lines += ["", "## Regressions against the baseline", ""]
        for r in regressions:
            if r.get("kind") == "cost":
                lines.append(f"- **{r['metric']}**: {_num(r['baseline'])} → {_num(r['current'])}")
            else:
                lines.append(f"- **{r['metric']}**: {_pct(r['baseline'])} → {_pct(r['current'])}")

    if case_flips:
        lines += [
            "",
            "## Cases that changed against the baseline",
            "",
            "Passed / decided attempts. A regression is gated once both runs decided "
            "at least two attempts of the case.",
            "",
            "| case | change | baseline | now | gated |",
            "|---|---|---|---|---|",
        ]
        lines += [
            f"| {f['case']} | {f['kind']} | {f['baseline']} | {f['current']} "
            f"| {'yes' if f['gated'] else 'no'} |"
            for f in case_flips
        ]

    if summary.get("flaky_cases"):
        lines += ["", "## Flaky cases", ""]
        outcomes = summary.get("case_outcomes") or {}
        lines += [
            f"- {case_id}: passed {outcomes[case_id]['passed']} of {outcomes[case_id]['decided']}"
            for case_id in summary["flaky_cases"]
        ]

    lines += [
        "",
        "## Cases",
        "",
        "| case | category | result | pass | cit. P | cit. R | tools | tokens | s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        case, score, record = r["case"], r.get("score"), r.get("record") or {}
        if not score:
            lines.append(
                f"| {case['id']} | {case['category']} | skipped: {r['skipped']} | | | | | | |"
            )
            continue
        label = case["id"] + (f" #{record['attempt'] + 1}" if record.get("attempt") else "")
        verdict = score.get("passed")
        usage = record.get("usage") or {}
        lines.append(
            f"| {label} | {case['category']} | {_result_cell(case, score, record)} "
            f"| {'—' if verdict is None else ('yes' if verdict else 'no')} "
            f"| {_pct(score.get('citation_precision'))} | {_pct(score.get('citation_recall'))} "
            f"| {usage.get('tool_calls', 0)} | {_num(usage.get('tokens'), 0)} "
            f"| {_num(record.get('elapsed_seconds'))} |"
        )
    return "\n".join(lines) + "\n"
