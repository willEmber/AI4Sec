"""The run's summary as Markdown: the table a person reads, and pastes."""

from __future__ import annotations

from typing import Any


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:,.{digits}f}"


def render_markdown(
    meta: dict[str, Any],
    summary: dict[str, Any],
    results: list[dict[str, Any]],
    regressions: list[dict[str, Any]] | None = None,
    baseline: dict[str, Any] | None = None,
) -> str:
    def versus(name: str) -> str:
        before = (baseline or {}).get(name)
        now = summary.get(name)
        if before is None or now is None:
            return ""
        return f" ({(now - before) * 100:+.1f})"

    cost = summary["cost"]
    lines = [
        f"# Eval: {meta['dataset']} v{meta['dataset_version']}",
        "",
        f"- model: `{meta['model']}` · judge: `{meta['judge_model']}` "
        f"({meta['judge_prompt_version']}) · prompt: `{meta['prompt_version']}`",
        f"- commit: `{meta.get('commit') or 'unknown'}` · {meta['generated_at']}",
        f"- cases: {summary['cases']} scored, {summary['skipped']} skipped",
        "",
        "| metric | value |",
        "|---|---|",
        f"| answer correctness ({summary['answerable_cases']} answerable) "
        f"| {_pct(summary['correctness'])}{versus('correctness')} |",
        f"| abstention accuracy ({summary['unanswerable_cases']} unanswerable) "
        f"| {_pct(summary['abstention_accuracy'])}{versus('abstention_accuracy')} |",
        f"| citation precision ({summary['citations_judged']} citations) "
        f"| {_pct(summary['citation_precision'])}{versus('citation_precision')} |",
        f"| citation recall ({summary['statements_needing_support']} sentences) "
        f"| {_pct(summary['citation_recall'])}{versus('citation_recall')} |",
        f"| turns completed | {_pct(summary['completion_rate'])}{versus('completion_rate')} |",
        f"| cases contradicting a reference fact | {summary['cases_with_contradiction']} |",
        f"| uncited factual sentences | {summary['uncited_claims']} |",
        f"| invented citation ids | {summary['citations_invented']} |",
        f"| grouped markers `[ev_a, ev_b]` (not rendered as badges) "
        f"| {summary.get('grouped_citation_markers', 0)} |",
        f"| tool calls / turn | {_num(cost['tool_calls_mean'])} |",
        f"| model calls / turn | {_num(cost['llm_calls_mean'])} |",
        f"| tokens / turn | {_num(cost['tokens_mean'], 0)} |",
        f"| wall seconds / turn (mean / median) "
        f"| {_num(cost['wall_seconds_mean'])} / {_num(cost['wall_seconds_median'])} |",
        f"| judge errors | {summary['judge_errors']} |",
        "",
        "## By category",
        "",
        "| category | cases | correctness | citation precision | citation recall |",
        "|---|---|---|---|---|",
    ]
    for name, row in summary["by_category"].items():
        lines.append(
            f"| {name} | {row['cases']} | {_pct(row['correctness'])} "
            f"| {_pct(row['citation_precision'])} | {_pct(row['citation_recall'])} |"
        )

    if regressions:
        lines += ["", "## Regressions against the baseline", ""]
        lines += [
            f"- **{r['metric']}**: {_pct(r['baseline'])} → {_pct(r['current'])}"
            for r in regressions
        ]

    lines += [
        "",
        "## Cases",
        "",
        "| case | category | result | cit. P | cit. R | tools | tokens | s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        case, score, record = r["case"], r.get("score"), r.get("record") or {}
        if not score:
            lines.append(f"| {case['id']} | {case['category']} | skipped: {r['skipped']} | | | | | |")
            continue
        if case["answerable"]:
            result = _pct(score["correctness"])
            if score["contradicted"]:
                result += f" ({score['contradicted']} contradicted)"
        else:
            verdict = score["abstention"] or {}
            result = "abstained" if verdict.get("correct") else (
                "fabricated" if verdict.get("fabricated") else "did not abstain"
            )
        if not record.get("completed"):
            result += f" · {record.get('status')}/{record.get('completion') or 'no answer'}"
            if record.get("stop_reason"):
                result += f" ({record['stop_reason']})"
        usage = record.get("usage") or {}
        lines.append(
            f"| {case['id']} | {case['category']} | {result} "
            f"| {_pct(score.get('citation_precision'))} | {_pct(score.get('citation_recall'))} "
            f"| {usage.get('tool_calls', 0)} | {_num(usage.get('tokens'), 0)} "
            f"| {_num(record.get('elapsed_seconds'))} |"
        )
    return "\n".join(lines) + "\n"
