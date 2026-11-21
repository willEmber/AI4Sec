"""Turn one recorded answer into scores, and many scores into a report."""

from __future__ import annotations

import asyncio
import statistics
from collections import defaultdict
from typing import Any

from evals import judges
from evals.dataset import EvalCase
from evals.judges import AskJudge, JudgeError
from evals.rank_claims import audit_rank_claims, pool_rank_audits
from evals.statements import Statement, split_statements

POINT_CREDIT = {"stated": 1.0, "partial": 0.5, "missing": 0.0, "contradicted": 0.0}

# An attempt passes when it finished on its own, got at least this share of
# its key points (and of its report's) with none contradicted, and called the
# tools it had to. A mean hides the run that dropped a step; this is what
# finds it, and what the flips between two runs are counted on.
PASS_CORRECTNESS = 0.75

# Metrics a baseline is compared on; all are "higher is better" fractions.
GATED_METRICS = (
    "completion_rate",
    "correctness",
    "abstention_accuracy",
    "citation_precision",
    "citation_recall",
    "pass_rate",
    "tool_selection_accuracy",
    "report_correctness",
)

# Cost metrics `--max-cost-rise` may gate; "lower is better", compared relatively.
COST_METRICS = ("tokens_mean", "wall_seconds_p95")


def _ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def check_tools(case: EvalCase, record: dict[str, Any]) -> dict[str, Any] | None:
    """Whether the scored turn called what it had to and nothing it must not.

    Read from the record alone, so a stored run is re-checked without a judge.
    """
    if not case.checks_tools:
        return None
    called = {str(call.get("tool") or "") for call in record.get("tool_calls") or []}
    missing = [name for name in case.expect_tools if name not in called]
    any_met = not case.expect_any_tools or bool(called & set(case.expect_any_tools))
    forbidden = sorted(called & set(case.forbid_tools))
    return {
        "called": sorted(called),
        "missing": missing,
        "any_met": any_met,
        "forbidden": forbidden,
        "ok": not missing and any_met and not forbidden,
    }


def scored_report(record: dict[str, Any]) -> str:
    """The text of the last report the scored turn put on screen, '' when none."""
    for report in reversed(record.get("reports") or []):
        if (report.get("markdown") or "").strip():
            return report["markdown"]
    return ""


def _credit(verdicts: list[str]) -> float:
    return round(sum(POINT_CREDIT[v] for v in verdicts) / len(verdicts), 4)


async def _score_citations(
    ask: AskJudge, case: EvalCase, statements: list[Statement], evidence: dict[str, Any]
) -> dict[str, Any]:
    """Judge every cited sentence against its passages, and find uncited claims.

    An id that resolves to no evidence — one the model made up, or one the
    runner dropped — is a citation that supports nothing: it needs no judge.
    """
    details: list[dict[str, Any]] = [
        {"index": s.index, "text": s.text, "citations": s.citations} for s in statements
    ]
    errors = 0

    async def check(position: int, statement: Statement) -> None:
        nonlocal errors
        known = [cid for cid in statement.citations if cid in evidence]
        detail = details[position]
        detail["invented"] = [cid for cid in statement.citations if cid not in evidence]
        if not known:
            detail.update(support="none", supporting=[])
            return
        try:
            verdict = await judges.judge_citation(
                ask,
                question=case.question,
                context=statements[position - 1].text if position else "",
                statement=statement.text,
                quotes=[evidence[cid]["quote"] for cid in known],
            )
        except JudgeError as exc:
            errors += 1
            detail["judge_error"] = str(exc)
            return
        detail["support"] = verdict["support"]
        detail["supporting"] = [cid for cid, ok in zip(known, verdict["passages"]) if ok]

    cited = [(i, s) for i, s in enumerate(statements) if s.citations]
    uncited = [s for s in statements if not s.citations]

    async def claims() -> set[int]:
        nonlocal errors
        try:
            return await judges.judge_claims(ask, question=case.question, statements=uncited)
        except JudgeError:
            errors += 1
            return set()

    results = await asyncio.gather(claims(), *(check(i, s) for i, s in cited))
    uncited_claims = results[0]
    for detail in details:
        if not detail["citations"]:
            detail["claim"] = detail["index"] in uncited_claims

    judged = [d for d in details if d["citations"] and "judge_error" not in d]
    citations_judged = sum(len(d["citations"]) for d in judged)
    citations_supporting = sum(len(d["supporting"]) for d in judged)
    full = sum(1 for d in judged if d["support"] == "full")
    return {
        "statements": {
            "total": len(statements),
            "cited": len(judged),
            "uncited_claims": len(uncited_claims),
            "full": full,
            "partial": sum(1 for d in judged if d["support"] == "partial"),
            "none": sum(1 for d in judged if d["support"] == "none"),
        },
        "citations": {
            "judged": citations_judged,
            "supporting": citations_supporting,
            "invented": sum(len(d["invented"]) for d in judged),
        },
        # Precision: of the citations written, how many back their sentence.
        "citation_precision": _ratio(citations_supporting, citations_judged),
        # Recall: of the sentences that needed backing, how many are fully backed.
        "citation_recall": _ratio(full, len(judged) + len(uncited_claims)),
        "judge_errors": errors,
        "details": details,
    }


def passed(case: EvalCase, record: dict[str, Any], score: dict[str, Any]) -> bool | None:
    """Whether this attempt did what the case asks; `None` when a judge could not say."""
    if "content_judge_error" in score or "report_judge_error" in score:
        return None
    if not record.get("completed"):
        return False
    if case.key_points and (
        score["correctness"] is None
        or score["correctness"] < PASS_CORRECTNESS
        or score["contradicted"]
    ):
        return False
    if not case.answerable and not (score["abstention"] or {}).get("correct"):
        return False
    if case.report_key_points and (
        score["report_correctness"] is None
        or score["report_correctness"] < PASS_CORRECTNESS
        or "contradicted" in score["report_points"]
    ):
        return False
    tools = score.get("tools")
    return not (tools and not tools["ok"])


def _report_evidence(record: dict[str, Any]) -> dict[str, Any]:
    """Evidence map for auditing a report: session evidence merged with frozen citations."""
    merged: dict[str, Any] = dict(record.get("evidence") or {})
    for rep in record.get("reports") or []:
        cits = rep.get("citations")
        if isinstance(cits, dict):
            for eid, entry in cits.items():
                if eid not in merged and isinstance(entry, dict):
                    merged[eid] = entry
    return merged


async def score_case(ask: AskJudge, case: EvalCase, record: dict[str, Any]) -> dict[str, Any]:
    """Score one recorded turn. A turn that produced no answer scores zero on
    what it was asked for; it is not dropped, or a crash would raise the mean."""
    answer = record.get("answer") or ""
    rep_text = scored_report(record)
    ev_map = record.get("evidence") or {}
    score: dict[str, Any] = {
        "answered": bool(answer.strip()),
        "correctness": None,
        "key_points": [],
        "contradicted": 0,
        "abstention": None,
        "report_correctness": None,
        "report_points": [],
        "tools": check_tools(case, record),
        "rank_claims": audit_rank_claims(answer, ev_map) if answer.strip() else audit_rank_claims("", {}),
        "report_rank_claims": (
            audit_rank_claims(rep_text, _report_evidence(record))
            if rep_text
            else audit_rank_claims("", {})
        ),
        "judge_errors": 0,
    }

    async def content() -> None:
        if not score["answered"]:
            if case.answerable and case.key_points:
                score["correctness"] = 0.0
                score["key_points"] = ["missing"] * len(case.key_points)
            elif not case.answerable:
                score["abstention"] = {"abstained": False, "fabricated": False, "correct": False}
            return
        try:
            if case.answerable and case.key_points:
                verdicts = await judges.judge_key_points(ask, case, answer)
                score["key_points"] = verdicts
                score["contradicted"] = verdicts.count("contradicted")
                score["correctness"] = _credit(verdicts)
            elif not case.answerable:
                verdict = await judges.judge_abstention(ask, case, answer)
                verdict["correct"] = verdict["abstained"] and not verdict["fabricated"]
                score["abstention"] = verdict
        except JudgeError as exc:
            score["judge_errors"] += 1
            score["content_judge_error"] = str(exc)

    async def report() -> None:
        if not case.report_key_points:
            return
        text = scored_report(record)
        if not text:
            # No report is a report that states none of its points.
            score["report_points"] = ["missing"] * len(case.report_key_points)
            score["report_correctness"] = 0.0
            return
        try:
            verdicts = await judges.judge_report_points(ask, case, text)
        except JudgeError as exc:
            score["judge_errors"] += 1
            score["report_judge_error"] = str(exc)
            return
        score["report_points"] = verdicts
        score["report_correctness"] = _credit(verdicts)

    async def citations() -> None:
        if not (score["answered"] and case.cites):
            return
        found = await _score_citations(
            ask, case, split_statements(answer), record.get("evidence") or {}
        )
        score["judge_errors"] += found.pop("judge_errors")
        score.update(found)

    await asyncio.gather(content(), report(), citations())
    score["passed"] = passed(case, record, score)
    return score


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def _percentile(values: list[float], q: float) -> float | None:
    """The q-th percentile (0–100), interpolated; the value itself for one sample."""
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 1)
    cuts = statistics.quantiles(values, n=100, method="inclusive")
    return round(cuts[int(q) - 1], 1)


def case_outcomes(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Per case: attempts scored, attempts a judge could decide, attempts passed."""
    outcomes: dict[str, dict[str, int]] = defaultdict(lambda: {"attempts": 0, "decided": 0, "passed": 0})
    for r in results:
        if not r.get("score"):
            continue
        row = outcomes[r["case"]["id"]]
        row["attempts"] += 1
        verdict = r["score"].get("passed")
        if verdict is not None:
            row["decided"] += 1
            row["passed"] += int(verdict)
    return dict(outcomes)


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Dataset-level metrics from per-attempt results (`{case, record, score}`).

    Correctness is a mean over attempts; the citation metrics pool every
    sentence and citation in the run, so a long answer weighs what it wrote.
    """
    scored = [r for r in results if r.get("score")]
    answerable = [r for r in scored if r["case"]["answerable"]]
    unanswerable = [r for r in scored if not r["case"]["answerable"]]

    def pooled(rows: list[dict[str, Any]]) -> dict[str, Any]:
        statements = [r["score"]["statements"] for r in rows if "statements" in r["score"]]
        citations = [r["score"]["citations"] for r in rows if "citations" in r["score"]]
        judged = sum(c["judged"] for c in citations)
        needed = sum(s["cited"] + s["uncited_claims"] for s in statements)
        return {
            "citation_precision": _ratio(sum(c["supporting"] for c in citations), judged),
            "citation_recall": _ratio(sum(s["full"] for s in statements), needed),
            "citations_judged": judged,
            "citations_invented": sum(c["invented"] for c in citations),
            "statements_needing_support": needed,
            "uncited_claims": sum(s["uncited_claims"] for s in statements),
        }

    def content(rows: list[dict[str, Any]]) -> dict[str, Any]:
        graded = [r["score"]["correctness"] for r in rows if r["score"]["correctness"] is not None]
        decided = [r["score"].get("passed") for r in rows if r["score"].get("passed") is not None]
        return {
            "cases": len(rows),
            "correctness": _mean(graded),
            "cases_with_contradiction": sum(1 for r in rows if r["score"]["contradicted"]),
            "pass_rate": _ratio(sum(1 for p in decided if p), len(decided)),
        }

    abstentions = [r["score"]["abstention"] for r in unanswerable if r["score"]["abstention"]]
    usage = [r["record"].get("usage") or {} for r in scored]
    walls = [r["record"]["elapsed_seconds"] for r in scored if r["record"].get("elapsed_seconds")]
    tokens = [float(u["tokens"]) for u in usage if u.get("tokens")]
    decided = [r["score"].get("passed") for r in scored if r["score"].get("passed") is not None]
    tools = [r["score"]["tools"] for r in scored if r["score"].get("tools")]
    reports = [
        r["score"]["report_correctness"] for r in scored
        if r["score"].get("report_correctness") is not None
    ]
    outcomes = case_outcomes(results)
    ans_rank_pool = pool_rank_audits(
        [r["score"]["rank_claims"] for r in scored if "rank_claims" in r["score"]]
    )
    rep_rank_pool = pool_rank_audits(
        [r["score"]["report_rank_claims"] for r in scored if "report_rank_claims" in r["score"]]
    )

    categories = sorted({r["case"]["category"] for r in answerable})
    summary: dict[str, Any] = {
        "cases": len(outcomes),
        "attempts": len(scored),
        "skipped": len(results) - len(scored),
        "completion_rate": _ratio(
            sum(1 for r in scored if r["record"].get("completed")), len(scored)
        ),
        "pass_rate": _ratio(sum(1 for p in decided if p), len(decided)),
        **{k: v for k, v in content(answerable).items() if k not in ("cases", "pass_rate")},
        "answerable_cases": len(answerable),
        "abstention_accuracy": _ratio(sum(1 for a in abstentions if a["correct"]), len(abstentions)),
        "fabrication_rate": _ratio(sum(1 for a in abstentions if a["fabricated"]), len(abstentions)),
        "unanswerable_cases": len(unanswerable),
        "tool_selection_accuracy": _ratio(sum(1 for t in tools if t["ok"]), len(tools)),
        "tool_checked_attempts": len(tools),
        "report_correctness": _mean(reports),
        "report_attempts": len(reports),
        "reports_missing": sum(
            1 for r in scored
            if r["case"].get("report_key_points") and not scored_report(r["record"])
        ),
        **pooled(scored),
        "answers_without_citation": sum(
            1 for r in answerable
            if r["case"].get("cites", True)
            and r["score"]["answered"]
            and not r["score"].get("citations", {}).get("judged")
        ),
        "grouped_citation_markers": sum(
            int(r["record"].get("grouped_citation_markers") or 0) for r in scored
        ),
        "rank_claims": ans_rank_pool["claims"],
        "rank_support": ans_rank_pool["support_rate"],
        "rank_supported": ans_rank_pool["supported"],
        "rank_unsupported": ans_rank_pool["unsupported"],
        "rank_uncited": ans_rank_pool["uncited"],
        "rank_contradicted": ans_rank_pool["contradicted"],
        "report_rank_claims": rep_rank_pool["claims"],
        "report_rank_support": rep_rank_pool["support_rate"],
        "report_rank_supported": rep_rank_pool["supported"],
        "report_rank_unsupported": rep_rank_pool["unsupported"],
        "report_rank_uncited": rep_rank_pool["uncited"],
        "report_rank_contradicted": rep_rank_pool["contradicted"],
        "judge_errors": sum(r["score"]["judge_errors"] for r in scored),
        "flaky_cases": sorted(
            case_id for case_id, row in outcomes.items() if 0 < row["passed"] < row["decided"]
        ),
        "case_outcomes": outcomes,
        "cost": {
            "tool_calls_mean": _mean([float(u.get("tool_calls") or 0) for u in usage]),
            "llm_calls_mean": _mean([float(u.get("llm_calls") or 0) for u in usage]),
            "tokens_mean": _mean(tokens),
            "tokens_p95": _percentile(tokens, 95),
            "tokens_total": sum(int(u.get("tokens") or 0) for u in usage),
            "wall_seconds_mean": _mean(walls),
            "wall_seconds_median": round(statistics.median(walls), 1) if walls else None,
            "wall_seconds_p95": _percentile(walls, 95),
        },
        "by_category": {
            name: {
                **content([r for r in answerable if r["case"]["category"] == name]),
                **{
                    k: v
                    for k, v in pooled(
                        [r for r in answerable if r["case"]["category"] == name]
                    ).items()
                    if k in ("citation_precision", "citation_recall")
                },
            }
            for name in categories
        },
    }
    return summary


def compare(
    summary: dict[str, Any],
    baseline: dict[str, Any],
    *,
    max_drop: float,
    max_cost_rise: float = 0.0,
) -> list[dict[str, Any]]:
    """Gated metrics that fell more than `max_drop` (absolute) below the baseline,
    and, with `max_cost_rise`, cost metrics that rose more than that fraction.

    A metric missing on either side is not a regression — a subset run has no
    unanswerable cases, and that says nothing about abstention.
    """
    regressions = []
    for name in GATED_METRICS:
        now, before = summary.get(name), baseline.get(name)
        if now is None or before is None:
            continue
        if before - now > max_drop:
            regressions.append({"metric": name, "baseline": before, "current": now})
    for count_key in ("rank_contradicted", "report_rank_contradicted"):
        now_c = int(summary.get(count_key) or 0)
        before_c = int(baseline.get(count_key) or 0)
        if now_c > before_c:
            regressions.append(
                {"metric": count_key, "baseline": before_c, "current": now_c, "kind": "count"}
            )
    if max_cost_rise > 0:
        for name in COST_METRICS:
            now = (summary.get("cost") or {}).get(name)
            before = (baseline.get("cost") or {}).get(name)
            if now is None or not before:
                continue
            if now > before * (1 + max_cost_rise):
                regressions.append(
                    {"metric": name, "baseline": before, "current": now, "kind": "cost"}
                )
    return regressions


def flips(summary: dict[str, Any], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    """Cases whose outcome changed against the baseline, worst first.

    `regressed`: passed every decided attempt before, passed none now. Gated
    only when both sides decided at least two attempts — one observation
    cannot tell a broken case from a flaky one. `flaky`: passed every attempt
    before, only some now. `fixed`: the reverse of `regressed`.
    """
    before_all = baseline.get("case_outcomes") or {}
    now_all = summary.get("case_outcomes") or {}
    out = []
    for case_id in sorted(set(before_all) & set(now_all)):
        before, now = before_all[case_id], now_all[case_id]
        if not before["decided"] or not now["decided"]:
            continue
        was_pass = before["passed"] == before["decided"]
        was_fail = before["passed"] == 0
        is_pass = now["passed"] == now["decided"]
        is_fail = now["passed"] == 0
        if was_pass and is_fail:
            kind = "regressed"
        elif was_pass and not is_pass:
            kind = "flaky"
        elif was_fail and is_pass:
            kind = "fixed"
        else:
            continue
        out.append({
            "case": case_id,
            "kind": kind,
            "baseline": f"{before['passed']}/{before['decided']}",
            "current": f"{now['passed']}/{now['decided']}",
            "gated": kind == "regressed" and min(before["decided"], now["decided"]) >= 2,
        })
    order = {"regressed": 0, "flaky": 1, "fixed": 2}
    return sorted(out, key=lambda f: (order[f["kind"]], f["case"]))
