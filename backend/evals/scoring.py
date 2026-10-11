"""Turn one recorded answer into scores, and many scores into a report."""

from __future__ import annotations

import asyncio
import statistics
from typing import Any

from evals import judges
from evals.dataset import EvalCase
from evals.judges import AskJudge, JudgeError
from evals.statements import Statement, split_statements

POINT_CREDIT = {"stated": 1.0, "partial": 0.5, "missing": 0.0, "contradicted": 0.0}

# Metrics a baseline is compared on; all are "higher is better" fractions.
GATED_METRICS = (
    "completion_rate",
    "correctness",
    "abstention_accuracy",
    "citation_precision",
    "citation_recall",
)


def _ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


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


async def score_case(ask: AskJudge, case: EvalCase, record: dict[str, Any]) -> dict[str, Any]:
    """Score one recorded turn. A turn that produced no answer scores zero on
    what it was asked for; it is not dropped, or a crash would raise the mean."""
    answer = record.get("answer") or ""
    score: dict[str, Any] = {
        "answered": bool(answer.strip()),
        "correctness": None,
        "key_points": [],
        "contradicted": 0,
        "abstention": None,
        "judge_errors": 0,
    }
    if not score["answered"]:
        if case.answerable:
            score["correctness"] = 0.0
            score["key_points"] = ["missing"] * len(case.key_points)
        else:
            score["abstention"] = {"abstained": False, "fabricated": False, "correct": False}
        return score

    async def content() -> None:
        try:
            if case.answerable:
                verdicts = await judges.judge_key_points(ask, case, answer)
                score["key_points"] = verdicts
                score["contradicted"] = verdicts.count("contradicted")
                score["correctness"] = round(
                    sum(POINT_CREDIT[v] for v in verdicts) / len(verdicts), 4
                )
            else:
                verdict = await judges.judge_abstention(ask, case, answer)
                verdict["correct"] = verdict["abstained"] and not verdict["fabricated"]
                score["abstention"] = verdict
        except JudgeError as exc:
            score["judge_errors"] += 1
            score["content_judge_error"] = str(exc)

    citation_scores, _ = await asyncio.gather(
        _score_citations(ask, case, split_statements(answer), record.get("evidence") or {}),
        content(),
    )
    score["judge_errors"] += citation_scores.pop("judge_errors")
    score.update(citation_scores)
    return score


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Dataset-level metrics from per-case results (`{case, record, score}`).

    Correctness is a mean over cases; the citation metrics pool every sentence
    and citation in the run, so a long answer weighs what it wrote.
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
        return {
            "cases": len(rows),
            "correctness": _mean(graded),
            "cases_with_contradiction": sum(1 for r in rows if r["score"]["contradicted"]),
        }

    abstentions = [r["score"]["abstention"] for r in unanswerable if r["score"]["abstention"]]
    usage = [r["record"].get("usage") or {} for r in scored]
    walls = [r["record"]["elapsed_seconds"] for r in scored if r["record"].get("elapsed_seconds")]

    categories = sorted({r["case"]["category"] for r in answerable})
    summary: dict[str, Any] = {
        "cases": len(scored),
        "skipped": len(results) - len(scored),
        "completion_rate": _ratio(
            sum(1 for r in scored if r["record"].get("completed")), len(scored)
        ),
        **{k: v for k, v in content(answerable).items() if k != "cases"},
        "answerable_cases": len(answerable),
        "abstention_accuracy": _ratio(sum(1 for a in abstentions if a["correct"]), len(abstentions)),
        "fabrication_rate": _ratio(sum(1 for a in abstentions if a["fabricated"]), len(abstentions)),
        "unanswerable_cases": len(unanswerable),
        **pooled(scored),
        "answers_without_citation": sum(
            1 for r in answerable
            if r["score"]["answered"] and not r["score"].get("citations", {}).get("judged")
        ),
        "grouped_citation_markers": sum(
            int(r["record"].get("grouped_citation_markers") or 0) for r in scored
        ),
        "judge_errors": sum(r["score"]["judge_errors"] for r in scored),
        "cost": {
            "tool_calls_mean": _mean([float(u.get("tool_calls") or 0) for u in usage]),
            "llm_calls_mean": _mean([float(u.get("llm_calls") or 0) for u in usage]),
            "tokens_mean": _mean([float(u["tokens"]) for u in usage if u.get("tokens")]),
            "tokens_total": sum(int(u.get("tokens") or 0) for u in usage),
            "wall_seconds_mean": _mean(walls),
            "wall_seconds_median": round(statistics.median(walls), 1) if walls else None,
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
    summary: dict[str, Any], baseline: dict[str, Any], *, max_drop: float
) -> list[dict[str, Any]]:
    """Gated metrics that fell more than `max_drop` (absolute) below the baseline.

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
    return regressions
