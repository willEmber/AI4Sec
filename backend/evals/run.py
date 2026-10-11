"""Run the golden set against the agent and score it.

Talks to the live gateway and spends tokens (one agent turn per case, plus the
judge's calls on the utility model). The papers a case reads must already be
parsed in this database; a case whose paper is missing is skipped and reported,
never silently dropped.

    cd backend
    uv run python -m evals.run --validate                 # offline: dataset only
    uv run python -m evals.run --limit 3                  # smoke run
    uv run python -m evals.run                            # whole set
    uv run python -m evals.run --model <name> --category numeric,equation
    uv run python -m evals.run --rejudge data/evals/<run>/results.json
    uv run python -m evals.run --rejudge data/evals/<run>/results.json --control
    uv run python -m evals.run --baseline evals/baselines/reading_seed.json

Records land in `data/evals/<time>-<model>/` (`results.json`, `summary.md`).
With `--baseline`, a gated metric more than `--max-drop` below the baseline
makes the exit code 1. `--save-baseline` writes this run's summary there.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals import report, scoring  # noqa: E402
from evals.dataset import Dataset, EvalCase, load_dataset  # noqa: E402
from evals.judges import JUDGE_PROMPT_VERSION, AskJudge  # noqa: E402

logger = logging.getLogger("scholar.evals")


def _commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=Path(__file__).parent,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


async def _run_cases(
    dataset: Dataset, cases: list[EvalCase], *, model: str, concurrency: int, ask: AskJudge
) -> list[dict[str, Any]]:
    """Run and score the cases, a few at a time, keeping the dataset's order."""
    from app.services import identity

    from evals import runner

    needed = sorted({dataset.papers[key].paper_id for case in cases for key in case.papers})
    parsed = await runner.parsed_paper_ids(needed)
    owner_id, _ = await identity.create_principal(label=f"eval:{dataset.name}")
    slots = asyncio.Semaphore(concurrency)

    async def one(case: EvalCase) -> dict[str, Any]:
        result: dict[str, Any] = {"case": case.model_dump()}
        missing = [key for key in case.papers if dataset.papers[key].paper_id not in parsed]
        if missing:
            result["skipped"] = f"paper not parsed here: {', '.join(missing)}"
            print(f"  skip  {case.id}: {result['skipped']}")
            return result
        async with slots:
            try:
                record = await runner.run_case(dataset, case, owner_id=owner_id, model=model)
            except Exception as exc:  # noqa: BLE001 — one case must not end the run
                logger.exception("case %s failed outside the turn", case.id)
                record = {"status": "error", "completed": False, "answer": "",
                          "error_code": type(exc).__name__, "usage": {}}
        result["record"] = record
        result["score"] = await scoring.score_case(ask, case, record)
        print(f"  {'done' if record.get('completed') else record.get('status'):5} {case.id}"
              f" — {_headline(case, result['score'])}")
        return result

    return list(await asyncio.gather(*(one(case) for case in cases)))


def _headline(case: EvalCase, score: dict[str, Any]) -> str:
    if case.answerable:
        content = f"correct {report._pct(score['correctness'])}"
    else:
        content = "abstained" if (score["abstention"] or {}).get("correct") else "did not abstain"
    return (f"{content}, cit P {report._pct(score.get('citation_precision'))}"
            f" R {report._pct(score.get('citation_recall'))}")


def _shuffle_evidence(results: list[dict[str, Any]]) -> None:
    """Give every cited id the passage of an id half the run away.

    The negative control for the citation judge: the sentences are unchanged
    and now cite the wrong passages, so precision and recall should collapse.
    A judge that still calls them supported is reading the sentence, not the
    passage, and its numbers on the real run mean nothing.
    """
    slots = [
        (record["evidence"], evidence_id)
        for record in (r.get("record") or {} for r in results)
        for evidence_id in record.get("evidence") or {}
    ]
    quotes = [evidence[evidence_id]["quote"] for evidence, evidence_id in slots]
    shift = len(slots) // 2
    for position, (evidence, evidence_id) in enumerate(slots):
        evidence[evidence_id]["quote"] = quotes[(position + shift) % len(slots)]


async def _rejudge(
    path: Path, dataset: Dataset, ask: AskJudge, *, shuffle: bool = False
) -> tuple[dict, list[dict]]:
    """Score stored trajectories again; the agent is not run."""
    from evals import runner

    stored = json.loads(path.read_text(encoding="utf-8"))
    by_id = {case.id: case for case in dataset.cases}
    if shuffle:
        _shuffle_evidence(stored["results"])

    async def one(result: dict[str, Any]) -> dict[str, Any]:
        # The dataset's current key points apply, so a corrected reference
        # re-scores an old run; a case since removed keeps its stored form.
        case = by_id.get(result["case"]["id"]) or EvalCase.model_validate(result["case"])
        result["case"] = case.model_dump()
        if result.get("record") and not shuffle:
            await runner.refresh_evidence(result["record"])
        if result.get("record"):
            result["score"] = await scoring.score_case(ask, case, result["record"])
        return result

    return stored["meta"], list(await asyncio.gather(*(one(r) for r in stored["results"])))


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="reading_seed", help="Name in evals/datasets, or a path")
    parser.add_argument("--model", default="", help="Conversation model; defaults to the configured one")
    parser.add_argument("--judge-model", default="", help="Judge model; defaults to the utility model")
    parser.add_argument("--case", default="", help="Comma-separated case ids")
    parser.add_argument("--category", default="", help="Comma-separated categories")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N selected cases")
    parser.add_argument("--concurrency", type=int, default=2, help="Turns in flight at once")
    parser.add_argument("--out", default="", help="Output directory")
    parser.add_argument("--validate", action="store_true", help="Check the dataset and exit")
    parser.add_argument("--rejudge", default="", help="Re-score a stored results.json")
    parser.add_argument("--control", action="store_true",
                        help="With --rejudge: score against shuffled passages (judge sanity check)")
    parser.add_argument("--baseline", default="", help="Summary to compare against")
    parser.add_argument("--max-drop", type=float, default=0.05,
                        help="Largest tolerated fall of a gated metric (absolute)")
    parser.add_argument("--save-baseline", default="", help="Write this run's summary here")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    cases = dataset.select(
        ids=_split(args.case), categories=_split(args.category), limit=args.limit
    )
    if args.control and not args.rejudge:
        raise SystemExit("--control scores a stored run: pass --rejudge <results.json>.")
    if args.validate:
        by_category: dict[str, int] = {}
        for case in dataset.cases:
            by_category[case.category] = by_category.get(case.category, 0) + 1
        print(f"{dataset.name} v{dataset.version}: {len(dataset.cases)} cases, "
              f"{len(dataset.papers)} papers — {by_category}")
        return 0
    if not cases:
        raise SystemExit("No case matches the selection.")

    from app.agents.prompts import PROMPT_VERSION
    from app.config import get_settings
    from app.db import database as db
    from app.services import agent_runner
    from app.services.llm_gateway import get_registry
    from app.services.llm_service import close_shared_client

    from evals.judges import llm_judge

    settings = get_settings()
    db.configure(settings.database_url)
    await db.init_db()
    registry = get_registry()
    judge_model = args.judge_model or registry.utility_model
    ask = llm_judge(judge_model)

    try:
        if args.rejudge:
            meta, results = await _rejudge(Path(args.rejudge), dataset, ask, shuffle=args.control)
            meta = {**meta, "rejudged_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "control": "shuffled_evidence" if args.control else ""}
        else:
            model = args.model or settings.agent_model or registry.agent_default_model
            print(f"{dataset.name} v{dataset.version}: {len(cases)} cases · model {model} "
                  f"· judge {judge_model}\n")
            await agent_runner.open_agent_checkpointer()
            try:
                results = await _run_cases(
                    dataset, cases, model=model, concurrency=args.concurrency, ask=ask
                )
            finally:
                await agent_runner.close_agent_checkpointer()
            meta = {
                "dataset": dataset.name,
                "dataset_version": dataset.version,
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "commit": _commit(),
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "partial": len(cases) != len(dataset.cases),
            }
        meta.update(judge_model=judge_model, judge_prompt_version=JUDGE_PROMPT_VERSION)
    finally:
        await close_shared_client()
        await db.close_pool()

    summary = scoring.aggregate(results)
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8")) if args.baseline else None
    regressions = (
        scoring.compare(summary, baseline["summary"], max_drop=args.max_drop) if baseline else []
    )

    out = Path(args.out) if args.out else (
        Path(args.rejudge).parent if args.rejudge
        else settings.data_dir / "evals" / f"{time.strftime('%Y%m%d-%H%M%S')}-{meta['model']}"
    )
    out.mkdir(parents=True, exist_ok=True)
    # A control run is kept beside the real one, never over it.
    stem = ("control", "control") if args.control else ("results", "summary")
    payload = {"meta": meta, "summary": summary, "results": results}
    (out / f"{stem[0]}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = report.render_markdown(
        meta, summary, results, regressions, baseline["summary"] if baseline else None
    )
    (out / f"{stem[1]}.md").write_text(markdown, encoding="utf-8")
    if args.save_baseline:
        Path(args.save_baseline).write_text(
            json.dumps({"meta": meta, "summary": summary}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print("\n" + markdown.split("## Cases")[0])
    print(f"report: {out / f'{stem[1]}.md'}")
    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
