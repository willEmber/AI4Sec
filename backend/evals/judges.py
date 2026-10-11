"""LLM judges: what an answer got right, and whether its citations hold.

Each judge asks one narrow question and returns JSON, because a judge that is
asked for an overall grade hands back its own taste. A verdict that cannot be
parsed after one repeat is recorded as an error and left out of the metric's
denominator — counting it as a failure would let a flaky judge move the score.

The judge is a plain `async (prompt) -> str`, so the scoring code is tested
with a scripted one and no network.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from evals.dataset import EvalCase
from evals.statements import Statement

logger = logging.getLogger("scholar.evals.judges")

JUDGE_PROMPT_VERSION = "judge-1"
MAX_QUOTE_CHARS = 6000   # a table's quote limit; a judge shown half a table fails it

AskJudge = Callable[[str], Awaitable[str]]



class JudgeError(RuntimeError):
    """The judge did not return a usable verdict."""


def parse_json(text: str) -> dict[str, Any]:
    """The first JSON object in the reply.

    Tried from every `{`, not from the first one to the last `}`: a judge that
    explains itself in LaTeX (`\\hat{x}_T`) before the verdict writes braces
    that are not JSON.
    """
    text = text or ""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            data, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            return data
        start = text.find("{", start + 1)
    raise JudgeError("no JSON object in the judge's reply")


async def _ask_json(ask: AskJudge, prompt: str, check: Callable[[dict[str, Any]], Any]) -> Any:
    """Ask, parse and validate; one repeat on a malformed reply."""
    last: Exception | None = None
    for _ in range(2):
        try:
            return check(parse_json(await ask(prompt)))
        except (JudgeError, KeyError, TypeError, ValueError) as exc:
            last = exc
    raise JudgeError(str(last))


KEY_POINTS_PROMPT = """You grade an answer to a question about a research paper against a list of \
reference facts. The reference facts are correct; do not second-guess them.

For each reference fact decide whether the ANSWER states it:
- "stated": the answer conveys this fact. Wording, language and rounding may differ \
(24.56 vs 24.6 is fine; 24.56 vs 25.4 is not).
- "partial": the answer conveys part of it, or conveys it without a specific value the fact gives.
- "missing": the answer does not mention it.
- "contradicted": the answer states something incompatible with it (a different number, the \
opposite direction, a different method).

QUESTION:
{question}

REFERENCE FACTS:
{points}

ANSWER:
{answer}

Reply with JSON only:
{{"points": [{{"n": 1, "verdict": "stated|partial|missing|contradicted"}}, ...]}}"""

ABSTENTION_PROMPT = """A reader asked a question that the attached paper(s) CANNOT answer: \
{note}

Decide two things about the ANSWER:
- "abstained": true if the answer tells the reader that the paper(s) do not report / contain \
this information (it may still offer related facts that the paper does contain).
- "fabricated": true if the answer presents a specific value or claim as the answer to the \
question, as if it came from the paper(s).

QUESTION:
{question}

ANSWER:
{answer}

Reply with JSON only:
{{"abstained": true|false, "fabricated": true|false}}"""

CITATION_PROMPT = """You check whether quoted source passages support a sentence taken from an \
answer about research papers. Judge only from the passages; do not use outside knowledge.

The sentence may be in a different language from the passages. Passages may contain HTML tables \
and LaTeX.

QUESTION the answer responds to:
{question}

PRECEDING SENTENCE (context only, for resolving "it" / "this method"; do not judge it):
{context}

SENTENCE to check:
{statement}

PASSAGES cited by the sentence:
{passages}

Decide:
- "support": "full" if every factual claim in the sentence is backed by the passages taken \
together; "partial" if some claims are backed and others are not; "none" if the passages do not \
back it or contradict it.
- for each passage, "supports": true if that passage on its own backs at least one claim of the \
sentence, false if it is irrelevant to the sentence.

Reply with JSON only:
{{"support": "full|partial|none", "passages": [{{"n": 1, "supports": true|false}}, ...]}}"""

CLAIMS_PROMPT = """Below are numbered sentences from an answer about research papers. None of \
them carries a citation.

List the sentences that assert a specific fact about a paper or its results — something a \
reader would need a source for (a number, a method detail, what an experiment showed, what the \
authors claim).

Do NOT list: section headings, transitions, restatements of the question, offers of further \
help, statements that the paper does not contain something, the answerer's own advice or \
opinion, or a summary sentence that only introduces the cited sentences after it.

QUESTION the answer responds to:
{question}

SENTENCES:
{sentences}

Reply with JSON only:
{{"claims": [n, ...]}}"""

_POINT_VERDICTS = {"stated", "partial", "missing", "contradicted"}
_SUPPORT_VERDICTS = {"full", "partial", "none"}


def _numbered(items: list[str]) -> str:
    return "\n".join(f"{n}. {item}" for n, item in enumerate(items, start=1))


async def judge_key_points(ask: AskJudge, case: EvalCase, answer: str) -> list[str]:
    """One verdict per key point, in the case's order."""

    def check(data: dict[str, Any]) -> list[str]:
        by_n = {int(p["n"]): str(p["verdict"]).lower() for p in data["points"]}
        verdicts = [by_n[n] for n in range(1, len(case.key_points) + 1)]
        if any(v not in _POINT_VERDICTS for v in verdicts):
            raise ValueError(f"unknown verdict in {verdicts}")
        return verdicts

    prompt = KEY_POINTS_PROMPT.format(
        question=case.question, points=_numbered(case.key_points), answer=answer
    )
    return await _ask_json(ask, prompt, check)


async def judge_abstention(ask: AskJudge, case: EvalCase, answer: str) -> dict[str, bool]:
    def check(data: dict[str, Any]) -> dict[str, bool]:
        return {"abstained": bool(data["abstained"]), "fabricated": bool(data["fabricated"])}

    prompt = ABSTENTION_PROMPT.format(
        note=case.note or "the information is not in them.",
        question=case.question,
        answer=answer,
    )
    return await _ask_json(ask, prompt, check)


async def judge_citation(
    ask: AskJudge, *, question: str, context: str, statement: str, quotes: list[str]
) -> dict[str, Any]:
    """`{"support": ..., "passages": [bool, ...]}` for one cited sentence."""

    def check(data: dict[str, Any]) -> dict[str, Any]:
        support = str(data["support"]).lower()
        if support not in _SUPPORT_VERDICTS:
            raise ValueError(f"unknown support verdict {support!r}")
        by_n = {int(p["n"]): bool(p["supports"]) for p in data["passages"]}
        return {"support": support, "passages": [by_n[n] for n in range(1, len(quotes) + 1)]}

    passages = "\n\n".join(
        f"[{n}] {quote[:MAX_QUOTE_CHARS]}" for n, quote in enumerate(quotes, start=1)
    )
    prompt = CITATION_PROMPT.format(
        question=question, context=context or "(none)", statement=statement, passages=passages
    )
    return await _ask_json(ask, prompt, check)


async def judge_claims(ask: AskJudge, *, question: str, statements: list[Statement]) -> set[int]:
    """Indices (`Statement.index`) of uncited sentences that needed a citation."""
    if not statements:
        return set()

    def check(data: dict[str, Any]) -> set[int]:
        picked = {int(n) for n in data["claims"]}
        if any(n < 1 or n > len(statements) for n in picked):
            raise ValueError(f"claim number out of range: {sorted(picked)}")
        return {statements[n - 1].index for n in picked}

    prompt = CLAIMS_PROMPT.format(
        question=question, sentences=_numbered([s.text for s in statements])
    )
    return await _ask_json(ask, prompt, check)


def llm_judge(model: str = "", *, concurrency: int = 6) -> AskJudge:
    """A judge on the application's own gateway.

    Thinking is off and the temperature is zero: a verdict is a short JSON
    object, and reasoning billed against the output ceiling is how utility
    calls here have come back empty before.
    """
    from app.services.llm_gateway.registry import get_registry
    from app.services.llm_service import get_llm_service

    llm = get_llm_service()
    name = model or get_registry().utility_model
    slots = asyncio.Semaphore(concurrency)

    async def ask(prompt: str) -> str:
        async with slots:
            return await llm.chat(
                messages=[{"role": "user", "content": prompt}],
                model=name,
                temperature=0.0,
                max_tokens=3000,
                enable_thinking=False,
            )

    return ask
