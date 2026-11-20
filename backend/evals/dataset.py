"""The golden set: papers, questions, and what a right answer must contain."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

DATASET_DIR = Path(__file__).resolve().parent / "datasets"

Category = Literal[
    "method", "fact", "numeric", "equation", "reasoning", "multi_paper", "unanswerable",
    # Chains (P16): a whole-paper report, a question answered after one, and a
    # paper the agent must find and fetch before it can read it.
    "report", "follow_up", "acquisition",
]
Mode = Literal["auto", "snap", "lens", "sphere"]

# The tools that make a report: the three report tools, and `begin_report`,
# which opens a report written as the answer (P16; both while modes switch
# over). A turn that runs one makes a report, which an eval must not take from
# a cache: the second attempt would measure a lookup.
MODE_TOOLS = frozenset(
    {"run_insight_snap", "run_logic_lens", "run_research_sphere", "begin_report"}
)

_PAPER_ID_RE = re.compile(r"^[0-9a-f]{40}$")


class EvalPaper(BaseModel):
    """A paper a case reads. `paper_id` is the SHA-1 of the PDF, as everywhere
    else, so a case is pinned to the exact file its reference was written from;
    `source` only tells a person where that file came from."""

    paper_id: str
    title: str
    source: str = ""

    @model_validator(mode="after")
    def _check(self) -> EvalPaper:
        if not _PAPER_ID_RE.match(self.paper_id):
            raise ValueError(f"paper_id is not a SHA-1: {self.paper_id!r}")
        return self


class SetupTurn(BaseModel):
    """A turn run in the case's conversation before the scored question.

    Not scored: it puts the conversation where the question needs it — a
    report already on screen, say — the way a reader's earlier message would.
    """

    question: str
    mode: Mode = "auto"


class EvalCase(BaseModel):
    id: str
    category: Category
    language: Literal["zh", "en"] = "zh"
    # Attached before the first turn. Empty only for an acquisition case,
    # whose paper the agent has to find.
    papers: list[str] = Field(default_factory=list)
    question: str
    # The mode the reader picked in the composer for the scored question.
    mode: Mode = "auto"
    setup: list[SetupTurn] = Field(default_factory=list)
    # False: the attached papers do not contain the answer, and the right
    # behaviour is to say so. Such a case has no key points.
    answerable: bool = True
    # Atomic facts a correct answer states. Each is judged on its own, so a
    # point should hold one checkable claim (one number, one mechanism).
    key_points: list[str] = Field(default_factory=list)
    # Facts the report made in the scored turn must contain, judged against
    # the report rather than the answer: the answer only summarises it.
    report_key_points: list[str] = Field(default_factory=list)
    # Whether the answer is expected to cite passages. An answer summarising a
    # report is told not to cite it, so its citations are not scored.
    cites: bool = True
    # Tool selection in the scored turn: every one of `expect_tools`, at least
    # one of `expect_any_tools`, none of `forbid_tools`.
    expect_tools: list[str] = Field(default_factory=list)
    expect_any_tools: list[str] = Field(default_factory=list)
    forbid_tools: list[str] = Field(default_factory=list)
    # Why an unanswerable case is unanswerable — shown to the judge.
    note: str = ""

    @property
    def runs_modes(self) -> bool:
        """Whether any turn of this case can make a report."""
        return (
            self.mode != "auto"
            or any(turn.mode != "auto" for turn in self.setup)
            or bool(MODE_TOOLS & set(self.expect_tools))
        )

    @property
    def checks_tools(self) -> bool:
        return bool(self.expect_tools or self.expect_any_tools or self.forbid_tools)

    def tool_names(self) -> set[str]:
        return {*self.expect_tools, *self.expect_any_tools, *self.forbid_tools}

    @model_validator(mode="after")
    def _check(self) -> EvalCase:
        if self.answerable and not (self.key_points or self.report_key_points):
            raise ValueError(f"{self.id}: an answerable case needs key_points or report_key_points")
        if not self.answerable and (self.key_points or self.report_key_points):
            raise ValueError(f"{self.id}: an unanswerable case has no key_points")
        if (self.category == "unanswerable") != (not self.answerable):
            raise ValueError(f"{self.id}: category and answerable disagree")
        if self.category == "multi_paper" and len(self.papers) < 2:
            raise ValueError(f"{self.id}: a multi_paper case reads at least two papers")
        if not self.papers and self.category != "acquisition":
            raise ValueError(f"{self.id}: only an acquisition case starts without a paper")
        if self.category == "follow_up" and not self.setup:
            raise ValueError(f"{self.id}: a follow_up case needs a setup turn")
        if self.report_key_points and not (
            self.mode != "auto" or MODE_TOOLS & set(self.expect_tools)
        ):
            raise ValueError(f"{self.id}: report_key_points need a scored turn that makes a report")
        clash = set(self.forbid_tools) & {*self.expect_tools, *self.expect_any_tools}
        if clash:
            raise ValueError(f"{self.id}: tools both expected and forbidden: {sorted(clash)}")
        return self


class Dataset(BaseModel):
    name: str
    version: str
    description: str = ""
    # How many times each case is run unless the command line says otherwise.
    # One run cannot tell a broken case from a flaky one.
    repeats: int = Field(default=1, ge=1)
    papers: dict[str, EvalPaper]
    cases: list[EvalCase]

    @model_validator(mode="after")
    def _check(self) -> Dataset:
        seen: set[str] = set()
        for case in self.cases:
            if case.id in seen:
                raise ValueError(f"duplicate case id: {case.id}")
            seen.add(case.id)
            for key in case.papers:
                if key not in self.papers:
                    raise ValueError(f"{case.id}: unknown paper {key!r}")
        return self

    def select(
        self, *, ids: list[str] | None = None, categories: list[str] | None = None, limit: int = 0
    ) -> list[EvalCase]:
        cases = [
            c for c in self.cases
            if (not ids or c.id in ids) and (not categories or c.category in categories)
        ]
        return cases[:limit] if limit else cases

    def unknown_tools(self, known: set[str]) -> dict[str, list[str]]:
        """Tool names a case expects or forbids that the agent does not have.

        A misspelt forbidden tool would pass every run, so names are checked
        against the agent's real tool list rather than trusted.
        """
        return {
            case.id: sorted(case.tool_names() - known)
            for case in self.cases
            if case.tool_names() - known
        }


def load_dataset(name_or_path: str) -> Dataset:
    """Load a dataset by file path, or by name from `evals/datasets/`."""
    path = Path(name_or_path)
    if not path.is_file():
        path = DATASET_DIR / f"{name_or_path}.json"
    if not path.is_file():
        raise FileNotFoundError(f"No such dataset: {name_or_path}")
    return Dataset.model_validate(json.loads(path.read_text(encoding="utf-8")))
