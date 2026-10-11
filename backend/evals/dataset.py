"""The golden set: papers, questions, and what a right answer must contain."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

DATASET_DIR = Path(__file__).resolve().parent / "datasets"

Category = Literal["method", "fact", "numeric", "equation", "reasoning", "multi_paper", "unanswerable"]

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


class EvalCase(BaseModel):
    id: str
    category: Category
    language: Literal["zh", "en"] = "zh"
    papers: list[str] = Field(min_length=1)
    question: str
    # False: the attached papers do not contain the answer, and the right
    # behaviour is to say so. Such a case has no key points.
    answerable: bool = True
    # Atomic facts a correct answer states. Each is judged on its own, so a
    # point should hold one checkable claim (one number, one mechanism).
    key_points: list[str] = Field(default_factory=list)
    # Why an unanswerable case is unanswerable — shown to the judge.
    note: str = ""

    @model_validator(mode="after")
    def _check(self) -> EvalCase:
        if self.answerable and not self.key_points:
            raise ValueError(f"{self.id}: an answerable case needs key_points")
        if not self.answerable and self.key_points:
            raise ValueError(f"{self.id}: an unanswerable case has no key_points")
        if (self.category == "unanswerable") != (not self.answerable):
            raise ValueError(f"{self.id}: category and answerable disagree")
        if self.category == "multi_paper" and len(self.papers) < 2:
            raise ValueError(f"{self.id}: a multi_paper case reads at least two papers")
        return self


class Dataset(BaseModel):
    name: str
    version: str
    description: str = ""
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


def load_dataset(name_or_path: str) -> Dataset:
    """Load a dataset by file path, or by name from `evals/datasets/`."""
    path = Path(name_or_path)
    if not path.is_file():
        path = DATASET_DIR / f"{name_or_path}.json"
    if not path.is_file():
        raise FileNotFoundError(f"No such dataset: {name_or_path}")
    return Dataset.model_validate(json.loads(path.read_text(encoding="utf-8")))
