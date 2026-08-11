"""Logic Lens structured digest models.

Logic Lens is deliberately a prose report: the point of a deep read-through is
the argument — why the method works, what the numbers actually demonstrate — and
no schema holds that. So the Lens report is *not* generated from a schema the way
Insight Snap is. The Markdown stays the primary output, and this digest is
extracted from it by a second pass.

That direction is the whole design. Generating a card view independently of the
prose would let the two drift, and a card that contradicts the paragraph next to
it is worse than no card at all. Extracting from the finished report means every
card is a pointer into text the reader can switch to and verify, which is exactly
what the structured/Markdown toggle is for.

Everything here is optional by construction: a failed or disabled digest pass
leaves ``available=False``, the run keeps its Markdown, and the UI renders what
it always did.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LensClaim(BaseModel):
    """A statement from the report, with the page that supports it."""

    text: str = ""
    page: int = 0


class LensSymbol(BaseModel):
    """One row of a formula's symbol table."""

    symbol: str = ""        # LaTeX body, no delimiters: "d_k", "\\alpha"
    meaning: str = ""


class LensFormula(BaseModel):
    """A key equation, kept as LaTeX so the card can typeset it.

    ``role`` is what makes this worth a card rather than a screenshot of the
    paper: the report's explanation of what the formula computes and why it is
    written that way.
    """

    name: str = ""
    latex: str = ""         # body only — no $, $$ or \\begin{equation}
    page: int = 0
    role: str = ""
    symbols: list[LensSymbol] = Field(default_factory=list)


class LensStage(BaseModel):
    """One module of the end-to-end pipeline, in data-flow order."""

    name: str = ""
    role: str = ""
    page: int = 0


class LensStep(BaseModel):
    """One step of the paper's algorithm, with the report's annotation."""

    step: str = ""
    note: str = ""


class LensAlgorithm(BaseModel):
    name: str = ""
    page: int = 0
    complexity: str = ""
    steps: list[LensStep] = Field(default_factory=list)


class LensDataset(BaseModel):
    """A dataset plus the metrics computed on it.

    ``measures`` carries the report's reading of the metric — what it actually
    measures and why it suits the task — because a bare metric name is the part
    a reader already knew.
    """

    name: str = ""
    metrics: str = ""
    measures: str = ""
    page: int = 0


class LensFinding(BaseModel):
    """One quantified result, in the same columns Insight Snap uses.

    Sharing the shape is deliberate: the frontend renders both with one table
    component, so a result reads identically whichever mode produced it.
    """

    metric: str = ""
    dataset: str = ""
    value: str = ""
    baseline: str = ""
    delta: str = ""
    page: int = 0
    note: str = ""


class LensReproducibility(BaseModel):
    """What the report says a reader would have to reconstruct themselves.

    ``score`` is 0-3 (nothing / partial / most / enough for the core result) so
    the card can render the same 0-3 meter as the Snap content review.
    """

    score: int = 0
    available: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)


class LensDigest(BaseModel):
    """The four-part Lens report as fields, extracted from its Markdown."""

    # Part 1 — Overview & motivation
    core_idea: str = ""
    problem: str = ""
    gap: str = ""
    contributions: list[LensClaim] = Field(default_factory=list)

    # Part 2 — Method deep-dive
    pipeline: list[LensStage] = Field(default_factory=list)
    formulas: list[LensFormula] = Field(default_factory=list)
    algorithm: LensAlgorithm | None = None

    # Part 3 — Experiments & results
    datasets: list[LensDataset] = Field(default_factory=list)
    setup: list[LensClaim] = Field(default_factory=list)
    findings: list[LensFinding] = Field(default_factory=list)
    takeaways: list[LensClaim] = Field(default_factory=list)

    # Part 4 — Critical assessment
    why_it_works: list[LensClaim] = Field(default_factory=list)
    limitations: list[LensClaim] = Field(default_factory=list)
    reproducibility: LensReproducibility = Field(default_factory=LensReproducibility)
    open_questions: list[LensClaim] = Field(default_factory=list)

    # False when the pass was disabled, failed, or returned nothing usable —
    # the frontend falls back to the Markdown report in all three cases.
    available: bool = False


class LensFigure(BaseModel):
    """An architecture figure the report may embed, for the structured view.

    Extracted deterministically from the PaperIR (same selection the prompt is
    handed), not from the model — a card must never show an invented image URL.
    """

    page: int = 0
    caption: str = ""
    url: str = ""
