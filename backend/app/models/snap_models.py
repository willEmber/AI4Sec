"""Insight Snap data models.

``TriageSignals`` is the evidence half of a triage verdict: everything about a
paper that can be established from *outside* its own prose — where it was
published and how that venue ranks, how often it has actually been cited,
whether the authors released code, whether it is open access, and whether it has
been retracted. The report renders these with their sources so a reader can
check the verdict instead of taking the model's word for it.

Keeping them in a model (rather than interpolating strings into a prompt) is
what lets the content review run *without* seeing them — see
``snap_subgraph.review_content`` for why that separation matters.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RepoHost(str, Enum):
    GITHUB = "github"
    GITLAB = "gitlab"
    BITBUCKET = "bitbucket"
    HUGGINGFACE = "huggingface"
    ZENODO = "zenodo"
    OSF = "osf"
    CODEOCEAN = "codeocean"
    PROJECT_PAGE = "project_page"


class CodeRepo(BaseModel):
    """A code/artifact link found in the paper.

    ``is_official`` is a heuristic (see ``triage_signals.extract_code_repos``):
    a repository named in the abstract, a first-page footnote, or a sentence
    with an ownership cue ("our code", "we release") is treated as the authors'
    own. Links harvested from Related Work or the bibliography stay
    ``is_official=False`` — they are somebody else's baseline, and counting them
    as released code would be the exact kind of unfounded signal this module
    exists to eliminate.
    """

    url: str = ""
    host: RepoHost = RepoHost.GITHUB
    slug: str = ""                # owner/name, when the host has that shape
    is_official: bool = False
    evidence_page: int = 0        # 1-based page the link was found on
    evidence_quote: str = ""      # the sentence that carried it

    # Populated only when SNAP_PROBE_REPOS is on (GitHub only).
    stars: int = 0
    last_push: str = ""           # ISO date
    archived: bool = False
    probe_ok: bool = False        # the probe ran and the repo exists


class CitationIntents(BaseModel):
    """How the citing literature uses this paper, per Semantic Scholar.

    S2 labels each citation edge with an *intent* — ``background``,
    ``methodology`` or ``result`` — which is a different (and narrower) thing
    than citation sentiment: it says which part of the citing paper leaned on
    this work, not whether it agreed. A paper cited 200 times as background and
    twice as methodology is a landmark nobody builds on; the reverse is a tool.
    That distinction is what makes this worth a request.

    ``sampled`` records how many edges were actually inspected, because the
    endpoint is paginated and these counts are a sample, not a census.
    """

    background: int = 0
    methodology: int = 0
    result: int = 0
    influential: int = 0
    sampled: int = 0

    @property
    def available(self) -> bool:
        return self.sampled > 0


class TriageSignals(BaseModel):
    """External, checkable evidence about a paper. No LLM judgement here."""

    # --- identity ---
    doi: str = ""
    arxiv_id: str = ""
    openalex_id: str = ""
    s2_paper_id: str = ""
    resolved: bool = False        # at least one external record was matched

    # --- publication venue ---
    venue: str = ""
    venue_normalized: str = ""    # conference proceedings → canonical acronym
    year: int = 0
    sci_rank: str = ""            # Q1..Q4 (EasyScholar)
    ccf_rank: str = ""            # A/B/C  (EasyScholar)
    is_preprint: bool = False
    is_survey: bool = False
    publication_types: list[str] = Field(default_factory=list)

    # --- impact ---
    cited_by_count: int = 0
    citations_per_year: float = 0.0
    influential_citation_count: int = 0
    reference_count: int = 0
    intents: CitationIntents = Field(default_factory=CitationIntents)

    # --- availability ---
    repos: list[CodeRepo] = Field(default_factory=list)
    is_open_access: bool = False
    oa_url: str = ""

    # --- red flags ---
    is_retracted: bool = False

    # --- derived scores, all in [0,1] ---
    venue_score: float = 0.0
    citation_score: float = 0.0
    code_score: float = 0.0
    external_score: float = 0.0

    # signal name → human-readable source ("OpenAlex", "EasyScholar", "p.1 footnote")
    provenance: dict[str, str] = Field(default_factory=dict)
    # Lookups that were attempted and failed, so the report can say "unknown"
    # rather than implying a zero.
    unavailable: list[str] = Field(default_factory=list)

    @property
    def official_repos(self) -> list[CodeRepo]:
        return [r for r in self.repos if r.is_official]

    @property
    def has_official_code(self) -> bool:
        return bool(self.official_repos)

    @property
    def citations_known(self) -> bool:
        return "citations" not in self.unavailable


# ──────────────────────────────────────────────────────────────────────
# The report itself, as fields rather than prose
# ──────────────────────────────────────────────────────────────────────

class SnapFinding(BaseModel):
    """One quantified experimental result.

    Free prose lets a model write "significantly outperforms baselines"; these
    columns do not. The model must name the metric, the dataset and the number,
    which is what makes findings comparable across papers (the comparison matrix
    is just these rows from several runs, stacked).
    """

    metric: str = ""        # "BLEU", "top-1 accuracy", "F1"
    dataset: str = ""       # "WMT14 EN-DE"
    value: str = ""         # "28.4" — a string: papers report ranges and ± too
    baseline: str = ""      # "26.3 (best prior)"
    delta: str = ""         # "+2.1"
    page: int = 0
    note: str = ""          # caveats: "single seed", "subset of the test split"


class SnapClaim(BaseModel):
    """A contribution or limitation, with the page that supports it."""

    text: str = ""
    page: int = 0


class SnapReport(BaseModel):
    """The model-authored half of an Insight Snap, as fields."""

    one_liner: str = ""
    contributions: list[SnapClaim] = Field(default_factory=list)
    findings: list[SnapFinding] = Field(default_factory=list)
    suitable_for: str = ""
    limitations: list[SnapClaim] = Field(default_factory=list)
    # True when the JSON pass failed and the raw text was kept as markdown.
    degraded: bool = False
    raw_markdown: str = ""
