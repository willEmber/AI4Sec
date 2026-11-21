"""Structured contracts for multi-system publication venue rankings (P17)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

_CCF_ORDER = {"A": 3, "B": 2, "C": 1}
_CORE_ORDER = {"A*": 4, "A": 3, "B": 2, "C": 1}
_JCR_ORDER = {"Q1": 4, "Q2": 3, "Q3": 2, "Q4": 1}
_CCFCN_ORDER = {"T1": 3, "T2": 2, "T3": 1, "中文A": 3, "中文B": 2, "中文C": 1}


def _rank_sort_score(entry: "RankingEntry") -> tuple[int, int]:
    up = entry.rank.strip().upper()
    sys_name = entry.ranking_system
    if sys_name in {"CCF", "TH_CPL", "CAAI"}:
        return (_CCF_ORDER.get(up, 0), int(entry.top))
    if sys_name == "CORE":
        return (_CORE_ORDER.get(up, 0), int(entry.top))
    if sys_name == "JCR":
        return (_JCR_ORDER.get(up, 0), int(entry.top))
    if sys_name == "CCF_CN":
        return (_CCFCN_ORDER.get(up, 0), int(entry.top))
    if sys_name == "CAS":
        for d, sc in (("1区", 4), ("2区", 3), ("3区", 2), ("4区", 1)):
            if d in up:
                return (sc, int(entry.top))
    return (0, int(entry.top))

SOURCE_PRIORITIES: dict[str, int] = {
    "local_catalog": 4,
    "easyscholar": 3,
    "openalex": 2,
    "llm_websearch": 1,
    "none": 0,
    "": 0,
}


def source_priority(source: str, *, success: bool = True) -> int:
    if not success:
        return 0
    return SOURCE_PRIORITIES.get((source or "").strip().lower(), 1)


class RankingEntry(BaseModel):
    """One rating in one evaluation system — systems are never collapsed together."""

    ranking_system: str          # CCF | CORE | TH_CPL | CAAI | CCF_CN | JCR | CAS
    rank: str                    # A | B | C | A* | T1..T3 | Q1..Q4 | 1区..4区 ...
    category: str = ""           # subfield or discipline name
    edition_year: str = ""       # e.g. "2026", "2025", "2023"
    top: bool = False            # CAS Top flag
    impact_factor: str = ""      # JCR Impact Factor (preserved as string)

    def compact_label(self) -> str:
        label = self.rank
        if self.ranking_system == "CAS" and self.top and "top" not in label.lower():
            label = f"{label} Top"
        if self.ranking_system == "JCR" and self.impact_factor:
            label = f"{label} (IF {self.impact_factor})"
        elif self.edition_year:
            label = f"{label} ({self.edition_year})"
        return label


class VenueLookupInput(BaseModel):
    """Everything known about where a paper appeared when resolving its rank."""

    venue: str = ""
    venue_aliases: list[str] = Field(default_factory=list)
    issns: list[str] = Field(default_factory=list)
    dblp_key: str = ""
    openalex_source_id: str = ""
    crossref_event: str = ""
    openreview_status: str = ""
    venue_type_hint: str = ""
    is_preprint_source: bool = False


class VenueProfile(BaseModel):
    """Canonical venue identity, subtrack isolation, multi-system ranks, and red flags."""

    query_venue: str = ""
    canonical_name: str = ""
    acronym: str = ""
    full_name: str = ""
    venue_type: str = ""         # conference | journal | book_series | preprint | ""
    subtrack: str = "main"       # main | findings | workshop | companion | preprint
    peer_reviewed: bool | None = None
    is_preprint_file: bool = False
    openreview_tier: str = ""    # oral | spotlight | poster | ""
    issn_l: str = ""
    issns: list[str] = Field(default_factory=list)
    dblp_prefix: str = ""
    openalex_source_id: str = ""
    rankings: list[RankingEntry] = Field(default_factory=list)
    warning: str = ""            # CAS International Journal Early Warning flag
    note: str = ""               # e.g. open-review curated journal note (TMLR/COLM)
    metrics: dict[str, Any] = Field(default_factory=dict)
    source: str = ""             # local_catalog | easyscholar | openalex | llm_websearch | ""
    verification_status: str = "unknown"  # verified | unverified | unknown

    @property
    def display_venue(self) -> str:
        if self.subtrack == "findings":
            base = self.acronym or self.canonical_name or self.query_venue
            if base and "findings" not in base.lower():
                return f"Findings of {base}"
            return base
        if self.venue_type == "conference" and self.acronym:
            return self.acronym
        if self.acronym in {"TPAMI", "IJCV", "JMLR", "TMLR", "PVLDB", "PACMPL", "PACMMOD", "IMWUT", "TOG"}:
            return self.acronym
        return self.canonical_name or self.acronym or self.query_venue

    @property
    def sci_rank(self) -> str:
        best = ""
        best_score = 0
        for r in self.rankings:
            if r.ranking_system == "JCR":
                val = r.rank.strip().upper()
                if _JCR_ORDER.get(val, 0) > best_score:
                    best = val
                    best_score = _JCR_ORDER[val]
        return best

    @property
    def ccf_rank(self) -> str:
        best = ""
        best_score = 0
        for r in self.rankings:
            if r.ranking_system == "CCF":
                val = r.rank.strip().upper()
                if _CCF_ORDER.get(val, 0) > best_score:
                    best = val
                    best_score = _CCF_ORDER[val]
        return best

    @property
    def core_rank(self) -> str:
        best = ""
        best_score = 0
        for r in self.rankings:
            if r.ranking_system == "CORE":
                val = r.rank.strip().upper()
                if _CORE_ORDER.get(val, 0) > best_score:
                    best = val
                    best_score = _CORE_ORDER[val]
        return best

    @property
    def cas_rank(self) -> str:
        for r in self.rankings:
            if r.ranking_system == "CAS":
                return r.rank.strip()
        return ""

    @property
    def has_any_rank(self) -> bool:
        return bool(self.rankings or self.warning or self.note)

    def compact_ranks(self) -> dict[str, str]:
        """System -> concise formatted rank string (first/highest entry per system)."""
        out: dict[str, str] = {}
        # Sort so highest CCF/CORE/JCR/CCF_CN/CAS tier wins when a venue spans multiple categories.
        ordered = sorted(self.rankings, key=_rank_sort_score, reverse=True)
        for entry in ordered:
            if entry.ranking_system not in out:
                out[entry.ranking_system] = entry.compact_label()
        return out

    def agent_summary(self) -> dict[str, Any]:
        """Compact dict embedded on candidate papers returned by discovery tools."""
        out: dict[str, Any] = {}
        venue_label = self.canonical_name or self.query_venue
        if venue_label:
            out["canonical_venue"] = venue_label
        if self.acronym and self.acronym != venue_label:
            out["acronym"] = self.acronym
        if self.venue_type:
            out["venue_type"] = self.venue_type
        if self.subtrack and self.subtrack != "main":
            out["subtrack"] = self.subtrack
        if self.peer_reviewed is not None:
            out["peer_reviewed"] = self.peer_reviewed
        if self.openreview_tier:
            out["presentation_tier"] = self.openreview_tier
        if self.is_preprint_file:
            out["preprint_file_only"] = True
        ranks = self.compact_ranks()
        if ranks:
            out["ranks"] = ranks
        if self.warning:
            out["cas_warning"] = self.warning
        if self.note:
            out["note"] = self.note
        if self.metrics:
            out["metrics"] = dict(self.metrics)
        if ranks or self.warning:
            out["source"] = self.source
            out["verification_status"] = self.verification_status
        return out

    def prompt_badge(self, language: str = "zh") -> str:
        """Short inline label for `_format_papers` in the reading system prompt."""
        zh = language == "zh"
        if self.subtrack == "preprint" and not self.rankings:
            return "arXiv 预印本" if zh else "arXiv preprint"
        label = self.display_venue or self.canonical_name or self.query_venue
        if not label:
            return ""
        bits: list[str] = []
        if self.ccf_rank:
            bits.append(f"CCF {self.ccf_rank}")
        elif self.core_rank in {"A*", "A", "B", "C"}:
            bits.append(f"CORE {self.core_rank}")
        for r in self.rankings:
            if r.ranking_system == "CCF_CN" and not self.ccf_rank:
                bits.append(f"CCF中文 {r.rank}" if zh else f"CCF-CN {r.rank}")
                break
        if self.sci_rank:
            bits.append(f"JCR {self.sci_rank}")
        if self.cas_rank and not self.ccf_rank:
            short_cas = self.cas_rank
            if r_top := any(r.ranking_system == "CAS" and r.top for r in self.rankings):
                if "top" not in short_cas.lower():
                    short_cas = f"{short_cas} Top"
                _ = r_top
            bits.append(f"中科院{short_cas}" if zh else f"CAS {short_cas}")
        if self.openreview_tier:
            bits.append(self.openreview_tier.capitalize())
        if self.subtrack in {"workshop", "companion"} and "workshop" not in label.lower():
            bits.append("Workshop" if self.subtrack == "workshop" else "Companion")
        badge = f"{label} · {' / '.join(bits)}" if bits else label
        if self.warning:
            badge += f" [{'预警:' + self.warning if zh else 'CAS Warning: ' + self.warning}]"
        if self.is_preprint_file and self.subtrack != "preprint":
            badge += " [预印本文件]" if zh else " [preprint PDF]"
        return badge

    def verified_fact_clauses(self) -> list[str]:
        """Citable clauses for `metadata` evidence snapshots (verified facts only)."""
        clauses: list[str] = []
        venue_name = self.display_venue or self.canonical_name
        if self.subtrack == "findings" and venue_name:
            clauses.append(f"venue: {venue_name} (peer-reviewed Findings track, distinct from main conference)")
        elif self.subtrack in {"workshop", "companion"} and venue_name:
            clauses.append(f"venue: {venue_name} ({self.subtrack} track; main-venue tier does not apply)")
        elif self.subtrack == "preprint":
            clauses.append("preprint (not peer-reviewed)")

        if self.verification_status == "verified" and self.rankings:
            rank_parts = []
            for sys_name, label in self.compact_ranks().items():
                rank_parts.append(f"{sys_name} {label}")
            if rank_parts:
                src_label = "venue catalog" if self.source == "local_catalog" else self.source
                prefix = f"{venue_name}: " if venue_name and self.subtrack == "main" else ""
                clauses.append(f"{prefix}{', '.join(rank_parts)} (source: {src_label})")

        if self.warning:
            clauses.append(f"CAS International Journal Early Warning: {self.warning}")
        if self.note and self.verification_status == "verified":
            clauses.append(self.note)
        return clauses
