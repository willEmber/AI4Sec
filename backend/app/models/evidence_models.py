"""Evidence contract: what an answer is allowed to cite, and how it resolves.

Two rules shape everything here.

*Immutability.* An evidence row is never updated. Its id is derived from the
content it records, so a changed quote is a different id and an answer given
last week still resolves to the text that was actually read (acceptance case
A17).

*Stable location.* `blocks.block_id` is an autoincrement key that
`build_and_store_paper_ir` deletes and reinserts on every re-parse, so it
cannot anchor a long-lived citation. A locator pins the parse version plus a
position that is stable inside it (`order_idx`, section path, page), and the
parse version is carried on the evidence row.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SourceLevel(str, Enum):
    """How direct the evidence is. Answers must state this when it is weak."""

    FULLTEXT = "fulltext"          # read out of a parsed PDF
    ABSTRACT = "abstract"          # abstract only — full text unavailable
    METADATA = "metadata"          # bibliographic record (venue, year, counts)
    EXTERNAL_WEB = "external_web"  # a web page or API response


class Locator(BaseModel):
    """Position of a quote inside one parse version, or inside an external source.

    Page numbers are 0-based here, matching `blocks.page_idx` and MinerU's own
    indexing. The 1-based value shown to a reader is produced by `page_label`
    and by `to_api()` — the conversion lives in this one place so no caller has
    to remember which convention it holds (development plan §5 rule 4).
    """

    section_id: str = ""
    section_path: str = ""
    page_index: int | None = None        # 0-based, inclusive
    page_end_index: int | None = None    # 0-based, inclusive
    # Position within the parse version's block order. Stable across restarts
    # for a given version, unlike the autoincrement `blocks.block_id`.
    block_start: int | None = None
    block_end: int | None = None
    node_id: str = ""                    # paper_nodes id, when the hit came from one
    bbox: list[float] = Field(default_factory=list)   # [x0, y0, x1, y1]

    @property
    def page_label(self) -> int | None:
        """1-based page for display; `None` when the source has no pages."""
        return None if self.page_index is None else self.page_index + 1

    @property
    def page_end_label(self) -> int | None:
        return None if self.page_end_index is None else self.page_end_index + 1

    def canonical(self) -> str:
        """Stable string form, used when deriving an evidence id."""
        return json.dumps(
            self.model_dump(exclude_defaults=False), sort_keys=True, ensure_ascii=False
        )

    def to_api(self) -> dict[str, Any]:
        """Serialization for clients: adds the 1-based page fields."""
        data = self.model_dump()
        data["page_label"] = self.page_label
        data["page_end_label"] = self.page_end_label
        return data


def content_hash(text: str) -> str:
    """Hash of a quote, used to detect that a source changed under a citation."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def compute_evidence_id(
    *,
    source_level: SourceLevel | str,
    quote: str,
    literature_id: str = "",
    paper_id: str = "",
    parse_version: str = "",
    locator: Locator | None = None,
    source_url: str = "",
    scope: str = "",
) -> str:
    """Derive the evidence id from what the evidence *is*.

    Deterministic so that reading the same passage twice yields one row rather
    than two, and so that a different quote can never reuse an id an answer has
    already cited.

    `scope`, when given, makes the id private to one owner. Web evidence needs
    it: two readers running the same web search get the same snippet, and an
    unscoped id would make the second reader's citation resolve to a row the
    first reader owns — which the ownership check then refuses. Empty leaves
    the id exactly as it has always been computed.
    """
    level = source_level.value if isinstance(source_level, SourceLevel) else str(source_level)
    material = "\x1f".join(
        [
            level,
            literature_id,
            paper_id,
            parse_version,
            source_url,
            (locator or Locator()).canonical(),
            content_hash(quote),
        ]
        + ([scope] if scope else [])
    )
    return "ev_" + hashlib.sha1(material.encode("utf-8")).hexdigest()[:20]


class Evidence(BaseModel):
    """One immutable, citable excerpt."""

    evidence_id: str
    owner_id: str = ""
    session_id: str = ""
    literature_id: str = ""
    paper_id: str = ""            # empty for metadata / web evidence
    parse_version: str = ""       # empty for non-PDF sources
    source_level: SourceLevel
    locator: Locator = Field(default_factory=Locator)
    quote: str = ""
    content_hash: str = ""
    source_url: str = ""
    provider: str = ""
    retrieved_at: str = ""

    def to_api(self) -> dict[str, Any]:
        data = self.model_dump(exclude={"locator", "owner_id"})
        data["source_level"] = self.source_level.value
        data["locator"] = self.locator.to_api()
        return data


class CitationCheck(BaseModel):
    """Result of validating one citation in an answer.

    Only mechanical checks live here: that the id exists, that the caller may
    read it, and that the stored quote still matches its source. Whether the
    evidence actually *supports* the sentence is a question for evaluation, not
    for this validator (development plan §5 rule 6).
    """

    evidence_id: str
    exists: bool = False
    authorized: bool = False
    quote_matches: bool = False
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.exists and self.authorized and self.quote_matches
