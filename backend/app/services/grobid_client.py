"""GROBID fallback for header metadata (DOI, title, authors, year, venue).

The primary path scans the first few pages of the MinerU text for a DOI or arXiv
id with a regex (``main_graph._extract_doi_from_ir``). That works on a
well-typeset preprint and fails on plenty of real PDFs: scans, two-column layouts
whose reading order interleaves the header, journals that print the DOI only in a
running footer MinerU drops, and any paper where the identifier is split across
blocks. When the DOI is missed everything downstream degrades to title matching,
which is the least reliable way to resolve a paper.

GROBID's ``processHeaderDocument`` parses the header as a structure instead of
searching for a pattern, so it recovers these cases. It is optional: without
``GROBID_URL`` configured, ``extract_header`` reports itself unavailable and the
pipeline keeps its existing behaviour unchanged.

Only the header is requested (not the full text) — it is the cheap, fast endpoint
and the only part this pipeline is missing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

import httpx

from app.config import get_settings

logger = logging.getLogger("scholar.grobid")

_TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
_DOI_CLEAN_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/)?", re.IGNORECASE)


@dataclass
class GrobidHeader:
    """Parsed header fields. ``available`` is False when GROBID was not used."""

    available: bool = False
    title: str = ""
    doi: str = ""
    arxiv_id: str = ""
    year: int = 0
    venue: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""

    @property
    def authors_str(self) -> str:
        return ", ".join(self.authors)


def _text(node: ElementTree.Element | None) -> str:
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())


def _parse_tei_header(xml: str) -> GrobidHeader:
    """Pull the fields we need out of a TEI header document."""
    header = GrobidHeader(available=True)
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        logger.warning("grobid: TEI parse failed: %s", exc)
        return GrobidHeader()

    header.title = _text(root.find(".//tei:titleStmt/tei:title", _TEI_NS))

    for idno in root.findall(".//tei:idno", _TEI_NS):
        kind = (idno.get("type") or "").lower()
        value = _text(idno)
        if not value:
            continue
        if kind == "doi" and not header.doi:
            header.doi = _DOI_CLEAN_RE.sub("", value).strip().lower()
        elif kind in ("arxiv", "arxivid") and not header.arxiv_id:
            header.arxiv_id = value.replace("arXiv:", "").strip()

    for date in root.findall(".//tei:publicationStmt/tei:date", _TEI_NS) + root.findall(
        ".//tei:monogr//tei:date", _TEI_NS
    ):
        when = date.get("when") or _text(date)
        match = re.search(r"(19|20)\d{2}", when)
        if match:
            header.year = int(match.group(0))
            break

    # Journal or proceedings title lives in monogr/title; prefer the full form.
    for title in root.findall(".//tei:monogr/tei:title", _TEI_NS):
        level = (title.get("level") or "").lower()
        value = _text(title)
        if value and level in ("j", "m"):
            header.venue = value
            break

    for author in root.findall(".//tei:sourceDesc//tei:author", _TEI_NS):
        forename = " ".join(_text(f) for f in author.findall("./tei:persName/tei:forename", _TEI_NS))
        surname = _text(author.find("./tei:persName/tei:surname", _TEI_NS))
        name = " ".join(part for part in (forename.strip(), surname) if part).strip()
        if name and name not in header.authors:
            header.authors.append(name)

    header.abstract = _text(root.find(".//tei:profileDesc//tei:abstract", _TEI_NS))
    return header


async def extract_header(pdf_path: str | Path) -> GrobidHeader:
    """Run ``processHeaderDocument`` on a PDF. Never raises.

    Returns an unavailable header when GROBID is not configured, the file is
    missing, or the request fails — callers keep whatever the regex scan found.
    """
    settings = get_settings()
    base = settings.grobid_url.strip().rstrip("/")
    if not base:
        return GrobidHeader()

    path = Path(pdf_path)
    if not path.exists():
        logger.debug("grobid: pdf missing at %s", path)
        return GrobidHeader()

    url = f"{base}/api/processHeaderDocument"
    try:
        with path.open("rb") as handle:
            files = {"input": (path.name, handle, "application/pdf")}
            async with httpx.AsyncClient(timeout=settings.grobid_timeout_seconds) as client:
                resp = await client.post(
                    url,
                    files=files,
                    data={"consolidateHeader": "1"},
                    headers={"Accept": "application/xml"},
                )
        if resp.status_code != 200:
            logger.warning("grobid: %s -> HTTP %s", url, resp.status_code)
            return GrobidHeader()
    except Exception as exc:
        logger.warning("grobid: request to %s failed: %s", url, exc)
        return GrobidHeader()

    header = _parse_tei_header(resp.text)
    if header.available:
        logger.info(
            "grobid: header — doi=%s arxiv=%s year=%s venue=%s authors=%d",
            header.doi or "-", header.arxiv_id or "-", header.year or "-",
            header.venue or "-", len(header.authors),
        )
    return header
