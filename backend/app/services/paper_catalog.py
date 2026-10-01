"""Three separate identities for a paper, and the mapping between them.

The platform already has one identity — `papers.paper_id`, the SHA-1 of an
uploaded PDF. That is a *file* identity, and it is the wrong thing to hang a
research task on:

* a candidate found by search may never be downloaded, so it has no file;
* one work can have several files (preprint, version of record);
* a DOI identifies the work, not any particular file.

So a work gets a `literature_id`, a file keeps its `paper_id`, and
`literature_files` joins them. External identifiers (DOI, arXiv, OpenAlex, S2)
stay as attributes of the work — none of them is promoted to the primary key,
because plenty of real papers have none of them.

A parse of a file gets a third identity, `paper_versions.version_id`, which is
what evidence points at so a re-parse cannot invalidate an old citation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from typing import Any

from app.db import database as db
from app.services.paper_search.utils import normalize_doi, normalize_whitespace, title_fingerprint

logger = logging.getLogger("scholar.paper_catalog")

_ARXIV_PREFIX_RE = re.compile(
    r"^(?:https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/|arxiv:)", re.IGNORECASE
)
_ARXIV_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)


def normalize_arxiv_id(value: str) -> str:
    """Strip URL prefixes, the `arXiv:` label and the version suffix.

    Versions are dropped so `2103.12345v1` and `2103.12345v3` resolve to one
    work; if the two files differ they are still two `paper_id`s underneath.
    """
    raw = (value or "").strip()
    raw = _ARXIV_PREFIX_RE.sub("", raw)
    raw = raw.removesuffix(".pdf").strip()
    raw = _ARXIV_VERSION_RE.sub("", raw)
    return raw.lower()


def derive_literature_id(
    *,
    doi: str = "",
    arxiv_id: str = "",
    openalex_id: str = "",
    s2_paper_id: str = "",
    title: str = "",
    fallback: str = "",
) -> str:
    """Derive a stable id from the strongest identifier available.

    Preference order matches how reliable the identifier is in practice. Two
    records that share a DOI get one id without any fuzzy matching; records with
    only a title fall back to a fingerprint, which is weaker but still stable.
    """
    doi = normalize_doi(doi)
    if doi:
        return "lit_doi_" + hashlib.sha1(doi.encode("utf-8")).hexdigest()[:20]
    arxiv = normalize_arxiv_id(arxiv_id)
    if arxiv:
        return "lit_arx_" + hashlib.sha1(arxiv.encode("utf-8")).hexdigest()[:20]
    if openalex_id.strip():
        return "lit_oa_" + hashlib.sha1(openalex_id.strip().lower().encode("utf-8")).hexdigest()[:20]
    if s2_paper_id.strip():
        return "lit_s2_" + hashlib.sha1(s2_paper_id.strip().lower().encode("utf-8")).hexdigest()[:20]
    if normalize_whitespace(title):
        return "lit_t_" + title_fingerprint(title)[:20]
    if fallback:
        return "lit_f_" + hashlib.sha1(fallback.encode("utf-8")).hexdigest()[:20]
    return "lit_x_" + uuid.uuid4().hex[:20]


async def find_literature(
    *, doi: str = "", arxiv_id: str = "", title: str = ""
) -> dict[str, Any] | None:
    """Look up an existing work by identifier, then by title fingerprint."""
    doi = normalize_doi(doi)
    if doi:
        row = await db.fetch_one("SELECT * FROM literature_items WHERE doi = ?", (doi,))
        if row:
            return row
    arxiv = normalize_arxiv_id(arxiv_id)
    if arxiv:
        row = await db.fetch_one("SELECT * FROM literature_items WHERE arxiv_id = ?", (arxiv,))
        if row:
            return row
    if normalize_whitespace(title):
        row = await db.fetch_one(
            "SELECT * FROM literature_items WHERE title_fingerprint = ?",
            (title_fingerprint(title),),
        )
        if row:
            return row
    return None


async def upsert_literature_item(
    *,
    title: str = "",
    doi: str = "",
    arxiv_id: str = "",
    openalex_id: str = "",
    s2_paper_id: str = "",
    pmid: str = "",
    authors: list[str] | None = None,
    year: int | None = None,
    venue: str = "",
    abstract: str = "",
    url: str = "",
    oa_pdf_url: str = "",
    source: str = "",
    fallback_key: str = "",
) -> str:
    """Create or enrich a work record. Returns its `literature_id`.

    Enriching never blanks a field: a later record that omits the venue must not
    erase a venue an earlier one supplied. `year` is only written together with
    `year_known`, so a missing year stays distinguishable from year zero
    (acceptance case A04).
    """
    existing = await find_literature(doi=doi, arxiv_id=arxiv_id, title=title)
    literature_id = (
        existing["literature_id"]
        if existing
        else derive_literature_id(
            doi=doi,
            arxiv_id=arxiv_id,
            openalex_id=openalex_id,
            s2_paper_id=s2_paper_id,
            title=title,
            fallback=fallback_key,
        )
    )

    year_known = year is not None and year > 0
    values = {
        "doi": normalize_doi(doi),
        "arxiv_id": normalize_arxiv_id(arxiv_id),
        "openalex_id": openalex_id.strip(),
        "s2_paper_id": s2_paper_id.strip(),
        "pmid": pmid.strip(),
        "title": normalize_whitespace(title),
        "title_fingerprint": title_fingerprint(title) if normalize_whitespace(title) else "",
        "authors_json": json.dumps(authors or [], ensure_ascii=False) if authors else "",
        "venue": normalize_whitespace(venue),
        "abstract": (abstract or "").strip(),
        "url": url.strip(),
        "oa_pdf_url": oa_pdf_url.strip(),
        "source": source.strip(),
    }

    if existing is None:
        await db.execute(
            """INSERT INTO literature_items
                   (literature_id, doi, arxiv_id, openalex_id, s2_paper_id, pmid, title,
                    title_fingerprint, authors_json, year, year_known, venue, abstract,
                    url, oa_pdf_url, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                literature_id,
                values["doi"],
                values["arxiv_id"],
                values["openalex_id"],
                values["s2_paper_id"],
                values["pmid"],
                values["title"],
                values["title_fingerprint"],
                values["authors_json"] or "[]",
                year if year_known else 0,
                1 if year_known else 0,
                values["venue"],
                values["abstract"],
                values["url"],
                values["oa_pdf_url"],
                values["source"],
            ),
        )
        return literature_id

    # Fill blanks only.
    assignments: list[str] = []
    params: list[Any] = []
    for column, value in values.items():
        if value and not (existing.get(column) or ""):
            assignments.append(f"{column} = ?")
            params.append(value)
    if year_known and not existing.get("year_known"):
        assignments.extend(["year = ?", "year_known = 1"])
        params.append(year)
    if assignments:
        assignments.append("updated_at = now()")
        params.append(literature_id)
        await db.execute(
            f"UPDATE literature_items SET {', '.join(assignments)} WHERE literature_id = ?",
            tuple(params),
        )
    return literature_id


async def get_literature_item(literature_id: str) -> dict[str, Any] | None:
    return await db.fetch_one(
        "SELECT * FROM literature_items WHERE literature_id = ?", (literature_id,)
    )


async def link_paper_file(
    *, literature_id: str, paper_id: str, origin: str = "", is_primary: bool = True
) -> None:
    """Record that a local PDF is a copy of a work.

    Marking a file primary demotes the previous primary, so a session always has
    one obvious file to open even when a work has several.
    """
    if is_primary:
        await db.execute(
            "UPDATE literature_files SET is_primary = 0 WHERE literature_id = ?",
            (literature_id,),
        )
    await db.execute(
        """INSERT INTO literature_files (literature_id, paper_id, origin, is_primary)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(literature_id, paper_id) DO UPDATE SET
               origin = CASE WHEN excluded.origin <> '' THEN excluded.origin
                             ELSE literature_files.origin END,
               is_primary = excluded.is_primary""",
        (literature_id, paper_id, origin, 1 if is_primary else 0),
    )


async def literature_for_paper(paper_id: str) -> str:
    """The work a local file belongs to, or `""` if it is not linked yet."""
    row = await db.fetch_one(
        "SELECT literature_id FROM literature_files WHERE paper_id = ? "
        "ORDER BY is_primary DESC LIMIT 1",
        (paper_id,),
    )
    return row["literature_id"] if row else ""


async def primary_paper_for_literature(literature_id: str) -> str:
    row = await db.fetch_one(
        "SELECT paper_id FROM literature_files WHERE literature_id = ? "
        "ORDER BY is_primary DESC, created_at LIMIT 1",
        (literature_id,),
    )
    return row["paper_id"] if row else ""


async def ensure_literature_for_local_paper(paper_id: str, *, origin: str = "upload") -> str:
    """Give an uploaded PDF a work record, reusing one if its metadata matches.

    An upload frequently has no DOI at the point it arrives; the paper row is
    then the only thing known about it, so `paper_id` seeds the fallback key.
    Once a DOI is extracted later, `upsert_literature_item` merges the record.
    """
    existing = await literature_for_paper(paper_id)
    if existing:
        return existing

    paper = await db.fetch_one("SELECT * FROM papers WHERE paper_id = ?", (paper_id,))
    if paper is None:
        raise ValueError(f"No such paper: {paper_id}")

    literature_id = await upsert_literature_item(
        title=paper["title"] or "",
        doi=paper["doi"] or "",
        venue=paper["venue"] or "",
        year=paper["year"] or None,
        source=origin,
        fallback_key=paper_id,
    )
    await link_paper_file(literature_id=literature_id, paper_id=paper_id, origin=origin)
    return literature_id


# ── Parse versions ──────────────────────────────────────────────────────────


def compute_parse_content_hash(payload: str | bytes) -> str:
    """Hash of a parse's raw content, used to recognise an identical re-parse."""
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    return hashlib.sha256(data).hexdigest()


async def register_paper_version(
    *,
    paper_id: str,
    parse_id: str = "",
    parser: str = "mineru",
    parser_config: dict[str, Any] | None = None,
    output_dir: str = "",
    content_hash: str = "",
    block_count: int = 0,
    page_count: int = 0,
) -> str:
    """Record a parse of a paper and make it current. Returns the `version_id`.

    An identical re-parse — same parser, same configuration, same content hash —
    reuses the existing version rather than creating a duplicate, so evidence
    gathered before it stays attached to a live version.
    """
    config_json = json.dumps(parser_config or {}, sort_keys=True, ensure_ascii=False)
    if content_hash:
        twin = await db.fetch_one(
            """SELECT version_id FROM paper_versions
                WHERE paper_id = ? AND parser = ? AND parser_config = ? AND content_hash = ?
                ORDER BY created_at DESC LIMIT 1""",
            (paper_id, parser, config_json, content_hash),
        )
        if twin is not None:
            await _make_current(paper_id, twin["version_id"])
            return twin["version_id"]

    version_id = f"pv_{uuid.uuid4().hex[:24]}"
    # Demote first: the partial unique index allows only one current version.
    await db.execute(
        "UPDATE paper_versions SET is_current = 0, status = 'superseded' "
        "WHERE paper_id = ? AND is_current = 1",
        (paper_id,),
    )
    await db.execute(
        """INSERT INTO paper_versions
               (version_id, paper_id, parse_id, parser, parser_config, output_dir,
                content_hash, block_count, page_count, status, is_current)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ready', 1)""",
        (
            version_id,
            paper_id,
            parse_id,
            parser,
            config_json,
            output_dir,
            content_hash,
            block_count,
            page_count,
        ),
    )
    logger.info("Registered parse version %s for paper %s", version_id, paper_id)
    return version_id


async def _make_current(paper_id: str, version_id: str) -> None:
    await db.execute(
        "UPDATE paper_versions SET is_current = 0, status = 'superseded' "
        "WHERE paper_id = ? AND is_current = 1 AND version_id <> ?",
        (paper_id, version_id),
    )
    await db.execute(
        "UPDATE paper_versions SET is_current = 1, status = 'ready' WHERE version_id = ?",
        (version_id,),
    )


async def get_current_version(paper_id: str) -> dict[str, Any] | None:
    return await db.fetch_one(
        "SELECT * FROM paper_versions WHERE paper_id = ? AND is_current = 1", (paper_id,)
    )


async def get_version(version_id: str) -> dict[str, Any] | None:
    return await db.fetch_one(
        "SELECT * FROM paper_versions WHERE version_id = ?", (version_id,)
    )


async def list_versions(paper_id: str) -> list[dict[str, Any]]:
    return await db.fetch_all(
        "SELECT * FROM paper_versions WHERE paper_id = ? ORDER BY created_at DESC",
        (paper_id,),
    )
