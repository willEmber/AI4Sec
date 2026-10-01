"""Turn a known work into a local PDF the reading tools can open.

The DOI chain in `paper_downloader` was built for a batch of DOIs. An agent
does not start from a DOI: it starts from something it found — an arXiv id, an
open-access link a search returned, sometimes a DOI, sometimes only a title. So
this module orders the attempts by what is actually known about the work, and
ends at the same place every upload ends: a file at
`papers/{paper_id}/original.pdf`, a `papers` row, and a link from the work to
that file.

Two things it deliberately does not do:

*Invent identity.* `paper_id` stays the SHA-1 of the file's bytes, exactly as
for an upload. The same PDF fetched from arXiv and from a publisher is one
file, and a DOI never substitutes for file identity.

*Pretend.* When no route yields a PDF, the failure names the routes tried. The
agent's correct next move is to answer from the abstract and say so
(acceptance case A08), which it can only do if it is told the full text is not
available rather than that something broke.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from app.config import get_settings
from app.db import database as db
from app.services import paper_catalog
from app.services.paper_downloader import (
    DEFAULT_USER_AGENT,
    DownloaderCredentials,
    download_paper as download_by_doi,
    download_via_arxiv,
    download_via_direct_url,
)

logger = logging.getLogger("scholar.agent.acquisition")


@dataclass
class AcquisitionResult:
    ok: bool
    paper_id: str = ""
    source: str = ""
    detail: str = ""
    already_local: bool = False
    # Every route that was tried and what it said, so a failure is explainable
    # rather than just "unavailable".
    attempts: list[dict[str, str]] = field(default_factory=list)


def _credentials() -> DownloaderCredentials:
    settings = get_settings()
    return DownloaderCredentials(
        unpaywall_email=settings.unpaywall_email,
        core_api_key=settings.core_api_key,
        elsevier_api_key=settings.elsevier_api_key,
        elsevier_inst_token=settings.elsevier_inst_token,
        wiley_tdm_token=settings.wiley_tdm_token,
    )


async def _register_local_pdf(tmp_path: Path, *, literature_id: str, origin: str) -> str:
    """Give a freshly downloaded file the same identity an upload would get."""
    content = tmp_path.read_bytes()
    paper_id = hashlib.sha1(content).hexdigest()

    settings = get_settings()
    paper_dir = settings.data_dir / "papers" / paper_id
    paper_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = paper_dir / "original.pdf"
    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        tmp_path.unlink(missing_ok=True)   # same bytes already stored
    else:
        tmp_path.replace(pdf_path)

    existing = await db.fetch_one("SELECT paper_id FROM papers WHERE paper_id = ?", (paper_id,))
    if existing is None:
        item = await paper_catalog.get_literature_item(literature_id) or {}
        await db.execute(
            "INSERT INTO papers (paper_id, file_path, title, doi, venue, year) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                paper_id,
                f"papers/{paper_id}/original.pdf",
                item.get("title") or "",
                item.get("doi") or "",
                item.get("venue") or "",
                item.get("year") or 0,
            ),
        )

    await paper_catalog.link_paper_file(
        literature_id=literature_id, paper_id=paper_id, origin=origin
    )
    return paper_id


async def acquire_pdf(literature_id: str, *, timeout: float = 90.0) -> AcquisitionResult:
    """Get a PDF for a work, trying every route its identifiers allow.

    Order is by likelihood and cost: an arXiv id is one predictable request; an
    open-access URL a provider already handed us is one more; the DOI chain is
    several API calls and is tried last.
    """
    item = await paper_catalog.get_literature_item(literature_id)
    if item is None:
        return AcquisitionResult(ok=False, detail=f"No such work: {literature_id}")

    existing = await paper_catalog.primary_paper_for_literature(literature_id)
    if existing:
        pdf = get_settings().data_dir / "papers" / existing / "original.pdf"
        if pdf.exists() and pdf.stat().st_size > 0:
            return AcquisitionResult(
                ok=True, paper_id=existing, source="local", already_local=True
            )

    settings = get_settings()
    tmp_dir = settings.data_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # Downloaded under a scratch name: the real name is the hash of the bytes,
    # which is not known until they have all arrived.
    tmp_path = tmp_dir / f"acq_{uuid.uuid4().hex}.pdf"

    attempts: list[dict[str, str]] = []
    client = httpx.AsyncClient(headers={"User-Agent": DEFAULT_USER_AGENT}, timeout=timeout)
    try:
        arxiv_id = (item.get("arxiv_id") or "").strip()
        if arxiv_id:
            result = await download_via_arxiv(client, arxiv_id, tmp_path, timeout=timeout)
            if result and result.ok:
                paper_id = await _register_local_pdf(
                    tmp_path, literature_id=literature_id, origin="arxiv"
                )
                return AcquisitionResult(
                    ok=True, paper_id=paper_id, source="arxiv", attempts=attempts
                )
            if result:
                attempts.append({"source": "arxiv", "detail": result.detail or "failed"})

        oa_url = (item.get("oa_pdf_url") or "").strip()
        if oa_url:
            result = await download_via_direct_url(client, oa_url, tmp_path, timeout=timeout)
            if result and result.ok:
                paper_id = await _register_local_pdf(
                    tmp_path, literature_id=literature_id, origin="oa_url"
                )
                return AcquisitionResult(
                    ok=True, paper_id=paper_id, source="oa_url", attempts=attempts
                )
            if result:
                attempts.append({"source": "oa_url", "detail": result.detail or "failed"})

        doi = (item.get("doi") or "").strip()
        if doi:
            result = await download_by_doi(
                doi,
                tmp_path,
                credentials=_credentials(),
                title=item.get("title") or "",
                client=client,
                timeout=timeout,
            )
            if result.ok:
                paper_id = await _register_local_pdf(
                    tmp_path, literature_id=literature_id, origin=result.source
                )
                return AcquisitionResult(
                    ok=True, paper_id=paper_id, source=result.source, attempts=attempts
                )
            attempts.append({"source": result.source, "detail": result.detail or "failed"})
    finally:
        await client.aclose()
        tmp_path.unlink(missing_ok=True)

    if not attempts:
        return AcquisitionResult(
            ok=False,
            detail="This work has no DOI, arXiv id or open-access link, so there is "
            "nothing to download from.",
        )
    return AcquisitionResult(
        ok=False,
        detail="; ".join(f"{a['source']}: {a['detail']}" for a in attempts),
        attempts=attempts,
    )
