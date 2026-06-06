"""Build a Zotero RDF import bundle (item + report note + PDF) for a run.

Zotero has no native Markdown import and its local desktop Web API is currently
read-only, so the most robust *offline, no-cloud* way to get an item together
with the AI report (as a note) and the original PDF into a user's local Zotero
is a "Zotero RDF + files" bundle: a ``.rdf`` describing the item, plus the PDF
under a ``files/`` folder that the ``.rdf`` references. The user runs
File → Import on the ``.rdf`` and gets the item, a child note, and the stored
PDF — all locally.

The RDF shape here mirrors Zotero's own ``Zotero RDF`` export translator exactly
(``bib:Article`` + ``bib:Journal`` container, ``foaf:Person`` authors,
``bib:Memo`` child note linked via ``dcterms:isReferencedBy``, ``z:Attachment``
with a relative ``z:path``) so it round-trips through Zotero's importer.
"""

from __future__ import annotations

import html
import io
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

import httpx

# --- Markdown -> HTML, with a safe fallback when python-markdown is absent ---
try:  # pragma: no cover - exercised by import availability, not unit tests
    import markdown as _markdown

    def _md_to_html(md: str) -> str:
        return _markdown.markdown(
            md,
            extensions=["extra", "sane_lists"],
            output_format="html5",
        )
except Exception:  # pragma: no cover - dependency missing

    def _md_to_html(md: str) -> str:
        # Readable, lossless fallback: preserve structure as preformatted text.
        return "<pre>" + html.escape(md) + "</pre>"


# Zotero RDF namespaces (must match the Zotero RDF export translator).
_RDF_NS: dict[str, str] = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "z": "http://www.zotero.org/namespaces/export#",
    "bib": "http://purl.org/net/biblio#",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "link": "http://purl.org/rss/1.0/modules/link/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
    "prism": "http://prismstandard.org/namespaces/1.2/basic/",
    "vcard": "http://nwalsh.com/rdf/vCard#",
}

# Legacy citation badge spans the renderer injects — downgrade to plain "[p.N]"
# so the Zotero note (which can't run the renderer) reads cleanly.
_CITE_BADGE_RE = re.compile(
    r'<span\s+class=["\']cite-badge["\']\s+data-page=["\'](\d+)["\']>\[p\.\1\]</span>',
    re.IGNORECASE,
)


def _clean_report_md(md: str) -> str:
    return _CITE_BADGE_RE.sub(r"[p.\1]", md)


def _safe_name(title: str, fallback: str) -> str:
    """ASCII-safe base for filenames / Content-Disposition (avoids header encoding
    pitfalls). The *item* title inside the RDF keeps the real Unicode value."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", (title or "").strip()).strip("._-")
    return name[:60] or fallback


async def fetch_crossref_meta(doi: str) -> dict:
    """Best-effort authors + abstract from Crossref by DOI. Never raises.

    The ``papers`` table stores only title/venue/year/DOI, so authors and the
    abstract are pulled here at export time to produce a complete Zotero item.
    """
    if not doi:
        return {}
    url = f"https://api.crossref.org/works/{doi}"
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                url, headers={"User-Agent": "scholar/1.0 (zotero-export)"}
            )
            resp.raise_for_status()
            msg = resp.json().get("message", {})
    except Exception:
        return {}

    authors: list[tuple[str, str]] = []
    for a in msg.get("author", []) or []:
        family = (a.get("family") or "").strip()
        given = (a.get("given") or "").strip()
        if family:
            authors.append((family, given))
        elif given:
            authors.append((given, ""))

    abstract = msg.get("abstract") or ""
    if abstract:
        abstract = re.sub(r"<[^>]+>", "", abstract)  # strip JATS XML tags
        abstract = html.unescape(abstract).strip()

    return {"authors": authors, "abstract": abstract}


def build_rdf(
    *,
    title: str,
    authors: list[tuple[str, str]],
    year: int,
    venue: str,
    doi: str,
    abstract: str,
    note_html: str,
    pdf_rel_path: str | None,
    pdf_title: str = "Full Text PDF",
) -> str:
    """Render the Zotero RDF/XML for one journalArticle, optional child note,
    and optional file attachment."""
    e = _xml_escape
    out: list[str] = ['<?xml version="1.0" encoding="UTF-8"?>']
    ns = " ".join(f'xmlns:{k}="{v}"' for k, v in _RDF_NS.items())
    out.append(f"<rdf:RDF {ns}>")

    # --- Article (item_1) ---
    out.append('    <bib:Article rdf:about="#item_1">')
    out.append("        <z:itemType>journalArticle</z:itemType>")
    if pdf_rel_path:
        out.append('        <link:link rdf:resource="#item_2"/>')
    if note_html:
        out.append('        <dcterms:isReferencedBy rdf:resource="#item_3"/>')

    if authors:
        out.append("        <bib:authors>")
        out.append("            <rdf:Seq>")
        for last, first in authors:
            out.append("                <rdf:li>")
            out.append("                    <foaf:Person>")
            out.append(f"                        <foaf:surname>{e(last)}</foaf:surname>")
            if first:
                out.append(
                    f"                        <foaf:givenName>{e(first)}</foaf:givenName>"
                )
            out.append("                    </foaf:Person>")
            out.append("                </rdf:li>")
        out.append("            </rdf:Seq>")
        out.append("        </bib:authors>")

    # Journal container carries venue title + DOI (matches Zotero's exporter,
    # which writes "DOI ..." as a dc:identifier on the container).
    out.append("        <dcterms:isPartOf>")
    out.append("            <bib:Journal>")
    if venue:
        out.append(f"                <dc:title>{e(venue)}</dc:title>")
    if doi:
        out.append(f"                <dc:identifier>DOI {e(doi)}</dc:identifier>")
    out.append("            </bib:Journal>")
    out.append("        </dcterms:isPartOf>")

    if title:
        out.append(f"        <dc:title>{e(title)}</dc:title>")
    if year:
        out.append(f"        <dc:date>{e(str(year))}</dc:date>")
    if abstract:
        out.append(f"        <dcterms:abstract>{e(abstract)}</dcterms:abstract>")
    out.append("    </bib:Article>")

    # --- Attachment (item_2) ---
    if pdf_rel_path:
        out.append('    <z:Attachment rdf:about="#item_2">')
        out.append("        <z:itemType>attachment</z:itemType>")
        out.append(f"        <z:path>{e(pdf_rel_path)}</z:path>")
        out.append("        <link:type>application/pdf</link:type>")
        out.append(f"        <dc:title>{e(pdf_title)}</dc:title>")
        out.append("    </z:Attachment>")

    # --- Note (item_3): report HTML escaped into rdf:value ---
    if note_html:
        out.append('    <bib:Memo rdf:about="#item_3">')
        out.append("        <z:itemType>note</z:itemType>")
        out.append(f"        <rdf:value>{e(note_html)}</rdf:value>")
        out.append("    </bib:Memo>")

    out.append("</rdf:RDF>")
    return "\n".join(out)


async def build_zotero_bundle(
    *,
    paper: dict,
    markdown_report: str,
    pdf_path: Path | None,
) -> tuple[bytes, str]:
    """Build the ``.zip`` bundle (``<name>.rdf`` + ``files/2/<name>.pdf``).

    Returns ``(zip_bytes, download_filename)``.
    """
    paper_id = paper.get("paper_id") or "paper"
    title = paper.get("title") or paper_id
    doi = paper.get("doi") or ""
    venue = paper.get("venue") or ""
    year = int(paper.get("year") or 0)

    extra = await fetch_crossref_meta(doi)
    authors = extra.get("authors") or []
    abstract = extra.get("abstract") or ""

    note_html = _md_to_html(_clean_report_md(markdown_report)) if markdown_report else ""

    base = _safe_name(title, fallback=f"paper_{paper_id[:8]}")

    pdf_bytes: bytes | None = None
    pdf_rel: str | None = None
    if pdf_path is not None and pdf_path.exists():
        pdf_bytes = pdf_path.read_bytes()
        pdf_rel = f"files/2/{base}.pdf"

    rdf_xml = build_rdf(
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        doi=doi,
        abstract=abstract,
        note_html=note_html,
        pdf_rel_path=pdf_rel,
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{base}.rdf", rdf_xml)
        if pdf_bytes is not None and pdf_rel is not None:
            zf.writestr(pdf_rel, pdf_bytes)

    return buf.getvalue(), f"{base}_zotero.zip"
