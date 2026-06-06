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

Authors and other bibliographic fields are not stored in the ``papers`` table
(only title/venue/year/DOI), so they are recovered here at export time via a
multi-source, best-effort lookup (Crossref by DOI → OpenAlex by DOI → OpenAlex /
Crossref by title). This is why an item can otherwise import with only the
title, venue and date: the single DOI-only Crossref call it used before returned
nothing whenever the paper had no DOI (preprints, conference papers, scanned
PDFs) or Crossref had no author record for it.
"""

from __future__ import annotations

import html
import io
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

import httpx

from app.services.paper_search.utils import (
    jaccard_similarity,
    normalize_whitespace,
    openalex_abstract_from_inverted_index,
)

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

# --- Bibliographic enrichment -------------------------------------------------
_CR_BASE = "https://api.crossref.org/works"
_OA_BASE = "https://api.openalex.org/works"
_HTTP_HEADERS = {"User-Agent": "scholar/1.0 (zotero-export; mailto:scholar@example.com)"}
# A title-search hit is only trusted when this similar to the query title, so a
# DOI-less paper never imports another paper's authors by mistake.
_TITLE_MATCH_THRESHOLD = 0.55


def _clean_report_md(md: str) -> str:
    return _CITE_BADGE_RE.sub(r"[p.\1]", md)


def _safe_name(title: str, fallback: str) -> str:
    """ASCII-safe base for filenames / Content-Disposition (avoids header encoding
    pitfalls). The *item* title inside the RDF keeps the real Unicode value."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", (title or "").strip()).strip("._-")
    return name[:60] or fallback


def _meta_skeleton() -> dict:
    return {
        "authors": [],  # list[tuple[family, given]]
        "abstract": "",
        "venue": "",
        "year": 0,
        "volume": "",
        "issue": "",
        "pages": "",
        "issn": "",
        "url": "",
        "doi": "",
        "_title": "",  # for title-similarity guard only; popped before return
    }


def _split_name(display_name: str) -> tuple[str, str]:
    """Best-effort split of a single display name into (family, given).

    Sources that only give a full name (OpenAlex) are split on the last space —
    ``"Ashish Vaswani"`` -> ``("Vaswani", "Ashish")``, single-token names keep an
    empty given name."""
    name = normalize_whitespace(display_name)
    if not name:
        return ("", "")
    parts = name.rsplit(" ", 1)
    if len(parts) == 2:
        return (parts[1], parts[0])
    return (name, "")


def _parse_crossref(msg: dict) -> dict:
    """Crossref ``message`` (a /works/{doi} record or a search item) -> meta."""
    meta = _meta_skeleton()

    authors: list[tuple[str, str]] = []
    for a in msg.get("author") or []:
        family = (a.get("family") or "").strip()
        given = (a.get("given") or "").strip()
        if family:
            authors.append((family, given))
        elif given:
            authors.append((given, ""))
        elif (a.get("name") or "").strip():
            authors.append((a["name"].strip(), ""))
    meta["authors"] = authors

    abstract = msg.get("abstract") or ""
    if abstract:
        abstract = re.sub(r"<[^>]+>", "", abstract)  # strip JATS XML tags
        meta["abstract"] = html.unescape(abstract).strip()

    container = msg.get("container-title") or []
    if container:
        meta["venue"] = (container[0] or "").strip()

    for date_field in ("published-print", "published-online", "issued"):
        parts = (msg.get(date_field) or {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            meta["year"] = int(parts[0][0])
            break

    meta["volume"] = str(msg.get("volume") or "").strip()
    meta["issue"] = str(msg.get("issue") or "").strip()
    meta["pages"] = str(msg.get("page") or "").strip()
    issn = msg.get("ISSN") or []
    if issn:
        meta["issn"] = (issn[0] or "").strip()
    meta["url"] = (msg.get("URL") or "").strip()
    meta["doi"] = (msg.get("DOI") or "").strip()

    title = msg.get("title")
    if isinstance(title, list):
        meta["_title"] = (title[0] if title else "") or ""
    else:
        meta["_title"] = title or ""
    return meta


def _parse_openalex(work: dict) -> dict:
    """OpenAlex ``work`` record -> meta."""
    meta = _meta_skeleton()

    authors: list[tuple[str, str]] = []
    for a in work.get("authorships") or []:
        name = ((a.get("author") or {}).get("display_name") or "").strip()
        if name:
            authors.append(_split_name(name))
    meta["authors"] = authors

    meta["abstract"] = openalex_abstract_from_inverted_index(
        work.get("abstract_inverted_index")
    )

    loc = work.get("primary_location") or {}
    source = loc.get("source") or {}
    meta["venue"] = normalize_whitespace(source.get("display_name") or "")
    issn_l = (source.get("issn_l") or "").strip()
    if not issn_l:
        issn = source.get("issn") or []
        if isinstance(issn, list) and issn:
            issn_l = (issn[0] or "").strip()
    meta["issn"] = issn_l

    meta["year"] = int(work.get("publication_year") or 0)

    biblio = work.get("biblio") or {}
    meta["volume"] = str(biblio.get("volume") or "").strip()
    meta["issue"] = str(biblio.get("issue") or "").strip()
    first = str(biblio.get("first_page") or "").strip()
    last = str(biblio.get("last_page") or "").strip()
    if first and last:
        meta["pages"] = f"{first}-{last}"
    elif first:
        meta["pages"] = first

    doi = ((work.get("ids") or {}).get("doi") or "").strip()
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]
    meta["doi"] = doi
    meta["url"] = (loc.get("landing_page_url") or "").strip() or (
        f"https://doi.org/{doi}" if doi else ""
    )
    meta["_title"] = normalize_whitespace(work.get("title") or "")
    return meta


def _merge_meta(base: dict, extra: dict) -> dict:
    """Fill empty fields of ``base`` from ``extra`` (first source wins).

    Authors are filled only when ``base`` still has none, so a structured
    Crossref author list is never overwritten by a coarser split-name list."""
    if not base.get("authors") and extra.get("authors"):
        base["authors"] = extra["authors"]
    for key in ("abstract", "venue", "volume", "issue", "pages", "issn", "url", "doi"):
        if not base.get(key) and extra.get(key):
            base[key] = extra[key]
    if not base.get("year") and extra.get("year"):
        base["year"] = extra["year"]
    return base


def _oa_params() -> dict:
    """OpenAlex ``mailto`` for the polite pool, if an email is configured."""
    try:
        from app.services.paper_search.config import Settings

        email = Settings.from_env().pick_openalex_mailto()
    except Exception:
        email = ""
    return {"mailto": email} if email else {}


async def _get_json(client: httpx.AsyncClient, url: str, params: dict | None = None):
    try:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


async def fetch_item_meta(doi: str = "", title: str = "") -> dict:
    """Best-effort bibliographic metadata for a Zotero item. Never raises.

    Combines several keyless sources so an item still gets authors / abstract /
    volume / issue / pages even when one source has gaps or there is no DOI at
    all:

      1. Crossref by DOI    — authoritative, structured ``family``/``given`` names
      2. OpenAlex by DOI    — fills gaps (esp. preprints & conferences)
      3. OpenAlex by title  — recovers authors when there is no DOI
      4. Crossref by title  — last-resort author source

    Title-search hits (3, 4) are accepted only when the returned title is close
    enough to the query title, so a DOI-less paper never adopts another paper's
    authors.
    """
    meta = _meta_skeleton()
    doi = (doi or "").strip()
    title = (title or "").strip()
    if not doi and not title:
        meta.pop("_title", None)
        return meta

    async with httpx.AsyncClient(timeout=10.0, headers=_HTTP_HEADERS) as client:
        # 1. Crossref by DOI
        if doi:
            data = await _get_json(client, f"{_CR_BASE}/{doi}")
            if data:
                _merge_meta(meta, _parse_crossref(data.get("message") or {}))

        # 2. OpenAlex by DOI (fills authors/abstract Crossref may lack)
        if doi and (not meta["authors"] or not meta["abstract"]):
            data = await _get_json(client, f"{_OA_BASE}/doi:{doi}", params=_oa_params())
            if data:
                _merge_meta(meta, _parse_openalex(data))

        # 3. OpenAlex by title (no DOI, or DOI gave us no authors)
        if not meta["authors"] and title:
            data = await _get_json(
                client,
                _OA_BASE,
                params={**_oa_params(), "search": title, "per_page": "1"},
            )
            results = (data or {}).get("results") or []
            if results:
                cand = _parse_openalex(results[0])
                if jaccard_similarity(cand.get("_title", ""), title) >= _TITLE_MATCH_THRESHOLD:
                    _merge_meta(meta, cand)

        # 4. Crossref bibliographic query (last-resort author source)
        if not meta["authors"] and title:
            data = await _get_json(
                client, _CR_BASE, params={"query.bibliographic": title, "rows": "1"}
            )
            items = ((data or {}).get("message") or {}).get("items") or []
            if items:
                cand = _parse_crossref(items[0])
                if jaccard_similarity(cand.get("_title", ""), title) >= _TITLE_MATCH_THRESHOLD:
                    _merge_meta(meta, cand)

    meta.pop("_title", None)
    return meta


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
    volume: str = "",
    issue: str = "",
    pages: str = "",
    issn: str = "",
    url: str = "",
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

    # Journal container carries the venue title + ISSN (matches Zotero's
    # exporter, which writes ISSN — not the article DOI — on the container).
    out.append("        <dcterms:isPartOf>")
    out.append("            <bib:Journal>")
    if venue:
        out.append(f"                <dc:title>{e(venue)}</dc:title>")
    if issn:
        out.append(f"                <dc:identifier>ISSN {e(issn)}</dc:identifier>")
    out.append("            </bib:Journal>")
    out.append("        </dcterms:isPartOf>")

    if title:
        out.append(f"        <dc:title>{e(title)}</dc:title>")
    if year:
        out.append(f"        <dc:date>{e(str(year))}</dc:date>")
    if volume:
        out.append(f"        <prism:volume>{e(volume)}</prism:volume>")
    if issue:
        out.append(f"        <prism:number>{e(issue)}</prism:number>")
    if pages:
        out.append(f"        <bib:pages>{e(pages)}</bib:pages>")
    # The article DOI belongs on the item itself; Zotero's importer maps a
    # "DOI ..." dc:identifier on the item to the DOI field.
    if doi:
        out.append(f"        <dc:identifier>DOI {e(doi)}</dc:identifier>")
    if url:
        out.append("        <dc:identifier>")
        out.append("            <dcterms:URI>")
        out.append(f"                <rdf:value>{e(url)}</rdf:value>")
        out.append("            </dcterms:URI>")
        out.append("        </dc:identifier>")
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
    doi = (paper.get("doi") or "").strip()
    venue = paper.get("venue") or ""
    year = int(paper.get("year") or 0)

    # Don't title-search on a sha1 fallback "title" — it would match nothing
    # useful and risks importing a wrong paper's authors.
    lookup_title = title if title != paper_id else ""
    extra = await fetch_item_meta(doi=doi, title=lookup_title)

    authors = extra.get("authors") or []
    abstract = extra.get("abstract") or ""
    # The DB venue/year are curated for ranking — keep them, fill gaps only.
    venue = venue or extra.get("venue") or ""
    if not year:
        year = int(extra.get("year") or 0)
    doi = doi or (extra.get("doi") or "")
    volume = extra.get("volume") or ""
    issue = extra.get("issue") or ""
    pages = extra.get("pages") or ""
    issn = extra.get("issn") or ""
    url = extra.get("url") or ""

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
        volume=volume,
        issue=issue,
        pages=pages,
        issn=issn,
        url=url,
    )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{base}.rdf", rdf_xml)
        if pdf_bytes is not None and pdf_rel is not None:
            zf.writestr(pdf_rel, pdf_bytes)

    return buf.getvalue(), f"{base}_zotero.zip"
