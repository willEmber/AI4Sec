from __future__ import annotations

import hashlib
import html
import re
import xml.etree.ElementTree as ET


_DOI_PREFIX_RE = re.compile(r"^https?://(dx\.)?doi\.org/", re.IGNORECASE)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def strip_html(text: str) -> str:
    text = text or ""
    text = re.sub(r"<[^>]+>", " ", text)
    return normalize_whitespace(html.unescape(text))


def normalize_doi(doi: str) -> str:
    doi = (doi or "").strip()
    doi = _DOI_PREFIX_RE.sub("", doi)
    doi = doi.strip()
    return doi.lower()


def title_fingerprint(title: str) -> str:
    raw = normalize_whitespace(title).lower()

    ascii_norm = normalize_whitespace(re.sub(r"[^a-z0-9]+", " ", raw))
    if ascii_norm:
        return hashlib.sha1(ascii_norm.encode("utf-8")).hexdigest()

    # Fallback for non-Latin titles (e.g. Chinese): keep unicode word chars to avoid collisions.
    unicode_norm = normalize_whitespace(re.sub(r"[^\w]+", " ", raw, flags=re.UNICODE)).replace("_", " ")
    unicode_norm = normalize_whitespace(unicode_norm)
    return hashlib.sha1(unicode_norm.encode("utf-8")).hexdigest()


def safe_filename_component(text: str) -> str:
    text = normalize_whitespace(text)
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text)
    return text.strip("._-") or "paper"


def openalex_abstract_from_inverted_index(inverted_index: dict[str, list[int]] | None) -> str:
    if not inverted_index:
        return ""
    position_to_word: dict[int, str] = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            # Later duplicates are fine; order is recovered by position anyway.
            position_to_word[pos] = word
    words = [position_to_word[i] for i in sorted(position_to_word)]
    return normalize_whitespace(" ".join(words))


def jaccard_similarity(a: str, b: str) -> float:
    a_tokens = set(re.findall(r"[a-z0-9]+", (a or "").lower()))
    b_tokens = set(re.findall(r"[a-z0-9]+", (b or "").lower()))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def safe_xml_fromstring(xml_text: str, *, max_chars: int = 5_000_000) -> ET.Element:
    """
    Parse XML defensively to reduce DoS risk from DTD/entity expansion.
    - Strips DOCTYPE declarations (common in PubMed XML) to avoid DTD processing.
    - Rejects ENTITY declarations.
    - Rejects overly large payloads.
    """
    xml_text = xml_text or ""
    if not xml_text:
        raise ValueError("empty xml")
    if max_chars > 0 and len(xml_text) > max_chars:
        raise ValueError("xml too large")
    low = xml_text.casefold()
    # PubMed efetch XML contains a DOCTYPE with an external DTD. We don't need the DTD for parsing,
    # and keeping it can enable DTD-related attack surface in some XML parsers.
    if "<!doctype" in low:
        # Best-effort: remove the DOCTYPE (with optional internal subset) while keeping the XML content.
        xml_text = re.sub(r"(?is)<!doctype[^>]*(\[[\s\S]*?\])?\s*>", "", xml_text, count=1)
        low = xml_text.casefold()
    if "<!entity" in low:
        raise ValueError("xml contains disallowed entity declarations")
    return ET.fromstring(xml_text)
