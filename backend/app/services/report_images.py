"""Resolve report figures for portable export.

Reports embed MinerU-extracted figures as Markdown images pointing at the app's
own ``/api/papers/{paper_id}/images/{name}`` endpoint. Those links only work
inside the running app, so any exported artifact (a Zotero note, a downloaded
``.md``) shows broken images.

This module rewrites those figure links so the export is self-contained:

* When object storage (R2) is configured, each figure is uploaded once to
  ``scholar/{paper_id}/{name}`` and the link is rewritten to its public URL.
* Otherwise, callers may opt into inlining the figure as a base64 ``data:`` URI
  (used for the offline Zotero note) or leave the original link untouched (used
  for the plain Markdown download, i.e. the original behaviour).

Everything here is best-effort and never raises: a figure that can't be resolved
or uploaded is either dropped or left as-is, but never breaks the export.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
from pathlib import Path

from app.services import r2_storage

logger = logging.getLogger("scholar.report_images")

# MinerU figure files are a content hash + extension (no path separators).
_SAFE_IMAGE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+\.(jpg|jpeg|png|gif|webp)$", re.IGNORECASE)
_IMAGE_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
# Markdown image: ![alt](url)
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(\s*([^)\s]+)\s*\)")
# Don't inline anything enormous as a data URI (Zotero note-size limits).
_MAX_EMBED_BYTES = 5 * 1024 * 1024


def _figure_name(url: str) -> str | None:
    """Return the figure filename if ``url`` is an internal figure ref, else None."""
    if "/images/" not in url:
        return None
    name = url.split("?", 1)[0].split("#", 1)[0].rsplit("/", 1)[-1]
    return name if _SAFE_IMAGE_NAME_RE.match(name) else None


def _resolve_image_file(paper_id: str, name: str, data_dir: Path) -> Path | None:
    """Locate a MinerU figure on disk (mirrors the image-serving endpoint)."""
    data_root = data_dir.resolve()
    mineru_dir = (data_dir / "papers" / paper_id / "mineru").resolve()
    if not mineru_dir.is_relative_to(data_root) or not mineru_dir.exists():
        return None
    for cand in mineru_dir.rglob(name):
        if cand.is_file() and cand.parent.name == "images":
            resolved = cand.resolve()
            if resolved.is_relative_to(data_root):
                return resolved
    return None


def _data_uri(content_type: str, data: bytes) -> str:
    return f"data:{content_type};base64,{base64.b64encode(data).decode('ascii')}"


async def _resolve_one(
    url: str,
    name: str,
    *,
    paper_id: str,
    data_dir: Path,
    embed_fallback: bool,
    drop_unresolved: bool,
) -> tuple[str, str | None]:
    """Decide the replacement for one figure URL.

    Returns ``(url, new_src)`` where ``new_src`` is the rewritten link, an empty
    string to drop the image, or ``None`` to leave it unchanged.
    """
    path = _resolve_image_file(paper_id, name, data_dir)
    if path is None:
        return url, ("" if drop_unresolved else None)

    content_type = _IMAGE_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
    new_src: str | None = None

    if r2_storage.is_enabled():
        try:
            data = path.read_bytes()
            new_src = await r2_storage.upload_file(
                f"scholar/{paper_id}/{name}", data, content_type
            )
        except Exception as e:  # pragma: no cover - IO error path
            logger.warning("figure read/upload failed for %s: %s", name, e)

    if new_src is None and embed_fallback:
        try:
            data = path.read_bytes()
            if len(data) <= _MAX_EMBED_BYTES:
                new_src = _data_uri(content_type, data)
        except Exception as e:  # pragma: no cover - IO error path
            logger.warning("figure embed failed for %s: %s", name, e)

    if new_src is None:
        # Couldn't host or embed: drop for notes, keep original for plain md.
        return url, ("" if drop_unresolved else None)
    return url, new_src


async def process_report_markdown(
    markdown: str,
    *,
    paper_id: str,
    data_dir: Path | None,
    embed_fallback: bool = False,
    drop_unresolved: bool = False,
) -> str:
    """Rewrite internal figure links in ``markdown`` for portable export.

    ``embed_fallback`` inlines figures as data URIs when object storage is off
    (or upload fails); ``drop_unresolved`` removes images that can't be hosted or
    embedded (preferred for notes, where a broken link shows an error icon).
    """
    if not markdown or data_dir is None:
        return markdown

    refs: dict[str, str] = {}
    for m in _MD_IMAGE_RE.finditer(markdown):
        url = m.group(2).strip()
        if url in refs:
            continue
        name = _figure_name(url)
        if name:
            refs[url] = name
    if not refs:
        return markdown

    results = await asyncio.gather(
        *(
            _resolve_one(
                url,
                name,
                paper_id=paper_id,
                data_dir=data_dir,
                embed_fallback=embed_fallback,
                drop_unresolved=drop_unresolved,
            )
            for url, name in refs.items()
        )
    )
    actions = dict(results)

    def _repl(m: re.Match) -> str:
        alt, url = m.group(1), m.group(2).strip()
        new_src = actions.get(url)
        if new_src is None:  # not a figure ref, or left unchanged
            return m.group(0)
        if new_src == "":  # drop the image
            return ""
        return f"![{alt}]({new_src})"

    return _MD_IMAGE_RE.sub(_repl, markdown)
