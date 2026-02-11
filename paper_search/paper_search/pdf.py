from __future__ import annotations

from pathlib import Path

from .config import Settings
from .http_client import HTTPClient
from .utils import safe_filename_component


def _is_pdf_file(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(4)
        return head == b"%PDF"
    except OSError:
        return False


async def download_pdf_to_cache(
    client: HTTPClient,
    *,
    url: str,
    settings: Settings,
    file_stem: str,
) -> Path | None:
    url = (url or "").strip()
    if not url:
        return None

    settings.pdf_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_filename_component(file_stem)
    dest = settings.pdf_dir / f"{stem}.pdf"
    if dest.exists() and dest.stat().st_size > 0 and _is_pdf_file(dest):
        return dest

    ok = await client.download_to_file(url, dest_path=dest)
    if not ok or not (dest.exists() and _is_pdf_file(dest)):
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    return dest
