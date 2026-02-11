from __future__ import annotations

"""
Wiley Online Library Text & Data Mining (TDM) PDF downloader.

Endpoint shape (as used by the user's verified script):
  GET https://api.wiley.com/onlinelibrary/tdm/v1/articles/{urlencoded-doi}
  Headers:
    - Wiley-TDM-Client-Token: <token>
"""

import os
import time
from pathlib import Path
from urllib.parse import quote

import requests

from .credentials import env_first


WILEY_TDM_ARTICLE_DOI_URL = "https://api.wiley.com/onlinelibrary/tdm/v1/articles/{doi}"


def env_default_wiley_tdm_token() -> str:
    return env_first("WILEY_TDM_TOKEN", "WILEY_TDM_CLIENT_TOKEN", "WILEY_CLIENT_TOKEN")


def looks_like_wiley_doi(doi: str) -> bool:
    """
    Best-effort heuristic to avoid calling Wiley TDM for obviously non-Wiley DOIs.

    Most Wiley DOIs commonly start with:
      - 10.1002/
      - 10.1111/
    """
    d = (doi or "").strip().lower()
    return d.startswith("10.1002/") or d.startswith("10.1111/")


def _looks_like_pdf_bytes(data: bytes) -> bool:
    return data.lstrip().startswith(b"%PDF")


def download_wiley_tdm_pdf_to_path(
    session: requests.Session,
    doi: str,
    dest_path: Path,
    *,
    token: str,
    timeout: int = 60,
    retries: int = 3,
    backoff_seconds: float = 1.5,
) -> tuple[bool, str | None]:
    token = (token or "").strip()
    if not token:
        return False, "Missing Wiley TDM token"

    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    url = WILEY_TDM_ARTICLE_DOI_URL.format(doi=quote(doi, safe=""))
    headers: dict[str, str] = {
        "Wiley-TDM-Client-Token": token,
        "Accept": "application/pdf",
    }

    last_err: str | None = None
    for attempt in range(1, max(1, int(retries)) + 1):
        try:
            with session.get(
                url,
                headers=headers,
                stream=True,
                timeout=timeout,
                allow_redirects=True,
            ) as resp:
                if resp.status_code == 429:
                    sleep_s = min(60.0, backoff_seconds * attempt * 2)
                    time.sleep(sleep_s)
                    continue

                if resp.status_code == 404:
                    return False, "Wiley TDM: article not found (404)"
                if resp.status_code == 401:
                    return False, "Wiley TDM: unauthorized (401) - check token"
                if resp.status_code == 403:
                    return False, "Wiley TDM: forbidden (403) - check subscription/token"

                resp.raise_for_status()

                it = resp.iter_content(chunk_size=1024 * 1024)
                first = next(it, b"")
                if not first:
                    raise ValueError("Empty response body")

                content_type = (resp.headers.get("Content-Type") or "").lower()
                if ("application/pdf" not in content_type) and (
                    not _looks_like_pdf_bytes(first[:2048])
                ):
                    raise ValueError(f"Not a PDF (Content-Type={content_type!r})")

                with open(tmp_path, "wb") as f:
                    f.write(first)
                    for chunk in it:
                        if chunk:
                            f.write(chunk)

            os.replace(tmp_path, dest_path)
            return True, None
        except Exception as e:
            last_err = str(e)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            sleep_s = min(60.0, backoff_seconds * attempt)
            time.sleep(sleep_s)

    return False, last_err

