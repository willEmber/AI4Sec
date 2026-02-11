from __future__ import annotations

"""
Elsevier (ScienceDirect) Text & Data Mining (TDM) API PDF downloader.

Docs / endpoint shape (as used in the user's verified script):
  GET https://api.elsevier.com/content/article/doi/{doi}
  Headers:
    - X-ELS-APIKey: <api key>
    - X-ELS-Insttoken: <optional institution token>
    - Accept: application/pdf

This downloader is only attempted when an Elsevier API key is configured.
"""

import time
import os
from pathlib import Path

import requests

from .credentials import env_first

ELSEVIER_ARTICLE_DOI_URL = "https://api.elsevier.com/content/article/doi/{doi}"


def env_default_elsevier_api_key() -> str:
    return env_first("ELSEVIER_API_KEY", "ELSEVIER_TDM_API_KEY", "X_ELS_APIKEY")


def env_default_elsevier_inst_token() -> str:
    return env_first("ELSEVIER_INSTTOKEN", "ELSEVIER_INST_TOKEN", "X_ELS_INSTTOKEN")


def _looks_like_pdf_bytes(data: bytes) -> bool:
    return data.lstrip().startswith(b"%PDF")


def download_elsevier_tdm_pdf_to_path(
    session: requests.Session,
    doi: str,
    dest_path: Path,
    *,
    api_key: str,
    inst_token: str = "",
    timeout: int = 60,
    retries: int = 3,
    backoff_seconds: float = 1.5,
) -> tuple[bool, str | None]:
    """
    Download a PDF via Elsevier TDM API.

    Returns (ok, detail). When ok=False, detail contains a human-readable error.
    """
    api_key = (api_key or "").strip()
    if not api_key:
        return False, "Missing Elsevier API key"

    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    url = ELSEVIER_ARTICLE_DOI_URL.format(doi=doi)
    headers: dict[str, str] = {
        "X-ELS-APIKey": api_key,
        "Accept": "application/pdf",
    }
    if inst_token:
        headers["X-ELS-Insttoken"] = inst_token

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
                    return False, "Elsevier: article not found (404)"
                if resp.status_code == 401:
                    return False, "Elsevier: unauthorized (401) - check API key"
                if resp.status_code == 403:
                    return False, "Elsevier: forbidden (403) - check subscription/InstToken"

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
