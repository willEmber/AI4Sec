from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Mapping

import requests


class HTTPStatusError(RuntimeError):
    def __init__(
        self,
        method: str,
        url: str,
        status_code: int,
        body: str,
        *,
        retry_after: float | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(f"{method} {_redact(url)} -> HTTP {status_code}: {body[:500]}")
        self.method = method
        self.url = _redact(url)
        self.status_code = status_code
        self.body = body
        self.retry_after = retry_after
        self.headers = dict(headers or {})


def _redact(url: str) -> str:
    """Keep API keys out of exception text, which reaches logs and tool results."""
    return _KEY_PARAM_RE.sub(r"\1=***", url or "")


_KEY_PARAM_RE = re.compile(r"\b(api_key|apikey|key|token)=[^&\s]+", re.IGNORECASE)


@dataclass
class HTTPClient:
    timeout: float = 30.0
    headers: Mapping[str, str] | None = None

    async def get_json(
        self, url: str, *, params: Mapping[str, str] | None = None, headers: Mapping[str, str] | None = None
    ) -> Any:
        return await asyncio.to_thread(self._get_json_sync, url, params, headers)

    async def get_json_with_headers(
        self, url: str, *, params: Mapping[str, str] | None = None, headers: Mapping[str, str] | None = None
    ) -> tuple[Any, dict[str, str]]:
        """Like `get_json`, but also returns the response headers (rate-limit budgets live there)."""
        return await asyncio.to_thread(self._get_json_with_headers_sync, url, params, headers)

    async def get_text(
        self, url: str, *, params: Mapping[str, str] | None = None, headers: Mapping[str, str] | None = None
    ) -> str:
        return await asyncio.to_thread(self._get_text_sync, url, params, headers)

    async def post_json(
        self,
        url: str,
        *,
        json_body: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        return await asyncio.to_thread(self._post_json_sync, url, json_body, headers)

    async def get_status_and_json(
        self, url: str, *, params: Mapping[str, str] | None = None, headers: Mapping[str, str] | None = None
    ) -> tuple[int, Any]:
        return await asyncio.to_thread(self._get_status_and_json_sync, url, params, headers)

    async def download_to_file(
        self,
        url: str,
        *,
        dest_path: Path,
        headers: Mapping[str, str] | None = None,
    ) -> bool:
        return await asyncio.to_thread(self._download_to_file_sync, url, dest_path, headers)

    def _merged_headers(self, extra: Mapping[str, str] | None) -> dict[str, str]:
        merged: dict[str, str] = {}
        if self.headers:
            merged.update(dict(self.headers))
        if extra:
            merged.update(dict(extra))
        return merged

    @staticmethod
    def _retry_after_seconds(resp: requests.Response) -> float | None:
        value = (resp.headers.get("Retry-After") or "").strip()
        if not value:
            return None
        try:
            return max(float(value), 0.0)
        except ValueError:
            pass
        try:
            dt = parsedate_to_datetime(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            return max((dt - now).total_seconds(), 0.0)
        except Exception:
            return None

    def _get_status_and_json_sync(
        self, url: str, params: Mapping[str, str] | None, headers: Mapping[str, str] | None
    ) -> tuple[int, Any]:
        resp = requests.get(
            url, params=params, headers=self._merged_headers(headers), timeout=self.timeout, allow_redirects=True
        )
        status = resp.status_code
        try:
            data = resp.json()
        except Exception:
            data = None
        return status, data

    def _get_json_sync(
        self, url: str, params: Mapping[str, str] | None, headers: Mapping[str, str] | None
    ) -> Any:
        return self._get_json_with_headers_sync(url, params, headers)[0]

    def _get_json_with_headers_sync(
        self, url: str, params: Mapping[str, str] | None, headers: Mapping[str, str] | None
    ) -> tuple[Any, dict[str, str]]:
        resp = requests.get(
            url, params=params, headers=self._merged_headers(headers), timeout=self.timeout, allow_redirects=True
        )
        if not resp.ok:
            raise self._status_error("GET", resp)
        return resp.json(), dict(resp.headers)

    def _status_error(self, method: str, resp: requests.Response) -> HTTPStatusError:
        return HTTPStatusError(
            method,
            resp.url,
            resp.status_code,
            resp.text,
            retry_after=self._retry_after_seconds(resp),
            headers=dict(resp.headers),
        )

    def _get_text_sync(
        self, url: str, params: Mapping[str, str] | None, headers: Mapping[str, str] | None
    ) -> str:
        resp = requests.get(
            url, params=params, headers=self._merged_headers(headers), timeout=self.timeout, allow_redirects=True
        )
        if not resp.ok:
            raise self._status_error("GET", resp)
        resp.encoding = resp.encoding or "utf-8"
        return resp.text

    def _post_json_sync(
        self, url: str, json_body: Mapping[str, Any], headers: Mapping[str, str] | None
    ) -> Any:
        resp = requests.post(
            url,
            json=json_body,
            headers=self._merged_headers(headers),
            timeout=self.timeout,
            allow_redirects=True,
        )
        if not resp.ok:
            raise self._status_error("POST", resp)
        return resp.json()

    def _download_to_file_sync(
        self, url: str, dest_path: Path, headers: Mapping[str, str] | None
    ) -> bool:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

        try:
            with requests.get(
                url,
                headers=self._merged_headers(headers),
                timeout=self.timeout,
                allow_redirects=True,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                with tmp_path.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            f.write(chunk)
            os.replace(tmp_path, dest_path)
            return True
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            return False
