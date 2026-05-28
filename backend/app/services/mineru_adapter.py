from __future__ import annotations

import asyncio
import json
import logging
import random
import shutil
import sqlite3
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Optional

from app.config import get_settings
from app.db import database as db

logger = logging.getLogger("scholar.mineru")

API_BASE = "https://mineru.net/api/v4"

_MAX_ZIP_EXTRACTED_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB


class MinerUPollTimeoutError(TimeoutError):
    def __init__(
        self,
        batch_id: str,
        elapsed_s: float,
        poll_count: int,
        last_state_counts: dict[str, int],
        timeout_s: int,
    ) -> None:
        self.batch_id = batch_id
        self.elapsed_s = elapsed_s
        self.poll_count = poll_count
        self.last_state_counts = last_state_counts
        self.timeout_s = timeout_s
        super().__init__(
            "MinerU batch timed out "
            f"batch={batch_id} elapsed={elapsed_s:.0f}s timeout={timeout_s}s "
            f"polls={poll_count} last_states={last_state_counts}. "
            "The remote MinerU task may still finish later; retry this paper after checking parse status."
        )


def _safe_zip_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract zip with path traversal and zip bomb protection."""
    dest_resolved = dest.resolve()
    total_size = 0
    for info in zf.infolist():
        # Reject absolute paths and path traversal
        member_path = (dest / info.filename).resolve()
        if not member_path.is_relative_to(dest_resolved):
            raise ValueError(f"Zip path traversal detected: {info.filename}")
        # Accumulate uncompressed size for zip bomb check
        total_size += info.file_size
        if total_size > _MAX_ZIP_EXTRACTED_SIZE:
            raise ValueError(
                f"Zip extracted size exceeds limit "
                f"({_MAX_ZIP_EXTRACTED_SIZE // (1024 * 1024)} MB)"
            )
    # All checks passed, extract
    zf.extractall(dest)


class MinerUClient:
    """Async-friendly MinerU API client. Reimplemented to avoid importing
    paper_converter which has heavy dependencies (mineru_clean_markdown)."""

    def __init__(self, token: str, api_base: str = API_BASE):
        self.token = token
        self.api_base = api_base
        self.timeout = 60
        self.retries = 8
        self.backoff = 2.0
        self.max_sleep = 120.0

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

    def _compute_sleep(self, attempt: int, retry_after: Optional[float] = None) -> float:
        if retry_after is not None:
            return max(0.0, min(float(retry_after), self.max_sleep))
        jitter = 0.7 + random.random() * 0.6
        return max(0.5, min(self.max_sleep, (self.backoff ** attempt) * jitter))

    def _request_json_sync(
        self, method: str, url: str, json_body: Optional[dict] = None
    ) -> dict[str, Any]:
        """Synchronous HTTP request with retry."""
        import requests
        retryable = {408, 425, 429, 500, 502, 503, 504}

        for attempt in range(self.retries + 1):
            try:
                resp = requests.request(
                    method, url, headers=self.headers,
                    json=json_body, timeout=self.timeout,
                )
                if resp.status_code in retryable and attempt < self.retries:
                    time.sleep(self._compute_sleep(attempt))
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception:
                if attempt == self.retries:
                    raise
                time.sleep(self._compute_sleep(attempt))
        raise RuntimeError(f"Request failed: {url}")

    def create_upload_urls_batch(
        self, files: list[dict[str, Any]], model_version: str = "vlm"
    ) -> tuple[str, list[str]]:
        payload = {"files": files, "model_version": model_version}
        data = self._request_json_sync("POST", f"{self.api_base}/file-urls/batch", payload)
        if data.get("code") != 0:
            raise RuntimeError(f"create_upload_urls_batch failed: {data}")
        return data["data"]["batch_id"], data["data"]["file_urls"]

    def get_batch_results(self, batch_id: str) -> dict[str, Any]:
        data = self._request_json_sync("GET", f"{self.api_base}/extract-results/batch/{batch_id}")
        if data.get("code") != 0:
            raise RuntimeError(f"get_batch_results failed: {data}")
        return data["data"]


def _put_upload_sync(upload_url: str, file_path: Path) -> None:
    """Upload file to presigned URL. No Content-Type header (MinerU requirement)."""
    import requests
    retryable = {408, 425, 429, 500, 502, 503, 504}

    for attempt in range(7):
        try:
            with open(file_path, "rb") as f:
                r = requests.put(upload_url, data=f, timeout=600)
            if r.status_code in retryable and attempt < 6:
                jitter = 0.7 + random.random() * 0.6
                time.sleep(max(0.5, min(120.0, (2.0 ** attempt) * jitter)))
                continue
            if r.status_code not in (200, 201):
                raise RuntimeError(f"Upload HTTP {r.status_code}: {r.text[:200]}")
            return
        except Exception:
            if attempt >= 6:
                raise
            time.sleep(max(0.5, min(120.0, (2.0 ** attempt) * 0.9)))
    raise RuntimeError(f"Upload failed: {file_path.name}")


def _download_file_sync(url: str, out_path: Path) -> None:
    """Download file with retry and atomic write."""
    import requests
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    retryable = {408, 425, 429, 500, 502, 503, 504}

    for attempt in range(7):
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                if r.status_code in retryable and attempt < 6:
                    time.sleep(max(0.5, min(120.0, (2.0 ** attempt) * 0.9)))
                    continue
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            tmp.replace(out_path)
            return
        except Exception:
            tmp.unlink(missing_ok=True)
            if attempt >= 6:
                raise
            time.sleep(max(0.5, min(120.0, (2.0 ** attempt) * 0.9)))
    raise RuntimeError(f"Download failed: {url}")


def _update_parse_poll_sync(
    parse_id: str,
    *,
    remote_batch_id: str = "",
    poll_count: int | None = None,
    state_counts: dict[str, int] | None = None,
) -> None:
    if not parse_id:
        return

    assignments = ["updated_at = datetime('now')"]
    params: list[Any] = []
    if remote_batch_id:
        assignments.append("remote_batch_id = ?")
        params.append(remote_batch_id)
    if poll_count is not None:
        assignments.append("poll_count = ?")
        params.append(poll_count)
    if state_counts is not None:
        assignments.append("last_state_counts = ?")
        params.append(json.dumps(state_counts, ensure_ascii=True))
        assignments.append("last_poll_at = datetime('now')")

    params.append(parse_id)
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(db.get_db_path(), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            f"UPDATE mineru_parses SET {', '.join(assignments)} WHERE parse_id = ?",
            params,
        )
        conn.commit()
    except Exception as exc:
        logger.debug("MinerU poll metadata update skipped parse_id=%s: %s", parse_id, exc)
    finally:
        if conn is not None:
            conn.close()


def _poll_until_done_sync(
    client: MinerUClient,
    batch_id: str,
    sleep_s: int = 6,
    timeout_s: int = 3600,
    *,
    time_fn: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], None] = time.sleep,
    on_poll: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Poll MinerU batch until all tasks are done or failed."""
    from collections import Counter
    t0 = time_fn()
    poll_count = 0
    last_state_counts: dict[str, int] = {}
    while True:
        poll_count += 1
        data = client.get_batch_results(batch_id)
        results = data.get("extract_result") or data.get("extract_results") or []
        if results:
            states = [r.get("state") for r in results]
            last_state_counts = dict(Counter(states))
            elapsed_s = time_fn() - t0
            if on_poll:
                on_poll({
                    "batch_id": batch_id,
                    "poll_count": poll_count,
                    "state_counts": last_state_counts,
                    "elapsed_s": elapsed_s,
                })
            if poll_count % 5 == 1:  # Log every 5th poll to avoid spam
                logger.info(
                    f"MinerU poll #{poll_count} batch={batch_id}: "
                    f"{last_state_counts} ({elapsed_s:.0f}s elapsed)"
                )
            if all(s in ("done", "failed") for s in states):
                logger.info(
                    f"MinerU poll DONE after {poll_count} polls in "
                    f"{elapsed_s:.0f}s: {last_state_counts}"
                )
                return results
        else:
            elapsed_s = time_fn() - t0
            if on_poll:
                on_poll({
                    "batch_id": batch_id,
                    "poll_count": poll_count,
                    "state_counts": {},
                    "elapsed_s": elapsed_s,
                })
        if elapsed_s >= timeout_s:
            raise MinerUPollTimeoutError(
                batch_id=batch_id,
                elapsed_s=elapsed_s,
                poll_count=poll_count,
                last_state_counts=last_state_counts,
                timeout_s=timeout_s,
            )
        sleep_fn(sleep_s)


def _get_client() -> MinerUClient:
    settings = get_settings()
    return MinerUClient(token=settings.mineru_token)


async def parse_pdf(paper_id: str, parse_id: str) -> Path:
    """Parse a single PDF via MinerU batch API. Returns the output directory."""
    settings = get_settings()
    paper_dir = settings.data_dir / "papers" / paper_id
    pdf_path = paper_dir / "original.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir = paper_dir / "mineru" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    await db.execute(
        "UPDATE mineru_parses SET status = 'running', updated_at = datetime('now') WHERE parse_id = ?",
        (parse_id,),
    )

    try:
        result_dir = await asyncio.to_thread(_parse_pdf_sync, pdf_path, output_dir, paper_id, parse_id)
        await db.execute(
            "UPDATE mineru_parses SET status = 'done', output_dir = ?, updated_at = datetime('now') WHERE parse_id = ?",
            (str(result_dir), parse_id),
        )
        return result_dir
    except Exception as e:
        await db.execute(
            "UPDATE mineru_parses SET status = 'failed', error_msg = ?, updated_at = datetime('now') WHERE parse_id = ?",
            (str(e), parse_id),
        )
        raise


def _parse_pdf_sync(pdf_path: Path, output_dir: Path, paper_id: str, parse_id: str) -> Path:
    """Synchronous MinerU parsing."""
    t_total = time.perf_counter()
    settings = get_settings()
    client = _get_client()
    data_id = paper_id[:20]
    files_payload = [{"name": pdf_path.name, "data_id": data_id}]

    # Step 1: Create upload URLs
    t0 = time.perf_counter()
    batch_id, upload_urls = client.create_upload_urls_batch(
        files=files_payload,
        model_version=settings.mineru_model_version,
    )
    _update_parse_poll_sync(parse_id, remote_batch_id=batch_id)
    logger.info(f"[{paper_id}] MinerU create_upload_urls: {time.perf_counter()-t0:.2f}s batch_id={batch_id}")

    # Step 2: Upload PDF
    t0 = time.perf_counter()
    size_mb = pdf_path.stat().st_size / 1024 / 1024
    _put_upload_sync(upload_urls[0], pdf_path)
    logger.info(f"[{paper_id}] MinerU upload: {time.perf_counter()-t0:.2f}s ({size_mb:.1f} MB)")

    # Step 3: Poll until done
    t0 = time.perf_counter()
    results = _poll_until_done_sync(
        client,
        batch_id,
        sleep_s=max(1, settings.mineru_poll_interval_seconds),
        timeout_s=max(1, settings.mineru_parse_timeout_seconds),
        on_poll=lambda event: _update_parse_poll_sync(
            parse_id,
            remote_batch_id=batch_id,
            poll_count=int(event["poll_count"]),
            state_counts=event["state_counts"],
        ),
    )
    logger.info(f"[{paper_id}] MinerU poll: {time.perf_counter()-t0:.1f}s (remote processing)")

    for r in results:
        if r.get("state") == "failed":
            raise RuntimeError(f"MinerU parse failed: {r.get('err_msg', 'unknown')}")

        zip_url = r.get("full_zip_url")
        if not zip_url:
            raise RuntimeError("No zip URL in MinerU response")

        # Step 4: Download zip
        t0 = time.perf_counter()
        zip_path = output_dir / f"{data_id}.zip"
        _download_file_sync(zip_url, zip_path)
        zip_mb = zip_path.stat().st_size / 1024 / 1024
        logger.info(f"[{paper_id}] MinerU download: {time.perf_counter()-t0:.2f}s ({zip_mb:.1f} MB)")

        # Step 5: Extract zip
        t0 = time.perf_counter()
        extract_dir = output_dir / data_id
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            _safe_zip_extract(zf, extract_dir)
        n_files = sum(1 for _ in extract_dir.rglob("*") if _.is_file())
        logger.info(f"[{paper_id}] MinerU extract: {time.perf_counter()-t0:.2f}s ({n_files} files)")

        logger.info(f"[{paper_id}] MinerU TOTAL: {time.perf_counter()-t_total:.1f}s")
        return extract_dir

    raise RuntimeError("No results from MinerU batch")


async def parse_pdf_batch(paper_ids: list[str], parse_ids: list[str]) -> list[Path]:
    """Parse multiple PDFs in a single MinerU batch."""
    settings = get_settings()

    pdf_paths: list[Path] = []
    data_ids: list[str] = []
    files_payload: list[dict[str, Any]] = []

    for paper_id in paper_ids:
        paper_dir = settings.data_dir / "papers" / paper_id
        pdf_path = paper_dir / "original.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        data_id = paper_id[:20]
        pdf_paths.append(pdf_path)
        data_ids.append(data_id)
        files_payload.append({"name": pdf_path.name, "data_id": data_id})

    for parse_id in parse_ids:
        await db.execute(
            "UPDATE mineru_parses SET status = 'running', updated_at = datetime('now') WHERE parse_id = ?",
            (parse_id,),
        )

    def _batch_sync() -> list[Path]:
        client = _get_client()
        batch_id, upload_urls = client.create_upload_urls_batch(
            files=files_payload,
            model_version=settings.mineru_model_version,
        )
        for parse_id in parse_ids:
            _update_parse_poll_sync(parse_id, remote_batch_id=batch_id)
        for p, u in zip(pdf_paths, upload_urls):
            _put_upload_sync(u, p)

        def _on_batch_poll(event: dict[str, Any]) -> None:
            for parse_id in parse_ids:
                _update_parse_poll_sync(
                    parse_id,
                    remote_batch_id=batch_id,
                    poll_count=int(event["poll_count"]),
                    state_counts=event["state_counts"],
                )

        results = _poll_until_done_sync(
            client,
            batch_id,
            sleep_s=max(1, settings.mineru_poll_interval_seconds),
            timeout_s=max(1, settings.mineru_batch_timeout_seconds),
            on_poll=_on_batch_poll,
        )

        out_dirs: list[Path] = []
        for r, pid, did in zip(results, paper_ids, data_ids):
            out = settings.data_dir / "papers" / pid / "mineru" / "raw"
            out.mkdir(parents=True, exist_ok=True)

            if r.get("state") == "failed":
                out_dirs.append(out)
                continue

            zip_url = r.get("full_zip_url")
            if not zip_url:
                out_dirs.append(out)
                continue

            zp = out / f"{did}.zip"
            _download_file_sync(zip_url, zp)

            ed = out / did
            if ed.exists():
                shutil.rmtree(ed)
            with zipfile.ZipFile(zp, "r") as zf:
                _safe_zip_extract(zf, ed)
            out_dirs.append(ed)

        return out_dirs

    result_dirs = await asyncio.to_thread(_batch_sync)

    for parse_id, result_dir in zip(parse_ids, result_dirs):
        has_content = (result_dir / "content_list.json").exists() or list(result_dir.glob("**/content_list.json"))
        if has_content:
            await db.execute(
                "UPDATE mineru_parses SET status = 'done', output_dir = ?, updated_at = datetime('now') WHERE parse_id = ?",
                (str(result_dir), parse_id),
            )
        else:
            await db.execute(
                "UPDATE mineru_parses SET status = 'failed', error_msg = 'No content_list.json found', updated_at = datetime('now') WHERE parse_id = ?",
                (parse_id,),
            )

    return result_dirs
