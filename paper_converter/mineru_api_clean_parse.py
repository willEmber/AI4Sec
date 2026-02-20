#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MinerU Official API: parse PDF -> download zip -> extract -> clean -> export *_clean.md

Supports:
1) URL mode: POST /api/v4/extract/task (URL only; does NOT support direct file upload)
2) Local batch upload: POST /api/v4/file-urls/batch -> PUT upload -> auto submit -> poll /api/v4/extract-results/batch/{batch_id}

Cleaning goals (for RAG / Dify):
- Remove boilerplate (IEEE/ACM copyright / permissions / DOI / ISSN, etc.)
- Remove repeated header/footer-like short blocks across many pages
- Truncate at "References/Bibliography/参考文献/致谢/Acknowledgements"
- Drop reference-like entries (optional heuristic)
- Keep equations / captions in lightweight form

Notes:
- MinerU outputs zip that contains markdown/json by default (extra formats optional).
- Prefer content_list.json for deterministic cleaning (reading-order flattened list).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse
from collections import Counter

import requests

from mineru_clean_markdown import CleanConfig, process_zip_to_clean_md


API_BASE = "https://mineru.net/api/v4"
STATE_VERSION = 1

LOG_FILE: Optional[Path] = None


def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def log(msg: str, *, level: str = "INFO") -> None:
    line = f"{_now_str()} [{level}] {msg}"
    print(line, flush=True)
    if LOG_FILE:
        try:
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            # Logging must never break the pipeline.
            pass


def human_bytes(n: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n)
    for u in units:
        if abs(v) < 1024.0 or u == units[-1]:
            return f"{v:.1f}{u}" if u != "B" else f"{int(v)}B"
        v /= 1024.0
    return f"{v:.1f}TB"


def human_duration(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def save_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))


def safe_load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def results_list_from_batch_data(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return data.get("extract_result") or data.get("extract_results") or []


def build_state_index(files: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for f in files:
        data_id = f.get("data_id")
        if isinstance(data_id, str) and data_id:
            idx[data_id] = f
    return idx


def default_state_file(args: argparse.Namespace, mode: str) -> Path:
    if getattr(args, "state_file", None):
        return Path(args.state_file)
    return Path(args.workdir) / f"state_{mode}.json"


def load_processed_local_metas(out_root: Path) -> Dict[str, Any]:
    processed_relpaths: set = set()
    processed_abspaths: set = set()
    processed_file_names: set = set()
    processed_data_ids: set = set()

    for meta_path in sorted(out_root.glob("*.meta.json")):
        meta = safe_load_json(meta_path)
        if not isinstance(meta, dict):
            continue
        if meta.get("mode") != "local":
            continue
        rel = meta.get("source_pdf_relpath")
        absp = meta.get("source_pdf_path")
        name = meta.get("file_name")
        data_id = meta.get("data_id")
        if isinstance(rel, str) and rel:
            processed_relpaths.add(rel)
        if isinstance(absp, str) and absp:
            processed_abspaths.add(absp)
        if isinstance(name, str) and name:
            processed_file_names.add(name)
        if isinstance(data_id, str) and data_id:
            processed_data_ids.add(data_id)

    return {
        "relpaths": processed_relpaths,
        "abspaths": processed_abspaths,
        "file_names": processed_file_names,
        "data_ids": processed_data_ids,
    }


def guess_stem_from_url(file_url: str) -> str:
    try:
        parsed = urlparse(file_url)
        name = Path(parsed.path).name
        if not name:
            return ""
        name = unquote(name)
        return Path(name).stem or name
    except Exception:
        return ""


def pick_unique_path(path: Path, *, salt: str, max_tries: int = 50) -> Path:
    if not path.exists():
        return path

    h = sha1_bytes(salt.encode("utf-8"))[:8]
    stem = path.stem
    suffix = path.suffix
    for i in range(0, max_tries + 1):
        extra = f"__{h}" if i == 0 else f"__{h}_{i}"
        cand = path.with_name(f"{stem}{extra}{suffix}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"Failed to pick unique path for: {path}")


def parse_retry_after_s(resp: requests.Response) -> Optional[float]:
    ra = resp.headers.get("Retry-After")
    if not ra:
        return None
    try:
        return float(ra)
    except ValueError:
        return None


def parse_json_retry_after_s(data: Any) -> Optional[float]:
    if not isinstance(data, dict):
        return None

    top = data.get("retry_after") or data.get("retryAfter")
    if isinstance(top, (int, float)):
        return float(top)

    d = data.get("data")
    if isinstance(d, dict):
        for k in ("retry_after", "retryAfter", "retry_after_s", "retry_after_seconds"):
            v = d.get(k)
            if isinstance(v, (int, float)):
                return float(v)
    return None


def is_retryable_mineru_error(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    code = data.get("code")
    if isinstance(code, int) and code in (429, 500, 502, 503, 504):
        return True
    msg = str(data.get("msg") or data.get("message") or "")
    msg_l = msg.lower()
    return any(
        key in msg_l
        for key in (
            "rate limit",
            "too many",
            "too frequent",
            "frequency",
            "throttle",
            "busy",
            "temporarily unavailable",
            "限速",
            "频繁",
            "过于频繁",
        )
    )


def compute_sleep_s(
    attempt: int,
    *,
    backoff: float,
    max_sleep_s: float,
    retry_after_s: Optional[float] = None,
) -> float:
    if retry_after_s is not None:
        return max(0.0, min(float(retry_after_s), max_sleep_s))
    jitter = 0.7 + random.random() * 0.6
    return max(0.5, min(max_sleep_s, (backoff ** attempt) * jitter))


def request_json(
    method: str,
    url: str,
    headers: Dict[str, str],
    json_body: Optional[Dict[str, Any]] = None,
    timeout: int = 60,
    retries: int = 8,
    backoff: float = 2.0,
    max_sleep_s: float = 120.0,
) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    retryable_statuses = {408, 425, 429, 500, 502, 503, 504}
    for attempt in range(retries + 1):
        try:
            resp = requests.request(
                method,
                url,
                headers=headers,
                json=json_body,
                timeout=timeout,
            )

            if resp.status_code in retryable_statuses:
                retry_after_s = parse_retry_after_s(resp)
                if attempt < retries:
                    sleep_s = compute_sleep_s(
                        attempt,
                        backoff=backoff,
                        max_sleep_s=max_sleep_s,
                        retry_after_s=retry_after_s,
                    )
                    if resp.status_code == 429:
                        log(f"RATE_LIMIT HTTP 429. sleep={sleep_s:.1f}s then retry ({attempt+1}/{retries})", level="WARN")
                    else:
                        log(
                            f"RETRY HTTP {resp.status_code}. sleep={sleep_s:.1f}s then retry ({attempt+1}/{retries})",
                            level="WARN",
                        )
                    time.sleep(sleep_s)
                    continue

            resp.raise_for_status()
            data = resp.json()

            if is_retryable_mineru_error(data):
                retry_after_s = parse_json_retry_after_s(data)
                if attempt < retries:
                    sleep_s = compute_sleep_s(
                        attempt,
                        backoff=backoff,
                        max_sleep_s=max_sleep_s,
                        retry_after_s=retry_after_s,
                    )
                    log(f"RATE_LIMIT API response. sleep={sleep_s:.1f}s then retry ({attempt+1}/{retries})", level="WARN")
                    time.sleep(sleep_s)
                    continue

            return data
        except Exception as e:
            last_err = e
            if attempt == retries:
                break
            sleep_s = compute_sleep_s(attempt, backoff=backoff, max_sleep_s=max_sleep_s)
            log(
                f"RETRY exception={type(e).__name__}. sleep={sleep_s:.1f}s then retry ({attempt+1}/{retries})",
                level="WARN",
            )
            time.sleep(sleep_s)
    raise RuntimeError(f"Request failed after retries: {url}. Last error: {last_err}")


def download_file(
    url: str,
    out_path: Path,
    *,
    timeout: int = 300,
    retries: int = 6,
    backoff: float = 2.0,
    max_sleep_s: float = 120.0,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    retryable_statuses = {408, 425, 429, 500, 502, 503, 504}

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                if r.status_code in retryable_statuses:
                    if attempt < retries:
                        retry_after_s = parse_retry_after_s(r)
                        sleep_s = compute_sleep_s(
                            attempt,
                            backoff=backoff,
                            max_sleep_s=max_sleep_s,
                            retry_after_s=retry_after_s,
                        )
                        if r.status_code == 429:
                            log(
                                f"RATE_LIMIT download 429. sleep={sleep_s:.1f}s then retry ({attempt+1}/{retries})",
                                level="WARN",
                            )
                        else:
                            log(
                                f"RETRY download HTTP {r.status_code}. sleep={sleep_s:.1f}s then retry ({attempt+1}/{retries})",
                                level="WARN",
                            )
                        time.sleep(sleep_s)
                        continue

                r.raise_for_status()

                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

            tmp_path.replace(out_path)
            return
        except Exception as e:
            last_err = e
            if attempt == retries:
                break
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            sleep_s = compute_sleep_s(attempt, backoff=backoff, max_sleep_s=max_sleep_s)
            log(
                f"RETRY download exception={type(e).__name__}. sleep={sleep_s:.1f}s then retry ({attempt+1}/{retries})",
                level="WARN",
            )
            time.sleep(sleep_s)

    raise RuntimeError(f"Download failed after retries: {url}. Last error: {last_err}")


# ---------------- MinerU API client ----------------

@dataclasses.dataclass
class MinerUClient:
    token: str
    api_base: str = API_BASE
    timeout_s: int = 60
    retries: int = 8
    backoff: float = 2.0
    max_sleep_s: float = 120.0

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

    # URL single file -> task_id
    def create_task_from_url(self, file_url: str, model_version: str = "vlm", data_id: Optional[str] = None) -> str:
        payload: Dict[str, Any] = {
            "url": file_url,
            "model_version": model_version,
        }
        if data_id:
            payload["data_id"] = data_id
        data = request_json(
            "POST",
            f"{self.api_base}/extract/task",
            headers=self.headers,
            json_body=payload,
            timeout=self.timeout_s,
            retries=self.retries,
            backoff=self.backoff,
            max_sleep_s=self.max_sleep_s,
        )
        if data.get("code") != 0:
            raise RuntimeError(f"create_task_from_url failed: {data}")
        return data["data"]["task_id"]

    def get_task(self, task_id: str) -> Dict[str, Any]:
        data = request_json(
            "GET",
            f"{self.api_base}/extract/task/{task_id}",
            headers=self.headers,
            timeout=self.timeout_s,
            retries=self.retries,
            backoff=self.backoff,
            max_sleep_s=self.max_sleep_s,
        )
        if data.get("code") != 0:
            raise RuntimeError(f"get_task failed: {data}")
        return data["data"]

    # Local batch upload -> batch_id + upload urls
    def create_upload_urls_batch(
        self,
        files: List[Dict[str, Any]],
        model_version: str = "vlm",
        enable_formula: Optional[bool] = None,
        enable_table: Optional[bool] = None,
        language: Optional[str] = None,
    ) -> Tuple[str, List[str]]:
        payload: Dict[str, Any] = {
            "files": files,
            "model_version": model_version,
        }
        if enable_formula is not None:
            payload["enable_formula"] = enable_formula
        if enable_table is not None:
            payload["enable_table"] = enable_table
        if language is not None:
            payload["language"] = language

        data = request_json(
            "POST",
            f"{self.api_base}/file-urls/batch",
            headers=self.headers,
            json_body=payload,
            timeout=self.timeout_s,
            retries=self.retries,
            backoff=self.backoff,
            max_sleep_s=self.max_sleep_s,
        )
        if data.get("code") != 0:
            raise RuntimeError(f"create_upload_urls_batch failed: {data}")
        batch_id = data["data"]["batch_id"]
        file_urls = data["data"]["file_urls"]
        return batch_id, file_urls

    def get_batch_results(self, batch_id: str) -> Dict[str, Any]:
        data = request_json(
            "GET",
            f"{self.api_base}/extract-results/batch/{batch_id}",
            headers=self.headers,
            timeout=self.timeout_s,
            retries=self.retries,
            backoff=self.backoff,
            max_sleep_s=self.max_sleep_s,
        )
        if data.get("code") != 0:
            raise RuntimeError(f"get_batch_results failed: {data}")
        return data["data"]


def put_upload(
    upload_url: str,
    file_path: Path,
    *,
    timeout: int = 600,
    retries: int = 6,
    backoff: float = 2.0,
    max_sleep_s: float = 120.0,
) -> None:
    # Docs note: uploading does NOT need Content-Type header (simple PUT) in batch upload mode.
    retryable_statuses = {408, 425, 429, 500, 502, 503, 504}

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            with open(file_path, "rb") as f:
                r = requests.put(upload_url, data=f, timeout=timeout)

            if r.status_code in retryable_statuses:
                if attempt < retries:
                    retry_after_s = parse_retry_after_s(r)
                    sleep_s = compute_sleep_s(
                        attempt,
                        backoff=backoff,
                        max_sleep_s=max_sleep_s,
                        retry_after_s=retry_after_s,
                    )
                    if r.status_code == 429:
                        log(
                            f"RATE_LIMIT upload 429. sleep={sleep_s:.1f}s then retry ({attempt+1}/{retries})",
                            level="WARN",
                        )
                    else:
                        log(
                            f"RETRY upload HTTP {r.status_code}. sleep={sleep_s:.1f}s then retry ({attempt+1}/{retries})",
                            level="WARN",
                        )
                    time.sleep(sleep_s)
                    continue

            if r.status_code not in (200, 201):
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            return
        except Exception as e:
            last_err = e
            if attempt == retries:
                break
            sleep_s = compute_sleep_s(attempt, backoff=backoff, max_sleep_s=max_sleep_s)
            log(
                f"RETRY upload exception={type(e).__name__}. sleep={sleep_s:.1f}s then retry ({attempt+1}/{retries})",
                level="WARN",
            )
            time.sleep(sleep_s)

    raise RuntimeError(f"Upload failed after retries: {file_path.name}. Last error: {last_err}")


# ---------------- Pipeline: parse -> download -> clean ----------------

def poll_until_done_task(
    client: MinerUClient,
    task_id: str,
    sleep_s: int = 6,
    timeout_s: int = 3600,
) -> Dict[str, Any]:
    t0 = time.time()
    last_state: Optional[str] = None
    while True:
        data = client.get_task(task_id)
        state = data.get("state")
        if state != last_state:
            elapsed = time.time() - t0
            log(f"TASK task_id={task_id} state={state} elapsed={human_duration(elapsed)}")
            last_state = state
        if state == "done":
            return data
        if state == "failed":
            raise RuntimeError(f"Task failed: task_id={task_id}, err_msg={data.get('err_msg')}")
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Task timeout: task_id={task_id}, last_state={state}")
        time.sleep(sleep_s)


def poll_until_done_batch(
    client: MinerUClient,
    batch_id: str,
    sleep_s: int = 8,
    timeout_s: int = 7200,
) -> List[Dict[str, Any]]:
    t0 = time.time()
    last_summary: Optional[str] = None
    while True:
        data = client.get_batch_results(batch_id)
        results = results_list_from_batch_data(data)
        if not results:
            # Some API responses may differ; keep polling
            pass
        states = [r.get("state") for r in results]
        counts = Counter([s or "unknown" for s in states])
        done = counts.get("done", 0)
        failed = counts.get("failed", 0)
        total = len(results) if results else 0
        elapsed = time.time() - t0
        summary = f"{done}/{total} done, {failed} failed"
        if summary != last_summary:
            rate = (done / elapsed * 60.0) if elapsed > 0 else 0.0
            log(
                f"BATCH batch_id={batch_id} progress={summary} elapsed={human_duration(elapsed)} remote_rate={rate:.2f}/min"
            )
            last_summary = summary
        if results and all(s in ("done", "failed") for s in states):
            return results
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Batch timeout: batch_id={batch_id}, states={states[:10]}")
        time.sleep(sleep_s)


def poll_and_process_batch(
    *,
    client: MinerUClient,
    batch_id: str,
    jobs: List[Dict[str, Any]],
    state: Dict[str, Any],
    cfg: CleanConfig,
    out_root: Path,
    zip_dir: Path,
    state_path: Path,
    args: argparse.Namespace,
) -> None:
    t0 = time.time()
    out_root.mkdir(parents=True, exist_ok=True)
    zip_dir.mkdir(parents=True, exist_ok=True)

    idx = build_state_index(jobs)
    total = len(jobs)

    def cleaned_count() -> int:
        return sum(1 for j in jobs if j.get("cleaned") is True)

    last_poll_t = t0
    last_remote_done = 0
    last_local_cleaned = 0
    try:
        while True:
            elapsed = time.time() - t0
            if elapsed > args.timeout_s:
                raise TimeoutError(f"Batch timeout: batch_id={batch_id}")

            data = client.get_batch_results(batch_id)
            results = results_list_from_batch_data(data)

            states = [r.get("state") for r in results]
            counts = Counter([s or "unknown" for s in states])
            remote_done = counts.get("done", 0)
            remote_failed = counts.get("failed", 0)
            remote_total = len(results) if results else 0
            local_cleaned = cleaned_count()

            now = time.time()
            dt = max(0.001, now - last_poll_t)
            remote_delta = remote_done - last_remote_done
            local_delta = local_cleaned - last_local_cleaned

            remote_rate_avg = (remote_done / elapsed * 60.0) if elapsed > 0 else 0.0
            local_rate_avg = (local_cleaned / elapsed * 60.0) if elapsed > 0 else 0.0
            remote_rate_inst = remote_delta / dt * 60.0
            local_rate_inst = local_delta / dt * 60.0

            eta_s: Optional[float] = None
            if local_rate_avg > 0 and total > local_cleaned:
                eta_s = (total - local_cleaned) / (local_rate_avg / 60.0)

            eta_str = human_duration(eta_s) if eta_s is not None else "n/a"
            log(
                "POLL "
                f"batch_id={batch_id} "
                f"remote_done={remote_done}/{remote_total} remote_failed={remote_failed} "
                f"local_cleaned={local_cleaned}/{total} "
                f"elapsed={human_duration(elapsed)} eta={eta_str} "
                f"remote_rate(avg/inst)={remote_rate_avg:.2f}/{remote_rate_inst:.2f}/min "
                f"local_rate(avg/inst)={local_rate_avg:.2f}/{local_rate_inst:.2f}/min"
            )
            last_poll_t = now
            last_remote_done = remote_done
            last_local_cleaned = local_cleaned

            state["updated_at"] = _now_iso()
            state["batch_id"] = batch_id
            state["stage"] = "polling_processing"
            state["summary"] = {
                "remote_total": remote_total,
                "remote_done": remote_done,
                "remote_failed": remote_failed,
                "local_cleaned": local_cleaned,
                "elapsed_s": elapsed,
            }
            save_json(state_path, state)

            # Update jobs with remote states, and process newly done items.
            for r in results:
                name = r.get("file_name") or "unknown.pdf"
                data_id = r.get("data_id") or Path(name).stem
                job = idx.get(str(data_id))
                if not job:
                    continue

                job["result_state"] = r.get("state")
                if r.get("err_msg"):
                    job["err_msg"] = r.get("err_msg")

                zip_url = r.get("full_zip_url")
                if isinstance(zip_url, str) and zip_url:
                    job["zip_url"] = zip_url

                if job.get("result_state") == "failed":
                    job.setdefault("last_error", job.get("err_msg") or "remote_failed")
                    continue

                if job.get("result_state") != "done":
                    continue

                if job.get("cleaned") is True:
                    continue

                if not zip_url:
                    job.setdefault("last_error", "missing_full_zip_url")
                    continue

                try:
                    zip_path = zip_dir / f"{data_id}.zip"
                    job["zip_path"] = str(zip_path)

                    # download zip (skip if already downloaded)
                    dl_t0 = time.time()
                    if not zip_path.exists() or zip_path.stat().st_size == 0:
                        download_file(
                            zip_url,
                            zip_path,
                            timeout=args.io_timeout,
                            retries=args.io_retries,
                            backoff=args.api_backoff,
                            max_sleep_s=args.api_max_sleep,
                        )
                    dl_s = time.time() - dl_t0

                    paper_stem = Path(name).stem
                    clean_md_path_str = job.get("clean_md_path")
                    clean_md_path = (
                        Path(clean_md_path_str)
                        if isinstance(clean_md_path_str, str) and clean_md_path_str
                        else None
                    )
                    if clean_md_path is None:
                        clean_md_path = pick_unique_path(out_root / f"{paper_stem}_clean.md", salt=str(data_id))
                        job["clean_md_path"] = str(clean_md_path)

                    raw_dir = out_root / "_raw" / str(data_id)
                    clean_t0 = time.time()
                    clean_md_path = process_zip_to_clean_md(zip_path, raw_dir, cfg, clean_path=clean_md_path)
                    clean_s = time.time() - clean_t0

                    meta_path = clean_md_path.with_suffix(".meta.json")
                    meta = {
                        "mode": "local",
                        "file_name": name,
                        "data_id": str(data_id),
                        "batch_id": batch_id,
                        "model_version": args.model_version,
                        "zip_url": zip_url,
                        "source_pdf_path": job.get("source_pdf_path"),
                        "source_pdf_relpath": job.get("source_pdf_relpath"),
                        "timing": {
                            "download_s": dl_s,
                            "clean_s": clean_s,
                        },
                    }
                    save_json(meta_path, meta)

                    job["meta_path"] = str(meta_path)
                    job["cleaned"] = True
                    job["cleaned_at"] = _now_iso()
                    job.pop("last_error", None)

                    local_cleaned = cleaned_count()
                    local_rate = (local_cleaned / (time.time() - t0) * 60.0) if (time.time() - t0) > 0 else 0.0
                    log(
                        f"CLEAN {local_cleaned}/{total} name={name} zip={zip_path.name} dl={human_duration(dl_s)} clean={human_duration(clean_s)} local_rate={local_rate:.2f}/min"
                    )
                    state["updated_at"] = _now_iso()
                    state["summary"]["local_cleaned"] = local_cleaned
                    save_json(state_path, state)
                except Exception as e:
                    job["last_error"] = f"{type(e).__name__}: {e}"
                    state["updated_at"] = _now_iso()
                    save_json(state_path, state)
                    log(f"PROCESS_ERROR name={name} err={e}", level="WARN")
                    continue

            # stop if all remote tasks are terminal (done/failed)
            if results and all((r.get("state") in ("done", "failed")) for r in results):
                break

            time.sleep(args.poll_s)

    except KeyboardInterrupt:
        state["updated_at"] = _now_iso()
        state["batch_id"] = batch_id
        state["stage"] = "interrupted"
        save_json(state_path, state)
        log(f"INTERRUPT batch_id={batch_id}. state saved to {state_path}", level="WARN")
        raise

    elapsed = time.time() - t0
    local_cleaned = cleaned_count()
    pending_local = [
        j for j in jobs
        if j.get("result_state") == "done" and j.get("cleaned") is not True
    ]
    stage = "done" if not pending_local else "pending_local"
    level = "INFO" if stage == "done" else "WARN"
    log(
        f"DONE batch_id={batch_id} stage={stage} local_cleaned={local_cleaned}/{total} pending_local={len(pending_local)} "
        f"elapsed={human_duration(elapsed)} avg_local_rate={(local_cleaned/elapsed*60.0 if elapsed>0 else 0.0):.2f}/min",
        level=level,
    )
    state["updated_at"] = _now_iso()
    state["batch_id"] = batch_id
    state["stage"] = stage
    state["summary"] = {
        "local_cleaned": local_cleaned,
        "pending_local": len(pending_local),
        "total": total,
        "elapsed_s": elapsed,
    }
    save_json(state_path, state)


def run_url_mode(args: argparse.Namespace) -> None:
    client = MinerUClient(
        token=args.token,
        timeout_s=args.api_timeout,
        retries=args.api_retries,
        backoff=args.api_backoff,
        max_sleep_s=args.api_max_sleep,
    )
    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    state_path = default_state_file(args, "url")
    state: Dict[str, Any] = {
        "version": STATE_VERSION,
        "mode": "url",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "stage": "init",
        "url": args.url,
        "outdir": str(out_root.resolve()),
        "workdir": str(Path(args.workdir).resolve()),
        "model_version": args.model_version,
    }

    data_id = args.data_id or f"url_{sha1_bytes(args.url.encode('utf-8'))[:10]}"
    state["data_id"] = data_id

    # Resume if possible
    if args.resume and state_path.exists():
        prev = safe_load_json(state_path)
        if isinstance(prev, dict) and prev.get("mode") == "url" and prev.get("url") == args.url:
            same_outdir = str(out_root.resolve()) == str(Path(prev.get("outdir", "")).resolve()) if prev.get("outdir") else False
            if same_outdir and prev.get("stage") not in ("done", "completed"):
                state = prev
                log(f"RESUME state={state_path} data_id={state.get('data_id')} stage={state.get('stage')}", level="WARN")

                clean_md_path_str = state.get("clean_md_path")
                if isinstance(clean_md_path_str, str) and clean_md_path_str and Path(clean_md_path_str).exists():
                    log(f"URL already cleaned: {clean_md_path_str}")
                    return

    data_id = str(state.get("data_id") or data_id)

    task_id = state.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        task_id = client.create_task_from_url(args.url, model_version=args.model_version, data_id=data_id)
        state["task_id"] = task_id
        state["stage"] = "task_created"
        state["updated_at"] = _now_iso()
        save_json(state_path, state)
    log(f"URL task_id={task_id}")

    try:
        task = poll_until_done_task(client, task_id, sleep_s=args.poll_s, timeout_s=args.timeout_s)
        zip_url = task.get("full_zip_url")
        if not zip_url:
            raise RuntimeError(f"No full_zip_url in done task: {task}")
        state["zip_url"] = zip_url
        state["stage"] = "task_done"
        state["updated_at"] = _now_iso()
        save_json(state_path, state)
        log(f"URL zip_url={zip_url}")

        # download zip
        zip_dir = Path(args.workdir) / "zips"
        zip_path = zip_dir / f"{data_id}.zip"
        state["zip_path"] = str(zip_path)
        if not zip_path.exists() or zip_path.stat().st_size == 0:
            download_file(
                zip_url,
                zip_path,
                timeout=args.io_timeout,
                retries=args.io_retries,
                backoff=args.api_backoff,
                max_sleep_s=args.api_max_sleep,
            )
        state["stage"] = "zip_downloaded"
        state["updated_at"] = _now_iso()
        save_json(state_path, state)
        log(f"URL downloaded zip={zip_path}")

        # clean
        paper_stem = guess_stem_from_url(args.url) or data_id
        clean_md_path_str = state.get("clean_md_path")
        clean_md_path = (
            Path(clean_md_path_str)
            if isinstance(clean_md_path_str, str) and clean_md_path_str
            else pick_unique_path(out_root / f"{paper_stem}_clean.md", salt=data_id)
        )
        raw_dir = out_root / "_raw" / data_id
        cfg = CleanConfig(
            repeated_block_page_ratio=args.repeat_ratio,
            repeated_block_max_chars=args.repeat_max_chars,
        )
        clean_md_path = process_zip_to_clean_md(zip_path, raw_dir, cfg, clean_path=clean_md_path)
        state["clean_md_path"] = str(clean_md_path)
        state["stage"] = "cleaned"
        state["updated_at"] = _now_iso()
        save_json(state_path, state)

        meta = {
            "mode": "url",
            "url": args.url,
            "task_id": task_id,
            "data_id": data_id,
            "model_version": args.model_version,
            "zip_url": zip_url,
        }
        save_json(clean_md_path.with_suffix(".meta.json"), meta)
        log(f"URL clean_md={clean_md_path}")

        state["stage"] = "done"
        state["updated_at"] = _now_iso()
        save_json(state_path, state)
    except KeyboardInterrupt:
        state["stage"] = "interrupted"
        state["updated_at"] = _now_iso()
        save_json(state_path, state)
        log(f"INTERRUPT url state saved to {state_path}", level="WARN")
        raise


def run_local_batch_mode(args: argparse.Namespace) -> None:
    client = MinerUClient(
        token=args.token,
        timeout_s=args.api_timeout,
        retries=args.api_retries,
        backoff=args.api_backoff,
        max_sleep_s=args.api_max_sleep,
    )
    pdf_dir = Path(args.pdf_dir)
    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    state_path = default_state_file(args, "local")

    # Resume from state (best-effort)
    if args.resume and state_path.exists():
        prev = safe_load_json(state_path)
        if isinstance(prev, dict) and prev.get("mode") == "local" and prev.get("batch_id"):
            same_pdf_dir = str(pdf_dir.resolve()) == str(Path(prev.get("pdf_dir", "")).resolve()) if prev.get("pdf_dir") else False
            same_outdir = str(out_root.resolve()) == str(Path(prev.get("outdir", "")).resolve()) if prev.get("outdir") else False
            if same_pdf_dir and same_outdir and prev.get("stage") not in ("done", "completed"):
                batch_id = str(prev["batch_id"])
                jobs = prev.get("files") if isinstance(prev.get("files"), list) else []
                if jobs:
                    log(f"RESUME state={state_path} batch_id={batch_id} files={len(jobs)}", level="WARN")

                    # Continue uploads if needed (URLs may expire; best-effort)
                    pending_uploads = [
                        j for j in jobs
                        if j.get("upload_state") != "uploaded" and j.get("upload_url") and j.get("source_pdf_path")
                    ]
                    if pending_uploads:
                        log(f"RESUME upload pending={len(pending_uploads)}")
                        total_bytes = 0
                        up_t0 = time.time()
                        for i, j in enumerate(pending_uploads, start=1):
                            p = Path(j["source_pdf_path"])
                            if not p.exists():
                                j["upload_state"] = "failed"
                                j["last_error"] = "source_pdf_missing"
                                save_json(state_path, prev)
                                continue
                            size = p.stat().st_size
                            log(f"UPLOAD {i}/{len(pending_uploads)} name={p.name} size={human_bytes(size)}")
                            try:
                                put_upload(
                                    str(j["upload_url"]),
                                    p,
                                    timeout=args.io_timeout,
                                    retries=args.io_retries,
                                    backoff=args.api_backoff,
                                    max_sleep_s=args.api_max_sleep,
                                )
                                j["upload_state"] = "uploaded"
                                j["uploaded_at"] = _now_iso()
                                total_bytes += size
                            except Exception as e:
                                j["upload_state"] = "failed"
                                j["last_error"] = f"upload_error: {e}"
                            prev["updated_at"] = _now_iso()
                            save_json(state_path, prev)

                        up_elapsed = time.time() - up_t0
                        log(
                            f"UPLOAD resume_done files={len(pending_uploads)} bytes={human_bytes(total_bytes)} elapsed={human_duration(up_elapsed)} avg_speed={(human_bytes(total_bytes/up_elapsed)+'/s' if up_elapsed>0 else 'n/a')}"
                        )

                    cfg = CleanConfig(
                        repeated_block_page_ratio=args.repeat_ratio,
                        repeated_block_max_chars=args.repeat_max_chars,
                    )
                    zip_dir = Path(args.workdir) / "zips"
                    poll_and_process_batch(
                        client=client,
                        batch_id=batch_id,
                        jobs=jobs,
                        state=prev,
                        cfg=cfg,
                        out_root=out_root,
                        zip_dir=zip_dir,
                        state_path=state_path,
                        args=args,
                    )
                    return

    # Fresh run
    pdfs = sorted([p for p in pdf_dir.rglob("*.pdf") if p.is_file()])
    if not pdfs:
        raise RuntimeError(f"No PDFs found under: {pdf_dir}")

    if args.resume:
        processed = load_processed_local_metas(out_root)
        rel_done = processed["relpaths"]
        abs_done = processed["abspaths"]
        name_done = processed["file_names"]
        name_counts = Counter([p.name for p in pdfs])
        before = len(pdfs)
        kept: List[Path] = []
        for p in pdfs:
            rel = str(p.relative_to(pdf_dir))
            absp = str(p.resolve())
            if rel in rel_done or absp in abs_done:
                continue
            if name_counts.get(p.name, 0) == 1 and p.name in name_done:
                continue
            kept.append(p)
        pdfs = kept
        skipped = before - len(pdfs)
        if skipped:
            log(f"RESUME skip_existing={skipped} remaining={len(pdfs)}")

    # Respect API limit: max 200 per batch (docs)
    if args.max_files is not None:
        pdfs = pdfs[: args.max_files]
    if not pdfs:
        log("No remaining PDFs to process (all done).")
        return
    if len(pdfs) > 200:
        raise RuntimeError("Too many files for one batch (>200). Use --max-files <= 200 or split batches.")

    files_payload: List[Dict[str, Any]] = []
    jobs: List[Dict[str, Any]] = []
    used_data_ids: set = set()
    for p in pdfs:
        base = p.stem
        data_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", base)[:120] or "paper"
        if data_id in used_data_ids:
            short = sha1_bytes(str(p).encode("utf-8"))[:8]
            base_trim = data_id[: max(0, 120 - (2 + len(short)))]
            data_id = f"{base_trim}__{short}"
            for i in range(1, 100):
                if data_id not in used_data_ids:
                    break
                suffix = f"_{i}"
                data_id = (f"{base_trim}__{short}{suffix}")[:120]
        used_data_ids.add(data_id)

        entry: Dict[str, Any] = {"name": p.name, "data_id": data_id}
        if args.page_ranges:
            entry["page_ranges"] = args.page_ranges
        if args.is_ocr is not None:
            entry["is_ocr"] = args.is_ocr
        files_payload.append(entry)

        jobs.append(
            {
                "data_id": data_id,
                "file_name": p.name,
                "source_pdf_path": str(p.resolve()),
                "source_pdf_relpath": str(p.relative_to(pdf_dir)),
                "upload_state": "pending",
                "cleaned": False,
            }
        )

    state_obj: Dict[str, Any] = {
        "version": STATE_VERSION,
        "mode": "local",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "stage": "selected",
        "pdf_dir": str(pdf_dir.resolve()),
        "outdir": str(out_root.resolve()),
        "workdir": str(Path(args.workdir).resolve()),
        "model_version": args.model_version,
        "files": jobs,
    }
    save_json(state_path, state_obj)

    batch_id, upload_urls = client.create_upload_urls_batch(
        files=files_payload,
        model_version=args.model_version,
        enable_formula=args.enable_formula,
        enable_table=args.enable_table,
        language=args.language,
    )
    log(f"BATCH created batch_id={batch_id} files={len(pdfs)}")

    # Save upload URLs for resume.
    for j, u in zip(jobs, upload_urls):
        j["upload_url"] = u
    state_obj["batch_id"] = batch_id
    state_obj["stage"] = "uploading"
    state_obj["updated_at"] = _now_iso()
    save_json(state_path, state_obj)

    # Upload with progress + speed
    total_bytes = 0
    up_t0 = time.time()
    upload_failed: List[str] = []
    for i, (p, u) in enumerate(zip(pdfs, upload_urls), start=1):
        size = p.stat().st_size
        log(f"UPLOAD {i}/{len(pdfs)} name={p.name} size={human_bytes(size)}")
        try:
            put_upload(
                u,
                p,
                timeout=args.io_timeout,
                retries=args.io_retries,
                backoff=args.api_backoff,
                max_sleep_s=args.api_max_sleep,
            )
            total_bytes += size
            jobs[i - 1]["upload_state"] = "uploaded"
            jobs[i - 1]["uploaded_at"] = _now_iso()
        except Exception as e:
            upload_failed.append(p.name)
            jobs[i - 1]["upload_state"] = "failed"
            jobs[i - 1]["last_error"] = f"upload_error: {e}"
            log(f"UPLOAD_ERROR name={p.name} err={e}", level="WARN")
        state_obj["updated_at"] = _now_iso()
        save_json(state_path, state_obj)

    up_elapsed = time.time() - up_t0
    log(
        f"UPLOAD done files={len(pdfs)} failed={len(upload_failed)} bytes={human_bytes(total_bytes)} elapsed={human_duration(up_elapsed)} avg_speed={(human_bytes(total_bytes/up_elapsed)+'/s' if up_elapsed>0 else 'n/a')}"
    )

    cfg = CleanConfig(
        repeated_block_page_ratio=args.repeat_ratio,
        repeated_block_max_chars=args.repeat_max_chars,
    )
    zip_dir = Path(args.workdir) / "zips"
    poll_and_process_batch(
        client=client,
        batch_id=batch_id,
        jobs=jobs,
        state=state_obj,
        cfg=cfg,
        out_root=out_root,
        zip_dir=zip_dir,
        state_path=state_path,
        args=args,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--token", default="eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiIxNTYwMDg3OCIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc2ODUzMDcwOCwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiMTMyNjgzNjg0MjUiLCJvcGVuSWQiOm51bGwsInV1aWQiOiI3OWU0NzI2Zi1lNjg2LTQwZmUtODFiNC0zNGJlNDZmMWRjMmIiLCJlbWFpbCI6IiIsImV4cCI6MTc2OTc0MDMwOH0.9IMoZf_fSVBB_k1UIaPSmfG-zF3vkx_dD-E28lTjUF6Wh5rRaHJ5rxTWWZbTf91CyvagewSu8P0EfAWu6yqAAg", help="MinerU API token (or env MINERU_TOKEN)")
    # common.add_argument("--token", default="eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI3NzIwMDE1MCIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc2ODQ0NTExOSwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiMTg3ODIyMDIxODciLCJvcGVuSWQiOm51bGwsInV1aWQiOiJiOTUzZTc4NC02ZWE3LTQ4YzUtODMzMi1mZWRjNWNlZmRjZDkiLCJlbWFpbCI6IiIsImV4cCI6MTc2OTY1NDcxOX0.P4g7yl1Lo_6Q_M8uszcSNyzZoRAP6sjYLTKCpKHa3YGuq31DyFnuY1ZMvnvDHDzM3mmdI8V92nCDIzDF7R43pw", help="MinerU API token (or env MINERU_TOKEN)")
    common.add_argument("--model-version", default="vlm", choices=["vlm", "pipeline"], help="MinerU model_version")
    common.add_argument("--workdir", default="./mineru_work", help="Working directory for downloaded zips")
    common.add_argument("--outdir", default="./mineru_clean_out", help="Output directory (clean md + meta)")
    common.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume from checkpoint and skip existing outputs (recommended)",
    )
    common.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume/skip logic and reprocess everything",
    )
    common.set_defaults(resume=True)
    common.add_argument(
        "--state-file",
        default=None,
        help="Optional checkpoint state JSON path (default: <workdir>/state_<mode>.json)",
    )
    common.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path (append)",
    )

    # HTTP / rate limit handling
    common.add_argument("--api-timeout", type=int, default=60, help="MinerU API HTTP timeout (seconds)")
    common.add_argument("--api-retries", type=int, default=8, help="MinerU API retries on transient errors/rate limits")
    common.add_argument("--api-backoff", type=float, default=2.0, help="Exponential backoff base for retries")
    common.add_argument("--api-max-sleep", type=float, default=120.0, help="Max sleep (seconds) between retries")
    common.add_argument("--io-timeout", type=int, default=600, help="Upload/download timeout (seconds)")
    common.add_argument("--io-retries", type=int, default=6, help="Upload/download retries on transient errors/rate limits")

    # Polling
    common.add_argument("--poll-s", type=int, default=8, help="Polling interval seconds")
    common.add_argument("--timeout-s", type=int, default=7200, help="Timeout seconds")

    # Cleaning knobs
    common.add_argument("--repeat-ratio", type=float, default=0.30, help="Drop short blocks repeated on >= ratio pages")
    common.add_argument("--repeat-max-chars", type=int, default=220, help="Only consider blocks <= chars for repeat-drop")

    p = argparse.ArgumentParser(
        description="MinerU API PDF parse & clean -> clean markdown for RAG/Dify",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[common],
    )

    sub = p.add_subparsers(dest="mode", required=True)

    p_url = sub.add_parser(
        "url",
        help="Parse a PDF accessible by URL (extract/task)",
        parents=[common],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_url.add_argument("--url", required=True, help="File URL to parse")
    p_url.add_argument("--data-id", default=None, help="Optional data_id for task")

    p_local = sub.add_parser(
        "local",
        help="Batch upload local PDFs (file-urls/batch)",
        parents=[common],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_local.add_argument("--pdf-dir", required=True, help="Directory containing PDFs")
    p_local.add_argument("--max-files", type=int, default=None, help="Max files per batch (<=200 recommended)")
    p_local.add_argument("--page-ranges", default=None, help='Optional page ranges like "2,4-6"')
    p_local.add_argument("--is-ocr", type=lambda x: x.lower() == "true", default=None, help="pipeline only: true/false")
    p_local.add_argument("--enable-formula", type=lambda x: x.lower() == "true", default=None, help="pipeline only")
    p_local.add_argument("--enable-table", type=lambda x: x.lower() == "true", default=None, help="pipeline only")
    p_local.add_argument("--language", default=None, help="pipeline only: doc language (e.g., en/ch)")

    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    global LOG_FILE
    if args.log_file:
        LOG_FILE = Path(args.log_file)

    log(f"START mode={args.mode} workdir={args.workdir} outdir={args.outdir} resume={args.resume}")

    args.token = (args.token or "").strip() or os.getenv("MINERU_TOKEN")
    if not args.token:
        raise RuntimeError("Missing token. Provide --token or set env MINERU_TOKEN.")

    if args.mode == "url":
        run_url_mode(args)
    elif args.mode == "local":
        run_local_batch_mode(args)
    else:
        raise RuntimeError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
