from __future__ import annotations

"""
Download open-access PDFs by DOI.

Primary strategy:
  - Europe PMC: DOI -> PMCID -> PDF

Optional fallback:
  - Resolve OA URL via Unpaywall/CORE (if credentials are available), then download.
  - Elsevier TDM: Download via ScienceDirect TDM API (if ELSEVIER_API_KEY is available).
  - Wiley TDM: Download via Wiley Online Library TDM API (if WILEY_TDM_TOKEN is available).
  - Sci-Hub: Try multiple Sci-Hub mirrors as last resort.
"""

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests

from .elsevier_tdm import (
    download_elsevier_tdm_pdf_to_path,
    env_default_elsevier_api_key,
    env_default_elsevier_inst_token,
)
from .wiley_tdm import (
    download_wiley_tdm_pdf_to_path,
    env_default_wiley_tdm_token,
    looks_like_wiley_doi,
)
from .oa_resolver import (
    env_default_core_api_key,
    env_default_unpaywall_emails,
    parse_unpaywall_emails,
    resolve_open_access_url,
)
from .sci_hub import download as scihub_download

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

DEFAULT_HEADERS = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept": "application/pdf,application/xhtml+xml,text/html;q=0.9,*/*;q=0.8",
}

EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPEPMC_PDF_URL = "https://europepmc.org/backend/ptpmcrender.fcgi?accid={}&blobtype=pdf"


def normalize_doi(raw: str) -> str:
    doi = (raw or "").strip()
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    return doi.strip().strip("/").lower()


def safe_filename(stem: str, *, max_len: int = 180) -> str:
    stem = (stem or "").strip()
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", stem)
    stem = stem.replace(" ", "_")
    stem = re.sub(r"_+", "_", stem).strip("_")
    if not stem:
        stem = "paper"
    return stem[:max_len]


def looks_like_pdf_bytes(data: bytes) -> bool:
    # Some servers include a leading newline before %PDF.
    return data.lstrip().startswith(b"%PDF")


def _request_json(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    resp = session.get(url, params=params, timeout=timeout, allow_redirects=True)
    if resp.status_code == 404:
        return {}
    resp.raise_for_status()
    return resp.json()


def europepmc_find_pmcid(
    session: requests.Session,
    doi: str,
    *,
    timeout: int = 30,
) -> str | None:
    params = {"query": f'DOI:"{doi}"', "format": "json", "pageSize": 1}
    data = _request_json(session, EUROPEPMC_SEARCH_URL, params=params, timeout=timeout)
    results = (data.get("resultList") or {}).get("result") or []
    if not results:
        return None
    first = results[0] if isinstance(results[0], dict) else {}
    pmcid = first.get("pmcid") or first.get("pmcId") or first.get("pmc_id")
    if not pmcid:
        return None
    pmcid = str(pmcid).strip()
    return pmcid or None


def download_pdf_to_path(
    session: requests.Session,
    url: str,
    dest_path: Path,
    *,
    timeout: int = 60,
    retries: int = 3,
    backoff_seconds: float = 1.5,
) -> tuple[bool, str | None]:
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    last_err: str | None = None
    for attempt in range(1, max(1, int(retries)) + 1):
        try:
            with session.get(
                url,
                stream=True,
                timeout=timeout,
                allow_redirects=True,
            ) as resp:
                if resp.status_code == 429:
                    sleep_s = min(60.0, backoff_seconds * attempt * 2)
                    time.sleep(sleep_s)
                    continue
                resp.raise_for_status()

                it = resp.iter_content(chunk_size=1024 * 1024)
                first = next(it, b"")
                if not first:
                    raise ValueError("Empty response body")

                content_type = (resp.headers.get("Content-Type") or "").lower()
                if ("application/pdf" not in content_type) and (
                    not looks_like_pdf_bytes(first[:2048])
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


@dataclass(frozen=True)
class DownloadResult:
    doi: str
    ok: bool
    source: str
    pdf_path: str | None
    detail: str | None


def download_one(
    doi: str,
    *,
    out_dir: Path,
    timeout: int,
    retries: int,
    prefer_europepmc: bool,
    unpaywall_email: str,
    core_api_key: str,
    elsevier_api_key: str,
    elsevier_inst_token: str,
    wiley_tdm_token: str,
    user_agent: str,
    use_scihub: bool = True,
    use_elsevier: bool = True,
    use_wiley: bool = True,
) -> DownloadResult:
    doi_norm = normalize_doi(doi)
    if not doi_norm:
        return DownloadResult(doi=doi, ok=False, source="input", pdf_path=None, detail="Empty DOI")

    dest_path = Path(out_dir) / f"{safe_filename(doi_norm)}.pdf"
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return DownloadResult(doi=doi_norm, ok=True, source="local", pdf_path=str(dest_path), detail="Already exists")

    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    if user_agent:
        session.headers["User-Agent"] = user_agent

    def try_europepmc() -> DownloadResult | None:
        pmcid = europepmc_find_pmcid(session, doi_norm, timeout=timeout)
        if not pmcid:
            return None
        url = EUROPEPMC_PDF_URL.format(pmcid)
        ok, err = download_pdf_to_path(session, url, dest_path, timeout=timeout, retries=retries)
        if ok:
            return DownloadResult(doi=doi_norm, ok=True, source="europepmc", pdf_path=str(dest_path), detail=pmcid)
        return DownloadResult(doi=doi_norm, ok=False, source="europepmc", pdf_path=None, detail=err)

    def try_resolved_oa_url() -> DownloadResult | None:
        if not unpaywall_email and not core_api_key:
            return None
        resolved = resolve_open_access_url(
            doi_norm,
            unpaywall_email=unpaywall_email,
            core_api_key=core_api_key,
            timeout=float(timeout),
            user_agent=user_agent or "papersdownload/0.1",
        )
        if not resolved.oa_url:
            return None
        ok, err = download_pdf_to_path(session, resolved.oa_url, dest_path, timeout=timeout, retries=retries)
        if ok:
            return DownloadResult(
                doi=doi_norm,
                ok=True,
                source=resolved.source or "oa_resolver",
                pdf_path=str(dest_path),
                detail=resolved.oa_url,
            )
        return DownloadResult(
            doi=doi_norm,
            ok=False,
            source=resolved.source or "oa_resolver",
            pdf_path=None,
            detail=err,
        )

    def try_scihub() -> DownloadResult | None:
        if not use_scihub:
            return None
        ok, err, mirror = scihub_download(session, doi_norm, dest_path, timeout=timeout)
        if ok:
            return DownloadResult(
                doi=doi_norm,
                ok=True,
                source="scihub",
                pdf_path=str(dest_path),
                detail=mirror,
            )
        if err:
            return DownloadResult(
                doi=doi_norm,
                ok=False,
                source="scihub",
                pdf_path=None,
                detail=err,
            )
        return None

    def try_elsevier_tdm() -> DownloadResult | None:
        if not use_elsevier:
            return None
        if not elsevier_api_key:
            return None
        ok, err = download_elsevier_tdm_pdf_to_path(
            session,
            doi_norm,
            dest_path,
            api_key=elsevier_api_key,
            inst_token=elsevier_inst_token,
            timeout=timeout,
            retries=retries,
        )
        if ok:
            return DownloadResult(
                doi=doi_norm,
                ok=True,
                source="elsevier_tdm",
                pdf_path=str(dest_path),
                detail=None,
            )
        return DownloadResult(
            doi=doi_norm,
            ok=False,
            source="elsevier_tdm",
            pdf_path=None,
            detail=err,
        )

    def try_wiley_tdm() -> DownloadResult | None:
        if not use_wiley:
            return None
        if not wiley_tdm_token:
            return None
        if not looks_like_wiley_doi(doi_norm):
            return None
        ok, err = download_wiley_tdm_pdf_to_path(
            session,
            doi_norm,
            dest_path,
            token=wiley_tdm_token,
            timeout=timeout,
            retries=retries,
        )
        if ok:
            return DownloadResult(
                doi=doi_norm,
                ok=True,
                source="wiley_tdm",
                pdf_path=str(dest_path),
                detail=None,
            )
        return DownloadResult(
            doi=doi_norm,
            ok=False,
            source="wiley_tdm",
            pdf_path=None,
            detail=err,
        )

    if prefer_europepmc:
        r = try_europepmc()
        if r and r.ok:
            return r
        u = try_resolved_oa_url()
        if u and u.ok:
            return u
        e = try_elsevier_tdm()
        if e and e.ok:
            return e
        w = try_wiley_tdm()
        if w and w.ok:
            return w
        # 第三步：尝试 Sci-Hub
        s = try_scihub()
        if s and s.ok:
            return s
        # 返回最后一个失败结果（按尝试顺序）
        for candidate in (s, w, e, u, r):
            if candidate:
                return candidate
        return DownloadResult(doi=doi_norm, ok=False, source="lookup", pdf_path=None, detail="No OA source found")

    u = try_resolved_oa_url()
    if u and u.ok:
        return u
    r = try_europepmc()
    if r and r.ok:
        return r
    e = try_elsevier_tdm()
    if e and e.ok:
        return e
    w = try_wiley_tdm()
    if w and w.ok:
        return w
    # 第三步：尝试 Sci-Hub
    s = try_scihub()
    if s and s.ok:
        return s
    for candidate in (s, w, e, r, u):
        if candidate:
            return candidate
    return DownloadResult(doi=doi_norm, ok=False, source="lookup", pdf_path=None, detail="No OA source found")


def download_pdfs(
    dois: Iterable[str],
    *,
    out_dir: str | Path = "pdfs",
    workers: int = 4,
    timeout: int = 60,
    retries: int = 3,
    prefer_europepmc: bool = True,
    unpaywall_email: str | None = None,
    core_api_key: str | None = None,
    elsevier_api_key: str | None = None,
    elsevier_inst_token: str | None = None,
    wiley_tdm_token: str | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
    report_path: str | Path | None = "download_report.jsonl",
    log_level: int = logging.INFO,
    use_scihub: bool = True,
    use_elsevier: bool = True,
    use_wiley: bool = True,
) -> list[DownloadResult]:
    logging.basicConfig(level=log_level, format="%(asctime)s | %(levelname)s | %(message)s")
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    normalized: list[str] = [normalize_doi(d) for d in dois]
    normalized = [d for d in normalized if d]
    normalized = list(dict.fromkeys(normalized))  # de-dup, keep order
    if not normalized:
        raise ValueError("No DOI provided")

    if unpaywall_email is None:
        unpaywall_emails = env_default_unpaywall_emails()
    else:
        unpaywall_emails = parse_unpaywall_emails(unpaywall_email)
    core_key = (core_api_key if core_api_key is not None else env_default_core_api_key()).strip()
    els_key = (elsevier_api_key if elsevier_api_key is not None else env_default_elsevier_api_key()).strip()
    els_inst = (
        elsevier_inst_token if elsevier_inst_token is not None else env_default_elsevier_inst_token()
    ).strip()
    wiley_token = (wiley_tdm_token if wiley_tdm_token is not None else env_default_wiley_tdm_token()).strip()

    report_f = None
    report_path_obj: Path | None = None
    if report_path:
        report_path_obj = Path(report_path)
        report_f = open(report_path_obj, "w", encoding="utf-8")

    results: list[DownloadResult] = []
    ok_count = 0
    fail_count = 0

    try:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = {
                pool.submit(
                    download_one,
                    doi,
                    out_dir=out_dir_path,
                    timeout=int(timeout),
                    retries=int(retries),
                    prefer_europepmc=bool(prefer_europepmc),
                    unpaywall_email=unpaywall_emails[i % len(unpaywall_emails)] if unpaywall_emails else "",
                    core_api_key=core_key,
                    elsevier_api_key=els_key,
                    elsevier_inst_token=els_inst,
                    wiley_tdm_token=wiley_token,
                    user_agent=user_agent,
                    use_scihub=bool(use_scihub),
                    use_elsevier=bool(use_elsevier),
                    use_wiley=bool(use_wiley),
                ): doi
                for i, doi in enumerate(normalized)
            }

            for fut in as_completed(futures):
                result = fut.result()
                results.append(result)

                if report_f:
                    report_f.write(json.dumps(result.__dict__, ensure_ascii=False) + "\n")
                    report_f.flush()

                if result.ok:
                    ok_count += 1
                    logging.info("OK  | %s | %s", result.doi, result.source)
                else:
                    fail_count += 1
                    logging.info("FAIL| %s | %s | %s", result.doi, result.source, result.detail)
    finally:
        if report_f:
            report_f.close()

    logging.info(
        "Done. ok=%s fail=%s report=%s out=%s",
        ok_count,
        fail_count,
        report_path_obj,
        out_dir_path,
    )
    return results
