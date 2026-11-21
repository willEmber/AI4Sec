"""
期刊/会议等级查询客户端 - EasyScholar API（支持 httpx 异步并发限速、Singleflight 去重与 ISSN 优先查询）
并兼容原有的同步接口契约。
"""
from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Any, Optional

import httpx
import requests

from .models import RankingEntry, VenueProfile
from .normalizer import (
    clean_venue_text,
    extract_issns,
    is_preprint_venue,
    is_umbrella_series,
    strip_venue_decorations,
)

# EasyScholar API 配置
DEFAULT_EASYSCHOLAR_API_URL = "https://www.easyscholar.cc/open/getPublicationRank"

# 同步保守限速（保留旧行为）与异步全局限速
MIN_REQUEST_INTERVAL = 1.0
ASYNC_MIN_REQUEST_INTERVAL = 0.65

_CAS_KEYS = ("sciUpSmall", "sciUp", "sciBase")


def _get_env_value(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else default


def _get_default_api_url() -> str:
    """Resolve EasyScholar API URL: prefer AppSettings (.env), fall back to OS env, then default."""
    try:
        from app.config import get_settings

        url = (get_settings().easyscholar_api_url or "").strip()
        if url:
            return url
    except Exception:
        pass
    return _get_env_value("EASYSCHOLAR_API_URL", DEFAULT_EASYSCHOLAR_API_URL) or DEFAULT_EASYSCHOLAR_API_URL


def _get_default_secret_key() -> str:
    """Resolve EasyScholar secret key: prefer AppSettings (.env), fall back to OS env."""
    try:
        from app.config import get_settings

        key = (get_settings().easyscholar_secret_key or "").strip()
        if key:
            return key
    except Exception:
        pass
    return _get_env_value("EASYSCHOLAR_SECRET_KEY")


class RateLimiter:
    """同步速率限制器（保留向后兼容）。"""

    def __init__(self, min_interval: float = MIN_REQUEST_INTERVAL):
        self.min_interval = min_interval
        self._last_request_time: float = 0

    def wait(self) -> None:
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_time = time.time()


class _AsyncRateLimiter:
    """协程安全的进程级速率限制器（避免 time.sleep 阻塞事件循环）。"""

    def __init__(self, min_interval: float = ASYNC_MIN_REQUEST_INTERVAL) -> None:
        self.min_interval = min_interval
        self._lock: asyncio.Lock | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_time: float = 0.0

    def _get_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._loop is not loop:
            self._lock = asyncio.Lock()
            self._loop = loop
        return self._lock

    async def wait(self) -> None:
        async with self._get_lock():
            now = time.monotonic()
            elapsed = now - self._last_time
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self._last_time = time.monotonic()


_GLOBAL_ASYNC_ES_LIMITER = _AsyncRateLimiter()
_INFLIGHT_ES_QUERIES: dict[tuple[int, str], asyncio.Future["PublicationRankResult"]] = {}


class PublicationRankResult:
    """期刊/会议等级查询结果（兼容旧字段，并支持转换为多体系 VenueProfile）。"""

    def __init__(
        self,
        name: str,
        sci: Optional[str] = None,
        ccf: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
        source: str = "",
    ):
        self.name = name
        self.sci = sci
        self.ccf = ccf
        self.success = success
        self.error = error
        self.extra = extra or {}
        self.source = source

    def __repr__(self) -> str:
        if not self.success:
            return f"PublicationRankResult(name='{self.name}', error='{self.error}')"
        return (
            f"PublicationRankResult(name='{self.name}', "
            f"SCI='{self.sci or '未收录'}', CCF='{self.ccf or '未收录'}')"
        )

    def to_dict(self) -> dict[str, Any]:
        if not self.success:
            return {"name": self.name, "success": False, "error": self.error}
        return {
            "name": self.name,
            "success": True,
            "sci": self.sci,
            "ccf": self.ccf,
            "extra": self.extra,
            "source": self.source,
        }

    def to_venue_profile(self, *, query_venue: str = "") -> VenueProfile:
        """Convert this lookup result into a structured multi-system ``VenueProfile``."""
        qname = query_venue or self.name
        if not self.success:
            return VenueProfile(
                query_venue=qname,
                canonical_name=self.name or qname,
                source=self.source,
                verification_status="unknown",
            )

        extra = self.extra or {}
        editions = extra.get("_editions") if isinstance(extra.get("_editions"), dict) else {}
        src = (self.source or "").strip()
        verified = src in {"local_catalog", "easyscholar"}

        # If converted from a cached/serialized VenueProfile with explicit `_rankings`
        if isinstance(extra.get("_rankings"), list):
            rankings = []
            for r in extra["_rankings"]:
                entry = RankingEntry.model_validate(r)
                if not entry.source and src:
                    entry = entry.model_copy(update={"source": src, "verified": verified})
                elif "verified" not in (r if isinstance(r, dict) else {}):
                    entry = entry.model_copy(
                        update={"verified": (entry.source or src) in {"local_catalog", "easyscholar"}}
                    )
                rankings.append(entry)
        else:
            rankings = []
            if self.sci:
                rankings.append(
                    RankingEntry(
                        ranking_system="JCR",
                        rank=self.sci,
                        category=str(extra.get("jcr_category") or "all"),
                        edition_year=str(editions.get("JCR") or ""),
                        impact_factor=str(extra.get("sciif") or ""),
                        source=src,
                        verified=verified,
                    )
                )
            for key in _CAS_KEYS:
                val = extra.get(key)
                if val:
                    rankings.append(
                        RankingEntry(
                            ranking_system="CAS",
                            rank=str(val).rstrip("。."),
                            category="subject" if key == "sciUpSmall" else "broad",
                            edition_year=str(editions.get("CAS") or ""),
                            top=bool(extra.get("sciUpTop")),
                            source=src,
                            verified=verified,
                        )
                    )
                    break
            if extra.get("cas") and not any(r.ranking_system == "CAS" for r in rankings):
                rankings.append(
                    RankingEntry(
                        ranking_system="CAS",
                        rank=str(extra["cas"]),
                        category="broad",
                        edition_year=str(editions.get("CAS") or ""),
                        top=bool(extra.get("sciUpTop")),
                        source=src,
                        verified=verified,
                    )
                )
            if self.ccf:
                rankings.append(
                    RankingEntry(
                        ranking_system="CCF",
                        rank=self.ccf,
                        category=str(extra.get("ccf_category") or "all"),
                        edition_year=str(editions.get("CCF") or ""),
                        source=src,
                        verified=verified,
                    )
                )
            if extra.get("core"):
                rankings.append(
                    RankingEntry(
                        ranking_system="CORE",
                        rank=str(extra["core"]),
                        edition_year=str(editions.get("CORE") or ""),
                        source=src,
                        verified=verified,
                    )
                )
            if extra.get("th_cpl"):
                rankings.append(
                    RankingEntry(
                        ranking_system="TH_CPL",
                        rank=str(extra["th_cpl"]),
                        edition_year=str(editions.get("TH_CPL") or ""),
                        source=src,
                        verified=verified,
                    )
                )
            if extra.get("caai"):
                rankings.append(
                    RankingEntry(
                        ranking_system="CAAI",
                        rank=str(extra["caai"]),
                        edition_year=str(editions.get("CAAI") or ""),
                        source=src,
                        verified=verified,
                    )
                )
            if extra.get("ccf_cn"):
                rankings.append(
                    RankingEntry(
                        ranking_system="CCF_CN",
                        rank=str(extra["ccf_cn"]),
                        edition_year=str(editions.get("CCF_CN") or ""),
                        source=src,
                        verified=verified,
                    )
                )

        any_verified = any(r.verified for r in rankings) or verified
        metrics = extra.get("metrics") if isinstance(extra.get("metrics"), dict) else {}
        return VenueProfile(
            query_venue=qname,
            canonical_name=str(extra.get("canonical_name") or self.name or qname),
            acronym=str(extra.get("acronym") or ""),
            venue_type=str(extra.get("venue_type") or ""),
            subtrack=str(extra.get("subtrack") or "main"),
            peer_reviewed=extra.get("peer_reviewed") if isinstance(extra.get("peer_reviewed"), bool) else (True if rankings else None),
            is_preprint_file=bool(extra.get("is_preprint_file", False)),
            openreview_tier=str(extra.get("openreview_tier") or ""),
            issn_l=str(extra.get("issn_l") or ""),
            issns=list(extra.get("issns") or []),
            dblp_prefix=str(extra.get("dblp_prefix") or ""),
            rankings=rankings,
            warning=str(extra.get("cas_warning") or ""),
            note=str(extra.get("note") or ""),
            metrics=metrics,
            source=src,
            verification_status="verified" if any_verified else ("unverified" if rankings else "unknown"),
        )

    @classmethod
    def from_venue_profile(cls, profile: VenueProfile) -> "PublicationRankResult":
        """Bridge a ``VenueProfile`` back into legacy ``PublicationRankResult``."""
        extra: dict[str, Any] = {
            "_rankings": [r.model_dump(mode="json") for r in profile.rankings],
            "canonical_name": profile.canonical_name,
            "acronym": profile.acronym,
            "venue_type": profile.venue_type,
            "subtrack": profile.subtrack,
            "peer_reviewed": profile.peer_reviewed,
            "is_preprint_file": profile.is_preprint_file,
            "openreview_tier": profile.openreview_tier,
            "issn_l": profile.issn_l,
            "issns": profile.issns,
            "dblp_prefix": profile.dblp_prefix,
        }
        if profile.warning:
            extra["cas_warning"] = profile.warning
        if profile.note:
            extra["note"] = profile.note
        if profile.metrics:
            extra["metrics"] = dict(profile.metrics)
        for r in profile.rankings:
            if r.ranking_system == "JCR" and r.impact_factor:
                extra["sciif"] = r.impact_factor
            elif r.ranking_system == "CAS":
                extra["sciUp"] = r.rank
                if r.top:
                    extra["sciUpTop"] = "TOP"
            elif r.ranking_system == "CORE":
                extra["core"] = r.rank
            elif r.ranking_system == "TH_CPL":
                extra["th_cpl"] = r.rank
            elif r.ranking_system == "CAAI":
                extra["caai"] = r.rank
            elif r.ranking_system == "CCF_CN":
                extra["ccf_cn"] = r.rank

        has_signal = bool(
            profile.rankings
            or profile.warning
            or profile.note
            or profile.subtrack in {"workshop", "companion", "findings", "preprint"}
        )
        return cls(
            name=profile.canonical_name or profile.query_venue,
            sci=profile.sci_rank or None,
            ccf=profile.ccf_rank or None,
            success=has_signal,
            error=None if has_signal else "未收录等级信息",
            extra=extra,
            source=profile.source or "local_catalog",
        )


def _normalize_publication_name(name: str) -> str:
    """Legacy whitespace/control-char normalizer (kept exact for SQLite cache compatibility)."""
    if not isinstance(name, str):
        raise ValueError(f"期刊名称必须是字符串，收到了 {type(name).__name__}")
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", name)
    return name


def _validate_publication_name(name: str) -> str:
    """校验期刊名称合法性，解码 HTML 实体并返回规范化后的名称。"""
    if not isinstance(name, str):
        raise ValueError(f"期刊名称必须是字符串，收到了 {type(name).__name__}")
    cleaned = clean_venue_text(name)
    if not cleaned:
        raise ValueError("期刊名称不能为空")
    if len(cleaned) > 500:
        raise ValueError(f"期刊名称过长（{len(cleaned)} 字符），最大允许 500 字符")
    return cleaned


class EasyScholarClient:
    """EasyScholar 期刊等级查询客户端（支持同步与异步 ISSN 优先查询）。"""

    def __init__(
        self,
        secret_key: str | None = None,
        api_url: str | None = None,
        min_interval: float = MIN_REQUEST_INTERVAL,
        max_retries: int = 3,
        timeout: float = 10.0,
    ):
        self.secret_key = secret_key.strip() if isinstance(secret_key, str) else _get_default_secret_key()
        self.api_url = api_url.strip() if isinstance(api_url, str) else _get_default_api_url()
        self.max_retries = max_retries
        self.timeout = timeout
        self._rate_limiter = RateLimiter(min_interval)
        self._session = requests.Session()

    @property
    def configured(self) -> bool:
        return bool(self.secret_key)

    @staticmethod
    def _extract_sci_ccf(data: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
        official_rank = data.get("officialRank") or {}
        all_rank = official_rank.get("all") or {}
        sci = all_rank.get("sci")
        ccf = all_rank.get("ccf")
        return sci, ccf

    @classmethod
    def _parse_response_payload(cls, result: Any, display_name: str) -> PublicationRankResult:
        if not isinstance(result, dict):
            return PublicationRankResult(name=display_name, success=False, error="API 返回格式异常")
        code = result.get("code")
        msg = result.get("msg", "")
        if code != 200:
            return PublicationRankResult(
                name=display_name,
                success=False,
                error=f"API 错误 (code={code}): {msg}",
            )
        data = result.get("data")
        if data is None:
            return PublicationRankResult(
                name=display_name,
                success=False,
                error="API 返回数据为空",
            )
        sci, ccf = cls._extract_sci_ccf(data)
        official_all = (data.get("officialRank") or {}).get("all") or {}
        return PublicationRankResult(
            name=display_name,
            sci=sci,
            ccf=ccf,
            success=True,
            extra=dict(official_all),
            source="easyscholar",
        )

    async def _aquery_single_term(
        self,
        term: str,
        *,
        display_name: str,
        client: httpx.AsyncClient | None = None,
    ) -> PublicationRankResult:
        loop = asyncio.get_running_loop()
        inflight_key = (id(loop), term.lower())
        existing = _INFLIGHT_ES_QUERIES.get(inflight_key)
        if existing is not None and not existing.done():
            res = await asyncio.shield(existing)
            return PublicationRankResult(
                name=display_name,
                sci=res.sci,
                ccf=res.ccf,
                success=res.success,
                error=res.error,
                extra=dict(res.extra),
                source=res.source,
            )

        fut: asyncio.Future[PublicationRankResult] = loop.create_future()
        _INFLIGHT_ES_QUERIES[inflight_key] = fut

        own_client = client is None
        http = client or httpx.AsyncClient(timeout=self.timeout)
        params = {"secretKey": self.secret_key, "publicationName": term}
        last_error: str | None = None
        final_res: PublicationRankResult | None = None
        try:
            for attempt in range(1, self.max_retries + 1):
                await _GLOBAL_ASYNC_ES_LIMITER.wait()
                try:
                    resp = await http.get(self.api_url, params=params)
                    if resp.status_code == 429:
                        last_error = f"请求过于频繁，被限速 (尝试 {attempt}/{self.max_retries})"
                        await asyncio.sleep(2.0)
                        continue
                    resp.raise_for_status()
                    final_res = self._parse_response_payload(resp.json(), display_name)
                    break
                except httpx.TimeoutException:
                    last_error = f"请求超时 (尝试 {attempt}/{self.max_retries})"
                except httpx.HTTPError as exc:
                    last_error = f"HTTP 错误: {exc} (尝试 {attempt}/{self.max_retries})"
                except ValueError:
                    last_error = f"API 返回的数据不是合法 JSON (尝试 {attempt}/{self.max_retries})"
                except Exception as exc:
                    last_error = f"未知错误: {exc} (尝试 {attempt}/{self.max_retries})"

                if attempt < self.max_retries:
                    await asyncio.sleep(min(2**attempt, 6))

            if final_res is None:
                final_res = PublicationRankResult(
                    name=display_name,
                    success=False,
                    error=f"重试 {self.max_retries} 次后仍然失败: {last_error}",
                )
            fut.set_result(final_res)
            return final_res
        except BaseException as exc:
            if not fut.done():
                fut.set_exception(exc)
            raise
        finally:
            _INFLIGHT_ES_QUERIES.pop(inflight_key, None)
            if own_client:
                await http.aclose()

    async def aquery(
        self,
        publication_name: str,
        *,
        issns: list[str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> PublicationRankResult:
        """异步查询：优先尝试精确 ISSN，再尝试剥离修饰词后的标准刊名/简称，最后尝试原名。"""
        try:
            clean_name = _validate_publication_name(publication_name)
        except ValueError as exc:
            return PublicationRankResult(
                name=publication_name if isinstance(publication_name, str) else str(publication_name),
                success=False,
                error=str(exc),
            )

        if not self.secret_key:
            return PublicationRankResult(
                name=clean_name,
                success=False,
                error="缺少 EasyScholar 密钥配置，请在项目根目录 .env 中设置 EASYSCHOLAR_SECRET_KEY",
            )

        if is_preprint_venue(clean_name) or is_umbrella_series(clean_name):
            return PublicationRankResult(
                name=clean_name,
                success=False,
                error="预印本或会议丛书壳不直接对应单一期刊等级",
            )

        candidate_terms: list[str] = []
        for issn in extract_issns(issns):
            if issn not in candidate_terms:
                candidate_terms.append(issn)

        stripped, acronyms = strip_venue_decorations(clean_name)
        for term in (*acronyms, stripped, clean_name):
            t = term.strip()
            if t and t not in candidate_terms:
                candidate_terms.append(t)

        own_client = client is None
        http = client or httpx.AsyncClient(timeout=self.timeout)
        last_res: PublicationRankResult | None = None
        try:
            for term in candidate_terms[:3]:
                res = await self._aquery_single_term(term, display_name=clean_name, client=http)
                last_res = res
                if res.success and (res.sci or res.ccf or res.extra):
                    return res
                # Transient network/HTTP errors should stop immediately rather than
                # hammering EasyScholar three times with alternate spellings.
                if not res.success and any(
                    k in (res.error or "") for k in ("超时", "限速", "HTTP 错误", "网络连接")
                ):
                    return res
            return last_res or PublicationRankResult(name=clean_name, success=False, error="未找到等级信息")
        finally:
            if own_client:
                await http.aclose()

    def query(self, publication_name: str) -> PublicationRankResult:
        """同步查询单个期刊的 SCI 和 CCF 等级（保留向后兼容）。"""
        try:
            publication_name = _validate_publication_name(publication_name)
        except ValueError as e:
            return PublicationRankResult(
                name=publication_name if isinstance(publication_name, str) else str(publication_name),
                success=False,
                error=str(e),
            )

        if not self.secret_key:
            return PublicationRankResult(
                name=publication_name,
                success=False,
                error="缺少 EasyScholar 密钥配置，请在项目根目录 .env 中设置 EASYSCHOLAR_SECRET_KEY",
            )

        self._rate_limiter.wait()
        params = {
            "secretKey": self.secret_key,
            "publicationName": publication_name,
        }

        last_error: str | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.get(self.api_url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                return self._parse_response_payload(resp.json(), publication_name)
            except requests.exceptions.Timeout:
                last_error = f"请求超时 (尝试 {attempt}/{self.max_retries})"
            except requests.exceptions.ConnectionError:
                last_error = f"网络连接错误 (尝试 {attempt}/{self.max_retries})"
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response is not None else "unknown"
                if status_code == 429:
                    last_error = f"请求过于频繁，被限速 (尝试 {attempt}/{self.max_retries})"
                    time.sleep(2)
                else:
                    last_error = f"HTTP 错误 {status_code} (尝试 {attempt}/{self.max_retries})"
            except requests.exceptions.JSONDecodeError:
                last_error = f"API 返回的数据不是合法 JSON (尝试 {attempt}/{self.max_retries})"
            except Exception as e:
                last_error = f"未知错误: {e} (尝试 {attempt}/{self.max_retries})"

            if attempt < self.max_retries:
                time.sleep(min(2**attempt, 8))

        return PublicationRankResult(
            name=publication_name,
            success=False,
            error=f"重试 {self.max_retries} 次后仍然失败: {last_error}",
        )

    def query_batch(self, publication_names: list[str]) -> list[PublicationRankResult]:
        """批量查询多个期刊的 SCI 和 CCF 等级（同步、串行，遵守速率限制）。"""
        results: list[PublicationRankResult] = []
        seen: set[str] = set()
        unique_names: list[str] = []
        for name in publication_names:
            try:
                normalized = _validate_publication_name(name)
            except ValueError:
                normalized = name
            if normalized not in seen:
                seen.add(normalized)
                unique_names.append(name)

        cache: dict[str, PublicationRankResult] = {}
        for name in unique_names:
            result = self.query(name)
            cache[_validate_publication_name(name) if isinstance(name, str) else name] = result
            results.append(result)

        if len(publication_names) > len(unique_names):
            full_results: list[PublicationRankResult] = []
            for name in publication_names:
                try:
                    normalized = _validate_publication_name(name)
                except (ValueError, TypeError):
                    normalized = name
                if normalized in cache:
                    full_results.append(cache[normalized])
                else:
                    full_results.append(
                        PublicationRankResult(name=str(name), success=False, error="未找到缓存结果")
                    )
            return full_results

        return results

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "EasyScholarClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
