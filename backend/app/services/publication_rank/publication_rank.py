"""
期刊等级查询工具 - 基于 EasyScholar API
查询期刊的 SCI 分区和 CCF 等级
"""
from __future__ import annotations

import os
import re
import time
from typing import Optional

import requests


# EasyScholar API 配置
DEFAULT_EASYSCHOLAR_API_URL = "https://www.easyscholar.cc/open/getPublicationRank"


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
        # AppSettings unavailable (e.g. used as a standalone module) — fall through to env.
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


# 速率限制：每秒最多 1 次请求（保守策略）
MIN_REQUEST_INTERVAL = 1.0


class RateLimiter:
    """简单的速率限制器"""

    def __init__(self, min_interval: float = MIN_REQUEST_INTERVAL):
        self.min_interval = min_interval
        self._last_request_time: float = 0

    def wait(self):
        """等待到满足限速要求后再继续"""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
        self._last_request_time = time.time()


class PublicationRankResult:
    """期刊等级查询结果"""

    def __init__(self, name: str, sci: Optional[str] = None, ccf: Optional[str] = None,
                 success: bool = True, error: Optional[str] = None):
        self.name = name
        self.sci = sci
        self.ccf = ccf
        self.success = success
        self.error = error

    def __repr__(self):
        if not self.success:
            return f"PublicationRankResult(name='{self.name}', error='{self.error}')"
        return (f"PublicationRankResult(name='{self.name}', "
                f"SCI='{self.sci or '未收录'}', CCF='{self.ccf or '未收录'}')")

    def to_dict(self) -> dict:
        if not self.success:
            return {"name": self.name, "success": False, "error": self.error}
        return {
            "name": self.name,
            "success": True,
            "sci": self.sci,
            "ccf": self.ccf,
        }


def _normalize_publication_name(name: str) -> str:
    """
    对期刊名称进行规范化处理：
    1. 去除首尾空白字符
    2. 合并内部多余空格
    3. 去除特殊控制字符
    """
    if not isinstance(name, str):
        raise ValueError(f"期刊名称必须是字符串，收到了 {type(name).__name__}")
    name = name.strip()
    name = re.sub(r'\s+', ' ', name)
    name = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', name)
    return name


def _validate_publication_name(name: str) -> str:
    """校验期刊名称合法性，返回规范化后的名称。"""
    name = _normalize_publication_name(name)
    if not name:
        raise ValueError("期刊名称不能为空")
    if len(name) > 500:
        raise ValueError(f"期刊名称过长（{len(name)} 字符），最大允许 500 字符")
    return name


class EasyScholarClient:
    """EasyScholar 期刊等级查询客户端"""

    def __init__(self, secret_key: str | None = None,
                 api_url: str | None = None,
                 min_interval: float = MIN_REQUEST_INTERVAL,
                 max_retries: int = 3,
                 timeout: float = 10.0):
        self.secret_key = secret_key.strip() if isinstance(secret_key, str) else _get_default_secret_key()
        self.api_url = api_url.strip() if isinstance(api_url, str) else _get_default_api_url()
        self.max_retries = max_retries
        self.timeout = timeout
        self._rate_limiter = RateLimiter(min_interval)
        self._session = requests.Session()

    def _extract_sci_ccf(self, data: dict) -> tuple[Optional[str], Optional[str]]:
        """从 API 返回的 data 中提取 SCI 和 CCF 等级"""
        official_rank = data.get("officialRank") or {}
        all_rank = official_rank.get("all") or {}
        sci = all_rank.get("sci")
        ccf = all_rank.get("ccf")
        return sci, ccf

    def query(self, publication_name: str) -> PublicationRankResult:
        """查询单个期刊的 SCI 和 CCF 等级。"""
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

                result = resp.json()
                code = result.get("code")
                msg = result.get("msg", "")

                if code != 200:
                    return PublicationRankResult(
                        name=publication_name,
                        success=False,
                        error=f"API 错误 (code={code}): {msg}",
                    )

                data = result.get("data")
                if data is None:
                    return PublicationRankResult(
                        name=publication_name,
                        success=False,
                        error="API 返回数据为空",
                    )

                sci, ccf = self._extract_sci_ccf(data)
                return PublicationRankResult(
                    name=publication_name,
                    sci=sci,
                    ccf=ccf,
                    success=True,
                )

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
                time.sleep(min(2 ** attempt, 8))

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
            cache[_normalize_publication_name(name) if isinstance(name, str) else name] = result
            results.append(result)

        if len(publication_names) > len(unique_names):
            full_results: list[PublicationRankResult] = []
            for name in publication_names:
                try:
                    normalized = _normalize_publication_name(name)
                except (ValueError, TypeError):
                    normalized = name
                if normalized in cache:
                    full_results.append(cache[normalized])
                else:
                    full_results.append(PublicationRankResult(
                        name=str(name), success=False, error="未找到缓存结果"))
            return full_results

        return results

    def close(self):
        """关闭底层的 HTTP 会话"""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
