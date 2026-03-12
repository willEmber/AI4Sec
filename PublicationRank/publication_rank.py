"""
期刊等级查询工具 - 基于 EasyScholar API
查询期刊的 SCI 分区和 CCF 等级
"""

import os
import time
import re
from pathlib import Path
from typing import Optional
import requests
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# EasyScholar API 配置
DEFAULT_EASYSCHOLAR_API_URL = "https://www.easyscholar.cc/open/getPublicationRank"


def _get_env_value(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else default


def _get_default_api_url() -> str:
    return _get_env_value("EASYSCHOLAR_API_URL", DEFAULT_EASYSCHOLAR_API_URL) or DEFAULT_EASYSCHOLAR_API_URL


def _get_default_secret_key() -> str:
    return _get_env_value("EASYSCHOLAR_SECRET_KEY")

# 速率限制：每秒最多1次请求（保守策略）
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
    # 去除首尾空白
    name = name.strip()
    # 将制表符、换行等空白字符替换为空格，再去除不可见控制字符
    name = re.sub(r'\s+', ' ', name)
    name = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', name)
    return name


def _validate_publication_name(name: str) -> str:
    """
    校验期刊名称合法性，返回规范化后的名称。
    """
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
        """从API返回的data中提取 SCI 和 CCF 等级"""
        sci = None
        ccf = None

        official_rank = data.get("officialRank") or {}
        all_rank = official_rank.get("all") or {}

        # 提取 SCI 分区 (JCR)
        sci = all_rank.get("sci")

        # 提取 CCF 等级
        ccf = all_rank.get("ccf")

        return sci, ccf

    def query(self, publication_name: str) -> PublicationRankResult:
        """
        查询单个期刊的 SCI 和 CCF 等级。

        Args:
            publication_name: 期刊名称

        Returns:
            PublicationRankResult 对象
        """
        # 输入校验
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

        # 速率限制
        self._rate_limiter.wait()

        # 构建请求参数
        params = {
            "secretKey": self.secret_key,
            "publicationName": publication_name,
        }

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.get(
                    self.api_url,
                    params=params,
                    timeout=self.timeout,
                )
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
                    # 429 时额外等待
                    time.sleep(2)
                else:
                    last_error = f"HTTP 错误 {status_code} (尝试 {attempt}/{self.max_retries})"
            except requests.exceptions.JSONDecodeError:
                last_error = f"API 返回的数据不是合法 JSON (尝试 {attempt}/{self.max_retries})"
            except Exception as e:
                last_error = f"未知错误: {e} (尝试 {attempt}/{self.max_retries})"

            # 重试前等待（指数退避）
            if attempt < self.max_retries:
                time.sleep(min(2 ** attempt, 8))

        return PublicationRankResult(
            name=publication_name,
            success=False,
            error=f"重试 {self.max_retries} 次后仍然失败: {last_error}",
        )

    def query_batch(self, publication_names: list[str]) -> list[PublicationRankResult]:
        """
        批量查询多个期刊的 SCI 和 CCF 等级。
        自动处理去重和限速。

        Args:
            publication_names: 期刊名称列表

        Returns:
            PublicationRankResult 列表
        """
        results = []
        # 去重但保持顺序
        seen = set()
        unique_names = []
        for name in publication_names:
            try:
                normalized = _validate_publication_name(name)
            except ValueError:
                normalized = name
            if normalized not in seen:
                seen.add(normalized)
                unique_names.append(name)

        print(f"共 {len(publication_names)} 个期刊名，去重后 {len(unique_names)} 个，开始查询...")

        cache = {}
        for i, name in enumerate(unique_names, 1):
            print(f"  [{i}/{len(unique_names)}] 查询: {name}")
            result = self.query(name)
            cache[_normalize_publication_name(name) if isinstance(name, str) else name] = result
            results.append(result)
            print(f"    -> SCI={result.sci or '未收录'}, CCF={result.ccf or '未收录'}"
                  if result.success else f"    -> 错误: {result.error}")

        # 对于重复项，使用缓存结果
        if len(publication_names) > len(unique_names):
            full_results = []
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


def query_publication_rank(publication_name: str,
                           secret_key: str | None = None) -> PublicationRankResult:
    """
    便捷函数：查询单个期刊的 SCI 和 CCF 等级。

    Args:
        publication_name: 期刊名称
        secret_key: EasyScholar API 密钥

    Returns:
        PublicationRankResult 对象
    """
    with EasyScholarClient(secret_key=secret_key) as client:
        return client.query(publication_name)


if __name__ == "__main__":
    # 简单命令行测试
    import sys

    if len(sys.argv) > 1:
        name = " ".join(sys.argv[1:])
        result = query_publication_rank(name)
        print(result)
    else:
        print("用法: python publication_rank.py <期刊名称>")
        print("示例: python publication_rank.py IEEE Transactions on Medical Imaging")
