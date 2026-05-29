"""
LLM web_search 查询出版物等级 + 统一客户端（Cache → EasyScholar → LLM fallback）。

使用 Qwen Responses API 内置 web_search 工具来确定会议/期刊的 SCI 分区和 CCF 等级。
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from typing import Any

import httpx

from app.config import get_settings

from .publication_rank import (
    EasyScholarClient,
    PublicationRankResult,
    _validate_publication_name,
)
from .rank_cache import RankCache

logger = logging.getLogger("scholar.llm_rank")

_SYSTEM_PROMPT = """\
你是一个学术出版物等级查询助手。请通过网络搜索确定学术期刊或会议的等级信息。

查询内容：
1. SCI 分区（中科院最新大类分区，Q1/Q2/Q3/Q4 之一）
2. CCF 等级（中国计算机学会推荐目录最新版，A/B/C 之一）

规则：
- 会议（如 CVPR、NeurIPS）通常有 CCF 等级但没有 SCI 分区
- 期刊通常有 SCI 分区，部分也有 CCF 等级
- 未被收录或无法确定的字段返回 null

请严格按以下 JSON 格式返回，不要包含其他文字：
{"sci": "Q1", "ccf": "A"}
"""

_VALID_SCI = {"Q1", "Q2", "Q3", "Q4"}
_VALID_CCF = {"A", "B", "C"}
_TRANSIENT_FAILURE_MARKERS = (
    "LLM 查询失败",
    "LLM_BASEURL",
    "UnsupportedProtocol",
    "ReadTimeout",
    "ConnectError",
    "HTTP ",
)


def _is_transient_failure(result: PublicationRankResult) -> bool:
    if result.success:
        return False
    error = result.error or ""
    return any(marker in error for marker in _TRANSIENT_FAILURE_MARKERS)


# ---------------------------------------------------------------------------
# Response Parsing
# ---------------------------------------------------------------------------

def _parse_llm_response(text: str, publication_name: str) -> PublicationRankResult:
    """
    从 LLM 响应文本中解析 SCI/CCF 信息。

    支持：纯 JSON、markdown code fence 包裹的 JSON、文本中嵌入的 JSON 对象。
    """
    if not text or not text.strip():
        return PublicationRankResult(
            name=publication_name, success=False, error="LLM 返回空文本"
        )

    json_str = text.strip()

    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", json_str, re.DOTALL)
    if fence_match:
        json_str = fence_match.group(1).strip()

    if not json_str.startswith("{"):
        obj_match = re.search(r"\{[^{}]*\}", json_str)
        if obj_match:
            json_str = obj_match.group(0)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return PublicationRankResult(
            name=publication_name, success=False,
            error=f"LLM 返回非法 JSON: {text[:200]}"
        )

    if not isinstance(data, dict):
        return PublicationRankResult(
            name=publication_name, success=False,
            error=f"LLM 返回非字典 JSON: {type(data).__name__}"
        )

    sci_raw = data.get("sci")
    sci: str | None = None
    if isinstance(sci_raw, str):
        sci_upper = sci_raw.strip().upper()
        if sci_upper in _VALID_SCI:
            sci = sci_upper

    ccf_raw = data.get("ccf")
    ccf: str | None = None
    if isinstance(ccf_raw, str):
        ccf_upper = ccf_raw.strip().upper()
        if ccf_upper in _VALID_CCF:
            ccf = ccf_upper

    return PublicationRankResult(name=publication_name, sci=sci, ccf=ccf, success=True)


# ---------------------------------------------------------------------------
# LLMRankClient
# ---------------------------------------------------------------------------

_VALID_API_STYLES = {"responses", "chat_completions"}


class LLMRankClient:
    """通过 LLM + web search 查询出版物等级。

    API 风格由 LLM_RANK_API_STYLE 显式选择（默认与 llm_service.py 对齐，使用 /responses）：

    - ``responses`` → POST ``{base_url}/responses`` + ``tools=[{type: web_search}]``
      （Qwen Responses API / DashScope Bailian apps 协议）
    - ``chat_completions`` → POST ``{base_url}/chat/completions`` + ``enable_search``
      （OpenAI 兼容接口，例如 ``https://dashscope.aliyuncs.com/compatible-mode/v1``）
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        api_style: str | None = None,
    ):
        settings = get_settings()
        self.base_url = (
            base_url if base_url is not None else settings.llm_base_url
        ).strip().rstrip("/")
        self.api_key = (
            api_key if api_key is not None else settings.llm_api_key
        ).strip()
        self.model = (
            model if model is not None else settings.thinking_model
        ).strip()
        self.max_retries = max_retries
        self.timeout = timeout

        style = (api_style if api_style is not None else getattr(settings, "llm_rank_api_style", "responses")).strip().lower()
        if style not in _VALID_API_STYLES:
            logger.warning(
                "Unknown LLM_RANK_API_STYLE=%r, falling back to 'responses'. Valid: %s",
                style, sorted(_VALID_API_STYLES),
            )
            style = "responses"
        self.api_style = style
        self._use_chat_completions = (style == "chat_completions")

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _build_payload(self, publication_name: str) -> tuple[str, dict[str, Any]]:
        """构建请求 URL 和 payload，根据 API 风格自动选择格式。"""
        user_prompt = f"请查询学术出版物「{publication_name}」的 SCI 分区和 CCF 等级。"

        if self._use_chat_completions:
            url = f"{self.base_url}/chat/completions"
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "enable_search": True,
                "temperature": 0.1,
            }
        else:
            url = f"{self.base_url}/responses"
            payload = {
                "model": self.model,
                "input": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "tools": [{"type": "web_search"}],
                "temperature": 0.1,
            }
        return url, payload

    @staticmethod
    def _extract_content(data: dict, use_chat_completions: bool) -> str:
        """从 API 响应中提取文本内容。"""
        if use_chat_completions:
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return ""
        else:
            content = ""
            for item in data.get("output", []):
                if item.get("type") == "message":
                    for part in item.get("content", []):
                        if part.get("type") == "output_text":
                            content += part.get("text", "")
                    break
            return content

    @staticmethod
    def _is_retryable(status_code: int) -> bool:
        return status_code in {408, 429, 500, 502, 503, 504}

    def _compute_delay(self, attempt: int) -> float:
        delay = min(1.0 * (2 ** attempt), 30.0)
        return delay * random.uniform(0.8, 1.2)

    async def query(self, publication_name: str) -> PublicationRankResult:
        """查询单个出版物的 SCI/CCF 等级。"""
        if not self.base_url.startswith(("http://", "https://")):
            return PublicationRankResult(
                name=publication_name,
                success=False,
                error=(
                    "LLM_BASEURL 配置缺失或无效，请设置包含 http:// 或 "
                    "https:// 协议的完整地址"
                ),
            )

        url, payload = self._build_payload(publication_name)

        attempt = 0
        last_error: str | None = None
        retry_exhausted = False
        while True:
            attempt += 1
            try:
                timeout = httpx.Timeout(
                    connect=15.0, read=self.timeout, write=15.0, pool=15.0
                )
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(
                        url,
                        headers=self._headers(),
                        json=payload,
                    )

                if resp.status_code == 429 and attempt <= self.max_retries:
                    retry_after = resp.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after else self._compute_delay(attempt) * 2
                    delay = min(delay, 120.0)
                    logger.warning("LLM rank: 429 rate-limited, retry in %.1fs", delay)
                    await asyncio.sleep(delay)
                    continue

                if self._is_retryable(resp.status_code) and attempt <= self.max_retries:
                    delay = self._compute_delay(attempt)
                    logger.warning(
                        "LLM rank: HTTP %d (attempt %d/%d), retry in %.1fs",
                        resp.status_code, attempt, self.max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                resp.raise_for_status()
                data = resp.json()
                content = self._extract_content(data, self._use_chat_completions)

                if not content:
                    return PublicationRankResult(
                        name=publication_name, success=False,
                        error="LLM 未返回文本内容",
                    )

                return _parse_llm_response(content, publication_name)

            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}"
                if attempt > self.max_retries:
                    retry_exhausted = True
                    break
                await asyncio.sleep(self._compute_delay(attempt))

            except (httpx.ReadTimeout, httpx.ConnectError) as e:
                last_error = f"{type(e).__name__}: {e}"
                if attempt > self.max_retries:
                    retry_exhausted = True
                    break
                await asyncio.sleep(self._compute_delay(attempt))

            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                break

        if retry_exhausted:
            error = f"LLM 查询失败 ({self.max_retries} 次重试后): {last_error}"
        else:
            error = f"LLM 查询失败: {last_error}"
        return PublicationRankResult(
            name=publication_name, success=False, error=error,
        )

    async def query_batch(
        self, names: list[str], concurrency: int = 3
    ) -> list[PublicationRankResult]:
        """并发批量查询。"""
        sem = asyncio.Semaphore(concurrency)

        async def _limited(name: str) -> PublicationRankResult:
            async with sem:
                return await self.query(name)

        return list(await asyncio.gather(*[_limited(n) for n in names]))


# ---------------------------------------------------------------------------
# UnifiedRankClient
# ---------------------------------------------------------------------------

class UnifiedRankClient:
    """统一查询：Cache → EasyScholar → LLM web_search。"""

    def __init__(
        self,
        cache: RankCache | None = None,
        easyscholar: EasyScholarClient | None = None,
        llm_client: LLMRankClient | None = None,
        *,
        use_easyscholar: bool = True,
        use_llm: bool = True,
    ):
        self._cache = cache or RankCache()
        self._easyscholar = easyscholar or EasyScholarClient()
        self._llm = llm_client or LLMRankClient()
        self._use_easyscholar = use_easyscholar
        self._use_llm = use_llm

    async def init(self) -> None:
        await self._cache.init()

    async def query(self, publication_name: str) -> PublicationRankResult:
        """Cache → EasyScholar → LLM fallback。"""
        try:
            publication_name = _validate_publication_name(publication_name)
        except ValueError as e:
            return PublicationRankResult(
                name=publication_name if isinstance(publication_name, str) else str(publication_name),
                success=False, error=str(e),
            )

        # 1) 查缓存
        cached = await self._cache.get(publication_name)
        if cached is not None:
            if not _is_transient_failure(cached):
                logger.debug("cache hit for %s", publication_name)
                return cached
            logger.info(
                "ignoring transient publication_rank cache failure for %s: %s",
                publication_name, cached.error,
            )

        # 2) EasyScholar
        if self._use_easyscholar:
            try:
                es_result = await asyncio.get_running_loop().run_in_executor(
                    None, self._easyscholar.query, publication_name,
                )
                if es_result.success and (es_result.sci or es_result.ccf):
                    await self._cache.put(es_result, source="easyscholar")
                    logger.info("easyscholar hit for %s", publication_name)
                    return es_result
                if not es_result.success:
                    logger.info(
                        "easyscholar unavailable for %s: %s",
                        publication_name, es_result.error,
                    )
            except Exception as e:
                logger.warning("easyscholar error for %s: %s", publication_name, e)

        # 3) LLM web_search fallback
        if self._use_llm:
            llm_result = await self._llm.query(publication_name)
            source = "llm_websearch"
            if llm_result.success:
                await self._cache.put(llm_result, source=source)
                logger.info("llm hit for %s", publication_name)
            else:
                logger.warning("llm failed for %s: %s", publication_name, llm_result.error)
            return llm_result

        fail = PublicationRankResult(
            name=publication_name, success=False,
            error="所有查询源均已禁用或失败",
        )
        await self._cache.put(fail, source="none")
        return fail

    async def query_batch(
        self, names: list[str], concurrency: int = 3
    ) -> list[PublicationRankResult]:
        """批量查询，缓存命中的直接返回，其余走 EasyScholar/LLM。"""
        results: list[PublicationRankResult | None] = [None] * len(names)

        cache_map = await self._cache.get_batch(names)
        missed_indices: list[int] = []
        for i, name in enumerate(names):
            if cache_map.get(name) is not None:
                results[i] = cache_map[name]
            else:
                missed_indices.append(i)

        sem = asyncio.Semaphore(concurrency)

        async def _query_one(idx: int) -> None:
            async with sem:
                results[idx] = await self.query(names[idx])

        await asyncio.gather(*[_query_one(i) for i in missed_indices])
        return results  # type: ignore[return-value]

    async def close(self) -> None:
        await self._cache.close()
        self._easyscholar.close()

    async def __aenter__(self) -> "UnifiedRankClient":
        await self.init()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

async def query_publication_rank(name: str) -> PublicationRankResult:
    """便捷函数：单条查询。"""
    async with UnifiedRankClient() as client:
        return await client.query(name)


async def query_publication_ranks(names: list[str]) -> list[PublicationRankResult]:
    """便捷函数：批量查询。"""
    async with UnifiedRankClient() as client:
        return await client.query_batch(names)
