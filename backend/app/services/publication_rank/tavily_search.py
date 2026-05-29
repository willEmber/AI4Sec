"""
Tavily Search API 异步客户端。

用于替代大模型 API 内置的 web_search 能力：先用 Tavily 拉取网页搜索结果，
再把结果交给 LLM 做结构化抽取（见 :mod:`llm_rank`）。
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger("scholar.tavily")

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class TavilySearchClient:
    """对 Tavily ``/search`` 接口的轻量封装。

    认证使用请求体中的 ``api_key`` 字段（兼容所有 Tavily API 版本）。
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout: float = 30.0,
        max_retries: int = 2,
        search_url: str = TAVILY_SEARCH_URL,
    ):
        settings = get_settings()
        self.api_key = (
            api_key if api_key is not None else settings.tavily_api_key
        ).strip()
        self.timeout = timeout
        self.max_retries = max_retries
        self.search_url = search_url

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def _is_retryable(status_code: int) -> bool:
        return status_code in {408, 429, 500, 502, 503, 504}

    @staticmethod
    def _compute_delay(attempt: int) -> float:
        return min(1.0 * (2 ** attempt), 20.0) * random.uniform(0.8, 1.2)

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        search_depth: str = "basic",
        include_answer: bool = True,
    ) -> dict[str, Any]:
        """调用 Tavily 搜索，返回原始 JSON。

        Raises:
            RuntimeError: 未配置 ``TAVILY_KEY``。
            httpx.HTTPError: 网络或服务端错误（重试耗尽后抛出）。
        """
        if not self.api_key:
            raise RuntimeError("TAVILY_KEY 未配置，无法进行 Tavily 网络搜索")

        payload: dict[str, Any] = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": include_answer,
        }
        timeout = httpx.Timeout(connect=15.0, read=self.timeout, write=15.0, pool=15.0)

        attempt = 0
        while True:
            attempt += 1
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(self.search_url, json=payload)
                if self._is_retryable(resp.status_code) and attempt <= self.max_retries:
                    delay = self._compute_delay(attempt)
                    logger.warning(
                        "Tavily: HTTP %d (attempt %d/%d), retry in %.1fs",
                        resp.status_code, attempt, self.max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                resp.raise_for_status()
                return resp.json()
            except (httpx.ReadTimeout, httpx.ConnectError) as e:
                if attempt > self.max_retries:
                    raise
                delay = self._compute_delay(attempt)
                logger.warning(
                    "Tavily: %s (attempt %d/%d), retry in %.1fs",
                    type(e).__name__, attempt, self.max_retries, delay,
                )
                await asyncio.sleep(delay)

    async def search_context(self, query: str, **kwargs: Any) -> str:
        """搜索并把结果整理为适合喂给 LLM 的纯文本上下文。"""
        data = await self.search(query, **kwargs)

        parts: list[str] = []
        answer = data.get("answer")
        if answer:
            parts.append(f"搜索摘要: {answer}")

        for i, result in enumerate(data.get("results", []), 1):
            title = (result.get("title") or "").strip()
            content = (result.get("content") or "").strip()
            url = (result.get("url") or "").strip()
            block = f"[{i}] {title}".rstrip()
            if content:
                block += f"\n{content}"
            if url:
                block += f"\n来源: {url}"
            parts.append(block)

        return "\n\n".join(parts)
