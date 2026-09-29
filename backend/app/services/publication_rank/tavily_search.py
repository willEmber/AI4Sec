"""
Tavily Search API 异步客户端。

用于替代大模型 API 内置的 web_search 能力：先用 Tavily 拉取网页搜索结果，
再把结果交给 LLM 做结构化抽取（见 :mod:`llm_rank`）。
"""
from __future__ import annotations

import logging
from typing import Any

from app.services.web_search.credentials import TAVILY, web_search_enabled
from app.services.web_search.http import new_client
from app.services.web_search.keypool import KeyPool, pool_for, single_key_pool
from app.services.web_search.providers.tavily import search_raw

logger = logging.getLogger("scholar.tavily")


class TavilySearchClient:
    """对 Tavily ``/search`` 接口的轻量封装。

    不显式传 ``api_key`` 时使用进程级 key 池（``TAVILY_API_KEYS``，兼容
    ``TAVILY_API_KEY`` / ``TAVILY_KEY``）：某个 key 被拒、限流或额度用尽时自动换下一个。
    以前只读 ``TAVILY_KEY``，而 ``.env`` 里配的是 ``TAVILY_API_KEYS``，这条兜底路径因此从未生效。
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout: float = 30.0,
        pool: KeyPool | None = None,
    ):
        if pool is not None:
            self._pool = pool
        elif api_key is not None:
            self._pool = single_key_pool(TAVILY, api_key)
        else:
            self._pool = pool_for(TAVILY)
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return self._pool.configured and web_search_enabled()

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
            RuntimeError: 未配置 Tavily key，或网页检索已关闭。
            ProviderError: 所有 key 都不可用，或服务端持续出错。
        """
        if not self.configured:
            raise RuntimeError("TAVILY_API_KEYS 未配置或网页检索已关闭，无法进行 Tavily 网络搜索")

        payload: dict[str, Any] = {
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": include_answer,
        }
        async with new_client(self.timeout) as client:
            return await search_raw(client, self._pool, payload)

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
