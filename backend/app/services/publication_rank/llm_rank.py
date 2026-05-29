"""
Tavily 网络搜索 + LLM 抽取 查询出版物等级 + 统一客户端（Cache → EasyScholar → Tavily+LLM）。

先用 Tavily Search 拉取期刊/会议的等级相关网页，再交给 LLM 抽取为
SCI 分区 / CCF 等级。取代旧版直接依赖大模型 API 内置 web_search 工具的实现。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from app.config import get_settings
from app.services.llm_service import LLMService

from .publication_rank import (
    EasyScholarClient,
    PublicationRankResult,
    _validate_publication_name,
)
from .rank_cache import RankCache
from .tavily_search import TavilySearchClient

logger = logging.getLogger("scholar.llm_rank")

_SYSTEM_PROMPT = """\
你是一个学术出版物等级查询助手。下面会提供该出版物的网络搜索结果，请仅依据搜索结果确定其等级信息。

查询内容：
1. SCI 分区（中科院最新大类分区，Q1/Q2/Q3/Q4 之一）
2. CCF 等级（中国计算机学会推荐目录最新版，A/B/C 之一）

规则：
- 会议（如 CVPR、NeurIPS）通常有 CCF 等级但没有 SCI 分区
- 期刊通常有 SCI 分区，部分也有 CCF 等级
- 未被收录、搜索结果中无法确定的字段返回 null
- 不要编造，只依据搜索结果作答

请严格按以下 JSON 格式返回，不要包含其他文字：
{"sci": "Q1", "ccf": "A"}
"""

_VALID_SCI = {"Q1", "Q2", "Q3", "Q4"}
_VALID_CCF = {"A", "B", "C"}
_TRANSIENT_FAILURE_MARKERS = (
    "LLM 查询失败",
    "LLM_BASEURL",
    "Tavily 搜索失败",
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

class LLMRankClient:
    """通过 Tavily 网络搜索 + LLM 结构化抽取查询出版物等级。

    流程：

    1. 用 :class:`TavilySearchClient` 搜索该期刊/会议的 SCI 分区与 CCF 等级相关网页；
    2. 把搜索结果摘要交给 LLM（标准 Responses API，无内置搜索）抽取为
       ``{"sci": ..., "ccf": ...}`` JSON。
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        *,
        timeout: float = 60.0,
        tavily_api_key: str | None = None,
        tavily_client: TavilySearchClient | None = None,
        llm_service: LLMService | None = None,
    ):
        settings = get_settings()
        self.base_url = (
            base_url if base_url is not None else settings.llm_base_url
        ).strip().rstrip("/")
        self.api_key = (
            api_key if api_key is not None else settings.llm_api_key
        ).strip()
        raw_model = model if model is not None else getattr(settings, "thinking_model", "")
        # THINKING_MODELNAME may be a comma-separated list; use the first entry.
        self.model = next(
            (m.strip() for m in (raw_model or "").split(",") if m.strip()), ""
        )
        self.timeout = timeout
        self._tavily = tavily_client or TavilySearchClient(api_key=tavily_api_key)
        self._llm = llm_service or LLMService(base_url=self.base_url, api_key=self.api_key)

    @staticmethod
    def _build_query(publication_name: str) -> str:
        return f"{publication_name} 期刊 会议 中科院 SCI 分区 CCF 推荐等级"

    async def query(self, publication_name: str) -> PublicationRankResult:
        """查询单个出版物的 SCI/CCF 等级。"""
        # Cheap config guards first (no network). Keep the transient markers so a
        # misconfiguration is not cached as a permanent negative result.
        if not self.base_url.startswith(("http://", "https://")):
            return PublicationRankResult(
                name=publication_name,
                success=False,
                error=(
                    "LLM_BASEURL 配置缺失或无效，请设置包含 http:// 或 "
                    "https:// 协议的完整地址"
                ),
            )
        if not self._tavily.configured:
            return PublicationRankResult(
                name=publication_name,
                success=False,
                error="TAVILY_KEY 配置缺失，无法进行 Tavily 网络搜索",
            )

        # 1) Tavily web search
        try:
            context = await self._tavily.search_context(
                self._build_query(publication_name),
                max_results=5,
                search_depth="basic",
            )
        except Exception as e:
            return PublicationRankResult(
                name=publication_name,
                success=False,
                error=f"Tavily 搜索失败: {type(e).__name__}: {e}",
            )

        if not context.strip():
            return PublicationRankResult(
                name=publication_name, success=False,
                error="Tavily 未返回搜索结果",
            )

        # 2) LLM structured extraction (standard chat, no built-in web search)
        user_prompt = (
            f"出版物名称：{publication_name}\n\n"
            f"以下是该出版物的网络搜索结果：\n{context}\n\n"
            "请依据上述搜索结果，提取其 SCI 分区与 CCF 等级，并按要求的 JSON 格式返回。"
        )
        try:
            content = await self._llm.chat(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.model,
                temperature=0.1,
                max_tokens=256,
            )
        except Exception as e:
            return PublicationRankResult(
                name=publication_name,
                success=False,
                error=f"LLM 查询失败: {type(e).__name__}: {e}",
            )

        if not content:
            return PublicationRankResult(
                name=publication_name, success=False,
                error="LLM 未返回文本内容",
            )

        return _parse_llm_response(content, publication_name)

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
    """统一查询：Cache → EasyScholar → Tavily+LLM。"""

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
        """Cache → EasyScholar → Tavily+LLM fallback。"""
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

        # 3) Tavily + LLM fallback
        if self._use_llm:
            llm_result = await self._llm.query(publication_name)
            source = "tavily_llm"
            if llm_result.success:
                await self._cache.put(llm_result, source=source)
                logger.info("tavily+llm hit for %s", publication_name)
            else:
                logger.warning("tavily+llm failed for %s: %s", publication_name, llm_result.error)
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
        """批量查询，缓存命中的直接返回，其余走 EasyScholar/Tavily+LLM。"""
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
