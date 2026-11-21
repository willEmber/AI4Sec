"""
统一出版物等级解析引擎（P17）：
  L1: 进程内只读权威目录 (`VenueCatalog` — CCF 2026 / CORE / TH-CPL / CAAI / CCF-CN / JCR / CAS / 预警名单)
  L2: PostgreSQL 动态持久缓存 (`RankCache`，含源等级偏序保护)
  L3: 异步实时接口 (`OpenAlex /sources` ISSN 对齐与定量指标 + `EasyScholar` ISSN 优先异步查询)
  L4: Tavily 网页检索 + Utility LLM 多体系分离抽取（仅单点主动查询开启，标 `unverified`）
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from app.config import get_settings
from app.services.llm_gateway import get_registry
from app.services.llm_service import LLMService

from .catalog import VenueCatalog, get_venue_catalog
from .models import VenueLookupInput, VenueProfile
from .normalizer import is_preprint_venue, is_umbrella_series
from .openalex_source import fetch_openalex_source
from .publication_rank import (
    EasyScholarClient,
    PublicationRankResult,
    _validate_publication_name,
)
from .rank_cache import RankCache
from .tavily_search import TavilySearchClient

logger = logging.getLogger("scholar.llm_rank")

_SYSTEM_PROMPT_EXTRACT = """\
你是一个学术出版物等级提取助手。下面会提供该出版物的网络搜索结果，请严格依据搜索结果分别提取其在各独立评价体系中的等级。

注意各体系严格独立，切勿混淆：
1. jcr: Clarivate JCR 四分位分区（"Q1" / "Q2" / "Q3" / "Q4" 之一，未提及则为 null）
2. cas: 中国科学院文献情报中心分区表大类分区（"1区" / "2区" / "3区" / "4区" 之一，未提及则为 null）
3. ccf: 中国计算机学会推荐国际学术会议和期刊目录等级（"A" / "B" / "C" 之一，未提及则为 null）
4. core: CORE / ICORE 国际计算机会议等级（"A*" / "A" / "B" / "C" 之一，未提及则为 null）

规则：
- 会议（如 CVPR、NeurIPS）通常只有 ccf / core 等级，jcr 与 cas 必须填 null
- 附属 Workshop / Findings 不继承主会等级，无独立等级时填 null
- 未被收录或搜索结果无法确认的字段一律填 null，严禁编造

请严格仅返回如下 JSON 对象，不要包含任何其他说明文字：
{"jcr": "Q1", "cas": "1区", "ccf": "A", "core": null}
"""

_VALID_SCI = {"Q1", "Q2", "Q3", "Q4"}
_VALID_CCF = {"A", "B", "C"}
_VALID_CORE = {"A*", "A", "B", "C"}
_VALID_CAS_RE = re.compile(r"([1-4])\s*区")

_TRANSIENT_FAILURE_MARKERS = (
    "LLM 查询失败",
    "LLM_BASEURL",
    "Tavily 搜索失败",
    "UnsupportedProtocol",
    "ReadTimeout",
    "ConnectError",
    "请求超时",
    "被限速",
    "网络连接错误",
    "HTTP ",
    "web_search",
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
    """从 LLM JSON 响应中提取分离的 JCR / CAS / CCF / CORE 等级（兼容旧版 `sci` 键）。"""
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
            name=publication_name,
            success=False,
            error=f"LLM 返回非法 JSON: {text[:200]}",
        )

    if not isinstance(data, dict):
        return PublicationRankResult(
            name=publication_name,
            success=False,
            error=f"LLM 返回非字典 JSON: {type(data).__name__}",
        )

    # Support both new `jcr` key and legacy `sci` key
    sci_raw = data.get("jcr") if "jcr" in data else data.get("sci")
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

    extra: dict[str, Any] = {}
    cas_raw = data.get("cas")
    if isinstance(cas_raw, str):
        m = _VALID_CAS_RE.search(cas_raw.strip())
        if m:
            extra["cas"] = f"{m.group(1)}区"

    core_raw = data.get("core")
    if isinstance(core_raw, str):
        core_up = core_raw.strip().upper()
        if core_up in _VALID_CORE:
            extra["core"] = core_up

    if not (sci or ccf or extra):
        return PublicationRankResult(
            name=publication_name,
            success=False,
            error="搜索结果中未找到权威等级信息",
            source="llm_websearch",
        )

    return PublicationRankResult(
        name=publication_name,
        sci=sci,
        ccf=ccf,
        success=True,
        extra=extra,
        source="llm_websearch",
    )


# ---------------------------------------------------------------------------
# LLMRankClient
# ---------------------------------------------------------------------------

class LLMRankClient:
    """Tavily 搜网页 + Utility LLM 结构化多体系抽取（L4 最终兜底）。"""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        *,
        timeout: float = 90.0,
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
        raw_model = model if model is not None else get_registry().utility_model
        self.model = next(
            (m.strip() for m in (raw_model or "").split(",") if m.strip()), ""
        )
        self.timeout = timeout
        self._tavily = tavily_client or TavilySearchClient(api_key=tavily_api_key)
        self._llm = llm_service or LLMService(base_url=self.base_url, api_key=self.api_key)

    @staticmethod
    def _build_tavily_query(publication_name: str) -> str:
        return f"{publication_name} 期刊 会议 JCR Quartile 中科院分区 CCF CORE 等级"

    async def _query_via_tavily(self, publication_name: str) -> PublicationRankResult:
        if not self._tavily.configured:
            return PublicationRankResult(
                name=publication_name,
                success=False,
                error="TAVILY_API_KEYS（或 TAVILY_KEY）配置缺失，无法进行 Tavily 网络搜索",
            )

        try:
            context = await self._tavily.search_context(
                self._build_tavily_query(publication_name),
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
                name=publication_name,
                success=False,
                error="Tavily 未返回搜索结果",
            )

        user_prompt = (
            f"出版物名称：{publication_name}\n\n"
            f"以下是该出版物的网络搜索结果：\n{context}\n\n"
            "请依据上述搜索结果，分别提取其 JCR 分区、中科院分区、CCF 等级与 CORE 等级，按要求的 JSON 格式返回。"
        )
        try:
            content = await self._llm.chat(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT_EXTRACT},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.model,
                temperature=0.1,
                max_tokens=4096,
                enable_thinking=False,
                tools=None,
            )
        except Exception as e:
            return PublicationRankResult(
                name=publication_name,
                success=False,
                error=f"LLM 查询失败: {type(e).__name__}: {e}",
            )

        if not content:
            return PublicationRankResult(
                name=publication_name,
                success=False,
                error="LLM 未返回文本内容",
            )

        return _parse_llm_response(content, publication_name)

    async def query(self, publication_name: str) -> PublicationRankResult:
        if not self.base_url.startswith(("http://", "https://")):
            return PublicationRankResult(
                name=publication_name,
                success=False,
                error=(
                    "LLM_BASEURL 配置缺失或无效，请设置包含 http:// 或 "
                    "https:// 协议的完整地址"
                ),
            )
        return await self._query_via_tavily(publication_name)

    async def query_batch(
        self, names: list[str], concurrency: int = 3
    ) -> list[PublicationRankResult]:
        sem = asyncio.Semaphore(concurrency)

        async def _limited(name: str) -> PublicationRankResult:
            async with sem:
                return await self.query(name)

        return list(await asyncio.gather(*[_limited(n) for n in names]))


# ---------------------------------------------------------------------------
# UnifiedRankClient (L1 Catalog -> L2 PG Cache -> L3 EasyScholar/OpenAlex -> L4 LLM)
# ---------------------------------------------------------------------------

class UnifiedRankClient:
    """四层统一等级查询客户端。"""

    def __init__(
        self,
        cache: RankCache | None = None,
        easyscholar: EasyScholarClient | None = None,
        llm_client: LLMRankClient | None = None,
        *,
        catalog: VenueCatalog | None = None,
        use_catalog: bool | None = None,
        use_easyscholar: bool = True,
        use_openalex: bool | None = None,
        use_llm: bool = True,
    ):
        # When unit tests inject fake `cache` or `llm_client` doubles without
        # specifying `use_catalog` or `catalog`, let them test the L2->L4 cascade directly.
        if use_catalog is None:
            use_catalog = catalog is not None or (cache is None and llm_client is None)
        if use_openalex is None:
            use_openalex = cache is None and llm_client is None
        self._catalog = catalog if catalog is not None else (get_venue_catalog() if use_catalog else None)
        self._use_catalog = bool(use_catalog and self._catalog is not None)
        self._cache = cache or RankCache()
        self._easyscholar = easyscholar or EasyScholarClient()
        self._llm = llm_client or LLMRankClient()
        self._use_easyscholar = use_easyscholar
        self._use_openalex = use_openalex
        self._use_llm = use_llm

    async def init(self) -> None:
        await self._cache.init()

    async def resolve_profile(
        self,
        inp: VenueLookupInput | str,
        *,
        allow_network: bool = True,
    ) -> VenueProfile:
        """Resolve full ``VenueProfile`` across L1 -> L2 -> (optional L3/L4)."""
        lookup = VenueLookupInput(venue=inp) if isinstance(inp, str) else inp

        # L1: Read-only in-process catalog (<0.1ms)
        l1_profile: VenueProfile | None = None
        if self._use_catalog and self._catalog is not None:
            l1_profile = self._catalog.resolve(lookup)
            if (
                l1_profile.has_any_rank
                or l1_profile.subtrack in {"workshop", "companion", "findings", "preprint"}
                or l1_profile.venue_type == "book_series"
            ):
                return l1_profile

        effective_name = (
            (l1_profile.canonical_name if l1_profile else "")
            or lookup.venue
            or lookup.crossref_event
            or (lookup.venue_aliases[0] if lookup.venue_aliases else "")
        ).strip()
        if not effective_name:
            return l1_profile or VenueProfile(query_venue=lookup.venue)

        # L2: Persistent cache (PostgreSQL)
        primary_issn = lookup.issns[0] if lookup.issns else ""
        try:
            cached = await self._cache.get(effective_name, issn=primary_issn)
        except TypeError:
            # Custom test-double cache whose `get` accepts `(name)` only
            cached = await self._cache.get(effective_name)
        if cached is not None and not _is_transient_failure(cached):
            prof = cached.to_venue_profile(query_venue=lookup.venue or effective_name)
            prof.is_preprint_file = bool(l1_profile.is_preprint_file if l1_profile else lookup.is_preprint_source)
            if l1_profile and l1_profile.openreview_tier and not prof.openreview_tier:
                prof.openreview_tier = l1_profile.openreview_tier
            return prof

        if not allow_network:
            return l1_profile or VenueProfile(query_venue=lookup.venue, canonical_name=effective_name)

        # L3 + L4 live lookup
        live_res = await self._query_live(
            effective_name,
            issns=list(lookup.issns),
            openalex_source_id=lookup.openalex_source_id,
        )
        prof = live_res.to_venue_profile(query_venue=lookup.venue or effective_name)
        prof.is_preprint_file = bool(l1_profile.is_preprint_file if l1_profile else lookup.is_preprint_source)
        if l1_profile and l1_profile.openreview_tier and not prof.openreview_tier:
            prof.openreview_tier = l1_profile.openreview_tier
        return prof

    async def _put_cache(self, result: PublicationRankResult, source: str, *, issn: str = "") -> None:
        try:
            await self._cache.put(result, source=source, issn=issn)
        except TypeError:
            await self._cache.put(result, source=source)

    async def _query_live(
        self,
        publication_name: str,
        *,
        issns: list[str] | None = None,
        openalex_source_id: str = "",
    ) -> PublicationRankResult:
        clean_issns = list(issns or [])
        oa_meta: dict[str, Any] | None = None

        # Step 3a: OpenAlex `/sources` resolves exact ISSNs + quantitative metrics
        if self._use_openalex and (not clean_issns or openalex_source_id):
            oa_meta = await fetch_openalex_source(
                venue_name=publication_name,
                issns=clean_issns,
                source_id=openalex_source_id,
            )
            if oa_meta:
                for i in oa_meta.get("issns") or []:
                    if i not in clean_issns:
                        clean_issns.append(i)

        primary_issn = clean_issns[0] if clean_issns else ""

        # Step 3b: Async EasyScholar (queries ISSN first when available!)
        if self._use_easyscholar:
            try:
                if hasattr(self._easyscholar, "aquery"):
                    es_result = await self._easyscholar.aquery(publication_name, issns=clean_issns)
                else:
                    es_result = await asyncio.get_running_loop().run_in_executor(
                        None, self._easyscholar.query, publication_name
                    )
                if es_result.success and (es_result.sci or es_result.ccf or es_result.extra):
                    if oa_meta and oa_meta.get("metrics"):
                        es_result.extra.setdefault("metrics", oa_meta["metrics"])
                    if primary_issn:
                        es_result.extra.setdefault("issn_l", primary_issn)
                    await self._put_cache(es_result, source="easyscholar", issn=primary_issn)
                    logger.info("easyscholar hit for %s", publication_name)
                    return es_result
                if not es_result.success:
                    logger.info("easyscholar unavailable for %s: %s", publication_name, es_result.error)
            except Exception as e:
                logger.warning("easyscholar error for %s: %s", publication_name, e)

        # Step 4: Tavily search + LLM extraction fallback
        if self._use_llm:
            llm_result = await self._llm.query(publication_name)
            if llm_result.success:
                if oa_meta and oa_meta.get("metrics"):
                    llm_result.extra.setdefault("metrics", oa_meta["metrics"])
                await self._put_cache(llm_result, source="llm_websearch", issn=primary_issn)
                logger.info("tavily+llm hit for %s", publication_name)
            else:
                logger.warning("tavily+llm failed for %s: %s", publication_name, llm_result.error)
            return llm_result

        # If neither EasyScholar nor LLM yielded tiers, but OpenAlex gave metrics, return them
        if oa_meta and oa_meta.get("metrics"):
            oa_res = PublicationRankResult(
                name=oa_meta.get("display_name") or publication_name,
                sci=None,
                ccf=None,
                success=True,
                extra={
                    "metrics": oa_meta["metrics"],
                    "issn_l": oa_meta.get("issn_l") or primary_issn,
                    "venue_type": oa_meta.get("venue_type") or "",
                },
                source="openalex",
            )
            await self._put_cache(oa_res, source="openalex", issn=primary_issn)
            return oa_res

        fail = PublicationRankResult(
            name=publication_name,
            success=False,
            error="所有查询源均已禁用或未收录",
        )
        await self._put_cache(fail, source="none", issn=primary_issn)
        return fail

    async def query(self, publication_name: str, *, issns: list[str] | None = None) -> PublicationRankResult:
        """L1 Catalog -> L2 Cache -> L3 EasyScholar/OpenAlex -> L4 Tavily+LLM."""
        try:
            publication_name = _validate_publication_name(publication_name)
        except ValueError as e:
            return PublicationRankResult(
                name=publication_name if isinstance(publication_name, str) else str(publication_name),
                success=False,
                error=str(e),
            )

        # 1) L1 Local authoritative catalog
        if self._use_catalog and self._catalog is not None:
            prof = self._catalog.resolve(VenueLookupInput(venue=publication_name, issns=list(issns or [])))
            if prof.has_any_rank:
                return PublicationRankResult.from_venue_profile(prof)
            if prof.subtrack in {"workshop", "companion", "preprint"} or prof.venue_type == "book_series":
                return PublicationRankResult(
                    name=prof.canonical_name or publication_name,
                    success=False,
                    error=f"属于 {prof.subtrack if prof.venue_type != 'book_series' else '会议论文集丛书'}，不适用主会/独立期刊等级",
                    source="local_catalog",
                )
            if prof.subtrack == "findings":
                return PublicationRankResult.from_venue_profile(prof)

        # Guard: even when L1 catalog is disabled, never waste API calls on preprints/LNCS
        if is_preprint_venue(publication_name) or is_umbrella_series(publication_name):
            return PublicationRankResult(
                name=publication_name,
                success=False,
                error="预印本或论文集丛书壳不直接对应单一出版物等级",
            )

        # 2) L2 Cache
        primary_issn = issns[0] if issns else ""
        try:
            cached = await self._cache.get(publication_name, issn=primary_issn)
        except TypeError:
            cached = await self._cache.get(publication_name)
        if cached is not None:
            if not _is_transient_failure(cached):
                if not cached.source:
                    cached.source = "cache"
                return cached
            logger.info(
                "ignoring transient publication_rank cache failure for %s: %s",
                publication_name,
                cached.error,
            )

        # 3) L3 + L4 Live cascade
        return await self._query_live(publication_name, issns=issns)

    async def query_batch(
        self, names: list[str], concurrency: int = 3
    ) -> list[PublicationRankResult]:
        """批量查询：先走 L1 内存库，未命中的批量查 L2 缓存，剩余走异步 L3/L4。"""
        results: list[PublicationRankResult | None] = [None] * len(names)
        after_l1_indices: list[int] = []

        if self._use_catalog and self._catalog is not None:
            for idx, name in enumerate(names):
                try:
                    clean = _validate_publication_name(name)
                except ValueError as e:
                    results[idx] = PublicationRankResult(name=str(name), success=False, error=str(e))
                    continue
                prof = self._catalog.resolve(clean)
                if prof.has_any_rank or prof.subtrack == "findings":
                    results[idx] = PublicationRankResult.from_venue_profile(prof)
                elif prof.subtrack in {"workshop", "companion", "preprint"} or prof.venue_type == "book_series":
                    results[idx] = PublicationRankResult(
                        name=prof.canonical_name or clean,
                        success=False,
                        error="附属轨道、预印本或丛书不适用等级",
                        source="local_catalog",
                    )
                else:
                    after_l1_indices.append(idx)
        else:
            after_l1_indices = list(range(len(names)))

        if not after_l1_indices:
            return results  # type: ignore[return-value]

        l2_names = [names[i] for i in after_l1_indices]
        cache_map = await self._cache.get_batch(l2_names)
        missed_indices: list[int] = []
        for idx in after_l1_indices:
            cached = cache_map.get(names[idx])
            if cached is not None and not _is_transient_failure(cached):
                results[idx] = cached
            else:
                missed_indices.append(idx)

        if missed_indices:
            sem = asyncio.Semaphore(concurrency)

            async def _query_one(i: int) -> None:
                async with sem:
                    results[i] = await self.query(names[i])

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
# High-level convenience functions
# ---------------------------------------------------------------------------

async def query_publication_rank(name: str, *, issns: list[str] | None = None) -> PublicationRankResult:
    """便捷函数：单条查询（穿透 L1 -> L2 -> L3 -> L4）。"""
    async with UnifiedRankClient() as client:
        return await client.query(name, issns=issns)


async def query_publication_ranks(names: list[str], *, use_llm: bool = True) -> list[PublicationRankResult]:
    """便捷函数：批量查询。"""
    async with UnifiedRankClient(use_llm=use_llm) as client:
        return await client.query_batch(names)


async def resolve_venue_profile(
    inp: VenueLookupInput | str,
    *,
    allow_network: bool = True,
    allow_llm: bool = True,
) -> VenueProfile:
    """Resolve one venue into a rich ``VenueProfile``."""
    async with UnifiedRankClient(use_llm=allow_llm) as client:
        return await client.resolve_profile(inp, allow_network=allow_network)


async def resolve_venue_profiles_batch(
    inputs: list[VenueLookupInput | str],
    *,
    allow_network: bool = False,
    concurrency: int = 4,
) -> list[VenueProfile]:
    """Resolve many candidate venues rapidly.

    With ``allow_network=False`` (the default used by search/discovery tools),
    only L1 in-memory catalog and L2 PostgreSQL cache are consulted (<1 ms total),
    so paper search never blocks on external rank APIs.
    """
    if not inputs:
        return []
    cat = get_venue_catalog()
    profiles: list[VenueProfile | None] = [None] * len(inputs)
    unresolved_indices: list[int] = []

    for idx, raw_in in enumerate(inputs):
        lookup = VenueLookupInput(venue=raw_in) if isinstance(raw_in, str) else raw_in
        p = cat.resolve(lookup)
        if (
            p.has_any_rank
            or p.subtrack in {"workshop", "companion", "findings", "preprint"}
            or p.venue_type == "book_series"
        ):
            profiles[idx] = p
        else:
            profiles[idx] = p
            if p.canonical_name:
                unresolved_indices.append(idx)

    if not unresolved_indices:
        return profiles  # type: ignore[return-value]

    async with UnifiedRankClient(use_llm=False) as client:
        lookup_names = [profiles[i].canonical_name for i in unresolved_indices if profiles[i]]
        cache_hits = await client._cache.get_batch(lookup_names)
        still_missing: list[int] = []
        for idx in unresolved_indices:
            base_p = profiles[idx]
            assert base_p is not None
            cached = cache_hits.get(base_p.canonical_name)
            if cached is not None and cached.success:
                resolved = cached.to_venue_profile(query_venue=base_p.query_venue)
                resolved.is_preprint_file = base_p.is_preprint_file
                resolved.openreview_tier = base_p.openreview_tier or resolved.openreview_tier
                profiles[idx] = resolved
            elif allow_network:
                still_missing.append(idx)

        if allow_network and still_missing:
            sem = asyncio.Semaphore(concurrency)

            async def _fetch(i: int) -> None:
                async with sem:
                    profiles[i] = await client.resolve_profile(inputs[i], allow_network=True)

            await asyncio.gather(*[_fetch(i) for i in still_missing])

    return profiles  # type: ignore[return-value]
