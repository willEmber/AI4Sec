from __future__ import annotations

import json
import re
import asyncio
import random
from dataclasses import dataclass
from typing import Any

from .models import Paper
from .utils import normalize_whitespace
from .http_client import HTTPClient
from .debug import debug
from .http_client import HTTPStatusError


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    max_retries: int = 5
    retry_base_delay: float = 1.0
    retry_max_delay: float = 30.0


def _endpoint(base_url: str, path: str) -> str:
    base = (base_url or "").rstrip("/")
    if not base:
        return path
    return f"{base}{path}"


def _parse_json_array(text: str) -> list[Any] | None:
    text = (text or "").strip()
    if not text:
        return None

    # Strip common Markdown code fences.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        data = json.loads(text)
        return data if isinstance(data, list) else None
    except Exception:
        pass

    # Try to extract the first JSON array substring.
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, list) else None
    except Exception:
        return None


async def chat_completion(
    client: HTTPClient,
    *,
    cfg: LLMConfig,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> str:
    url = _endpoint(cfg.base_url, "/chat/completions")
    headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
    payload: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    delay = max(cfg.retry_base_delay, 0.0)
    attempt = 0
    while True:
        attempt += 1
        try:
            data = await client.post_json(url, json_body=payload, headers=headers)
            return (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        except HTTPStatusError as e:
            retryable = e.status_code in {429, 500, 502, 503, 504}
            if retryable and attempt <= max(cfg.max_retries, 0):
                wait = e.retry_after if e.retry_after is not None else delay
                wait = min(max(wait, 0.0), cfg.retry_max_delay)
                wait *= random.uniform(0.8, 1.2)  # jitter
                debug(f"LLM HTTP {e.status_code}; retry in {wait:.2f}s (attempt {attempt}/{cfg.max_retries}).")
                await asyncio.sleep(wait)
                delay = min(max(delay * 2, 0.5), cfg.retry_max_delay)
                continue
            raise
        except Exception as e:
            if attempt <= max(cfg.max_retries, 0):
                wait = min(max(delay, 0.0), cfg.retry_max_delay)
                wait *= random.uniform(0.8, 1.2)
                debug(f"LLM request error; retry in {wait:.2f}s (attempt {attempt}/{cfg.max_retries}). error={e}")
                await asyncio.sleep(wait)
                delay = min(max(delay * 2, 0.5), cfg.retry_max_delay)
                continue
            raise


def _batched(items: list[str], batch_size: int) -> list[list[str]]:
    size = max(int(batch_size), 1)
    return [items[i : i + size] for i in range(0, len(items), size)]


async def embeddings(
    client: HTTPClient,
    *,
    cfg: LLMConfig,
    model: str,
    texts: list[str],
    batch_size: int = 32,
) -> list[list[float]]:
    if not texts:
        return []
    if not (cfg.base_url and model):
        return []

    url = _endpoint(cfg.base_url, "/embeddings")
    headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}

    all_vecs: list[list[float]] = []
    delay = max(cfg.retry_base_delay, 0.0)

    for chunk in _batched(texts, batch_size):
        payload: dict[str, Any] = {"model": model, "input": chunk}
        attempt = 0
        while True:
            attempt += 1
            try:
                data = await client.post_json(url, json_body=payload, headers=headers)
                items = data.get("data")
                if not isinstance(items, list):
                    raise RuntimeError(f"embeddings response missing data list: {type(items)}")
                items_sorted = sorted(
                    (it for it in items if isinstance(it, dict)),
                    key=lambda x: x.get("index", 0),
                )
                for it in items_sorted:
                    emb = it.get("embedding")
                    if not isinstance(emb, list):
                        continue
                    all_vecs.append([float(v) for v in emb])
                break
            except HTTPStatusError as e:
                retryable = e.status_code in {429, 500, 502, 503, 504}
                if retryable and attempt <= max(cfg.max_retries, 0):
                    wait = e.retry_after if e.retry_after is not None else delay
                    wait = min(max(wait, 0.0), cfg.retry_max_delay)
                    wait *= random.uniform(0.8, 1.2)  # jitter
                    debug(
                        f"Embeddings HTTP {e.status_code}; retry in {wait:.2f}s "
                        f"(attempt {attempt}/{cfg.max_retries})."
                    )
                    await asyncio.sleep(wait)
                    delay = min(max(delay * 2, 0.5), cfg.retry_max_delay)
                    continue
                raise
            except Exception as e:
                if attempt <= max(cfg.max_retries, 0):
                    wait = min(max(delay, 0.0), cfg.retry_max_delay)
                    wait *= random.uniform(0.8, 1.2)
                    debug(
                        f"Embeddings request error; retry in {wait:.2f}s "
                        f"(attempt {attempt}/{cfg.max_retries}). error={e}"
                    )
                    await asyncio.sleep(wait)
                    delay = min(max(delay * 2, 0.5), cfg.retry_max_delay)
                    continue
                raise

    if len(all_vecs) != len(texts):
        raise RuntimeError(f"embeddings length mismatch: got {len(all_vecs)} want {len(texts)}")
    return all_vecs


async def rerank(
    client: HTTPClient,
    *,
    cfg: LLMConfig,
    model: str,
    query: str,
    documents: list[str],
    top_n: int,
    instruction: str | None = None,
) -> list[tuple[int, float]]:
    if not documents:
        return []
    if not (cfg.base_url and model and query):
        return []

    url = _endpoint(cfg.base_url, "/rerank")
    headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
    payload: dict[str, Any] = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": min(max(int(top_n), 1), len(documents)),
        "return_documents": False,
    }
    if instruction:
        payload["instruction"] = instruction

    delay = max(cfg.retry_base_delay, 0.0)
    attempt = 0
    while True:
        attempt += 1
        try:
            data = await client.post_json(url, json_body=payload, headers=headers)
            results: Any = data.get("results")
            if results is None:
                results = data.get("data")
            if isinstance(results, dict):
                results = results.get("results") or results.get("data") or results.get("items")
            if not isinstance(results, list):
                raise RuntimeError(f"unexpected rerank response: {type(results)}")

            ranked: list[tuple[int, float]] = []
            for it in results:
                if not isinstance(it, dict):
                    continue
                idx = it.get("index")
                if idx is None:
                    idx = it.get("document_index")
                score = it.get("relevance_score")
                if score is None:
                    score = it.get("score")
                if isinstance(idx, int) and isinstance(score, (int, float)):
                    ranked.append((idx, float(score)))
            ranked.sort(key=lambda x: x[1], reverse=True)
            return ranked[: min(max(int(top_n), 1), len(ranked))]
        except HTTPStatusError as e:
            retryable = e.status_code in {429, 500, 502, 503, 504}
            if retryable and attempt <= max(cfg.max_retries, 0):
                wait = e.retry_after if e.retry_after is not None else delay
                wait = min(max(wait, 0.0), cfg.retry_max_delay)
                wait *= random.uniform(0.8, 1.2)  # jitter
                debug(f"Rerank HTTP {e.status_code}; retry in {wait:.2f}s (attempt {attempt}/{cfg.max_retries}).")
                await asyncio.sleep(wait)
                delay = min(max(delay * 2, 0.5), cfg.retry_max_delay)
                continue
            raise
        except Exception as e:
            if attempt <= max(cfg.max_retries, 0):
                wait = min(max(delay, 0.0), cfg.retry_max_delay)
                wait *= random.uniform(0.8, 1.2)
                debug(f"Rerank request error; retry in {wait:.2f}s (attempt {attempt}/{cfg.max_retries}). error={e}")
                await asyncio.sleep(wait)
                delay = min(max(delay * 2, 0.5), cfg.retry_max_delay)
                continue
            raise


def _rank_prompt(query: str, papers: list[Paper], limit: int) -> list[dict[str, str]]:
    items = []
    for idx, p in enumerate(papers):
        items.append(
            {
                "i": idx,
                "title": normalize_whitespace(p.title),
                "abstract": normalize_whitespace(p.abstract)[:1200],
                "doi": p.doi,
                "url": p.url,
            }
        )
    user = {
        "query": query,
        "limit": limit,
        "papers": items,
        "output": "Return ONLY a JSON array of integers: the selected indices in descending relevance order.",
    }
    return [
        {
            "role": "system",
            "content": "You are a scientific literature ranking assistant. Rank by relevance to the query.",
        },
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


async def rank_papers(
    client: HTTPClient,
    *,
    cfg: LLMConfig,
    model: str,
    query: str,
    papers: list[Paper],
    limit: int,
) -> tuple[list[Paper], bool]:
    if not (cfg.base_url and cfg.api_key and model):
        return papers[:limit], False

    try:
        content = await chat_completion(
            client,
            cfg=cfg,
            model=model,
            messages=_rank_prompt(query, papers, limit),
            temperature=0.0,
            max_tokens=256,
        )
        indices = _parse_json_array(content)
        if not indices:
            debug("LLM rank returned empty/invalid JSON; fallback to default order.")
            return papers[:limit], False
        picked: list[Paper] = []
        seen: set[int] = set()
        for x in indices:
            if not isinstance(x, int):
                continue
            if x < 0 or x >= len(papers) or x in seen:
                continue
            picked.append(papers[x])
            seen.add(x)
            if len(picked) >= limit:
                break
        if not picked:
            debug("LLM rank did not select any valid indices; fallback to default order.")
            return papers[:limit], False

        # Fill remaining slots with original order to ensure stable length.
        for i, p in enumerate(papers):
            if len(picked) >= limit:
                break
            if i in seen:
                continue
            picked.append(p)
        debug(f"LLM rank applied successfully. picked={len(picked)}/{limit}")
        return picked[:limit], True
    except Exception as e:
        debug(f"LLM rank failed; fallback to default order. error={e}")
        return papers[:limit], False


def _summary_prompt(paper: Paper, *, max_abstract_chars: int) -> list[dict[str, str]]:
    payload = {
        "title": normalize_whitespace(paper.title),
        "abstract": normalize_whitespace(paper.abstract)[: max(int(max_abstract_chars), 0)],
        "output": (
            "请仅根据给定论文的标题与摘要，用中文写一段客观摘要（3-6 句）。"
            "只输出摘要正文，不要提及查询/主题/相关性/搜索意图，不要添加“摘要：”等前缀，不要使用 Markdown。"
        ),
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a scientific paper summarization assistant. "
                "Summarize based only on the given title and abstract. "
                "Return only the Chinese summary text (no preface, no relevance commentary)."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _batch_summary_prompt(papers: list[Paper], *, max_abstract_chars: int) -> list[dict[str, str]]:
    items: list[dict[str, Any]] = []
    for idx, p in enumerate(papers):
        items.append(
            {
                "i": idx,
                "title": normalize_whitespace(p.title),
                "abstract": normalize_whitespace(p.abstract)[: max(int(max_abstract_chars), 0)],
                "doi": normalize_whitespace(p.doi),
                "url": normalize_whitespace(p.url),
            }
        )
    payload = {
        "papers": items,
        "output": (
            "Return ONLY a JSON array of objects. Each object must be: "
            '{"i": <int>, "agent_remark": <string>}. '
            "For each paper, write a Chinese abstract-only summary (3-6 sentences) based only on its title and abstract. "
            "Do NOT mention query/topic/relevance, do NOT add prefixes like “摘要：”, and do NOT use Markdown."
        ),
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a scientific paper summarization assistant. "
                "For each item, summarize based only on the given title and abstract. "
                "The agent_remark must contain only the summary text."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


async def generate_agent_remark(
    client: HTTPClient,
    *,
    cfg: LLMConfig,
    model: str,
    query: str,
    paper: Paper,
    max_abstract_chars: int = 3000,
) -> str:
    if not (cfg.base_url and cfg.api_key and model):
        return ""
    try:
        content = await chat_completion(
            client,
            cfg=cfg,
            model=model,
            messages=_summary_prompt(paper, max_abstract_chars=max_abstract_chars),
            temperature=0.3,
            max_tokens=512,
        )
        return normalize_whitespace(content)
    except Exception as e:
        debug(f"LLM summary failed; title={normalize_whitespace(paper.title)[:80]} error={e}")
        return ""


async def generate_agent_remarks_batch(
    client: HTTPClient,
    *,
    cfg: LLMConfig,
    model: str,
    query: str,
    papers: list[Paper],
    max_abstract_chars: int = 3000,
) -> dict[int, str] | None:
    if not papers:
        return {}
    if not (cfg.base_url and cfg.api_key and model):
        return None

    try:
        content = await chat_completion(
            client,
            cfg=cfg,
            model=model,
            messages=_batch_summary_prompt(papers, max_abstract_chars=max_abstract_chars),
            temperature=0.3,
            max_tokens=min(512 * len(papers), 4096),
        )
        parsed = _parse_json_array(content)
        if not parsed:
            debug("LLM batch summary returned empty/invalid JSON; fallback to per-paper.")
            return None

        mapping: dict[int, str] = {}
        for idx, item in enumerate(parsed):
            if isinstance(item, str):
                mapping[idx] = normalize_whitespace(item)
                continue
            if not isinstance(item, dict):
                continue
            i = item.get("i")
            remark = item.get("agent_remark") or item.get("remark") or item.get("summary") or ""
            if isinstance(i, int) and isinstance(remark, str):
                mapping[i] = normalize_whitespace(remark)

        if not mapping:
            return None
        return mapping
    except Exception as e:
        debug(f"LLM batch summary failed; fallback to per-paper. error={e}")
        return None
