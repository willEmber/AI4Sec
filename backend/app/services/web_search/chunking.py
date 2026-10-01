"""Splitting a fetched page into citable chunks and picking the relevant ones.

A page is not summarised. A summary is the model's paraphrase, and an answer
citing it cites something no reader can find on the page. Chunks are the
page's own text, cut at headings and paragraph breaks, so each one can become
a piece of evidence that quotes the source verbatim; what the agent gets is
the few chunks that match its question, not the whole page.

Ranking is lexical first, then the backend's rerank model over the best
lexical candidates when one is configured — the same model and gateway
`paper_search` reranks with.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

logger = logging.getLogger("scholar.web_search.chunking")

TARGET_CHUNK_CHARS = 1200
MAX_CHUNK_CHARS = 2000
# Lexical candidates handed to the rerank model; beyond this the rerank call
# costs more than the ordering it improves.
RERANK_CANDIDATES = 30

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_MD_LINK = re.compile(r"!?\[([^\]]*)\]\(([^)\s]*)(?:\s+\"[^\"]*\")?\)")
# A chunk whose visible text is mostly link labels is a menu, a table of
# contents or a footer. Extractors that keep the whole page (Tavily's basic
# extract returned 290k characters of navigation for a 3k-character docs page)
# produce hundreds of these, and they outrank real text on any query that
# happens to name a menu item.
NAV_LINK_SHARE = 0.6
NAV_MIN_LINKS = 3
_LATIN = re.compile(r"[a-z0-9][a-z0-9\-_.]*[a-z0-9]|[a-z0-9]")
_CJK = re.compile(r"[㐀-鿿]+")
_SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it its of on or that the this "
    "to was were what when where which who why with does do did can not".split()
)


@dataclass(frozen=True)
class PageChunk:
    index: int
    heading_path: str
    text: str
    # Character offsets into the page Markdown, so a caller can page through
    # the document by offset and a chunk can be located again.
    start: int
    end: int
    # Mostly links: navigation, not content. Skipped when choosing chunks.
    navigation: bool = False


def clean_heading(text: str) -> str:
    """A heading as a reader sees it: link labels kept, targets, permalink
    pilcrows and stray anchors dropped."""
    text = _MD_LINK.sub(lambda m: "" if m.group(1).strip() in ("¶", "#", "") else m.group(1), text)
    return re.sub(r"\s+", " ", text.replace("¶", "")).strip(" #")


def is_navigation(text: str) -> bool:
    links = _MD_LINK.findall(text)
    if len(links) < NAV_MIN_LINKS:
        return False
    visible = _MD_LINK.sub(lambda m: m.group(1), text)
    label_chars = sum(len(label.strip()) for label, _ in links)
    content = re.sub(r"[\s*\-|>#\[\]()`_]+", "", visible)
    labels = re.sub(r"[\s*\-|>#\[\]()`_]+", "", "".join(label for label, _ in links))
    return label_chars > 0 and len(labels) >= NAV_LINK_SHARE * max(len(content), 1)


def split_markdown(
    markdown: str,
    *,
    target_chars: int = TARGET_CHUNK_CHARS,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[PageChunk]:
    """Chunks that never cross a heading and stay near `target_chars`."""
    text = markdown or ""
    # (start, end, heading path) for each paragraph, in document order.
    paragraphs: list[tuple[int, int, str]] = []
    headings: list[tuple[int, str]] = []
    pos = 0
    para_start: int | None = None
    para_end = 0

    def close_paragraph() -> None:
        nonlocal para_start
        if para_start is not None:
            paragraphs.append((para_start, para_end, " › ".join(h for _, h in headings)))
            para_start = None

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        m = _HEADING.match(stripped)
        if m:
            close_paragraph()
            level = len(m.group(1))
            headings[:] = [(lvl, h) for lvl, h in headings if lvl < level]
            headings.append((level, clean_heading(m.group(2))))
        elif not stripped:
            close_paragraph()
        else:
            if para_start is None:
                para_start = pos
            para_end = pos + len(line.rstrip("\r\n"))
        pos += len(line)
    close_paragraph()

    chunks: list[PageChunk] = []

    def emit(start: int, end: int, path: str) -> None:
        body = text[start:end].strip()
        if body:
            chunks.append(PageChunk(len(chunks), path, body, start, end, is_navigation(body)))

    group_start: int | None = None
    group_end = 0
    group_path = ""
    for start, end, path in paragraphs:
        if end - start > max_chars:
            if group_start is not None:
                emit(group_start, group_end, group_path)
                group_start = None
            for s, e in _hard_split(text, start, end, max_chars):
                emit(s, e, path)
            continue
        if group_start is not None and (path != group_path or end - group_start > target_chars):
            emit(group_start, group_end, group_path)
            group_start = None
        if group_start is None:
            group_start, group_path = start, path
        group_end = end
    if group_start is not None:
        emit(group_start, group_end, group_path)
    return chunks


def _hard_split(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    """Cut an over-long paragraph at sentence ends, or at `max_chars` when a
    single sentence is longer than that (tables, minified text)."""
    spans: list[tuple[int, int]] = []
    cursor = start
    while end - cursor > max_chars:
        window = text[cursor : cursor + max_chars]
        cuts = [m.start() for m in _SENTENCE_END.finditer(window)]
        cut = cuts[-1] if cuts and cuts[-1] > max_chars // 3 else max_chars
        spans.append((cursor, cursor + cut))
        cursor += cut
    spans.append((cursor, end))
    return spans


def tokens(text: str) -> list[str]:
    """Latin words without stopwords, plus CJK character bigrams (Chinese has
    no spaces, and a bigram is the smallest unit that still means something)."""
    lowered = (text or "").lower()
    out = [t for t in _LATIN.findall(lowered) if t not in _STOPWORDS]
    for run in _CJK.findall(text or ""):
        if len(run) == 1:
            out.append(run)
        out.extend(run[i : i + 2] for i in range(len(run) - 1))
    return out


def lexical_scores(chunks: list[PageChunk], question: str) -> list[float]:
    """TF-IDF-style overlap between the question and each chunk; a match in
    the chunk's heading counts extra, since headings name what follows."""
    q_terms = set(tokens(question))
    if not q_terms or not chunks:
        return [0.0] * len(chunks)
    bodies = [tokens(c.text) for c in chunks]
    heads = [set(tokens(c.heading_path)) for c in chunks]
    n = len(chunks)
    df = {t: sum(1 for b in bodies if t in b) for t in q_terms}
    scores: list[float] = []
    for body, head in zip(bodies, heads):
        counts: dict[str, int] = {}
        for t in body:
            if t in q_terms:
                counts[t] = counts.get(t, 0) + 1
        score = 0.0
        for t in q_terms:
            idf = math.log(1 + n / (1 + df[t]))
            if counts.get(t):
                score += idf * (1 + math.log(counts[t]))
            if t in head:
                score += 0.5 * idf
        # Mild length normalisation: a long chunk matches more by chance.
        scores.append(score / math.sqrt(1 + len(body) / 200))
    return scores


async def rank_chunks(
    chunks: list[PageChunk],
    question: str,
    *,
    limit: int,
    use_rerank: bool = True,
) -> tuple[list[tuple[PageChunk, float]], str]:
    """The `limit` chunks most relevant to `question`, best first, and which
    ranker produced the order: `rerank`, `lexical`, or `order` (no question)."""
    if not chunks:
        return [], "order"
    content = [c for c in chunks if not c.navigation] or chunks
    if not question.strip():
        return [(c, 0.0) for c in content[:limit]], "order"

    chunks = content
    lexical = lexical_scores(chunks, question)
    ranked = sorted(zip(chunks, lexical), key=lambda cs: cs[1], reverse=True)
    if use_rerank and len(chunks) > 1:
        candidates = [c for c, _ in ranked[:RERANK_CANDIDATES]]
        reranked = await _rerank(candidates, question, limit)
        if reranked:
            return reranked, "rerank"
    return ranked[:limit], "lexical"


async def _rerank(
    candidates: list[PageChunk], question: str, limit: int
) -> list[tuple[PageChunk, float]] | None:
    from app.services.paper_search.http_client import HTTPClient
    from app.services.paper_search.llm import LLMConfig, rerank
    from app.services.search_settings import backend_search_settings

    settings = backend_search_settings()
    if not (settings.llm_base_url and settings.rerank_model):
        return None
    cfg = LLMConfig(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        max_retries=int(settings.rerank_max_retries),
        retry_base_delay=settings.llm_retry_base_delay,
        retry_max_delay=settings.llm_retry_max_delay,
        rerank_url=settings.rerank_url,
    )
    docs = [
        (f"{c.heading_path}\n" if c.heading_path else "") + c.text[: settings.rerank_max_doc_chars]
        for c in candidates
    ]
    try:
        ranked = await rerank(
            HTTPClient(timeout=settings.rerank_timeout_s),
            cfg=cfg,
            model=settings.rerank_model,
            query=question,
            documents=docs,
            top_n=limit,
        )
    except Exception as exc:  # noqa: BLE001 — lexical order is a working fallback
        logger.warning("page chunk rerank failed, using lexical order: %s", exc)
        return None
    picked = [(candidates[i], score) for i, score in ranked if 0 <= i < len(candidates)]
    return picked or None


def chunks_from_offset(
    chunks: list[PageChunk], offset: int, max_chars: int
) -> tuple[list[PageChunk], int | None]:
    """Consecutive content chunks starting at character `offset`, within
    `max_chars`, and the offset to continue from (`None` at the end of the
    page). Navigation chunks are passed over.

    Always returns at least one chunk when any remain, so paging cannot stall
    on a chunk larger than the budget.
    """
    picked: list[PageChunk] = []
    used = 0
    for c in chunks:
        if c.end <= offset or c.navigation:
            continue
        if picked and used + len(c.text) > max_chars:
            return picked, c.start
        picked.append(c)
        used += len(c.text)
    return picked, None
