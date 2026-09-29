"""The open web: searching it, and reading a page found there.

These reach what the paper tools cannot — code repositories and project pages,
leaderboards, blog posts, dataset licences, venue deadlines, news after the
model's cutoff. They follow the same three rules as the discovery tools.

*Evidence level.* Everything here is `external_web`: no page number, no peer
review. A search result's snippet and a page's chunk are recorded verbatim, so
an answer citing one cites text a reader can find at that URL. Pages are cut
into chunks and ranked for the question, never summarised: a summary is the
model's own paraphrase, and a citation to it would point at nothing.

*Coverage.* A provider that was rate-limited, out of credit or refused the key
has not searched. That is reported per provider and makes the result
`partial`; an undated page under a date filter the provider could not apply is
set apart, not passed off as matching.

*Reach.* `read_web_page` only opens URLs that already appeared in the reader's
messages or in a tool result (`services.web_search.provenance`), so text in a
paper or a page cannot make the agent send session data to an address it
composed. Paper URLs (arXiv, DOI, OpenReview, ACL Anthology) are turned back
toward `resolve_paper` / `download_paper`, which yield page-numbered full text
instead of page-less web text. Both tools have their own per-run ceiling.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any

from langchain.tools import ToolRuntime, tool

from app.agents.context import AgentContext
from app.db import agent_repository as repo
from app.db import database as db
from app.models.agent_models import ErrorCode, MessageRole, ToolResult
from app.models.evidence_models import Locator
from app.services import evidence_service
from app.services.paper_search.models import PlatformStatus
from app.services.web_search import (
    DEPTHS,
    TOPICS,
    PageChunk,
    WebPage,
    WebSearchRequest,
    chunks_from_offset,
    domain_of,
    fetch_page,
    fetchable_reason,
    rank_chunks,
    scholarly_identifiers,
    search_web,
    split_markdown,
)
from app.services.web_search.cache import PAGE_CACHE
from app.services.web_search.credentials import web_search_enabled
from app.services.web_search.provenance import extract_urls, url_key, urls_in_messages

logger = logging.getLogger("scholar.agents.tools.web")

MAX_QUERY_CHARS = 400
MAX_RESULTS = 10
MAX_DOMAINS = 10
# What the model sees of a snippet; the evidence keeps the whole of it.
SNIPPET_EXCERPT_CHARS = 700
DEFAULT_PAGE_CHARS = 6000
MIN_PAGE_CHARS = 1000
MAX_PAGE_CHARS = 12_000
# Undated results under a date filter shown beside the matches.
MAX_UNKNOWN_DATE = 3

# A provider that returned nothing because it could not search.
_NOT_SEARCHED = {"failed", "auth_failed", "rate_limited", "quota_exhausted", "skipped_no_key", "unsupported"}

_WEB_CAVEAT = (
    "Web content is not peer-reviewed. Attribute it to its site, and prefer a "
    "paper's own text when the two disagree."
)


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _disabled() -> str:
    return ToolResult.unavailable(
        ErrorCode.FORBIDDEN,
        "Web search is turned off on this deployment. Answer from the papers "
        "and paper-search tools, and say the web was not consulted.",
    ).to_json()


def _parse_date(value: str, field: str) -> date | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        raise ValueError(f"{field} must be YYYY-MM-DD, got {value!r}") from None


def _clean_domains(values: list[str] | None) -> tuple[str, ...]:
    out: list[str] = []
    for raw in values or []:
        if not isinstance(raw, str):
            continue
        d = raw.strip().lower()
        d = d.split("://", 1)[-1].split("/", 1)[0].lstrip(".")
        if d and d not in out:
            out.append(d)
    return tuple(out[:MAX_DOMAINS])


def _status_dicts(statuses: list[PlatformStatus]) -> list[dict[str, Any]]:
    out = []
    for s in statuses:
        entry: dict[str, Any] = {"provider": s.platform, "status": s.status}
        if s.count:
            entry["count"] = s.count
        if s.detail:
            entry["detail"] = s.detail[:200]
        out.append(entry)
    return out


def _coverage_note(statuses: list[PlatformStatus]) -> str:
    missing = [s for s in statuses if s.status in _NOT_SEARCHED]
    if not missing:
        return ""
    parts = "; ".join(f"{s.platform} ({s.status})" for s in missing)
    return f"Not every provider could search: {parts}."


def _excerpt(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


@tool(parse_docstring=False)
async def web_search(
    query: str,
    runtime: ToolRuntime[AgentContext],
    topic: str = "general",
    max_results: int = 6,
    start_date: str = "",
    end_date: str = "",
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    depth: str = "fast",
) -> str:
    """Search the open web — for what is not in a paper database.

    Use it for code repositories and project pages, leaderboards, blog posts,
    documentation, dataset licences, conference deadlines, and news after your
    knowledge cutoff. To find papers, use search_papers instead: it returns
    identifiers that download_paper can use.

    - topic: general (default), news (recent events), or technical (the page
      about a specific project, tool, person or dataset).
    - start_date / end_date: YYYY-MM-DD publication range. Pages whose date is
      unknown and could not be checked are listed apart, under unknown_date.
    - include_domains / exclude_domains: e.g. ["github.com"].
    - depth: fast (one provider) or deep (two providers merged; costs double —
      use it only when a fast search came back thin).

    Each result has a snippet and an evidence_id; cite it as [ev_xxx]. To read
    more of a page, pass its url to read_web_page. The query leaves this
    system: never paste unpublished text from the reader's papers into it.
    """
    ctx = runtime.context
    query = " ".join((query or "").split())
    if not query:
        return ToolResult.failed(ErrorCode.INVALID_ARGUMENT, "query must not be empty.", retryable=False).to_json()
    if len(query) > MAX_QUERY_CHARS:
        return ToolResult.failed(
            ErrorCode.INVALID_ARGUMENT,
            f"query is {len(query)} characters; keep it under {MAX_QUERY_CHARS} — a search "
            "query, not a passage.",
            retryable=False,
        ).to_json()
    topic = (topic or "general").strip().lower()
    depth = (depth or "fast").strip().lower()
    if topic not in TOPICS:
        return ToolResult.failed(
            ErrorCode.INVALID_ARGUMENT, f"topic must be one of: {', '.join(TOPICS)}", retryable=False
        ).to_json()
    if depth not in DEPTHS:
        return ToolResult.failed(
            ErrorCode.INVALID_ARGUMENT, f"depth must be one of: {', '.join(DEPTHS)}", retryable=False
        ).to_json()
    try:
        start = _parse_date(start_date, "start_date")
        end = _parse_date(end_date, "end_date")
    except ValueError as exc:
        return ToolResult.failed(ErrorCode.INVALID_ARGUMENT, str(exc), retryable=False).to_json()
    if start and end and start > end:
        return ToolResult.failed(
            ErrorCode.INVALID_ARGUMENT, "start_date is after end_date.", retryable=False
        ).to_json()

    if not web_search_enabled():
        return _disabled()
    cost = 2 if depth == "deep" else 1
    if ctx.usage.web_searches + cost > ctx.budget.max_web_searches:
        return ToolResult.failed(
            ErrorCode.BUDGET_EXCEEDED,
            f"This run has used {ctx.usage.web_searches} of its {ctx.budget.max_web_searches} "
            "web searches. Answer from what you have, and say what was not searched.",
            retryable=False,
        ).to_json()

    req = WebSearchRequest(
        query=query,
        max_results=max(1, min(int(max_results or 6), MAX_RESULTS)),
        topic=topic,
        depth=depth,
        start_date=start,
        end_date=end,
        include_domains=_clean_domains(include_domains),
        exclude_domains=_clean_domains(exclude_domains),
    )
    try:
        outcome = await search_web(req)
    except ValueError as exc:
        return ToolResult.failed(ErrorCode.INVALID_ARGUMENT, str(exc)[:300], retryable=False).to_json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("web_search failed for %r: %s", query, exc)
        return ToolResult.failed(ErrorCode.UPSTREAM_ERROR, f"Web search failed: {exc}"[:300]).to_json()

    if any(s.status not in {"skipped_no_key", "unsupported"} for s in outcome.providers):
        ctx.usage.web_searches += cost

    statuses = _status_dicts(outcome.providers)
    if not outcome.searched:
        if all(s.status == "skipped_no_key" for s in outcome.providers):
            return ToolResult.unavailable(
                ErrorCode.FORBIDDEN,
                "No web search provider has an API key configured. Answer without the web, "
                "and say so.",
                data={"providers": statuses},
            ).to_json()
        limited = all(s.status in {"rate_limited", "quota_exhausted"} for s in outcome.providers)
        return ToolResult.failed(
            ErrorCode.RATE_LIMITED if limited else ErrorCode.UPSTREAM_ERROR,
            "No provider could search: "
            + "; ".join(f"{s.platform} {s.status}" for s in outcome.providers),
            retryable=limited,
        ).to_json()

    evidence_ids: list[str] = []

    async def shape(result: Any) -> dict[str, Any]:
        quote = result.snippet or result.title
        evidence = await evidence_service.record_web_evidence(
            quote=quote,
            source_url=result.url,
            owner_id=ctx.owner_id,
            session_id=ctx.session_id,
            provider=f"{result.provider.lower()}:search",
            locator=Locator(section_path=result.title),
            owner_scoped=True,
        )
        evidence_ids.append(evidence.evidence_id)
        entry = result.to_dict()
        entry["snippet"] = _excerpt(entry["snippet"], SNIPPET_EXCERPT_CHARS)
        entry["evidence_id"] = evidence.evidence_id
        ids = scholarly_identifiers(result.url)
        if ids:
            # A paper found through the web: the paper tools give it page
            # numbers and a literature_id.
            entry["paper_identifiers"] = ids
        return entry

    results = [await shape(r) for r in outcome.results]
    unknown = [await shape(r) for r in outcome.unknown_date[:MAX_UNKNOWN_DATE]]

    data: dict[str, Any] = {
        "query": query,
        "topic": topic,
        "depth": depth,
        "as_of": _today(),
        "providers": statuses,
        "filter": {
            "start_date": start.isoformat() if start else None,
            "end_date": end.isoformat() if end else None,
            "include_domains": list(req.include_domains) or None,
            "exclude_domains": list(req.exclude_domains) or None,
            "applied_by_provider": outcome.remote_filters or None,
        },
        "results": results,
        "unknown_date": unknown,
        "counts": {
            "returned": len(results),
            "unknown_date": len(outcome.unknown_date),
            "excluded": outcome.excluded,
        },
    }

    notes: list[str] = []
    coverage = _coverage_note(outcome.providers)
    if coverage:
        notes.append(coverage)
    if unknown:
        notes.append(
            f"{len(outcome.unknown_date)} result(s) have no date the filter could be checked "
            "against; they are under unknown_date. Say the date is unknown if you use one."
        )
    if any("paper_identifiers" in r for r in results + unknown):
        notes.append(
            "Some results are papers (see paper_identifiers): use resolve_paper and "
            "download_paper to read them with page numbers."
        )
    if not results and not unknown:
        notes.append("Nothing matched. Try other terms, or relax the filters.")
    else:
        notes.append(_WEB_CAVEAT)
    note = " ".join(notes)

    if coverage or unknown:
        return ToolResult.partial(data, note=note, evidence_ids=evidence_ids).to_json()
    return ToolResult.ok(data, evidence_ids=evidence_ids, note=note).to_json()


async def _known_urls(runtime: ToolRuntime[AgentContext]) -> set[str]:
    """Normalised URLs the reader or a tool has put in this session.

    The live conversation state covers this thread; the stored user messages
    and evidence cover whatever a compaction has since summarised away.
    """
    ctx = runtime.context
    state = runtime.state or {}
    messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
    keys = urls_in_messages(messages)

    for message in await repo.list_messages(ctx.session_id):
        if message.role == MessageRole.USER:
            keys.update(url_key(u) for u in extract_urls(message.content, bare=True))
    rows = await db.fetch_all(
        """SELECT source_url, quote FROM evidence
           WHERE session_id = ? AND (source_url != '' OR quote LIKE '%http%')""",
        (ctx.session_id,),
    )
    for row in rows:
        if row["source_url"]:
            keys.add(url_key(row["source_url"]))
        keys.update(url_key(u) for u in extract_urls(row["quote"] or ""))
    return keys


def _pick_ranked(ranked: list[tuple[PageChunk, float]], max_chars: int) -> list[PageChunk]:
    picked: list[PageChunk] = []
    used = 0
    for chunk, _score in ranked:
        if picked and used + len(chunk.text) > max_chars:
            continue
        picked.append(chunk)
        used += len(chunk.text)
    return picked


@tool(parse_docstring=False)
async def read_web_page(
    url: str,
    runtime: ToolRuntime[AgentContext],
    question: str = "",
    max_chars: int = DEFAULT_PAGE_CHARS,
    offset: int = 0,
) -> str:
    """Read a web page as text, in citable chunks.

    Only URLs that already appeared in this conversation can be read — from the
    reader's messages, a web_search result, or another tool's output. To read
    a page you have not seen, find it with web_search first.

    - question: what you want from the page. The chunks most relevant to it
      are returned, best first. Without a question, the page is returned in
      order from `offset`; continue with the `next_offset` it gives back.
    - max_chars: how much text to return (1000–12000, default 6000).

    Paper links (arXiv, DOI, OpenReview, ACL Anthology) are not read here: use
    resolve_paper and download_paper, which give page-numbered full text. Each
    chunk has an evidence_id; cite it as [ev_xxx].
    """
    ctx = runtime.context
    url = (url or "").strip()
    reason = fetchable_reason(url)
    if reason:
        return ToolResult.failed(ErrorCode.INVALID_ARGUMENT, f"Cannot read {url!r}: {reason}.", retryable=False).to_json()

    ids = scholarly_identifiers(url)
    if ids:
        return ToolResult.unavailable(
            ErrorCode.INVALID_ARGUMENT,
            "This is a paper. Use resolve_paper with these identifiers, then download_paper "
            "and ensure_paper_parsed, to read it with page numbers.",
            data={"url": url, "paper_identifiers": ids},
        ).to_json()

    if url_key(url) not in await _known_urls(runtime):
        return ToolResult.failed(
            ErrorCode.FORBIDDEN,
            "This URL has not appeared in the conversation, so it cannot be read. Only "
            "URLs from the reader's messages or from tool results can be opened — use "
            "web_search to find the page, then read a URL it returns.",
            retryable=False,
        ).to_json()

    if not web_search_enabled():
        return _disabled()

    max_chars = max(MIN_PAGE_CHARS, min(int(max_chars or DEFAULT_PAGE_CHARS), MAX_PAGE_CHARS))
    offset = max(0, int(offset or 0))
    question = " ".join((question or "").split())[:MAX_QUERY_CHARS]

    page: WebPage | None = PAGE_CACHE.get(url)
    attempts: list[dict[str, Any]] = []
    cached = page is not None
    if page is None:
        if ctx.usage.web_fetches >= ctx.budget.max_web_fetches:
            return ToolResult.failed(
                ErrorCode.BUDGET_EXCEEDED,
                f"This run has read {ctx.usage.web_fetches} web page(s), which is its limit. "
                "Work with the pages already read.",
                retryable=False,
            ).to_json()
        try:
            outcome = await fetch_page(url)
        except ValueError as exc:
            return ToolResult.failed(ErrorCode.INVALID_ARGUMENT, str(exc), retryable=False).to_json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("read_web_page failed for %s: %s", url, exc)
            return ToolResult.failed(ErrorCode.UPSTREAM_ERROR, f"Reading the page failed: {exc}"[:300]).to_json()
        attempts = _status_dicts(outcome.attempts)
        if any(a.status not in {"skipped_no_key", "unsupported"} for a in outcome.attempts):
            ctx.usage.web_fetches += 1
        page = outcome.page
        if page is None:
            if not any(a.searched for a in outcome.attempts):
                limited = all(a.status in {"rate_limited", "quota_exhausted"} for a in outcome.attempts)
                return ToolResult.failed(
                    ErrorCode.RATE_LIMITED if limited else ErrorCode.UPSTREAM_ERROR,
                    "No provider could read the page: "
                    + "; ".join(f"{a.platform} {a.status}" for a in outcome.attempts),
                    retryable=limited,
                ).to_json()
            return ToolResult.unavailable(
                ErrorCode.UPSTREAM_ERROR,
                "The page could not be read (it may be gone, blocked or login-only). Use the "
                "search snippet if you have one, and say the page itself was not read.",
                data={"url": url, "attempts": attempts},
            ).to_json()
        PAGE_CACHE.put(page)

    chunks = split_markdown(page.markdown)
    if question:
        ranked, ranker = await rank_chunks(chunks, question, limit=max(1, max_chars // 400))
        picked = _pick_ranked(ranked, max_chars)
        next_offset = None
    else:
        picked, next_offset = chunks_from_offset(chunks, offset, max_chars)
        ranker = "order"

    source_url = page.final_url or page.url
    title = page.title or domain_of(source_url)
    evidence_ids: list[str] = []
    out_chunks: list[dict[str, Any]] = []
    for chunk in picked:
        path = " › ".join(p for p in (title, chunk.heading_path) if p)
        evidence = await evidence_service.record_web_evidence(
            quote=chunk.text,
            source_url=source_url,
            owner_id=ctx.owner_id,
            session_id=ctx.session_id,
            provider=f"{page.provider.lower()}:page",
            locator=Locator(section_path=path),
            owner_scoped=True,
        )
        evidence_ids.append(evidence.evidence_id)
        out_chunks.append(
            {
                "evidence_id": evidence.evidence_id,
                "heading": chunk.heading_path,
                "offset": chunk.start,
                "text": chunk.text,
            }
        )

    data: dict[str, Any] = {
        "url": url,
        "final_url": source_url if source_url != url else None,
        "domain": domain_of(source_url),
        "title": page.title,
        "published_date": page.published_date,
        "date_known": page.published_date is not None,
        "provider": page.provider,
        "cached": cached,
        "total_chars": len(page.markdown),
        "chunks_total": len(chunks),
        "navigation_chunks_skipped": sum(1 for c in chunks if c.navigation),
        "question": question or None,
        "ranker": ranker,
        "chunks": out_chunks,
        "next_offset": next_offset,
    }
    if attempts and len(attempts) > 1:
        data["attempts"] = attempts

    notes = [_WEB_CAVEAT]
    if len(page.markdown.strip()) < 200:
        notes.insert(0, "The page returned very little text; it may need JavaScript or a login.")
    if not out_chunks:
        notes.insert(0, "Nothing left to read at this offset.")
    return ToolResult.ok(data, evidence_ids=evidence_ids, note=" ".join(notes)).to_json()


WEB_TOOLS = [web_search, read_web_page]
