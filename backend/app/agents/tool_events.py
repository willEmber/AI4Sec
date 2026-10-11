"""What a tool call looks like to the reader while a turn runs.

The model gets a tool's full result; the browser gets enough to follow the
work: what was asked for (`summarise_input`), and the first few things that
came back (`preview_of`). Both are small and bounded — a section read can be
tens of kilobytes, and every event is a persisted row.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk

MAX_INPUT_KEYS = 10
MAX_INPUT_CHARS = 200
MAX_INPUT_LIST_ITEMS = 6
MAX_PREVIEW_ITEMS = 5
MAX_PREVIEW_TITLE_CHARS = 140
MAX_PREVIEW_TEXT_CHARS = 180

# The result lists a preview is taken from, first match wins.
_LIST_KEYS = (
    "hits", "results", "papers", "blocks", "chunks", "sections", "turns",
    "rankings", "reviews", "memories",
)
_TITLE_KEYS = ("title", "paper_title", "heading", "section", "question", "name", "level", "session_title")
_TEXT_KEYS = (
    "abstract_excerpt", "snippet", "text", "quote", "answer_excerpt", "content",
    "summary", "value",
)
_META_KEYS = ("year", "venue", "source", "domain", "published_date")


def _clip(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def summarise_input(args: dict[str, Any]) -> dict[str, Any]:
    """A call's arguments, cut down to what a status row can show.

    Unset arguments are dropped — `year_from=0` or an empty list is the tool's
    default, not something the model asked for.
    """
    summary: dict[str, Any] = {}
    for key, value in args.items():
        if len(summary) >= MAX_INPUT_KEYS:
            break
        if isinstance(value, bool):
            if value:
                summary[key] = True
        elif isinstance(value, (int, float)):
            if value:
                summary[key] = value
        elif isinstance(value, str):
            if value.strip():
                summary[key] = _clip(value, MAX_INPUT_CHARS)
        elif isinstance(value, list):
            items = [
                _clip(v, 80)
                for v in value[:MAX_INPUT_LIST_ITEMS]
                if isinstance(v, (str, int, float)) and not isinstance(v, bool) and str(v).strip()
            ]
            if items:
                summary[key] = items
    return summary


def _first(item: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = item.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _preview_item(item: dict[str, Any], *, parent_title: str = "") -> dict[str, Any] | None:
    title = _first(item, _TITLE_KEYS)
    text = _first(item, _TEXT_KEYS)
    if isinstance(title, list):
        title = " › ".join(str(part) for part in title)
    if title is None and text is None:
        return None
    entry: dict[str, Any] = {}
    if title is not None:
        entry["title"] = _clip(title, MAX_PREVIEW_TITLE_CHARS)
    if text is not None and not isinstance(text, (list, dict)):
        entry["text"] = _clip(text, MAX_PREVIEW_TEXT_CHARS)
    page = item.get("page")
    if isinstance(page, int) and not isinstance(page, bool) and page > 0:
        entry["page"] = page
    url = item.get("url")
    if isinstance(url, str) and url.startswith(("http://", "https://")):
        entry["url"] = url[:500]
    meta = [
        _clip(item[key], 60)
        for key in _META_KEYS
        if isinstance(item.get(key), (str, int)) and not isinstance(item.get(key), bool) and item.get(key)
    ]
    if parent_title:
        meta.insert(0, _clip(parent_title, 60))
    if meta:
        entry["meta"] = " · ".join(meta[:3])
    return entry or None


def preview_of(data: dict[str, Any]) -> list[dict[str, Any]]:
    """The first few entries of a result, as short labelled lines.

    A search across several papers nests its hits per paper; those are
    flattened one hit per paper first, so the preview shows every paper that
    answered rather than five passages of the first.
    """
    for key in _LIST_KEYS:
        items = data.get(key)
        if not isinstance(items, list) or not items:
            continue
        rows = [item for item in items if isinstance(item, dict)]
        preview: list[dict[str, Any]] = []
        if rows and all(isinstance(row.get("hits"), list) for row in rows):
            depth = 0
            while len(preview) < MAX_PREVIEW_ITEMS and any(len(r["hits"]) > depth for r in rows):
                for row in rows:
                    if depth < len(row["hits"]) and isinstance(row["hits"][depth], dict):
                        entry = _preview_item(
                            row["hits"][depth], parent_title=str(row.get("title") or "")
                        )
                        if entry and len(preview) < MAX_PREVIEW_ITEMS:
                            preview.append(entry)
                depth += 1
            return preview
        for row in rows:
            entry = _preview_item(row)
            if entry:
                preview.append(entry)
            if len(preview) >= MAX_PREVIEW_ITEMS:
                break
        return preview
    return []


def _parse_args(raw: str) -> dict[str, Any] | None:
    """The arguments once they are complete; None while they are still streaming."""
    if not raw.strip():
        return None
    try:
        parsed = json.loads(raw)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


class ToolCallTracker:
    """Announces each tool call of a turn once, together with its arguments.

    A streaming model sends a call as fragments: the name first, then the
    arguments a few characters at a time. A call is announced when its
    arguments parse — a prefix of a JSON object never does — so the reader
    sees *what* is being searched for while the search runs, not only that one
    is. A non-streaming model delivers finished `tool_calls`, announced as they
    are. `drain` covers a call whose arguments never parsed (some gateways send
    none for a tool that takes none): it is announced without them when its
    result arrives.
    """

    def __init__(self) -> None:
        self._pending: dict[str, dict[str, str]] = {}
        self._announced: set[str] = set()

    def _announce(self, key: str, call_id: str) -> bool:
        if key in self._announced or (call_id and call_id in self._announced):
            return False
        self._announced.add(key)
        if call_id:
            self._announced.add(call_id)
        return True

    def feed(self, message: AIMessage | AIMessageChunk) -> list[tuple[str, str, dict[str, Any]]]:
        """`(name, call_id, arguments)` for every call this message completes."""
        ready: list[tuple[str, str, dict[str, Any]]] = []
        chunks = getattr(message, "tool_call_chunks", None) or []
        for chunk in chunks:
            key = f"{message.id or ''}:{chunk.get('index') or 0}"
            entry = self._pending.setdefault(key, {"name": "", "id": "", "args": ""})
            entry["name"] = entry["name"] or (chunk.get("name") or "")
            entry["id"] = entry["id"] or (chunk.get("id") or "")
            entry["args"] += chunk.get("args") or ""
            if not entry["name"] or key in self._announced:
                continue
            args = _parse_args(entry["args"])
            if args is not None and self._announce(key, entry["id"]):
                ready.append((entry["name"], entry["id"], args))
        if chunks:
            return ready
        for position, call in enumerate(getattr(message, "tool_calls", None) or []):
            name = call.get("name") or ""
            if not name:
                continue
            call_id = call.get("id") or ""
            key = call_id or f"{message.id or ''}:{name}:{position}"
            if self._announce(key, call_id):
                args = call.get("args")
                ready.append((name, call_id, args if isinstance(args, dict) else {}))
        return ready

    def drain(self) -> list[tuple[str, str, dict[str, Any]]]:
        """Calls seen but never announced, without arguments."""
        ready: list[tuple[str, str, dict[str, Any]]] = []
        for key, entry in self._pending.items():
            if entry["name"] and self._announce(key, entry["id"]):
                ready.append((entry["name"], entry["id"], {}))
        return ready
