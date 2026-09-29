"""Fetched pages, kept for a while so paging through one does not re-bill it.

`read_web_page` is called repeatedly on the same URL — the next `offset`, a
different `question` — and every provider charges per fetch. This is an
in-process LRU with a TTL; P7 step 3 moves it into SQLite so it also survives
a restart and is shared between workers.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

from app.services.web_search.models import WebPage
from app.services.web_search.urls import normalize_url

PAGE_TTL_SECONDS = 7 * 24 * 3600.0
MAX_PAGES = 64


class PageCache:
    def __init__(self, *, ttl: float = PAGE_TTL_SECONDS, max_items: int = MAX_PAGES) -> None:
        self.ttl = ttl
        self.max_items = max_items
        self._items: OrderedDict[str, tuple[float, WebPage]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, url: str) -> WebPage | None:
        key = normalize_url(url)
        with self._lock:
            hit = self._items.get(key)
            if hit is None:
                return None
            stored, page = hit
            if time.monotonic() - stored > self.ttl:
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return page

    def put(self, page: WebPage) -> None:
        now = time.monotonic()
        keys = {normalize_url(page.url)}
        if page.final_url:
            keys.add(normalize_url(page.final_url))
        with self._lock:
            for key in keys:
                self._items[key] = (now, page)
                self._items.move_to_end(key)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


PAGE_CACHE = PageCache()
