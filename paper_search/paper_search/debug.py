from __future__ import annotations

import os
import sys


def debug_enabled() -> bool:
    return os.getenv("PAPERSEARCH_DEBUG", "").strip().lower() in {"1", "true", "yes", "y", "on"}


def debug(msg: str) -> None:
    if not debug_enabled():
        return
    print(f"[paper_search] {msg}", file=sys.stderr)

