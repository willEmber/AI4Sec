from __future__ import annotations

import logging
from typing import Any


class _Logger:
    """
    Minimal logger adapter.

    - Supports `logger.info("x={}", 1)` style formatting used by loguru.
    - Does not configure global logging; callers should configure `logging` as needed.
    """

    def __init__(self, name: str = "paper_search") -> None:
        self._logger = logging.getLogger(name)

    def debug(self, msg: str, *args: Any) -> None:
        self._logger.debug(self._fmt(msg, *args))

    def info(self, msg: str, *args: Any) -> None:
        self._logger.info(self._fmt(msg, *args))

    def warning(self, msg: str, *args: Any) -> None:
        self._logger.warning(self._fmt(msg, *args))

    def error(self, msg: str, *args: Any) -> None:
        self._logger.error(self._fmt(msg, *args))

    @staticmethod
    def _fmt(msg: str, *args: Any) -> str:
        if not args:
            return str(msg)
        try:
            return str(msg).format(*args)
        except Exception:
            # Best-effort: don't let logging formatting crash business logic.
            return f"{msg} {args}"


logger = _Logger()

