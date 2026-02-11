from __future__ import annotations

"""
Credential helpers.

This project intentionally does not hardcode any API keys/emails. Instead:
  - Prefer OS environment variables
  - Optionally load a local `.env` file (NOT committed) for convenience
"""

import os
from pathlib import Path

_DOTENV_LOADED = False


def find_dotenv(start_dir: str | Path | None = None, *, filename: str = ".env") -> Path | None:
    """
    Find a dotenv file by walking up from start_dir (or cwd) to filesystem root.
    Returns the first match, or None.
    """
    start = Path(start_dir) if start_dir is not None else Path.cwd()
    try:
        start = start.resolve()
    except OSError:
        start = start.absolute()

    for d in (start, *start.parents):
        p = d / filename
        if p.is_file():
            return p
    return None


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    s = (line or "").strip()
    if not s or s.startswith("#"):
        return None
    if s.startswith("export "):
        s = s[len("export ") :].strip()

    if "=" not in s:
        return None
    key, value = s.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def load_dotenv(dotenv_path: str | Path) -> None:
    """
    Load dotenv key-value pairs into os.environ (does NOT override existing vars).
    """
    path = Path(dotenv_path)
    if not path.is_file():
        return

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="ignore")

    for raw_line in text.splitlines():
        parsed = _parse_dotenv_line(raw_line)
        if not parsed:
            continue
        key, value = parsed
        if key not in os.environ:
            os.environ[key] = value


def load_dotenv_if_present() -> Path | None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return None
    _DOTENV_LOADED = True

    path = find_dotenv()
    if not path:
        return None
    load_dotenv(path)
    return path


def env_first(*names: str) -> str:
    """
    Return the first non-empty env var in names (after attempting `.env` load).
    """
    load_dotenv_if_present()
    for name in names:
        v = (os.getenv(name) or "").strip()
        if v:
            return v
    return ""

