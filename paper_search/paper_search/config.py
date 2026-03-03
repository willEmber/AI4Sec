from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
import random
from threading import Lock
from typing import Final


_LOADED_ENV_FILES: set[Path] = set()
_RR_LOCK: Lock = Lock()
_RR_COUNTERS: dict[str, int] = {}


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_env_file(env_path: str | os.PathLike[str] = ".env", *, override: bool = False) -> None:
    """
    Minimal .env loader (dotenv-compatible enough for this project).
    - Supports KEY=VALUE and KEY = "VALUE" (spaces around '=').
    - Ignores blank lines and comments starting with '#'.
    """
    path = Path(env_path)
    if not path.exists():
        return
    if not override and path in _LOADED_ENV_FILES:
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_quotes(value.strip())
        if not key:
            continue
        if not override and key in os.environ:
            continue
        os.environ[key] = value
    _LOADED_ENV_FILES.add(path)


DEFAULT_USER_AGENT: Final[str] = "PaperSearch/0.1"


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    val = raw.strip().lower()
    if val in {"1", "true", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        return default


def _maybe_env_int(key: str) -> int | None:
    raw = os.getenv(key)
    if raw is None:
        return None
    try:
        return int(str(raw).strip())
    except ValueError:
        return None


def _norm_platform(name: str) -> str:
    return "".join(ch for ch in (name or "").casefold() if ch.isalnum())


def _parse_csv(raw: str) -> list[str]:
    return [v.strip() for v in (raw or "").split(",") if v.strip()]


def _dedupe_keep_order(values: list[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        if v in seen:
            continue
        out.append(v)
        seen.add(v)
    return tuple(out)


def _pick_from_pool(pool: tuple[str, ...], *, key: str, strategy: str) -> str:
    if not pool:
        return ""
    if len(pool) == 1:
        return pool[0]

    s = (strategy or "").strip().casefold()
    if s in {"random", "rand"}:
        return random.choice(pool)
    if s in {"first"}:
        return pool[0]

    # default: round-robin
    with _RR_LOCK:
        idx = _RR_COUNTERS.get(key, 0)
        _RR_COUNTERS[key] = idx + 1
    return pool[idx % len(pool)]


@dataclass(frozen=True)
class Settings:
    per_platform: int = 10
    final_limit: int = 10
    final_limit_max: int = 50
    platform_limits: dict[str, int] = field(default_factory=dict)
    query_max_chars: int = 512
    platforms_max: int = 10

    # Contact identity (polite pool / rate-limit)
    contact_emails: tuple[str, ...] = ()
    tool_name: str = "papersearch"
    email_pick_strategy: str = "round_robin"  # first | round_robin | random

    # Semantic Scholar
    semantic_api_key: str = ""

    # IEEE Xplore
    ieee_api_key: str = ""
    ieee_per_second_limit: int = 10
    ieee_daily_limit: int = 200

    # OpenAlex/Crossref
    openalex_mailtos: tuple[str, ...] = ()
    crossref_mailtos: tuple[str, ...] = ()

    # LLM (OpenAI-compatible)
    llm_base_url: str = ""
    llm_api_key: str = ""
    rank_model: str = ""
    embed_model: str = ""
    rerank_model: str = ""

    # Rerank controls
    rerank_timeout_s: float = 30.0
    rerank_max_retries: int = 1
    rerank_max_doc_chars: int = 3000

    # LLM runtime controls
    llm_max_concurrency: int = 1
    llm_max_retries: int = 5
    llm_retry_base_delay: float = 1.0
    llm_retry_max_delay: float = 30.0

    # DOI enrichment knobs (Crossref)
    doi_enrich_enabled: bool = True
    doi_enrich_timeout_s: float = 5.0
    doi_enrich_max_concurrency: int = 2

    # Storage
    storage_dir: Path = Path("storage")
    pdf_dirname: str = "pdfs"

    @property
    def pdf_dir(self) -> Path:
        return self.storage_dir / self.pdf_dirname

    def limit_for_platform(self, platform: str) -> int:
        key = _norm_platform(platform)
        limit = self.platform_limits.get(key)
        if isinstance(limit, int) and limit > 0:
            return limit
        return max(int(self.per_platform), 1)

    def pick_contact_email(self) -> str:
        return _pick_from_pool(self.contact_emails, key="contact_email", strategy=self.email_pick_strategy)

    def pick_openalex_mailto(self) -> str:
        return _pick_from_pool(self.openalex_mailtos, key="openalex_mailto", strategy=self.email_pick_strategy)

    def pick_crossref_mailto(self) -> str:
        return _pick_from_pool(self.crossref_mailtos, key="crossref_mailto", strategy=self.email_pick_strategy)

    @staticmethod
    def from_env() -> "Settings":
        email_pick_strategy = (os.getenv("PAPERSEARCH_EMAIL_PICK_STRATEGY") or "").strip() or "round_robin"

        contact_emails = _dedupe_keep_order(
            _parse_csv(os.getenv("PAPERSEARCH_CONTACT_EMAIL") or "")
            + _parse_csv(os.getenv("PAPERSEARCH_CONTACT_EMAILS") or "")
        )

        openalex_mailtos = _dedupe_keep_order(
            _parse_csv(os.getenv("PAPERSEARCH_OPENALEX_MAILTO") or "")
            + _parse_csv(os.getenv("PAPERSEARCH_OPENALEX_MAILTOS") or "")
        )
        crossref_mailtos = _dedupe_keep_order(
            _parse_csv(os.getenv("PAPERSEARCH_CROSSREF_MAILTO") or "")
            + _parse_csv(os.getenv("PAPERSEARCH_CROSSREF_MAILTOS") or "")
        )

        # Fallback to the global contact email pool when per-service pool is not provided.
        if not openalex_mailtos:
            openalex_mailtos = contact_emails
        if not crossref_mailtos:
            crossref_mailtos = contact_emails

        storage_dir = Path(os.getenv("PAPERSEARCH_STORAGE_DIR", "storage")).expanduser()

        per_platform = _env_int("PAPERSEARCH_PER_PLATFORM_LIMIT", 10)
        final_limit_max = _env_int("PAPERSEARCH_FINAL_LIMIT_MAX", 50)
        if final_limit_max <= 0:
            final_limit_max = 50

        final_limit = _env_int("PAPERSEARCH_FINAL_LIMIT", 10)
        if final_limit <= 0:
            final_limit = 10
        final_limit = min(final_limit, final_limit_max)

        query_max_chars = _env_int("PAPERSEARCH_QUERY_MAX_CHARS", 512)
        platforms_max = _env_int("PAPERSEARCH_PLATFORMS_MAX", 10)

        llm_max_concurrency = _env_int("PAPERSEARCH_LLM_MAX_CONCURRENCY", 1)
        doi_enrich_enabled = _env_bool("PAPERSEARCH_DOI_ENRICH_ENABLED", True)
        doi_enrich_timeout_s = _env_float("PAPERSEARCH_DOI_ENRICH_TIMEOUT_S", 5.0)
        doi_enrich_max_concurrency = _env_int("PAPERSEARCH_DOI_ENRICH_MAX_CONCURRENCY", 2)
        rerank_timeout_s = _env_float("PAPERSEARCH_RERANK_TIMEOUT_S", 30.0)
        rerank_max_retries = _env_int("PAPERSEARCH_RERANK_MAX_RETRIES", 1)
        rerank_max_doc_chars = _env_int("PAPERSEARCH_RERANK_MAX_DOC_CHARS", 3000)
        ieee_per_second_limit = _env_int(
            "PAPERSEARCH_IEEE_PER_SECOND_LIMIT",
            _env_int("PAPERSEARCH_IEEE_RATE_LIMIT_PER_SECOND", 10),
        )
        ieee_daily_limit = _env_int(
            "PAPERSEARCH_IEEE_DAILY_LIMIT",
            _env_int("PAPERSEARCH_IEEE_RATE_LIMIT_DAILY", 200),
        )

        platform_limits: dict[str, int] = {}

        # Optional: compact mapping like "arxiv=10,openalex=5"
        raw_limits = (os.getenv("PAPERSEARCH_PLATFORM_LIMITS") or "").strip()
        if raw_limits:
            for part in raw_limits.split(","):
                part = part.strip()
                if not part or "=" not in part:
                    continue
                name, value = part.split("=", 1)
                name = _norm_platform(name.strip())
                try:
                    n = int(value.strip())
                except ValueError:
                    continue
                if name and n > 0:
                    platform_limits[name] = n

        # Explicit per-platform overrides (higher priority)
        explicit_limits = {
            "PAPERSEARCH_PLATFORM_LIMIT_OPENALEX": "openalex",
            "PAPERSEARCH_PLATFORM_LIMIT_SEMANTICSCHOLAR": "semanticscholar",
            "PAPERSEARCH_PLATFORM_LIMIT_ARXIV": "arxiv",
            "PAPERSEARCH_PLATFORM_LIMIT_CROSSREF": "crossref",
            "PAPERSEARCH_PLATFORM_LIMIT_IEEE": "ieeexplore",
            "PAPERSEARCH_PLATFORM_LIMIT_IEEEXPLORE": "ieeexplore",
        }
        for env_key, platform_key in explicit_limits.items():
            v = _maybe_env_int(env_key)
            if v is not None and v > 0:
                platform_limits[platform_key] = v

        return Settings(
            per_platform=per_platform,
            final_limit=final_limit,
            final_limit_max=final_limit_max,
            platform_limits=platform_limits,
            query_max_chars=max(int(query_max_chars), 0),
            platforms_max=max(int(platforms_max), 0),
            contact_emails=contact_emails,
            tool_name=(os.getenv("PAPERSEARCH_TOOL_NAME") or "papersearch").strip() or "papersearch",
            email_pick_strategy=email_pick_strategy,
            semantic_api_key=(os.getenv("PAPERSEARCH_SEMANTICSCHOLAR_API_KEY") or "").strip(),
            ieee_api_key=(os.getenv("PAPERSEARCH_IEEE_API_KEY") or "").strip(),
            ieee_per_second_limit=max(int(ieee_per_second_limit), 1),
            ieee_daily_limit=max(int(ieee_daily_limit), 1),
            openalex_mailtos=openalex_mailtos,
            crossref_mailtos=crossref_mailtos,
            llm_base_url=(os.getenv("PAPERSEARCH_LLM_BASEURL") or "").strip(),
            llm_api_key=(os.getenv("PAPERSEARCH_LLM_APIKEY") or "").strip(),
            rank_model=(os.getenv("PAPERSEARCH_RANK_MODELNAME") or "").strip(),
            embed_model=(os.getenv("PAPERSEARCH_EMBED_MODELNAME") or "").strip(),
            rerank_model=(os.getenv("PAPERSEARCH_RERANK_MODELNAME") or "").strip(),
            llm_max_concurrency=llm_max_concurrency,
            llm_max_retries=_env_int("PAPERSEARCH_LLM_MAX_RETRIES", 5),
            llm_retry_base_delay=_env_float("PAPERSEARCH_LLM_RETRY_BASE_DELAY", 1.0),
            llm_retry_max_delay=_env_float("PAPERSEARCH_LLM_RETRY_MAX_DELAY", 30.0),
            rerank_timeout_s=max(float(rerank_timeout_s), 0.1),
            rerank_max_retries=max(int(rerank_max_retries), 0),
            rerank_max_doc_chars=max(int(rerank_max_doc_chars), 200),
            doi_enrich_enabled=bool(doi_enrich_enabled),
            doi_enrich_timeout_s=max(float(doi_enrich_timeout_s), 0.1),
            doi_enrich_max_concurrency=max(int(doi_enrich_max_concurrency), 1),
            storage_dir=storage_dir,
        )
