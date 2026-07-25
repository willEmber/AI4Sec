"""Collect external evidence for the Insight Snap triage verdict.

The verdict used to rest on the model's own reading plus, at most, a venue name.
This module supplies the rest: citation impact from OpenAlex/Semantic Scholar,
open-access and retraction status, EasyScholar venue ranks (already resolved
upstream by ``enrich_metadata``), and whether the authors released code.

Two rules shape the design:

* **Nothing is inferred.** Every field either comes from a named source or is
  listed in ``unavailable`` so the report can say "unknown" instead of showing a
  zero that reads like "never cited".
* **Failure is never fatal.** Each lookup is independent and swallowed; a paper
  with no DOI still gets a verdict, just with fewer signals.

Scoring reuses ``sphere_scorer`` — the same age-normalized citation curve and
venue-rank table Research Sphere ranks its candidates with, so the two modes
cannot disagree about what counts as a strong venue.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any

import httpx

from app.models.paper_ir import PaperIR
from app.models.snap_models import CitationIntents, CodeRepo, RepoHost, TriageSignals
from app.services.sphere_scorer import (
    compute_citation_component,
    compute_venue_component,
    is_preprint_venue,
    is_survey_title,
    normalize_venue,
)

logger = logging.getLogger("scholar.triage")


def _current_year() -> int:
    return datetime.now(timezone.utc).year


# ──────────────────────────────────────────────────────────────────────
# Code / artifact links  (P1-2)
# ──────────────────────────────────────────────────────────────────────

_HOST_PATTERNS: tuple[tuple[RepoHost, str, int], ...] = (
    # (host, domain regex, minimum path segments to accept)
    (RepoHost.GITHUB, r"github\.com", 2),
    (RepoHost.GITLAB, r"gitlab\.com", 2),
    (RepoHost.BITBUCKET, r"bitbucket\.org", 2),
    (RepoHost.HUGGINGFACE, r"huggingface\.co", 1),
    (RepoHost.ZENODO, r"zenodo\.org", 1),
    (RepoHost.OSF, r"osf\.io", 1),
    (RepoHost.CODEOCEAN, r"codeocean\.com", 1),
    (RepoHost.PROJECT_PAGE, r"[\w\-]+\.github\.io", 0),
)

_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?"
    r"((?:[\w\-]+\.github\.io|github\.com|gitlab\.com|bitbucket\.org"
    r"|huggingface\.co|zenodo\.org|osf\.io|codeocean\.com)"
    r"(?:/[\w.\-~%+]+)*)",
    re.IGNORECASE,
)

# Sentences carrying one of these are the authors announcing their own release.
_OWNERSHIP_CUES = (
    "our code", "our implementation", "our models", "our data",
    "code is available", "code available", "code and", "codes are available",
    "we release", "we have released", "we make", "we provide", "we publicly",
    "publicly available at", "available at", "available online at",
    "can be found at", "is released at", "released at", "open-sourced",
    "open sourced", "project page", "source code", "reference implementation",
    "我们的代码", "代码已开源", "代码可在", "开源地址", "项目主页", "代码地址",
)

# Sections whose links belong to other people's work.
_FOREIGN_SECTION_CUES = ("reference", "bibliograph", "related work", "acknowledg")

# Trailing characters PDFs glue onto a URL.
_URL_TRAILING = ".,;:)]}>\"'`"

# GitHub paths that are not repositories.
_GITHUB_RESERVED = {
    "features", "pricing", "about", "topics", "collections", "trending",
    "explore", "marketplace", "sponsors", "settings", "login", "join",
    "blog", "search", "orgs", "apps", "notifications",
}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。!?])\s+")


def _sentence_around(text: str, index: int) -> str:
    """The sentence containing ``index`` — the ownership-cue search window."""
    start = 0
    for m in _SENTENCE_SPLIT_RE.finditer(text[:index]):
        start = m.end()
    rest = _SENTENCE_SPLIT_RE.split(text[index:], maxsplit=1)
    end = index + (len(rest[0]) if rest else 0)
    return text[start:end].strip()


def _classify_host(path: str) -> tuple[RepoHost, int] | None:
    low = path.lower()
    for host, domain_re, min_segments in _HOST_PATTERNS:
        if re.match(rf"^{domain_re}(?:/|$)", low):
            return host, min_segments
    return None


def _normalize_url(raw_path: str) -> tuple[RepoHost, str, str] | None:
    """``(host, canonical_url, slug)`` for a repo-shaped path, else ``None``."""
    path = raw_path.rstrip(_URL_TRAILING)
    classified = _classify_host(path)
    if not classified:
        return None
    host, min_segments = classified

    segments = [s for s in path.split("/")[1:] if s]
    if len(segments) < min_segments:
        return None

    if host is RepoHost.GITHUB:
        if segments[0].lower() in _GITHUB_RESERVED:
            return None
        # Trim deep links (/tree/main, /blob/…) back to owner/repo.
        segments = segments[:2]
        segments[1] = re.sub(r"\.git$", "", segments[1])

    slug = "/".join(segments[:2]) if min_segments >= 2 else "/".join(segments)
    domain = path.split("/")[0].lower()
    url = f"https://{domain}" + (f"/{'/'.join(segments)}" if segments else "")
    return host, url, slug


def extract_code_repos(paper_ir: PaperIR, *, max_repos: int = 8) -> list[CodeRepo]:
    """Find code/artifact links in the paper and judge which are the authors' own.

    A link counts as official when it appears on the first page (abstracts and
    footnotes are where releases are announced) or its sentence carries an
    ownership cue. Links inside References / Related Work / Acknowledgements are
    never official — those are baselines and third-party tools.
    """
    found: dict[str, CodeRepo] = {}

    for block in paper_ir.blocks:
        text = block.text or ""
        if not text or "." not in text:
            continue
        section_low = (block.section_path or "").lower()
        foreign_section = any(cue in section_low for cue in _FOREIGN_SECTION_CUES)

        for m in _URL_RE.finditer(text):
            normalized = _normalize_url(m.group(1))
            if not normalized:
                continue
            host, url, slug = normalized

            sentence = _sentence_around(text, m.start())
            sentence_low = sentence.lower()
            has_cue = any(cue in sentence_low for cue in _OWNERSHIP_CUES)
            on_first_page = block.page_idx == 0
            is_official = (not foreign_section) and (has_cue or on_first_page)

            existing = found.get(url)
            if existing:
                # Keep the strongest evidence we have seen for this URL.
                if is_official and not existing.is_official:
                    existing.is_official = True
                    existing.evidence_page = block.page_idx + 1
                    existing.evidence_quote = sentence[:300]
                continue

            found[url] = CodeRepo(
                url=url,
                host=host,
                slug=slug,
                is_official=is_official,
                evidence_page=block.page_idx + 1,
                evidence_quote=sentence[:300],
            )

    repos = sorted(
        found.values(),
        key=lambda r: (not r.is_official, r.evidence_page),
    )
    return repos[:max_repos]


async def probe_github_repo(
    client: httpx.AsyncClient, repo: CodeRepo, *, token: str = ""
) -> None:
    """Fill in stars / last push / archived for a GitHub repo, in place.

    Best-effort: unauthenticated GitHub allows 60 requests/hour per IP, so this
    runs only when ``SNAP_PROBE_REPOS`` is on. Failures leave ``probe_ok`` False
    and the report simply omits the star count.
    """
    if repo.host is not RepoHost.GITHUB or "/" not in repo.slug:
        return
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = await client.get(
            f"https://api.github.com/repos/{repo.slug}", headers=headers, timeout=10.0
        )
        if resp.status_code != 200:
            logger.debug("github probe %s -> HTTP %s", repo.slug, resp.status_code)
            return
        data = resp.json()
        repo.stars = int(data.get("stargazers_count") or 0)
        repo.last_push = (data.get("pushed_at") or "")[:10]
        repo.archived = bool(data.get("archived"))
        repo.probe_ok = True
    except Exception as exc:
        logger.debug("github probe %s failed: %s", repo.slug, exc)


# ──────────────────────────────────────────────────────────────────────
# External metadata lookups  (P1-1)
# ──────────────────────────────────────────────────────────────────────

# Only what triage needs. `open_access` carries is_oa/oa_url/oa_status;
# `is_retracted` is the cheapest high-value red flag OpenAlex exposes.
_OA_TRIAGE_SELECT = (
    "id,doi,title,publication_year,cited_by_count,is_retracted,open_access,"
    "primary_location,locations,type,referenced_works,ids"
)

_S2_TRIAGE_FIELDS = (
    "paperId,title,year,venue,publicationVenue,citationCount,"
    "influentialCitationCount,isOpenAccess,openAccessPdf,externalIds,"
    "publicationTypes,referenceCount"
)


async def _openalex_triage_lookup(
    client: httpx.AsyncClient, *, doi: str = "", title: str = ""
) -> dict[str, Any] | None:
    """Fetch the triage fields for a work by DOI, falling back to title search."""
    from app.services.citation_graph import (
        _OPENALEX_SEM,
        _get_json,
        _oa_mailto_param,
    )
    from app.services.paper_search import jaccard_similarity, normalize_whitespace

    base = {**_oa_mailto_param(), "select": _OA_TRIAGE_SELECT}

    if doi:
        data = await _get_json(
            client,
            f"https://api.openalex.org/works/doi:{doi}",
            params=base,
            semaphore=_OPENALEX_SEM,
        )
        if data and data.get("id"):
            return data

    if title:
        data = await _get_json(
            client,
            "https://api.openalex.org/works",
            params={**base, "search": title, "per_page": "1"},
            semaphore=_OPENALEX_SEM,
        )
        results = (data or {}).get("results") or []
        if results:
            candidate = results[0]
            # Guard against the search endpoint returning a loosely related work.
            if jaccard_similarity(title, normalize_whitespace(candidate.get("title") or "")) >= 0.6:
                return candidate
    return None


async def _s2_triage_lookup(
    client: httpx.AsyncClient, *, doi: str = "", arxiv_id: str = "", title: str = ""
) -> dict[str, Any] | None:
    """Fetch the triage fields from Semantic Scholar (DOI → arXiv → title)."""
    from app.services.citation_graph import (
        _S2_SEM,
        _get_json,
        _s2_headers,
        _s2_throttle,
    )
    from app.services.paper_search import jaccard_similarity, normalize_whitespace

    headers = _s2_headers()
    params = {"fields": _S2_TRIAGE_FIELDS}

    for ident in (f"DOI:{doi}" if doi else "", f"ARXIV:{arxiv_id}" if arxiv_id else ""):
        if not ident:
            continue
        await _s2_throttle()
        data = await _get_json(
            client,
            f"https://api.semanticscholar.org/graph/v1/paper/{ident}",
            params=params,
            headers=headers,
            semaphore=_S2_SEM,
        )
        if data and data.get("paperId"):
            return data

    if title:
        await _s2_throttle()
        data = await _get_json(
            client,
            "https://api.semanticscholar.org/graph/v1/paper/search/match",
            params={**params, "query": title[:200]},
            headers=headers,
            semaphore=_S2_SEM,
        )
        matches = (data or {}).get("data") or []
        if matches:
            candidate = matches[0]
            if jaccard_similarity(title, normalize_whitespace(candidate.get("title") or "")) >= 0.6:
                return candidate
    return None


def _apply_openalex(signals: TriageSignals, work: dict[str, Any]) -> None:
    signals.resolved = True
    signals.openalex_id = (work.get("id") or "").replace("https://openalex.org/", "")
    signals.provenance["openalex_id"] = "OpenAlex"

    if not signals.doi:
        doi = (work.get("doi") or "").replace("https://doi.org/", "").strip().lower()
        if doi:
            signals.doi = doi

    cited = work.get("cited_by_count")
    if cited is not None:
        signals.cited_by_count = max(signals.cited_by_count, int(cited))
        signals.provenance["citations"] = "OpenAlex"

    year = work.get("publication_year") or 0
    if year and not signals.year:
        signals.year = int(year)
        signals.provenance["year"] = "OpenAlex"

    if work.get("is_retracted"):
        signals.is_retracted = True
        signals.provenance["retraction"] = "OpenAlex"

    oa = work.get("open_access") or {}
    if oa.get("is_oa"):
        signals.is_open_access = True
        signals.oa_url = oa.get("oa_url") or ""
        signals.provenance["open_access"] = "OpenAlex"

    refs = work.get("referenced_works") or []
    if refs:
        signals.reference_count = len(refs)

    if not signals.venue:
        from app.services.citation_graph import _oa_extract_venue

        venue = _oa_extract_venue(work)
        if venue:
            signals.venue = venue
            signals.provenance["venue"] = "OpenAlex"


def _apply_s2(signals: TriageSignals, paper: dict[str, Any]) -> None:
    signals.resolved = True
    signals.s2_paper_id = paper.get("paperId") or signals.s2_paper_id

    cited = paper.get("citationCount")
    if cited is not None and int(cited) > signals.cited_by_count:
        signals.cited_by_count = int(cited)
        signals.provenance["citations"] = "Semantic Scholar"

    influential = paper.get("influentialCitationCount")
    if influential is not None:
        signals.influential_citation_count = int(influential)
        signals.provenance["influential_citations"] = "Semantic Scholar"

    if paper.get("referenceCount"):
        signals.reference_count = signals.reference_count or int(paper["referenceCount"])

    if paper.get("isOpenAccess") and not signals.is_open_access:
        signals.is_open_access = True
        pdf = paper.get("openAccessPdf") or {}
        signals.oa_url = signals.oa_url or (pdf.get("url") or "")
        signals.provenance["open_access"] = "Semantic Scholar"

    types = [t for t in (paper.get("publicationTypes") or []) if isinstance(t, str)]
    if types:
        signals.publication_types = types
        # S2 labels reviews explicitly — stronger than our title heuristic.
        if any(t.lower() == "review" for t in types):
            signals.is_survey = True
            signals.provenance["survey"] = "Semantic Scholar"

    external = paper.get("externalIds") or {}
    if not signals.doi and external.get("DOI"):
        signals.doi = str(external["DOI"]).strip().lower()
    if not signals.arxiv_id and external.get("ArXiv"):
        signals.arxiv_id = str(external["ArXiv"])

    if not signals.year and paper.get("year"):
        signals.year = int(paper["year"])
    if not signals.venue:
        venue_obj = paper.get("publicationVenue") or {}
        venue = venue_obj.get("name") or paper.get("venue") or ""
        if venue:
            signals.venue = venue
            signals.provenance["venue"] = "Semantic Scholar"


async def _s2_citation_intents(
    client: httpx.AsyncClient, s2_paper_id: str, *, limit: int = 200
) -> CitationIntents:
    """Tally the intents on this paper's incoming citation edges.

    One extra S2 request, and it runs inside the same concurrent block as the
    LLM calls, so it costs no wall-clock time. Returns an empty (``available``
    False) tally on any failure — the report then simply omits the row.
    """
    from app.services.citation_graph import _S2_SEM, _get_json, _s2_headers, _s2_throttle

    intents = CitationIntents()
    if not s2_paper_id:
        return intents

    await _s2_throttle()
    data = await _get_json(
        client,
        f"https://api.semanticscholar.org/graph/v1/paper/{s2_paper_id}/citations",
        params={"fields": "intents,isInfluential", "limit": str(min(limit, 1000))},
        headers=_s2_headers(),
        semaphore=_S2_SEM,
    )
    edges = (data or {}).get("data") or []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        intents.sampled += 1
        if edge.get("isInfluential"):
            intents.influential += 1
        for intent in edge.get("intents") or []:
            if intent == "background":
                intents.background += 1
            elif intent == "methodology":
                intents.methodology += 1
            elif intent in ("result", "resultclaim"):
                intents.result += 1
    return intents


# ──────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────

# Weights for the external composite. Citations lead because they are the one
# signal that reflects how the field actually received the work; code is
# weighted last because its absence is normal in many subfields.
_W_CITATION = 0.40
_W_VENUE = 0.35
_W_CODE = 0.25


def score_code_availability(signals: TriageSignals) -> float:
    """Artifact availability in [0,1]: official repo > any repo > open access only."""
    official = signals.official_repos
    if official:
        score = 0.8
        best = max(official, key=lambda r: r.stars)
        if best.probe_ok:
            if best.stars >= 100:
                score = 1.0
            elif best.stars >= 10:
                score = 0.9
            if best.archived:
                score -= 0.1
        return min(1.0, score)
    if signals.repos:
        return 0.35  # links exist but read as third-party
    if signals.is_open_access:
        return 0.2   # at least the paper itself is reachable
    return 0.0


def compute_external_score(signals: TriageSignals) -> TriageSignals:
    """Fill the derived score fields from the collected evidence."""
    current_year = _current_year()

    signals.venue_normalized = normalize_venue(signals.venue)
    signals.is_preprint = is_preprint_venue(signals.venue)
    signals.venue_score = compute_venue_component(
        signals.venue, signals.sci_rank, signals.ccf_rank
    )

    if signals.citations_known:
        signals.citation_score = compute_citation_component(
            signals.cited_by_count, signals.year, current_year
        )
        age = max(1, current_year - signals.year + 1) if signals.year else 5
        signals.citations_per_year = round(signals.cited_by_count / age, 1)
    else:
        signals.citation_score = 0.0
        signals.citations_per_year = 0.0

    signals.code_score = score_code_availability(signals)

    # Renormalize over the components we actually have, so a paper whose
    # citation count could not be resolved is not silently scored as uncited.
    parts: list[tuple[float, float]] = [
        (_W_VENUE, signals.venue_score),
        (_W_CODE, signals.code_score),
    ]
    if signals.citations_known:
        parts.append((_W_CITATION, signals.citation_score))
    total_weight = sum(w for w, _ in parts)
    signals.external_score = (
        round(sum(w * v for w, v in parts) / total_weight, 3) if total_weight else 0.0
    )
    return signals


# ──────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────

async def collect_triage_signals(
    paper_ir: PaperIR,
    *,
    pub_rank: dict[str, Any] | None = None,
    doi: str = "",
    arxiv_id: str = "",
    enable_network: bool = True,
    probe_repos: bool = False,
    github_token: str = "",
    fetch_intents: bool = True,
    log_label: str = "triage",
) -> TriageSignals:
    """Gather every external signal available for this paper.

    ``pub_rank`` is the payload ``enrich_metadata`` already resolved (venue,
    year, SCI/CCF) — passing it avoids a second EasyScholar round-trip. Network
    lookups run concurrently and never raise.
    """
    signals = TriageSignals(doi=doi.strip().lower(), arxiv_id=arxiv_id)

    pub_rank = pub_rank or {}
    if pub_rank.get("venue"):
        signals.venue = pub_rank["venue"]
        signals.provenance["venue"] = "Crossref / Semantic Scholar"
    if pub_rank.get("year"):
        signals.year = int(pub_rank["year"])
        signals.provenance["year"] = "Crossref / Semantic Scholar"
    if pub_rank.get("sci"):
        signals.sci_rank = pub_rank["sci"]
        signals.provenance["sci_rank"] = "EasyScholar"
    if pub_rank.get("ccf"):
        signals.ccf_rank = pub_rank["ccf"]
        signals.provenance["ccf_rank"] = "EasyScholar"

    # Code links come straight from the PDF — no network needed.
    signals.repos = extract_code_repos(paper_ir)
    if signals.repos:
        official = signals.official_repos
        anchor = official[0] if official else signals.repos[0]
        signals.provenance["code"] = f"paper p.{anchor.evidence_page}"

    if paper_ir.title and is_survey_title(paper_ir.title):
        signals.is_survey = True
        signals.provenance.setdefault("survey", "title heuristic")

    if not enable_network:
        signals.unavailable.extend(["citations", "open_access", "retraction"])
        return compute_external_score(signals)

    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        oa_result, s2_result = await asyncio.gather(
            _openalex_triage_lookup(client, doi=signals.doi, title=paper_ir.title),
            _s2_triage_lookup(
                client, doi=signals.doi, arxiv_id=signals.arxiv_id, title=paper_ir.title
            ),
            return_exceptions=True,
        )

        if isinstance(oa_result, dict):
            _apply_openalex(signals, oa_result)
        else:
            if isinstance(oa_result, Exception):
                logger.warning("%s: OpenAlex lookup failed: %s", log_label, oa_result)
            signals.unavailable.append("openalex")

        if isinstance(s2_result, dict):
            _apply_s2(signals, s2_result)
        else:
            if isinstance(s2_result, Exception):
                logger.warning("%s: S2 lookup failed: %s", log_label, s2_result)
            signals.unavailable.append("semantic_scholar")

        if not signals.resolved:
            # Neither index matched: citations/OA/retraction are unknown, not zero.
            signals.unavailable.extend(["citations", "open_access", "retraction"])

        # Follow-ups that need an id the lookups above had to resolve first.
        follow_ups: list[Any] = []
        if fetch_intents and signals.s2_paper_id:
            follow_ups.append(_s2_citation_intents(client, signals.s2_paper_id))
        if probe_repos:
            targets = [r for r in signals.repos if r.host is RepoHost.GITHUB][:3]
            follow_ups += [probe_github_repo(client, r, token=github_token) for r in targets]

        if follow_ups:
            results = await asyncio.gather(*follow_ups, return_exceptions=True)
            for result in results:
                if isinstance(result, CitationIntents) and result.available:
                    signals.intents = result
                    signals.provenance["intents"] = "Semantic Scholar"
                elif isinstance(result, Exception):
                    logger.debug("%s: signal follow-up failed: %s", log_label, result)

    compute_external_score(signals)
    logger.info(
        "%s: signals resolved=%s venue=%s sci=%s ccf=%s citations=%s (%.1f/yr) "
        "influential=%s intents=%s code=%d(official=%d) oa=%s retracted=%s external_score=%.2f",
        log_label, signals.resolved, signals.venue_normalized or "-",
        signals.sci_rank or "-", signals.ccf_rank or "-",
        signals.cited_by_count if signals.citations_known else "unknown",
        signals.citations_per_year, signals.influential_citation_count,
        f"bg={signals.intents.background}/meth={signals.intents.methodology}/res={signals.intents.result}"
        if signals.intents.available else "-",
        len(signals.repos), len(signals.official_repos),
        signals.is_open_access, signals.is_retracted, signals.external_score,
    )
    return signals


# ──────────────────────────────────────────────────────────────────────
# Per-paper cache  (P3)
# ──────────────────────────────────────────────────────────────────────

async def load_cached_signals(paper_id: str, *, ttl_hours: int) -> TriageSignals | None:
    """Return cached signals for this paper when still inside the TTL.

    Citation counts drift, so entries expire; the code links (which come from the
    PDF and cannot change) expire with them, which is a small waste but keeps the
    cache to one row and one rule.
    """
    from app.db import database as db

    row = await db.fetch_one(
        "SELECT signals_json, fetched_at,"
        "       (julianday('now') - julianday(fetched_at)) * 24.0 AS age_hours"
        " FROM paper_signals WHERE paper_id = ?",
        (paper_id,),
    )
    if not row or not row.get("signals_json"):
        return None
    age = row.get("age_hours")
    if age is None or age > ttl_hours:
        return None
    try:
        return TriageSignals.model_validate_json(row["signals_json"])
    except Exception as exc:
        # A schema change since the row was written; treat it as a miss.
        logger.debug("signals cache: unusable row for %s (%s)", paper_id, exc)
        return None


async def store_signals(paper_id: str, signals: TriageSignals) -> None:
    """Persist signals for reuse by later runs on the same paper."""
    from app.db import database as db

    try:
        await db.execute(
            "INSERT OR REPLACE INTO paper_signals"
            " (paper_id, signals_json, cited_by_count, is_retracted, fetched_at)"
            " VALUES (?, ?, ?, ?, datetime('now'))",
            (
                paper_id,
                signals.model_dump_json(),
                signals.cited_by_count,
                int(signals.is_retracted),
            ),
        )
    except Exception as exc:
        logger.warning("signals cache: store failed for %s: %s", paper_id, exc)


async def get_or_collect_triage_signals(
    paper_ir: PaperIR,
    *,
    paper_id: str,
    ttl_hours: int = 168,
    log_label: str = "triage",
    **kwargs: Any,
) -> TriageSignals:
    """Cache-backed ``collect_triage_signals``.

    A cache hit re-scores rather than trusting the stored scores, so changing a
    scoring weight takes effect immediately instead of waiting out every TTL.
    """
    if ttl_hours > 0:
        cached = await load_cached_signals(paper_id, ttl_hours=ttl_hours)
        if cached is not None:
            compute_external_score(cached)
            logger.info(
                "%s: signals CACHED (citations=%s code=%d external_score=%.2f)",
                log_label,
                cached.cited_by_count if cached.citations_known else "unknown",
                len(cached.repos),
                cached.external_score,
            )
            return cached

    signals = await collect_triage_signals(paper_ir, log_label=log_label, **kwargs)
    if ttl_hours > 0 and signals.resolved:
        # Only cache a resolved lookup: caching a total miss would pin "unknown"
        # onto the paper for a week, including across a transient outage.
        await store_signals(paper_id, signals)
    return signals
