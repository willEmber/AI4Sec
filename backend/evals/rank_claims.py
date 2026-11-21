"""Deterministic audit of publication-tier claims and their citations (P18 §3.2).

Every sentence that asserts a venue tier (`CCF A`, `CORE A*`, `JCR Q1`,
`中科院 1 区`, `TH-CPL A`, `CAAI A`, `CCF-CN T1`), a `Workshop` / `Findings`
subtrack, or a CAS early-warning flag is checked mechanically against the
evidence passages the sentence cites — no LLM judge involved.

Four outcomes partition `claims` (`claims == supported + unsupported + uncited + contradicted`):
  * `supported`    — at least one cited passage carries the same `(system, value)`;
  * `contradicted` — no cited passage carries `(system, value)`, and at least one
                     carries a *different* value for the same `system`;
  * `unsupported`  — the sentence carries citations, but none mentions `system`
                     (the signature of copying a tier from the prompt badge onto
                     an unrelated citation);
  * `uncited`      — the sentence carries no citation at all.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from evals.statements import split_statements

_MATH_OR_CODE_BLOCK_RE = re.compile(r"\$\$.+?\$\$|\$[^$\n]+?\$|```[\s\S]*?```")

# System-specific extractors. Order does not merge systems: a sentence naming
# both `CCF A` and `CORE A*` yields two claims, checked independently.
_CCF_CN_RE = re.compile(
    r"\bCCF\s*[-_ ]?(?:中文|CN|高质量(?:科技期刊)?)\s*[-:：]?\s*(T[1-3]|中文[ABC])(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_CCF_RE = re.compile(
    r"\bCCF(?!\s*[-_ ]?(?:中文|CN|高质量|科技期刊|T[1-3]))"
    r"(?:\s+\d{4})?(?:\s*[-:：]\s*|\s+(?:tier|rank|等级)\s+|\s+)([ABC])(?:\s*类)?(?![A-Za-z0-9*])",
    re.IGNORECASE,
)
_CORE_RE = re.compile(
    r"\b(?:ICORE|CORE)(?:\s+\d{4})?(?:\s*[-:：]\s*|\s+(?:tier|rank|等级)\s+|\s+)(A\*|[ABC])(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_JCR_RE = re.compile(
    r"\b(?:JCR|SCIE?)(?:\s+\d{4})?(?:\s*[-:：]\s*|\s+)(Q[1-4])\b",
    re.IGNORECASE,
)
_CAS_RE = re.compile(
    r"(?:中科院|CAS)(?:升级版)?(?:\s*\d{4}\s*年?)?\s*(?:大类|小类)?\s*([1-4])\s*区",
    re.IGNORECASE,
)
_THCPL_RE = re.compile(
    r"(?:清华\s*)?\bTH[-_ ]?CPL(?:\s+\d{4})?\s*[-:：]?\s*([AB])(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_CAAI_RE = re.compile(
    r"\bCAAI(?:\s+\d{4})?\s*[-:：]?\s*([ABC])(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# Venue subtracks: scoped to publication context so ordinary English prose
# ("key findings", "experimental findings") is never mistaken for ACL Findings.
_FINDINGS_RE = re.compile(
    r"(?:\b(?:ACL|EMNLP|NAACL|EACL)\s*\(?Findings\)?\b"
    r"|\bFindings\s+of\s+(?:the\s+Association\s+for\s+Computational\s+Linguistics|ACL|EMNLP|NAACL|EACL)\b"
    r"|(?:\bvenue\s*:\s*[^\n;]*?\bfindings\s+track\b)"
    r"|(?:peer-reviewed\s+Findings\s+track)"
    r"|(?:Findings\s*(?:子轨道|附属轨道)))",
    re.IGNORECASE,
)
_WORKSHOP_RE = re.compile(
    r"(?:\b(?:CVPR|ICCV|ECCV|WACV|NeurIPS|NIPS|ICML|ICLR|ACL|EMNLP|NAACL|AAAI|IJCAI|KDD|WWW|SIGIR)W\b"
    r"|\b(?:CVPR|ICCV|ECCV|WACV|NeurIPS|NIPS|ICML|ICLR|ACL|EMNLP|NAACL|AAAI|IJCAI|KDD|WWW|SIGIR)\s+Workshops?\b"
    r"|\b(?:workshop|companion)\s+track\b"
    r"|(?:Workshop\s*(?:子轨道|附属轨道|论文)))",
    re.IGNORECASE,
)
_NEGATED_SUBTRACK_RE = re.compile(
    r"(?:非|不是|不属于|而非|区别于|不同于|distinct\s+from|not\s+a|rather\s+than)\s*"
    r"(?:Workshop|Findings)",
    re.IGNORECASE,
)

_WARNING_RE = re.compile(
    r"(?:CAS\s+(?:International\s+Journal\s+)?Early\s+Warning"
    r"|中科院(?:国际)?期刊预警"
    r"|国际期刊预警名单"
    r"|预警期刊)",
    re.IGNORECASE,
)
_NEGATED_WARNING_RE = re.compile(
    r"(?:未(?:发现|命中|被列入|列入|见)|无|不在|没有|非|not\s+(?:in|on|listed|marked)|no\s+|neither\s+|without\s+)"
    r"[^。；.!?\n]{0,24}?(?:预警|early[- ]warning)",
    re.IGNORECASE,
)


def _clean_for_scan(text: str) -> str:
    return _MATH_OR_CODE_BLOCK_RE.sub(" ", text or "")


def extract_rank_claims(text: str, *, in_evidence: bool = False) -> list[tuple[str, str]]:
    """Extract ordered, deduplicated `(system, normalized_value)` claims from `text`."""
    cleaned = _clean_for_scan(text)
    if not cleaned.strip():
        return []

    found: list[tuple[str, str]] = []

    def _add(sys_name: str, val: str) -> None:
        pair = (sys_name, val)
        if pair not in found:
            found.append(pair)

    for m in _CCF_CN_RE.finditer(cleaned):
        raw = m.group(1).upper()
        _add("CCF_CN", raw)

    for m in _CCF_RE.finditer(cleaned):
        _add("CCF", m.group(1).upper())

    for m in _CORE_RE.finditer(cleaned):
        _add("CORE", m.group(1).upper())

    for m in _JCR_RE.finditer(cleaned):
        _add("JCR", m.group(1).upper())

    for m in _CAS_RE.finditer(cleaned):
        _add("CAS", f"{m.group(1)}区")

    for m in _THCPL_RE.finditer(cleaned):
        _add("TH_CPL", m.group(1).upper())

    for m in _CAAI_RE.finditer(cleaned):
        _add("CAAI", m.group(1).upper())

    if _FINDINGS_RE.search(cleaned) and (in_evidence or not _NEGATED_SUBTRACK_RE.search(cleaned)):
        _add("SUBTRACK", "findings")

    if _WORKSHOP_RE.search(cleaned) and (in_evidence or not _NEGATED_SUBTRACK_RE.search(cleaned)):
        _add("SUBTRACK", "workshop")

    if _WARNING_RE.search(cleaned) and (in_evidence or not _NEGATED_WARNING_RE.search(cleaned)):
        _add("CAS_WARNING", "warning")

    return found


def audit_rank_claims(
    text: str,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Check every rank claim in `text` against the quotes of its cited evidence ids."""
    statements = split_statements(text or "")
    supported = 0
    unsupported = 0
    uncited = 0
    contradicted = 0
    details: list[dict[str, Any]] = []

    for stmt in statements:
        claims = extract_rank_claims(stmt.text, in_evidence=False)
        if not claims:
            continue

        ev_claims: set[tuple[str, str]] = set()
        ev_by_sys: dict[str, set[str]] = {}
        for cid in stmt.citations:
            item = evidence.get(cid)
            quote = ""
            if isinstance(item, Mapping):
                quote = str(item.get("quote") or "")
            elif isinstance(item, str):
                quote = item
            if not quote:
                continue
            for ev_sys, ev_val in extract_rank_claims(quote, in_evidence=True):
                ev_claims.add((ev_sys, ev_val))
                ev_by_sys.setdefault(ev_sys, set()).add(ev_val)
            # If an evidence quote says `workshop track` / `companion track`, it
            # contradicts any main-venue tier attached exclusively to it when no
            # other cited passage supports that tier.

        for sys_name, val in claims:
            if not stmt.citations:
                outcome = "uncited"
                uncited += 1
            elif (sys_name, val) in ev_claims:
                outcome = "supported"
                supported += 1
            elif sys_name in ev_by_sys:
                outcome = "contradicted"
                contradicted += 1
            elif (
                sys_name in {"CCF", "CORE", "JCR", "CAS", "TH_CPL", "CAAI", "CCF_CN"}
                and ("SUBTRACK", "workshop") in ev_claims
                and not any(s in ev_by_sys for s in ("CCF", "CORE", "JCR", "CAS", "TH_CPL", "CAAI", "CCF_CN"))
            ):
                # Claiming a main-venue tier while citing only a Workshop record
                outcome = "contradicted"
                contradicted += 1
            else:
                outcome = "unsupported"
                unsupported += 1

            details.append(
                {
                    "statement_index": stmt.index,
                    "system": sys_name,
                    "value": val,
                    "outcome": outcome,
                    "citations": list(stmt.citations),
                    "statement": stmt.text[:240],
                }
            )

    total = supported + unsupported + uncited + contradicted
    return {
        "claims": total,
        "supported": supported,
        "unsupported": unsupported,
        "uncited": uncited,
        "contradicted": contradicted,
        "support_rate": round(supported / total, 4) if total else None,
        "details": details,
    }


def pool_rank_audits(audits: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-attempt rank audits into dataset-level counts and support rate."""
    claims = sum(int(a.get("claims") or 0) for a in audits)
    supported = sum(int(a.get("supported") or 0) for a in audits)
    unsupported = sum(int(a.get("unsupported") or 0) for a in audits)
    uncited = sum(int(a.get("uncited") or 0) for a in audits)
    contradicted = sum(int(a.get("contradicted") or 0) for a in audits)
    return {
        "claims": claims,
        "supported": supported,
        "unsupported": unsupported,
        "uncited": uncited,
        "contradicted": contradicted,
        "support_rate": round(supported / claims, 4) if claims else None,
    }
