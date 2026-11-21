"""Compile the authoritative multi-system venue catalog (`venue_catalog.json.gz`).

Merges open/community structured tables into one compact, deterministic seed file
loaded in-process by ``app.services.publication_rank.catalog``:
  1. CCF 2026 (7th Edition) international conferences & journals + DBLP prefixes
  2. ICORE 2026 & CORE 2023 international CS conference rankings (A* / A / B / C)
  3. Tsinghua TH-CPL recommended CS conferences & journals (A / B) via ccfddl + TH-CPL
  4. CAAI recommended AI/Robotics/Vision/NLP conferences & journals (A / B / C)
  5. CCF High-Quality Sci-Tech Journals 2025 (T1 / T2 / T3, incl. Chinese top journals)
  6. JCR 2025 quartiles & Impact Factors (~22.6k journals, matched by ISSN & title)
  7. CAS (中科院升级版) 2025 divisions (1区–4区 + Top, ~21.7k journals)
  8. CAS International Journal Early Warning Lists (2020–2025 red flags)
  9. Curated CS emerging venues & multi-alias / DBLP stream overrides (TMLR, COLM,
     ICLR, NeurIPS/NIPS, PVLDB, PACMPL, PACMMOD, IEEE S&P, USENIX Security, etc.)

Usage:
    cd backend && uv run python -m scripts.build_venue_catalog --raw-dir <dir>
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.services.publication_rank.normalizer import (
    clean_venue_text,
    extract_dblp_prefix,
    extract_issns,
    is_umbrella_series,
    normalize_issn,
    normalize_name_key,
    strip_venue_decorations,
)

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "services"
    / "publication_rank"
    / "data"
    / "venue_catalog.json.gz"
)

_JCR_Q_ORDER = {"Q1": 4, "Q2": 3, "Q3": 2, "Q4": 1}
_CCF_TIER_ORDER = {"A": 3, "B": 2, "C": 1}
_FORMER_NAME_PAREN_RE = re.compile(
    r"[（(]\s*(?:原\s*[:：]?|was\s+|formerly\s+|previously\s+)([^（）()]+)[）)]",
    re.IGNORECASE,
)


def _split_former_names(raw_name: str) -> tuple[str, list[str]]:
    """Split 'IEEE Transactions on Networking（原 IEEE/ACM Transactions on Networking）'
    into clean current name + extracted former-name aliases."""
    cleaned = clean_venue_text(raw_name)
    former: list[str] = []
    for m in _FORMER_NAME_PAREN_RE.finditer(cleaned):
        cand = clean_venue_text(m.group(1)).strip(" ,;:")
        # Ignore explanatory text like "changed in 2006"
        cand = re.sub(r";?\s*changed\s+(?:in\s+)?\d{4}.*$", "", cand, flags=re.IGNORECASE).strip()
        if len(cand) >= 4 and cand not in former:
            former.append(cand)
    base = _FORMER_NAME_PAREN_RE.sub("", cleaned)
    base = re.sub(r"\s+", " ", base).strip(" ,;:-")
    return (base or cleaned), former
_CAS_NUM_RE = re.compile(r"^([1-4])(?:\s*区|\s*\[|$)")

_CONF_DEDUP_STOPWORDS = frozenset(
    {
        "the", "of", "on", "in", "for", "and", "a", "an", "to", "at", "with",
        "international", "annual", "conference", "symposium", "meeting", "workshop",
        "proceedings", "ieee", "acm", "usenix", "ifip", "siam", "euromicro",
        "sigplan", "sigsoft", "sigmod", "sigir", "sigkdd", "sigsac", "sigchi",
        "sigops", "sigcomm", "siggraph", "sigmetrics", "sigmobile", "sigda", "sigbed",
    }
)
_UK_US_SPELLING = (
    ("modelling", "modeling"),
    ("visualisation", "visualization"),
    ("organisation", "organization"),
    ("optimisation", "optimization"),
    ("synchronisation", "synchronization"),
    ("behaviour", "behavior"),
    ("colour", "color"),
    ("programme", "program"),
    ("analyse", "analyze"),
    ("parallelism", "parallel"),
)

_ISO4_DROP_WORDS = frozenset({"on", "of", "the", "in", "for", "and", "to", "with", "by", "a", "an", "at", "&"})
_ISO4_WORD_MAP: dict[str, str] = {
    "transactions": "Trans.",
    "transaction": "Trans.",
    "journal": "J.",
    "journals": "J.",
    "proceedings": "Proc.",
    "letters": "Lett.",
    "communications": "Commun.",
    "communication": "Commun.",
    "magazine": "Mag.",
    "reviews": "Rev.",
    "review": "Rev.",
    "surveys": "Surv.",
    "survey": "Surv.",
    "bulletin": "Bull.",
    "annals": "Ann.",
    "advances": "Adv.",
    "international": "Int.",
    "european": "Eur.",
    "american": "Am.",
    "national": "Natl.",
    "academy": "Acad.",
    "computer": "Comput.",
    "computers": "Comput.",
    "computing": "Comput.",
    "computation": "Comput.",
    "computational": "Comput.",
    "information": "Inf.",
    "informational": "Inf.",
    "informatics": "Inf.",
    "system": "Syst.",
    "systems": "Syst.",
    "engineering": "Eng.",
    "software": "Softw.",
    "network": "Netw.",
    "networks": "Netw.",
    "networking": "Netw.",
    "security": "Secur.",
    "knowledge": "Knowl.",
    "programming": "Program.",
    "program": "Program.",
    "language": "Lang.",
    "languages": "Lang.",
    "graphics": "Graph.",
    "graphical": "Graph.",
    "analysis": "Anal.",
    "analytical": "Anal.",
    "machine": "Mach.",
    "machinery": "Mach.",
    "intelligence": "Intell.",
    "intelligent": "Intell.",
    "learning": "Learn.",
    "vision": "Vis.",
    "visual": "Vis.",
    "visualization": "Vis.",
    "processing": "Process.",
    "processes": "Process.",
    "distributed": "Distrib.",
    "distribution": "Distrib.",
    "mobile": "Mob.",
    "mobility": "Mob.",
    "cybernetics": "Cybern.",
    "technology": "Technol.",
    "technologies": "Technol.",
    "technological": "Technol.",
    "methodology": "Methodol.",
    "methodologies": "Methodol.",
    "nature": "Nat.",
    "science": "Sci.",
    "sciences": "Sci.",
    "scientific": "Sci.",
    "robotics": "Robot.",
    "medical": "Med.",
    "medicine": "Med.",
    "biomedical": "Biomed.",
    "bioinformatics": "Bioinform.",
    "biotechnology": "Biotechnol.",
    "automatic": "Autom.",
    "automation": "Autom.",
    "wireless": "Wirel.",
    "services": "Serv.",
    "service": "Serv.",
    "affective": "Affect.",
    "autonomous": "Auton.",
    "evolutionary": "Evol.",
    "mathematical": "Math.",
    "mathematics": "Math.",
    "statistical": "Stat.",
    "statistics": "Stat.",
    "architecture": "Archit.",
    "architectures": "Archit.",
    "hardware": "Hardw.",
    "embedded": "Embed.",
    "design": "Des.",
    "integrated": "Integr.",
    "reliability": "Reliab.",
    "management": "Manag.",
    "interactive": "Interact.",
    "human": "Hum.",
    "electronic": "Electron.",
    "electronics": "Electron.",
    "optical": "Opt.",
    "optics": "Opt.",
    "industrial": "Ind.",
    "vehicular": "Veh.",
    "geoscience": "Geosci.",
    "sensing": "Sens.",
    "parallel": "Parallel",
}


def _conf_content_tokens(name: str) -> set[str]:
    s = re.sub(r"\([^()]*\)", " ", name)
    nk = normalize_name_key(s)
    for uk, us in _UK_US_SPELLING:
        nk = re.sub(rf"\b{uk}\b", us, nk)
    return {t for t in nk.split() if t and t not in _CONF_DEDUP_STOPWORDS}


def _generate_iso4_alias(journal_name: str) -> str:
    """Generate standard ISO-4 / S2 / DBLP abbreviated journal title."""
    cleaned = clean_venue_text(re.sub(r"\([^()]*\)", " ", journal_name))
    tokens = re.findall(r"[A-Za-z0-9&/\-]+", cleaned)
    if len(tokens) < 2:
        return ""
    out: list[str] = []
    changed = False
    for tok in tokens:
        low = tok.lower()
        if low in _ISO4_DROP_WORDS:
            changed = True
            continue
        if "/" in tok:
            # e.g. "IEEE/ACM"
            parts = [_ISO4_WORD_MAP.get(p.lower(), p) for p in tok.split("/")]
            joined = "/".join(parts)
            if joined != tok:
                changed = True
            out.append(joined)
            continue
        if "-" in tok:
            parts = [_ISO4_WORD_MAP.get(p.lower(), p.capitalize() if p.islower() else p) for p in tok.split("-")]
            joined = "-".join(parts)
            if joined != tok:
                changed = True
            out.append(joined)
            continue
        mapped = _ISO4_WORD_MAP.get(low)
        if mapped:
            out.append(mapped)
            changed = True
        elif tok.isupper():
            out.append(tok)
        else:
            out.append(tok[0].upper() + tok[1:])
    if not changed or len(out) < 2:
        return ""
    return " ".join(out)


@dataclass
class _VenueBuilderRecord:
    vid: str
    name: str
    acronym: str = ""
    vtype: str = "j"  # "c" | "j"
    dblp_prefixes: list[str] = field(default_factory=list)
    issns: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    # (system, rank, category, edition_year, top_int, impact_factor)
    ranks: list[tuple[str, str, str, str, int, str]] = field(default_factory=list)
    warning: str = ""
    note: str = ""

    def add_dblp(self, raw: str) -> None:
        p = extract_dblp_prefix(raw)
        if p and p not in self.dblp_prefixes:
            self.dblp_prefixes.append(p)

    def add_issn(self, raw: str) -> None:
        for issn in extract_issns(raw):
            if issn not in self.issns:
                self.issns.append(issn)

    def add_alias(self, raw: str) -> None:
        cleaned = clean_venue_text(raw)
        if not cleaned or is_umbrella_series(cleaned):
            return
        nk = normalize_name_key(cleaned)
        if not nk:
            return
        if nk == normalize_name_key(self.name) or nk == normalize_name_key(self.acronym):
            return
        if all(normalize_name_key(x) != nk for x in self.aliases):
            self.aliases.append(cleaned)

    def add_rank(
        self,
        system: str,
        rank: str,
        category: str = "",
        edition_year: str = "",
        top: bool = False,
        impact_factor: str = "",
    ) -> None:
        rank = rank.strip()
        if not rank:
            return
        top_i = 1 if top else 0
        for idx, existing in enumerate(self.ranks):
            if existing[0] == system and (existing[2] == category or system in {"JCR", "CORE", "CCF_CN"}):
                # Keep better or newer entry
                if system == "JCR" and _JCR_Q_ORDER.get(rank, 0) <= _JCR_Q_ORDER.get(existing[1], 0):
                    if not existing[5] and impact_factor:
                        self.ranks[idx] = (existing[0], existing[1], existing[2], existing[3], existing[4], impact_factor)
                    return
                if system == "CCF" and _CCF_TIER_ORDER.get(rank, 0) < _CCF_TIER_ORDER.get(existing[1], 0):
                    return
                self.ranks[idx] = (system, rank, category or existing[2], edition_year or existing[3], max(top_i, existing[4]), impact_factor or existing[5])
                return
        self.ranks.append((system, rank, category.strip(), edition_year.strip(), top_i, impact_factor.strip()))

    def to_compact(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.vid,
            "n": self.name,
            "t": self.vtype,
        }
        if self.acronym:
            out["a"] = self.acronym
        if self.dblp_prefixes:
            out["d"] = self.dblp_prefixes
        if self.issns:
            out["i"] = self.issns
        if self.aliases:
            out["al"] = self.aliases[:18]
        if self.ranks:
            out["r"] = [list(r) for r in self.ranks]
        if self.warning:
            out["w"] = self.warning
        if self.note:
            out["note"] = self.note
        return out


class _CatalogBuilder:
    def __init__(self) -> None:
        self.records: list[_VenueBuilderRecord] = []
        self.by_issn: dict[str, _VenueBuilderRecord] = {}
        self.by_dblp: dict[str, _VenueBuilderRecord] = {}
        self.by_conf_name: dict[str, _VenueBuilderRecord] = {}
        self.by_conf_acr: dict[str, _VenueBuilderRecord] = {}
        self.by_journal_name: dict[str, _VenueBuilderRecord] = {}

    def _index(self, rec: _VenueBuilderRecord) -> None:
        for issn in rec.issns:
            self.by_issn.setdefault(issn, rec)
        for dp in rec.dblp_prefixes:
            self.by_dblp.setdefault(dp, rec)
        nk = normalize_name_key(rec.name)
        stripped, _ = strip_venue_decorations(rec.name)
        snk = normalize_name_key(stripped)
        if rec.vtype == "c":
            if nk:
                self.by_conf_name.setdefault(nk, rec)
            if snk:
                self.by_conf_name.setdefault(snk, rec)
            if rec.acronym:
                self.by_conf_acr.setdefault(normalize_name_key(rec.acronym), rec)
            for al in rec.aliases:
                self.by_conf_name.setdefault(normalize_name_key(al), rec)
        else:
            if nk:
                self.by_journal_name.setdefault(nk, rec)
            if snk:
                self.by_journal_name.setdefault(snk, rec)
            for al in rec.aliases:
                self.by_journal_name.setdefault(normalize_name_key(al), rec)

    def upsert(
        self,
        *,
        name: str,
        vtype: str,
        acronym: str = "",
        issns: list[str] | None = None,
        dblp_prefixes: list[str] | None = None,
        aliases: list[str] | None = None,
    ) -> _VenueBuilderRecord:
        clean_name, former_aliases = _split_former_names(name)
        clean_acr = clean_venue_text(acronym)
        extra_aliases = [*former_aliases, *(aliases or [])]
        norm_issns = extract_issns(issns or [])
        # Fix known upstream copy-paste DBLP URL typos in CCF/ShowJCR tables
        raw_dblps = list(dblp_prefixes or [])
        if clean_acr.lower() == "asiaccs":
            raw_dblps = ["conf/asiaccs"]
        elif clean_acr.lower() == "iscas":
            raw_dblps = ["conf/iscas"]
        norm_dblps = [extract_dblp_prefix(d) for d in raw_dblps if extract_dblp_prefix(d)]

        existing: _VenueBuilderRecord | None = None
        for dp in norm_dblps:
            if dp in self.by_dblp:
                cand_d = self.by_dblp[dp]
                # Guard against upstream copy-pasted DBLP URLs across unrelated conferences (e.g. ISCAS -> conf/hpdc)
                if (
                    vtype == "c"
                    and cand_d.vtype == "c"
                    and clean_acr
                    and cand_d.acronym
                    and normalize_name_key(clean_acr) != normalize_name_key(cand_d.acronym)
                    and {normalize_name_key(clean_acr), normalize_name_key(cand_d.acronym)}
                    not in ({"nips", "neurips"}, {"fse", "esec fse"}, {"mm", "acm mm"}, {"sp", "ieee s and p"}, {"uss", "usenix security"})
                    and len(_conf_content_tokens(clean_name) & _conf_content_tokens(cand_d.name)) == 0
                ):
                    continue
                existing = cand_d
                break
        if existing is None:
            for issn in norm_issns:
                if issn in self.by_issn:
                    existing = self.by_issn[issn]
                    break
        if existing is None:
            nk = normalize_name_key(clean_name)
            stripped, _ = strip_venue_decorations(clean_name)
            snk = normalize_name_key(stripped)
            if vtype == "c":
                existing = self.by_conf_name.get(nk) or self.by_conf_name.get(snk)
                if existing is None and clean_acr and len(clean_acr) >= 2:
                    # Match conference by acronym when content tokens agree (handles UK/US spelling & minor "the"/"on" diffs)
                    cand = self.by_conf_acr.get(normalize_name_key(clean_acr))
                    if cand is not None:
                        cand_tokens = _conf_content_tokens(cand.name)
                        new_tokens = _conf_content_tokens(stripped or clean_name)
                        if (
                            not new_tokens
                            or not cand_tokens
                            or cand_tokens == new_tokens
                            or (len(cand_tokens & new_tokens) >= min(2, len(new_tokens), len(cand_tokens))
                                and len(cand_tokens & new_tokens) / max(1, min(len(new_tokens), len(cand_tokens))) >= 0.66)
                        ):
                            existing = cand
            else:
                existing = self.by_journal_name.get(nk) or self.by_journal_name.get(snk)
                if existing is None:
                    for fa in former_aliases:
                        fak = normalize_name_key(fa)
                        if fak and fak in self.by_journal_name:
                            existing = self.by_journal_name[fak]
                            break

        if existing is None:
            vid = f"v_{len(self.records) + 1}"
            # Prefer mixed-case readable title over ALL-CAPS JCR title
            existing = _VenueBuilderRecord(
                vid=vid,
                name=clean_name.title() if clean_name.isupper() and len(clean_name) > 8 else clean_name,
                acronym=clean_acr,
                vtype=vtype,
            )
            self.records.append(existing)
        else:
            # Upgrade ALL-CAPS name to mixed-case name if incoming has proper casing
            if existing.name.isupper() and not clean_name.isupper() and len(clean_name) > 4:
                existing.add_alias(existing.name)
                existing.name = clean_name
            elif clean_name and normalize_name_key(clean_name) != normalize_name_key(existing.name):
                existing.add_alias(clean_name)
            if clean_acr and not existing.acronym:
                existing.acronym = clean_acr
            elif clean_acr and normalize_name_key(clean_acr) != normalize_name_key(existing.acronym):
                existing.add_alias(clean_acr)

        for issn in norm_issns:
            existing.add_issn(issn)
        for dp in norm_dblps:
            existing.add_dblp(dp)
        for al in extra_aliases:
            existing.add_alias(al)

        self._index(existing)
        return existing


# ---------------------------------------------------------------------------
# Parsers for each upstream source
# ---------------------------------------------------------------------------

def _load_ccf_2026_json(builder: _CatalogBuilder, raw_dir: Path) -> int:
    count = 0
    zpath = raw_dir / "ccf_2026_md.zip"
    if not zpath.exists():
        return 0
    with zipfile.ZipFile(zpath) as zf:
        data = json.loads(zf.read("ccf-recommended-list-2026-markdown-main/data/ccf-2026.json"))
    for dom in data.get("domains") or []:
        cat_name = dom.get("name") or ""
        for kind, vtype in (("journals", "j"), ("conferences", "c")):
            tiers = dom.get(kind) or {}
            for tier, items in tiers.items():
                for item in items or []:
                    abbr = (item.get("abbr") or "").strip()
                    full_name = (item.get("full_name") or "").strip() or abbr
                    url = (item.get("url") or "").strip()
                    rec = builder.upsert(
                        name=full_name,
                        vtype=vtype,
                        acronym=abbr,
                        dblp_prefixes=[url] if "dblp" in url else [],
                    )
                    rec.add_rank("CCF", tier.upper(), cat_name, "2026")
                    count += 1
    return count


def _load_showjcr_tables(builder: _CatalogBuilder, raw_dir: Path) -> dict[str, int]:
    stats: dict[str, int] = {}
    zpath = raw_dir / "showjcr_repo.zip"
    if not zpath.exists():
        return stats

    prefix = "ShowJCR-master/中科院分区表及JCR原始数据文件/"
    with zipfile.ZipFile(zpath) as zf:
        # 1) CCF2026-UTF8.csv (fills anything missed & cross-checks)
        rows = list(csv.DictReader(io.StringIO(zf.read(prefix + "CCF2026-UTF8.csv").decode("utf-8-sig"))))
        for r in rows:
            abbr = (r.get("刊物名称") or "").strip()
            full_name = (r.get("Journal") or "").strip() or abbr
            cat = (r.get("领域") or "").strip()
            kind = (r.get("CCF推荐类别（国际学术刊物/会议）") or "").strip()
            vtype = "c" if "会议" in kind else "j"
            tier = (r.get("CCF推荐类型") or "").replace("类", "").strip().upper()
            url = (r.get("网址") or "").strip()
            if tier in {"A", "B", "C"}:
                rec = builder.upsert(
                    name=full_name,
                    vtype=vtype,
                    acronym=abbr,
                    dblp_prefixes=[url] if "dblp" in url else [],
                )
                # Keep `ccf-2026.json` authoritative if already populated on this record
                if not any(rk[0] == "CCF" for rk in rec.ranks):
                    rec.add_rank("CCF", tier, cat, "2026")
        stats["ccf2026_csv"] = len(rows)

        # 2) CCFT2025-UTF8.csv (计算领域高质量科技期刊分级目录 2025 T1/T2/T3)
        rows = list(csv.DictReader(io.StringIO(zf.read(prefix + "CCFT2025-UTF8.csv").decode("utf-8-sig"))))
        for r in rows:
            zh_name = (r.get("中文刊名") or "").strip()
            en_name = (r.get("Journal") or "").strip()
            tier = (r.get("T分区") or "").strip().upper()
            primary = en_name or zh_name
            aliases = [zh_name] if (en_name and zh_name) else []
            # Also strip "（英文）" from Chinese alias so readers querying Chinese title match
            if zh_name.endswith("（英文）"):
                aliases.append(zh_name.removesuffix("（英文）").strip())
            if primary and tier in {"T1", "T2", "T3"}:
                rec = builder.upsert(name=primary, vtype="j", aliases=aliases)
                rec.add_rank("CCF_CN", tier, "计算领域高质量科技期刊", "2025")
        stats["ccft2025"] = len(rows)

        # 3) CCFChinese2019-UTF8.csv (CCF 中文科技期刊目录 A/B/C)
        rows = list(csv.DictReader(io.StringIO(zf.read(prefix + "CCFChinese2019-UTF8.csv").decode("utf-8-sig"))))
        for r in rows:
            jname = (r.get("Journal") or "").strip()
            tier = (r.get("CCF推荐类型") or "").replace("类", "").strip().upper()
            if jname and tier in {"A", "B", "C"}:
                rec = builder.upsert(name=jname, vtype="j")
                # Keep T1/T2/T3 if already present from 2025; otherwise record Chinese tier
                if not any(rk[0] == "CCF_CN" for rk in rec.ranks):
                    rec.add_rank("CCF_CN", f"中文{tier}", "CCF推荐中文科技期刊", "2022")
        stats["ccf_zh"] = len(rows)

        # 4) FQBJCR2025-UTF8.csv (中科院分区表升级版 2025 全量 21,772 期刊)
        rows = list(csv.DictReader(io.StringIO(zf.read(prefix + "FQBJCR2025-UTF8.csv").decode("utf-8-sig"))))
        for r in rows:
            jname = (r.get("Journal") or "").strip()
            if not jname or is_umbrella_series(jname):
                continue
            issn_pair = (r.get("ISSN/EISSN") or "").strip()
            issns = extract_issns(issn_pair)
            broad_cat = (r.get("大类") or "").strip()
            broad_raw = (r.get("大类分区") or "").strip()
            m = _CAS_NUM_RE.match(broad_raw)
            if not m:
                continue
            div_num = m.group(1)
            is_top = (r.get("Top") or "").strip() == "是"
            rank_label = f"{broad_cat}{div_num}区" if broad_cat else f"{div_num}区"
            rec = builder.upsert(name=jname, vtype="j", issns=issns)
            rec.add_rank("CAS", rank_label, broad_cat, "2025", top=is_top)
        stats["cas2025"] = len(rows)

        # 5) JCR2025-UTF8.csv (Clarivate JCR 2025 Q1-Q4 + Impact Factor, 22,643 期刊)
        rows = list(csv.DictReader(io.StringIO(zf.read(prefix + "JCR2025-UTF8.csv").decode("utf-8-sig"))))
        for r in rows:
            jname = (r.get("Journal") or "").strip()
            if not jname or is_umbrella_series(jname):
                continue
            issns = extract_issns([r.get("ISSN") or "", r.get("EISSN") or ""])
            if_val = (r.get("IF(2025)") or "").strip()
            if if_val in {"N/A", "-", "0", "0.0"}:
                if_val = ""
            best_q = ""
            best_cat = ""
            for idx in range(1, 7):
                q = (r.get(f"IF Quartile(2025)_{idx}") or "").strip().upper()
                cat = (r.get(f"Category_{idx}") or "").strip()
                if q in _JCR_Q_ORDER and _JCR_Q_ORDER[q] > _JCR_Q_ORDER.get(best_q, 0):
                    best_q = q
                    best_cat = cat
            if best_q or if_val:
                rec = builder.upsert(name=jname, vtype="j", issns=issns)
                if best_q:
                    rec.add_rank("JCR", best_q, best_cat, "2025", impact_factor=if_val)
        stats["jcr2025"] = len(rows)

        # 6) CAS International Journal Early Warning Lists (2025 -> 2024 -> 2023 -> 2021 -> 2020)
        # Newer warning year takes precedence; older warnings kept with their year so readers know history.
        warn_count = 0
        for yr in ("2025", "2024", "2023", "2021", "2020"):
            fname = f"GJQKYJMD{yr}.csv"
            wrows = list(csv.DictReader(io.StringIO(zf.read(prefix + fname).decode("utf-8-sig"))))
            for r in wrows:
                jname = (r.get("Journal") or "").strip()
                reason = (
                    r.get(f"预警原因（{yr}）")
                    or r.get(f"预警等级（{yr}）")
                    or "预警"
                ).strip()
                if not jname:
                    continue
                rec = builder.upsert(name=jname, vtype="j")
                if not rec.warning:
                    label = f"{yr}年预警（{reason}）" if len(reason) > 1 else f"{yr}年{reason}风险预警"
                    rec.warning = label
                    warn_count += 1
        stats["cas_warnings"] = warn_count

    return stats


def _load_core_rankings(builder: _CatalogBuilder, raw_dir: Path) -> int:
    count = 0
    for fname, ed_year in (("icore_2026.csv", "2026"), ("core_2023.csv", "2023")):
        fpath = raw_dir / fname
        if not fpath.exists():
            continue
        rows = list(csv.reader(io.StringIO(fpath.read_text(encoding="utf-8", errors="replace"))))
        for r in rows:
            if len(r) < 5:
                continue
            title = clean_venue_text(r[1])
            acr = clean_venue_text(r[2])
            rank = clean_venue_text(r[4]).upper()
            if rank not in {"A*", "A", "B", "C"} or not title:
                continue
            rec = builder.upsert(name=title, vtype="c", acronym=acr)
            if not any(rk[0] == "CORE" for rk in rec.ranks):
                rec.add_rank("CORE", rank, "", ed_year)
                count += 1
    return count


def _load_ccfddl_and_thcpl(builder: _CatalogBuilder, raw_dir: Path) -> dict[str, int]:
    stats = {"ccfddl": 0, "th_cpl_md": 0}
    yml_path = raw_dir / "ccfddl_allconf.yml"
    if yml_path.exists():
        # Minimal deterministic YAML block reader for ccfddl's flat conference records
        text = yml_path.read_text(encoding="utf-8", errors="replace")
        blocks = re.split(r"\n(?=- title:\s*)", "\n" + text)
        for blk in blocks:
            m_title = re.search(r"^- title:\s*(.+)$", blk, re.MULTILINE)
            m_desc = re.search(r"^\s*description:\s*(.+)$", blk, re.MULTILINE)
            m_dblp = re.search(r"^\s*dblp:\s*(.+)$", blk, re.MULTILINE)
            m_ccf = re.search(r"^\s*ccf:\s*([ABC])\b", blk, re.MULTILINE)
            m_core = re.search(r"^\s*core:\s*(A\*|[ABC])\b", blk, re.MULTILINE)
            m_th = re.search(r"^\s*thcpl:\s*([AB])\b", blk, re.MULTILINE)
            if not m_title:
                continue
            acr = m_title.group(1).strip().strip("'\"")
            desc = (m_desc.group(1).strip().strip("'\"") if m_desc else "") or acr
            dblp_slug = m_dblp.group(1).strip().strip("'\"") if m_dblp else ""
            dblp_pref = f"conf/{dblp_slug}" if dblp_slug and "/" not in dblp_slug else dblp_slug
            rec = builder.upsert(
                name=desc,
                vtype="c",
                acronym=acr,
                dblp_prefixes=[dblp_pref] if dblp_pref else [],
            )
            if m_ccf and not any(rk[0] == "CCF" for rk in rec.ranks):
                rec.add_rank("CCF", m_ccf.group(1), "", "2026")
            if m_core and not any(rk[0] == "CORE" for rk in rec.ranks):
                rec.add_rank("CORE", m_core.group(1), "", "2026")
            if m_th:
                rec.add_rank("TH_CPL", m_th.group(1), "", "2024")
            stats["ccfddl"] += 1

    th_zip = raw_dir / "th_cpl_master.zip"
    if th_zip.exists():
        with zipfile.ZipFile(th_zip) as zf:
            md = zf.read("TH-CPL-master/readme.md").decode("utf-8", errors="replace")
        current_cat = ""
        current_tier = ""
        current_vtype = "j"
        for line in md.splitlines():
            line = line.strip()
            if line.startswith("## ") and "分类类别" not in line:
                current_cat = line.removeprefix("## ").strip()
            elif re.match(r"^[AB]类(期刊|会议)", line):
                current_tier = line[0]
                current_vtype = "j" if "期刊" in line else "c"
            elif "|" in line and current_tier and not line.startswith(("名称|", "-|")):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 2 and parts[0] and parts[1]:
                    acr, full = parts[0], parts[1]
                    rec = builder.upsert(name=full, vtype=current_vtype, acronym=acr)
                    rec.add_rank("TH_CPL", current_tier, current_cat, "2024")
                    stats["th_cpl_md"] += 1

    return stats


# Curated CAAI ratings + critical DBLP multi-prefix / alias overrides + emerging venues
_CURATED_CS_VENUES: tuple[dict[str, Any], ...] = (
    # Top AI / ML / CV / NLP conferences with multi-DBLP prefixes & CAAI
    {
        "name": "Advances in Neural Information Processing Systems",
        "acronym": "NeurIPS",
        "vtype": "c",
        "dblp": ["conf/neurips", "conf/nips"],
        "aliases": ["Neural Information Processing Systems", "NIPS"],
        "ranks": [("CCF", "A", "人工智能", "2026"), ("CORE", "A*", "", "2026"), ("TH_CPL", "A", "", "2024"), ("CAAI", "A", "人工智能", "2024")],
    },
    {
        "name": "International Conference on Learning Representations",
        "acronym": "ICLR",
        "vtype": "c",
        "dblp": ["conf/iclr"],
        "aliases": ["ICLR"],
        "ranks": [("CCF", "A", "人工智能", "2026"), ("CORE", "A*", "", "2026"), ("TH_CPL", "A", "", "2024"), ("CAAI", "A", "人工智能", "2024")],
    },
    {
        "name": "International Conference on Machine Learning",
        "acronym": "ICML",
        "vtype": "c",
        "dblp": ["conf/icml"],
        "ranks": [("CCF", "A", "人工智能", "2026"), ("CORE", "A*", "", "2026"), ("TH_CPL", "A", "", "2024"), ("CAAI", "A", "人工智能", "2024")],
    },
    {
        "name": "IEEE/CVF Conference on Computer Vision and Pattern Recognition",
        "acronym": "CVPR",
        "vtype": "c",
        "dblp": ["conf/cvpr"],
        "aliases": ["Computer Vision and Pattern Recognition"],
        "ranks": [("CCF", "A", "人工智能", "2026"), ("CORE", "A*", "", "2026"), ("TH_CPL", "A", "", "2024"), ("CAAI", "A", "计算机视觉", "2024")],
    },
    {
        "name": "IEEE/CVF International Conference on Computer Vision",
        "acronym": "ICCV",
        "vtype": "c",
        "dblp": ["conf/iccv"],
        "aliases": ["International Conference on Computer Vision"],
        "ranks": [("CCF", "A", "人工智能", "2026"), ("CORE", "A*", "", "2026"), ("TH_CPL", "A", "", "2024"), ("CAAI", "A", "计算机视觉", "2024")],
    },
    {
        "name": "European Conference on Computer Vision",
        "acronym": "ECCV",
        "vtype": "c",
        "dblp": ["conf/eccv"],
        "ranks": [("CCF", "B", "人工智能", "2026"), ("CORE", "A*", "", "2026"), ("TH_CPL", "A", "", "2024"), ("CAAI", "A", "计算机视觉", "2024")],
    },
    {
        "name": "Annual Meeting of the Association for Computational Linguistics",
        "acronym": "ACL",
        "vtype": "c",
        "dblp": ["conf/acl"],
        "ranks": [("CCF", "A", "人工智能", "2026"), ("CORE", "A*", "", "2026"), ("TH_CPL", "A", "", "2024"), ("CAAI", "A", "自然语言处理", "2024")],
    },
    {
        "name": "Conference on Empirical Methods in Natural Language Processing",
        "acronym": "EMNLP",
        "vtype": "c",
        "dblp": ["conf/emnlp"],
        "aliases": ["Empirical Methods in Natural Language Processing"],
        "ranks": [("CCF", "B", "人工智能", "2026"), ("CORE", "A*", "", "2026"), ("TH_CPL", "A", "", "2024"), ("CAAI", "A", "自然语言处理", "2024")],
    },
    {
        "name": "North American Chapter of the Association for Computational Linguistics",
        "acronym": "NAACL",
        "vtype": "c",
        "dblp": ["conf/naacl"],
        "aliases": ["NAACL-HLT"],
        "ranks": [("CCF", "B", "人工智能", "2026"), ("CORE", "A", "", "2026"), ("CAAI", "B", "自然语言处理", "2024")],
    },
    {
        "name": "AAAI Conference on Artificial Intelligence",
        "acronym": "AAAI",
        "vtype": "c",
        "dblp": ["conf/aaai"],
        "ranks": [("CCF", "A", "人工智能", "2026"), ("CORE", "A*", "", "2026"), ("TH_CPL", "A", "", "2024"), ("CAAI", "A", "人工智能", "2024")],
    },
    {
        "name": "International Joint Conference on Artificial Intelligence",
        "acronym": "IJCAI",
        "vtype": "c",
        "dblp": ["conf/ijcai"],
        "ranks": [("CCF", "B", "人工智能", "2026"), ("CORE", "A*", "", "2026"), ("TH_CPL", "A", "", "2024"), ("CAAI", "A", "人工智能", "2024")],
    },
    {
        "name": "ACM International Conference on Multimedia",
        "acronym": "ACM MM",
        "vtype": "c",
        "dblp": ["conf/mm"],
        "aliases": ["MM", "ACM Multimedia"],
        "ranks": [("CCF", "A", "计算机图形学与多媒体", "2026"), ("CORE", "A*", "", "2026"), ("TH_CPL", "A", "", "2024"), ("CAAI", "A", "多媒体", "2024")],
    },
    {
        "name": "ACM SIGKDD Conference on Knowledge Discovery and Data Mining",
        "acronym": "KDD",
        "vtype": "c",
        "dblp": ["conf/kdd"],
        "aliases": ["SIGKDD"],
        "ranks": [("CCF", "A", "数据库/数据挖掘/内容检索", "2026"), ("CORE", "A*", "", "2026"), ("TH_CPL", "A", "", "2024"), ("CAAI", "A", "数据挖掘", "2024")],
    },
    {
        "name": "International ACM SIGIR Conference on Research and Development in Information Retrieval",
        "acronym": "SIGIR",
        "vtype": "c",
        "dblp": ["conf/sigir"],
        "ranks": [("CCF", "A", "数据库/数据挖掘/内容检索", "2026"), ("CORE", "A*", "", "2026"), ("TH_CPL", "A", "", "2024"), ("CAAI", "A", "信息检索", "2024")],
    },
    {
        "name": "The Web Conference",
        "acronym": "WWW",
        "vtype": "c",
        "dblp": ["conf/www"],
        "aliases": ["International World Wide Web Conference", "ACM Web Conference"],
        "ranks": [("CCF", "A", "交叉/综合/新兴", "2026"), ("CORE", "A*", "", "2026"), ("TH_CPL", "A", "", "2024")],
    },
    # Security / Systems / SE / DB tricky DBLP prefixes & PACM series
    {
        "name": "IEEE Symposium on Security and Privacy",
        "acronym": "IEEE S&P",
        "vtype": "c",
        "dblp": ["conf/sp"],
        "aliases": ["Oakland", "SP"],
        "ranks": [("CCF", "A", "网络与信息安全", "2026"), ("CORE", "A*", "", "2026"), ("TH_CPL", "A", "", "2024")],
    },
    {
        "name": "USENIX Security Symposium",
        "acronym": "USENIX Security",
        "vtype": "c",
        "dblp": ["conf/uss"],
        "ranks": [("CCF", "A", "网络与信息安全", "2026"), ("CORE", "A*", "", "2026"), ("TH_CPL", "A", "", "2024")],
    },
    {
        "name": "ACM SIGSAC Conference on Computer and Communications Security",
        "acronym": "CCS",
        "vtype": "c",
        "dblp": ["conf/ccs"],
        "aliases": ["ACM CCS"],
        "ranks": [("CCF", "A", "网络与信息安全", "2026"), ("CORE", "A*", "", "2026"), ("TH_CPL", "A", "", "2024")],
    },
    {
        "name": "Network and Distributed System Security Symposium",
        "acronym": "NDSS",
        "vtype": "c",
        "dblp": ["conf/ndss"],
        "ranks": [("CCF", "A", "网络与信息安全", "2026"), ("CORE", "A*", "", "2026"), ("TH_CPL", "A", "", "2024")],
    },
    {
        "name": "USENIX Symposium on Operating Systems Design and Implementation",
        "acronym": "OSDI",
        "vtype": "c",
        "dblp": ["conf/osdi"],
        "ranks": [("CCF", "A", "软件工程/系统软件/程序设计语言", "2026"), ("CORE", "A*", "", "2026"), ("TH_CPL", "A", "", "2024")],
    },
    {
        "name": "ACM Symposium on Operating Systems Principles",
        "acronym": "SOSP",
        "vtype": "c",
        "dblp": ["conf/sosp"],
        "ranks": [("CCF", "A", "软件工程/系统软件/程序设计语言", "2026"), ("CORE", "A*", "", "2026"), ("TH_CPL", "A", "", "2024")],
    },
    {
        "name": "ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering",
        "acronym": "FSE",
        "vtype": "c",
        "dblp": ["conf/sigsoft", "journals/pacmse"],
        "aliases": ["ESEC/FSE", "ACM SIGSOFT Symposium on the Foundations of Software Engineering", "PACMSE", "Proceedings of the ACM on Software Engineering"],
        "ranks": [("CCF", "A", "软件工程/系统软件/程序设计语言", "2026"), ("CORE", "A*", "", "2026"), ("TH_CPL", "A", "", "2024")],
    },
    {
        "name": "Proceedings of the VLDB Endowment",
        "acronym": "PVLDB",
        "vtype": "j",
        "dblp": ["journals/pvldb", "conf/vldb"],
        "issns": ["2150-8097"],
        "aliases": ["VLDB", "International Conference on Very Large Data Bases", "Proc. VLDB Endow."],
        "ranks": [("CCF", "A", "数据库/数据挖掘/内容检索", "2026"), ("CORE", "A*", "", "2026"), ("TH_CPL", "A", "", "2024")],
    },
    {
        "name": "ACM SIGMOD International Conference on Management of Data",
        "acronym": "SIGMOD",
        "vtype": "c",
        "dblp": ["conf/sigmod", "journals/pacmmod"],
        "aliases": ["PACMMOD", "Proceedings of the ACM on Management of Data"],
        "ranks": [("CCF", "A", "数据库/数据挖掘/内容检索", "2026"), ("CORE", "A*", "", "2026"), ("TH_CPL", "A", "", "2024")],
    },
    {
        "name": "Proceedings of the ACM on Programming Languages",
        "acronym": "PACMPL",
        "vtype": "j",
        "dblp": ["journals/pacmpl"],
        "issns": ["2475-1421"],
        "aliases": ["Proc. ACM Program. Lang."],
        "ranks": [("CCF", "A", "软件工程/系统软件/程序设计语言", "2026")],
    },
    {
        "name": "Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies",
        "acronym": "IMWUT",
        "vtype": "j",
        "dblp": ["journals/imwut", "conf/huc"],
        "issns": ["2474-9567"],
        "aliases": ["UbiComp"],
        "ranks": [("CCF", "A", "人机交互与普适计算", "2026"), ("CORE", "A*", "", "2026")],
    },
    {
        "name": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
        "acronym": "TPAMI",
        "vtype": "j",
        "dblp": ["journals/pami", "journals/tpami"],
        "issns": ["0162-8828", "1939-3539"],
        "aliases": ["IEEE TPAMI", "PAMI", "IEEE Trans. Pattern Anal. Mach. Intell."],
        "ranks": [("CCF", "A", "人工智能", "2026"), ("TH_CPL", "A", "", "2024"), ("CAAI", "A", "人工智能", "2024")],
    },
    {
        "name": "IEEE Transactions on Image Processing",
        "acronym": "TIP",
        "vtype": "j",
        "dblp": ["journals/tip"],
        "issns": ["1057-7149", "1941-0042"],
        "aliases": ["IEEE TIP", "IEEE Trans. Image Process."],
        "ranks": [("CCF", "A", "计算机图形学与多媒体", "2026"), ("TH_CPL", "A", "", "2024"), ("CAAI", "A", "图像处理", "2024")],
    },
    {
        "name": "International Journal of Computer Vision",
        "acronym": "IJCV",
        "vtype": "j",
        "dblp": ["journals/ijcv"],
        "issns": ["0920-5691", "1573-1405"],
        "ranks": [("CCF", "A", "人工智能", "2026"), ("TH_CPL", "A", "", "2024"), ("CAAI", "A", "计算机视觉", "2024")],
    },
    {
        "name": "Journal of Machine Learning Research",
        "acronym": "JMLR",
        "vtype": "j",
        "dblp": ["journals/jmlr"],
        "issns": ["1532-4435", "1533-7928"],
        "ranks": [("CCF", "A", "人工智能", "2026"), ("TH_CPL", "A", "", "2024"), ("CAAI", "A", "机器学习", "2024")],
    },
    # Robotics / AI important conferences
    {
        "name": "IEEE International Conference on Robotics and Automation",
        "acronym": "ICRA",
        "vtype": "c",
        "dblp": ["conf/icra"],
        "ranks": [("CCF", "B", "人工智能", "2026"), ("CORE", "A*", "", "2026"), ("CAAI", "A", "智能机器人", "2024")],
    },
    {
        "name": "IEEE/RSJ International Conference on Intelligent Robots and Systems",
        "acronym": "IROS",
        "vtype": "c",
        "dblp": ["conf/iros"],
        "ranks": [("CCF", "C", "人工智能", "2026"), ("CORE", "A", "", "2026"), ("CAAI", "A", "智能机器人", "2024")],
    },
    {
        "name": "Robotics: Science and Systems",
        "acronym": "RSS",
        "vtype": "c",
        "dblp": ["conf/rss"],
        "ranks": [("CORE", "A*", "", "2026"), ("CAAI", "A", "智能机器人", "2024")],
    },
    {
        "name": "Conference on Robot Learning",
        "acronym": "CoRL",
        "vtype": "c",
        "dblp": ["conf/corl"],
        "ranks": [("CORE", "A", "", "2026"), ("CAAI", "B", "智能机器人", "2024")],
    },
    {
        "name": "Medical Image Computing and Computer-Assisted Intervention",
        "acronym": "MICCAI",
        "vtype": "c",
        "dblp": ["conf/miccai"],
        "ranks": [("CCF", "B", "交叉/综合/新兴", "2026"), ("CORE", "A", "", "2026"), ("CAAI", "A", "生物医学图像", "2024")],
    },
    {
        "name": "IEEE/CVF Winter Conference on Applications of Computer Vision",
        "acronym": "WACV",
        "vtype": "c",
        "dblp": ["conf/wacv"],
        "ranks": [("CORE", "A", "", "2026"), ("CAAI", "B", "计算机视觉", "2024")],
    },
    # Emerging open-review / systems venues (honest note without fabricating CCF/SCI)
    {
        "name": "Transactions on Machine Learning Research",
        "acronym": "TMLR",
        "vtype": "j",
        "dblp": ["journals/tmlr"],
        "issns": ["2835-8856"],
        "aliases": ["Trans. Mach. Learn. Res."],
        "ranks": [],
        "note": "OpenReview double-blind peer-reviewed ML journal (JMLR sister venue; supports J2C certification)",
    },
    {
        "name": "Conference on Language Modeling",
        "acronym": "COLM",
        "vtype": "c",
        "dblp": ["conf/colm"],
        "ranks": [],
        "note": "OpenReview peer-reviewed conference dedicated to language modeling",
    },
    {
        "name": "Conference on Machine Learning and Systems",
        "acronym": "MLSys",
        "vtype": "c",
        "dblp": ["conf/mlsys"],
        "ranks": [("CCF", "B", "交叉/综合/新兴", "2026")],
    },
    {
        "name": "ACM SIGGRAPH Annual Conference",
        "acronym": "SIGGRAPH",
        "vtype": "c",
        "dblp": ["conf/siggraph"],
        "aliases": ["ACM SIGGRAPH", "SIGGRAPH Conference Papers"],
        "ranks": [("CCF", "A", "计算机图形学与多媒体", "2026"), ("CORE", "A*", "", "2026"), ("TH_CPL", "A", "", "2024")],
    },
    {
        "name": "ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems",
        "acronym": "SIGSPATIAL",
        "vtype": "c",
        "dblp": ["conf/gis", "journals/sigspatial"],
        "aliases": ["ACM Special Interest Group on Spatial Information", "ACM GIS"],
        "ranks": [("CCF", "C", "数据库/数据挖掘/内容检索", "2026"), ("CORE", "A", "", "2026")],
    },
    {
        "name": "Proceedings of the National Academy of Sciences",
        "acronym": "PNAS",
        "vtype": "j",
        "issns": ["0027-8424", "1091-6490"],
        "aliases": [
            "Proceedings of the National Academy of Sciences of the United States of America",
            "Proc. Natl. Acad. Sci. U.S.A.",
            "Proc. Natl. Acad. Sci.",
        ],
        "ranks": [("JCR", "Q1", "MULTIDISCIPLINARY SCIENCES", "2025")],
    },
)


def _apply_curated_overrides(builder: _CatalogBuilder) -> int:
    for spec in _CURATED_CS_VENUES:
        rec = builder.upsert(
            name=spec["name"],
            vtype=spec["vtype"],
            acronym=spec.get("acronym", ""),
            issns=spec.get("issns", []),
            dblp_prefixes=spec.get("dblp", []),
            aliases=spec.get("aliases", []),
        )
        for sys_name, rank, cat, ed in spec.get("ranks", []):
            # Never overwrite an official CCF 2026 rank loaded from `ccf-2026.json`
            if sys_name == "CCF" and any(rk[0] == "CCF" for rk in rec.ranks):
                continue
            rec.add_rank(sys_name, rank, cat, ed)
        if spec.get("note") and not rec.note:
            rec.note = spec["note"]
    return len(_CURATED_CS_VENUES)


def _populate_iso4_journal_aliases(builder: _CatalogBuilder) -> int:
    """Add deterministic ISO-4 abbreviated aliases (e.g. 'IEEE Trans. Knowl. Data Eng.')
    for CS/CCF/top journals without colliding with existing canonical titles."""
    existing_names = {normalize_name_key(r.name) for r in builder.records}
    added = 0
    for rec in builder.records:
        if rec.vtype != "j":
            continue
        is_cs_or_top = any(
            rk[0] in {"CCF", "TH_CPL", "CAAI", "CCF_CN"} or (rk[0] == "JCR" and rk[1] == "Q1")
            for rk in rec.ranks
        )
        if not is_cs_or_top and not rec.warning:
            continue
        for title_src in (rec.name, *list(rec.aliases)):
            iso4 = _generate_iso4_alias(title_src)
            if not iso4:
                continue
            ik = normalize_name_key(iso4)
            if ik and ik not in existing_names:
                if all(normalize_name_key(x) != ik for x in rec.aliases):
                    rec.aliases.insert(0, iso4)
                    added += 1
    return added


def build_catalog(raw_dir: Path, output_path: Path, *, profile: str = "full") -> dict[str, Any]:
    t0 = time.perf_counter()
    builder = _CatalogBuilder()

    # Load CS authoritative directories first so mixed-case full names & DBLP keys anchor records
    ccf_json_n = _load_ccf_2026_json(builder, raw_dir)
    core_n = _load_core_rankings(builder, raw_dir)
    th_stats = _load_ccfddl_and_thcpl(builder, raw_dir)
    show_stats = _load_showjcr_tables(builder, raw_dir)
    curated_n = _apply_curated_overrides(builder)
    iso4_n = _populate_iso4_journal_aliases(builder)

    if profile == "open":
        # Strip proprietary commercial IF strings while keeping open/community tiers & warnings
        for rec in builder.records:
            rec.ranks = [
                (s, r, c, e, t, "") for (s, r, c, e, t, _if) in rec.ranks
            ]

    kept = [rec.to_compact() for rec in builder.records if rec.ranks or rec.warning or rec.note]
    payload = {
        "version": f"2026.11-p17-{profile}",
        "editions": {
            "CCF": "2026",
            "CORE": "2026",
            "TH_CPL": "2024",
            "CAAI": "2024",
            "CCF_CN": "2025",
            "CAS": "2025",
            "JCR": "2025",
        },
        "venues": kept,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with gzip.open(output_path, "wb", compresslevel=9) as fh:
        fh.write(raw_bytes)

    elapsed = time.perf_counter() - t0
    return {
        "version": payload["version"],
        "total_venues": len(kept),
        "uncompressed_bytes": len(raw_bytes),
        "compressed_bytes": output_path.stat().st_size,
        "elapsed_s": round(elapsed, 2),
        "sources": {
            "ccf_2026_json": ccf_json_n,
            "core_icore": core_n,
            **th_stats,
            **show_stats,
            "curated": curated_n,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compressed authoritative venue catalog.")
    parser.add_argument("--raw-dir", required=True, type=Path, help="Directory holding downloaded raw datasets")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path, help="Output .json.gz path")
    parser.add_argument("--profile", choices=("full", "open"), default="full")
    args = parser.parse_args()

    summary = build_catalog(args.raw_dir, args.output, profile=args.profile)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
