"""Deterministic venue normalization, subtrack isolation, and series unmasking (P17).

Design rules:
1. Never apply unconstrained edit-distance or embedding fuzzy matching across
   full journal/conference titles: ``Pattern Recognition`` (CCF B / CAS 1区) and
   ``Pattern Recognition Letters`` (CCF C / CAS 3区) differ by one suffix word.
2. Isolate ``workshop``, ``companion``, and ``findings`` subtracks before any
   rank lookup so workshops never inherit a main conference's CCF/CORE tier and
   ACL/EMNLP Findings are never conflated with the main proceedings.
3. Detect umbrella book/proceedings series (Springer LNCS, PMLR, CCIS, CEUR-WS,
   ACM ICPS, LIPIcs, Procedia...) and their ISSNs so callers keep probing DBLP,
   Crossref ``event``, S2, or OpenReview for the actual conference.
"""
from __future__ import annotations

import html
import re
import unicodedata

# ---------------------------------------------------------------------------
# Preprint & Umbrella Series Guards
# ---------------------------------------------------------------------------

_PREPRINT_RE = re.compile(
    r"(?:\b(?:arxiv|corr|biorxiv|medrxiv|chemrxiv|techrxiv|ssrn|vixra|repec|"
    r"authorea|preprints\.org|osf\s+preprints)\b|"
    r"\bcomputing\s+research\s+repository\b|"
    r"\bresearch\s+square\b|"
    r"\bpreprint(?:s|\s+server)?\b)",
    re.IGNORECASE,
)

# Container book series / publisher proceedings umbrellas that obscure the
# actual conference in Crossref / OpenAlex metadata.
_UMBRELLA_SERIES_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^lecture\s+notes\s+in\s+(?:computer\s+science|artificial\s+intelligence|bioinformatics|electrical\s+engineering|networks\s+and\s+systems|business\s+information\s+processing)\b",
        r"^(?:lncs|lnai|lnbi|lnee|lnns|lnbip)$",
        r"^communications\s+in\s+computer\s+and\s+information\s+science\b",
        r"^ccis$",
        r"^advances\s+in\s+intelligent\s+systems\s+and\s+computing\b",
        r"^proceedings\s+of\s+machine\s+learning\s+research\b",
        r"^jmlr\s+workshop\s+and\s+conference\s+proceedings\b",
        r"^pmlr$",
        r"^ceur\s+workshop\s+proceedings\b",
        r"^ceur[\s\-]?ws$",
        r"^acm\s+international\s+conference\s+proceeding(?:s)?\s+series\b",
        r"^aaai\s+publications\b",
        r"^leibniz\s+international\s+proceedings\s+in\s+informatics\b",
        r"^lipics$",
        r"^openaccess\s+series\s+in\s+informatics\b",
        r"^oasics$",
        r"^electronic\s+proceedings\s+in\s+theoretical\s+computer\s+science\b",
        r"^eptcs$",
        r"^epic\s+series\s+in\s+computing\b",
        r"^procedia\s+computer\s+science\b",
        r"^ifac[\s\-]papersonline\b",
        r"^iop\s+conference\s+series\b",
        r"^journal\s+of\s+physics:\s*conference\s+series\b",
        r"^(?:matec|e3s|itm|shs)\s+web\s+of\s+conferences\b",
        r"^(?:proceedings\s+of\s+)?spie(?:\s+proceedings)?$",
        r"^ieee\s+conference\s+publications?$",
    )
)

# ISSNs belonging to umbrella series (LNCS, CCIS, PMLR, CEUR-WS, Procedia CS, LIPIcs…)
# Must NEVER be matched as standalone journals!
UMBRELLA_ISSNS: frozenset[str] = frozenset(
    {
        "0302-9743",  # Lecture Notes in Computer Science (print)
        "1611-3349",  # Lecture Notes in Computer Science (electronic)
        "1865-0929",  # CCIS (print)
        "1865-0937",  # CCIS (electronic)
        "2640-3498",  # Proceedings of Machine Learning Research (PMLR)
        "1613-0073",  # CEUR Workshop Proceedings
        "1877-0509",  # Procedia Computer Science
        "1868-8969",  # LIPIcs
        "2190-6807",  # OASICS
        "1876-1100",  # LNEE
        "2367-3370",  # LNNS
        "2194-5357",  # AISC
        "0277-786X",  # Proceedings of SPIE
    }
)

# ---------------------------------------------------------------------------
# Subtrack Guards (Workshop / Companion / Findings)
# ---------------------------------------------------------------------------

# Base conference acronyms where appending "W" denotes a workshop proceedings
# (e.g. CVPRW, ICCVW, ECCVW). Carefully curated so real conferences ending in W
# (WWW, CSCW, ICWSM) are NEVER misclassified as workshops!
_WORKSHOP_BASE_ACRONYMS: frozenset[str] = frozenset(
    {
        "CVPR", "ICCV", "ECCV", "WACV", "ACCV", "BMVC", "MICCAI",
        "ICPR", "ICIP", "ICASSP", "ICME", "ISBI",
        "ICLR", "ICML", "NEURIPS", "NIPS", "AAAI", "IJCAI", "ECAI",
        "ACL", "EMNLP", "NAACL", "EACL", "COLING",
        "KDD", "ICDM", "SDM", "PAKDD", "ECMLPKDD",
        "SIGIR", "WWW", "WSDM", "CIKM", "RECSYS", "ICWSM",
        "SIGMOD", "VLDB", "ICDE", "EDBT", "DASFAA",
        "INFOCOM", "ICDCS", "PERCOM", "MOBICOM", "MOBIHOC", "GLOBECOM", "ICC",
        "ICSE", "FSE", "ESECFSE", "ASE", "ISSTA", "SANER", "ICSME",
        "SP", "CCS", "NDSS", "RAID", "ESORICS", "ACSAC",
        "OSDI", "SOSP", "NSDI", "EUROSYS", "HPCA", "ISCA", "MICRO", "ASPLOS",
        "CHI", "UBICOMP",
    }
)

_WORKSHOP_TEXT_RE = re.compile(
    r"(?:\bworkshops?\b|\bwkshps?\b|\bdoctoral\s+consortium\b|"
    r"\bstudent\s+research\s+workshop\b|\btutorials?\s+track\b|"
    r"\bdemo(?:nstration)?s?\s+track\b|\blate[\s\-]breaking\b|"
    r"\bsatellite\s+events?\b|\bpost[\s\-]conference\s+proceedings\b|"
    r"\btiny\s+papers?(?:\s+track)?\b|\bblog\s*posts?\s+track\b)",
    re.IGNORECASE,
)

# Unambiguous satellite subtracks that are NEVER standalone CCF/CORE main venues
_STRICT_SATELLITE_WORKSHOP_RE = re.compile(
    r"(?:\bworkshops$|\bwkshps?\b|\bdoctoral\s+consortium\b|"
    r"\bstudent\s+research\s+workshop\b|\btutorials?\s+track\b|"
    r"\bdemo(?:nstration)?s?\s+track\b|\blate[\s\-]breaking\b|"
    r"\bsatellite\s+events?\b|\bpost[\s\-]conference\s+proceedings\b|"
    r"\btiny\s+papers?(?:\s+track)?\b|\bblog\s*posts?\s+track\b)",
    re.IGNORECASE,
)

_HISTORICAL_PAREN_RE = re.compile(
    r"\(\s*(?:was|previously|formerly|prior\s+to|amalgamation\s+of|原\s*[:：]?)\b[^()]*\)",
    re.IGNORECASE,
)

_COMPANION_TEXT_RE = re.compile(
    r"(?:\bcompanion(?:\s+proceedings|\s+volume|\s+publication)?\b|"
    r"\badjunct(?:\s+proceedings|\s+publication)?\b|"
    r"\bextended\s+abstracts?\b|"
    r"\bchi\s+ea\b|\bsiggraph\s+(?:talks|posters|appy|emerging\s+technologies)\b)",
    re.IGNORECASE,
)

_FINDINGS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\bfindings\s+of\s+(?:the\s+)?association\s+for\s+computational\s+linguistics\s*(?::\s*([A-Za-z\-]+))?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(acl|emnlp|naacl|eacl|ijcnlp|aacl)[\s\-]*(?:\d{4}\s*)?\(\s*findings\s*\)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bfindings\s+of\s+(?:the\s+)?(?:\d{4}\s+)?(acl|emnlp|naacl|eacl|ijcnlp|aacl)\b",
        re.IGNORECASE,
    ),
)

# ---------------------------------------------------------------------------
# OpenReview status parsing
# ---------------------------------------------------------------------------

_OPENREVIEW_UNACCEPTED_RE = re.compile(
    r"(?:\bsubmitted\s+to\b|\bsubmission\b|\bwithdrawn\b|\brejected\b|"
    r"\bdesk[\s\-]?reject(?:ed)?\b|\bunder\s+review\b|\binvite\s+to\s+workshop\b)",
    re.IGNORECASE,
)
_OPENREVIEW_TIER_RE = re.compile(
    r"\b(oral|spotlight|notable[\s\-]top[\s\-]\d+%?|best\s+paper|poster)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Canonicalization & Acronym Rules
# ---------------------------------------------------------------------------

_ISSN_RE = re.compile(r"\b(\d{4})-?(\d{3}[\dXx])\b")
_DBLP_PREFIX_RE = re.compile(
    r"^(?:https?://dblp\d*\.(?:org|uni-trier\.de|dagstuhl\.de)/(?:rec|db)/)?((?:conf|journals)/[a-z0-9_\-]+)",
    re.IGNORECASE,
)

# Leading ACM DL / conference tag: "CHI '24: ...", "EuroS&P 2024: ...", "SC24: ..."
_LEADING_CONF_TAG_RE = re.compile(
    r"^([A-Z][A-Za-z0-9&/\-]{1,15}?)(?:\s*['’]?\d{2,4})?\s*:\s+(.+)$"
)

# Leading proceedings noise: "Proceedings of the 2024 IEEE/CVF ...", "37th Annual ..."
_LEADING_PROC_RE = re.compile(
    r"^(?:(?:in\s+)?proceedings\s+of\s+(?:the\s+)?|"
    r"conference\s+record\s+of\s+(?:the\s+)?|"
    r"digest\s+of\s+papers\.?\s*|"
    r"official\s+proceedings\s+of\s+(?:the\s+)?)+",
    re.IGNORECASE,
)
_YEAR_ORDINAL_PREFIX_RE = re.compile(
    r"^(?:(?:\d{4}(?:\s*[/\-–]\s*\d{2,4})?)\s+|"
    r"(?:\d{1,3}(?:st|nd|rd|th))\s+|"
    r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|"
    r"eighteenth|nineteenth|twentieth|twenty[\s\-][a-z]+|thirtieth|thirty[\s\-][a-z]+)\s+|"
    r"(?:[ivxlcdm]{2,7})\s+(?=international|annual|european|asian|acm|ieee)|"
    r"(?:annual|biannual|biennial)\s+)+",
    re.IGNORECASE,
)
# Trailing year / volume / part noise: "... (CVPR 2024)", "ACL (1)", "AAAI 2023"
_DBLP_VOLUME_SUFFIX_RE = re.compile(
    r"\s*\(\s*(?:\d+|long\s+papers?|short\s+papers?|main\s+track|main|vol\.?\s*\d+|part\s+[ivx\d]+)\s*\)\s*$",
    re.IGNORECASE,
)
_TRAILING_PART_VOLUME_RE = re.compile(
    r"(?:,\s*(?:volume|vol\.?|part)\s+[ivx\d]+|\s+conference\s+papers|\s+technical\s+papers)\s*$",
    re.IGNORECASE,
)
_TRAILING_YEAR_RE = re.compile(r"[\s,/\-–']+(?:19|20)\d{2}[a-z]?\s*$", re.IGNORECASE)
_PAREN_GROUP_RE = re.compile(r"\(([^()]{2,32})\)")

# Common journal title abbreviations seen in DBLP / S2 / IEEE / ACM citations
_JOURNAL_ABBREV_MAP: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bproc\.?\s+vldb\s+endow\.?\b", re.IGNORECASE), "PVLDB"),
    (re.compile(r"\bpvldb\b", re.IGNORECASE), "PVLDB"),
    (re.compile(r"\bproc\.?\s+acm\s+manag\.?\s+data\b", re.IGNORECASE), "PACMMOD"),
    (re.compile(r"\bpacmmod\b", re.IGNORECASE), "PACMMOD"),
    (re.compile(r"\bproc\.?\s+acm\s+program\.?\s+lang\.?\b", re.IGNORECASE), "PACMPL"),
    (re.compile(r"\bpacmpl\b", re.IGNORECASE), "PACMPL"),
    (re.compile(r"\bproc\.?\s+acm\s+interact\.?\s+mob\.?\s+wearable\s+ubiquitous\s+technol\.?\b", re.IGNORECASE), "IMWUT"),
    (re.compile(r"\bproc\.?\s+acm\s+hum\.?[\s\-]*comput\.?\s+interact\.?\b", re.IGNORECASE), "PACMHCI"),
    (re.compile(r"\bproc\.?\s+acm\s+meas\.?\s+anal\.?\s+comput\.?\s+syst\.?\b", re.IGNORECASE), "POMACS"),
    (re.compile(r"\bproc\.?\s+acm\s+softw\.?\s+eng\.?\b", re.IGNORECASE), "PACMSE"),
    (re.compile(r"\btrans\.?\s+mach\.?\s+learn\.?\s+res\.?\b", re.IGNORECASE), "Transactions on Machine Learning Research"),
    (re.compile(r"\bj\.?\s+mach\.?\s+learn\.?\s+res\.?\b", re.IGNORECASE), "Journal of Machine Learning Research"),
    (re.compile(r"\bcommun\.?\s+acm\b", re.IGNORECASE), "Communications of the ACM"),
    (re.compile(r"\bj\.?\s+acm\b", re.IGNORECASE), "Journal of the ACM"),
    (re.compile(r"\bacm\s+comput\.?\s+surv\.?\b", re.IGNORECASE), "ACM Computing Surveys"),
    (re.compile(r"^proc\.?\s+natl\.?\s+acad\.?\s+sci\.?(?:\s+u\.?\s*s\.?\s*a\.?)?$", re.IGNORECASE), "Proceedings of the National Academy of Sciences"),
    (re.compile(r"^pnas$", re.IGNORECASE), "Proceedings of the National Academy of Sciences"),
)

# Anchored society/organizer prefix so regional variants ("Asia", "European",
# "Pacific-Asia") or modified topics ("Semantic Web", "Mathematical Foundations",
# "Software Engineering and Knowledge Engineering") NEVER hijack flagship rules.
_ORG_PREFIX = (
    r"^(?:(?:the|ieee|acm|ieee/cvf|ieee/acm|ieee/rsj|cvf|usenix|aaai|"
    r"sigmod|sigkdd|sigir|sigsac|sigplan|sigsoft|sigchi|sigops|sigcomm|siggraph|"
    r"ifip|international|annual|joint)\s+)*"
    r"(?:(?:conference|symposium|meeting|congress)\s+(?:on|of|for)\s+(?:the\s+)?)?"
)
_ORG_SUFFIX = r"(?:\s+(?:conference|symposium))?$"

# High-precision anchored conference full-name patterns
_CONFERENCE_CANONICAL_RULES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pat, re.IGNORECASE), acr)
    for pat, acr in (
        (_ORG_PREFIX + r"(?:ieee(?:/cvf)?\s+)?winter\s+conference\s+on\s+applications\s+of\s+computer\s+vision" + _ORG_SUFFIX, "WACV"),
        (_ORG_PREFIX + r"computer\s+vision\s+and\s+pattern\s+recognition" + _ORG_SUFFIX, "CVPR"),
        (_ORG_PREFIX + r"computer\s+vision" + _ORG_SUFFIX, "ICCV"),
        (r"^(?:the\s+)?european\s+conference\s+on\s+computer\s+vision$", "ECCV"),
        (r"^(?:the\s+)?asian\s+conference\s+on\s+computer\s+vision$", "ACCV"),
        (r"^(?:the\s+)?british\s+machine\s+vision\s+conference$", "BMVC"),
        (_ORG_PREFIX + r"medical\s+image\s+computing\s+and\s+computer[\s\-]assisted\s+intervention" + _ORG_SUFFIX, "MICCAI"),
        (r"^(?:advances\s+in\s+|conference\s+on\s+)?neural\s+information\s+processing\s+systems$", "NeurIPS"),
        (_ORG_PREFIX + r"machine\s+learning" + _ORG_SUFFIX, "ICML"),
        (_ORG_PREFIX + r"learning\s+representations" + _ORG_SUFFIX, "ICLR"),
        (_ORG_PREFIX + r"language\s+modeling" + _ORG_SUFFIX, "COLM"),
        (_ORG_PREFIX + r"empirical\s+methods\s+in\s+natural\s+language\s+processing" + _ORG_SUFFIX, "EMNLP"),
        (r"^(?:conference\s+of\s+the\s+)?north\s+american\s+chapter\s+of\s+the\s+association\s+for\s+computational\s+linguistics(?:\s*[:\-–]\s*human\s+language\s+technologies)?$", "NAACL"),
        (r"^(?:conference\s+of\s+the\s+)?european\s+chapter\s+of\s+the\s+association\s+for\s+computational\s+linguistics$", "EACL"),
        (r"^(?:annual\s+)?meeting\s+of\s+the\s+association\s+for\s+computational\s+linguistics$", "ACL"),
        (_ORG_PREFIX + r"computational\s+linguistics" + _ORG_SUFFIX, "COLING"),
        (r"^(?:the\s+)?aaai\s+conference\s+on\s+artificial\s+intelligence$", "AAAI"),
        (r"^(?:the\s+)?international\s+joint\s+conference\s+on\s+artificial\s+intelligence$", "IJCAI"),
        (r"^(?:the\s+)?european\s+conference\s+on\s+artificial\s+intelligence$", "ECAI"),
        (_ORG_PREFIX + r"uncertainty\s+in\s+artificial\s+intelligence" + _ORG_SUFFIX, "UAI"),
        (_ORG_PREFIX + r"artificial\s+intelligence\s+and\s+statistics" + _ORG_SUFFIX, "AISTATS"),
        (_ORG_PREFIX + r"learning\s+theory" + _ORG_SUFFIX, "COLT"),
        (_ORG_PREFIX + r"robot\s+learning" + _ORG_SUFFIX, "CoRL"),
        (r"^robotics:\s*science\s+and\s+systems$", "RSS"),
        (_ORG_PREFIX + r"robotics\s+and\s+automation" + _ORG_SUFFIX, "ICRA"),
        (_ORG_PREFIX + r"intelligent\s+robots\s+and\s+systems" + _ORG_SUFFIX, "IROS"),
        (_ORG_PREFIX + r"knowledge\s+discovery\s+and\s+data\s+mining" + _ORG_SUFFIX, "KDD"),
        (_ORG_PREFIX + r"research\s+and\s+development\s+in\s+information\s+retrieval" + _ORG_SUFFIX, "SIGIR"),
        (r"^(?:the\s+)?(?:acm\s+)?web\s+conference$|^(?:international\s+)?world\s+wide\s+web\s+conference$", "WWW"),
        (_ORG_PREFIX + r"web\s+search\s+and\s+data\s+mining" + _ORG_SUFFIX, "WSDM"),
        (_ORG_PREFIX + r"information\s+and\s+knowledge\s+management" + _ORG_SUFFIX, "CIKM"),
        (_ORG_PREFIX + r"recommender\s+systems" + _ORG_SUFFIX, "RecSys"),
        (r"^acm\s+(?:international\s+)?conference\s+on\s+multimedia$", "ACM MM"),
        (_ORG_PREFIX + r"acoustics,?\s+speech,?\s+and\s+signal\s+processing" + _ORG_SUFFIX, "ICASSP"),
        (r"^(?:annual\s+)?conference\s+of\s+the\s+international\s+speech\s+communication\s+association$|^interspeech$", "INTERSPEECH"),
        (_ORG_PREFIX + r"management\s+of\s+data" + _ORG_SUFFIX, "SIGMOD"),
        (_ORG_PREFIX + r"very\s+large\s+data\s+bases" + _ORG_SUFFIX, "VLDB"),
        (_ORG_PREFIX + r"data\s+engineering" + _ORG_SUFFIX, "ICDE"),
        (_ORG_PREFIX + r"software\s+engineering" + _ORG_SUFFIX, "ICSE"),
        (r"^(?:acm\s+)?(?:joint\s+european\s+software\s+engineering\s+conference\s+and\s+)?(?:sigsoft\s+)?(?:international\s+)?symposium\s+on\s+(?:the\s+)?foundations\s+of\s+software\s+engineering$", "FSE"),
        (_ORG_PREFIX + r"automated\s+software\s+engineering" + _ORG_SUFFIX, "ASE"),
        (_ORG_PREFIX + r"software\s+testing\s+and\s+analysis" + _ORG_SUFFIX, "ISSTA"),
        (_ORG_PREFIX + r"programming\s+language\s+design\s+and\s+implementation" + _ORG_SUFFIX, "PLDI"),
        (_ORG_PREFIX + r"principles\s+of\s+programming\s+languages" + _ORG_SUFFIX, "POPL"),
        (_ORG_PREFIX + r"functional\s+programming" + _ORG_SUFFIX, "ICFP"),
        (_ORG_PREFIX + r"object[\s\-]oriented\s+programming,?\s+systems,?\s+languages,?\s+and\s+applications" + _ORG_SUFFIX, "OOPSLA"),
        (_ORG_PREFIX + r"operating\s+systems\s+design\s+and\s+implementation" + _ORG_SUFFIX, "OSDI"),
        (_ORG_PREFIX + r"operating\s+systems\s+principles" + _ORG_SUFFIX, "SOSP"),
        (_ORG_PREFIX + r"networked\s+systems\s+design\s+and\s+implementation" + _ORG_SUFFIX, "NSDI"),
        (r"^(?:acm\s+)?european\s+conference\s+on\s+computer\s+systems$", "EuroSys"),
        (_ORG_PREFIX + r"architectural\s+support\s+for\s+programming\s+languages\s+and\s+operating\s+systems" + _ORG_SUFFIX, "ASPLOS"),
        (_ORG_PREFIX + r"high\s+performance\s+computing,?\s+networking,?\s+storage\s+and\s+analysis" + _ORG_SUFFIX, "SC"),
        (r"^acm\s+siggraph$", "SIGGRAPH"),
        (_ORG_PREFIX + r"computer\s+and\s+communications\s+security" + _ORG_SUFFIX, "CCS"),
        (r"^usenix\s+security\s+symposium$", "USENIX Security"),
        (_ORG_PREFIX + r"network\s+and\s+distributed\s+system\s+security" + _ORG_SUFFIX, "NDSS"),
        (r"^(?:ieee\s+)?symposium\s+on\s+security\s+and\s+privacy$", "IEEE S&P"),
        (_ORG_PREFIX + r"(?:chi\s+conference\s+on\s+)?human\s+factors\s+in\s+computing\s+systems" + _ORG_SUFFIX, "CHI"),
        (_ORG_PREFIX + r"computer[\s\-]supported\s+cooperative\s+work(?:\s+and\s+social\s+computing)?" + _ORG_SUFFIX, "CSCW"),
        (_ORG_PREFIX + r"theory\s+of\s+computing" + _ORG_SUFFIX, "STOC"),
        (_ORG_PREFIX + r"foundations\s+of\s+computer\s+science" + _ORG_SUFFIX, "FOCS"),
        (_ORG_PREFIX + r"discrete\s+algorithms" + _ORG_SUFFIX, "SODA"),
        (_ORG_PREFIX + r"computer\s+aided\s+verification" + _ORG_SUFFIX, "CAV"),
    )
)


# ---------------------------------------------------------------------------
# Public Normalization Functions
# ---------------------------------------------------------------------------

def clean_venue_text(raw: str) -> str:
    """Decode HTML entities, normalize Unicode NFKC, strip zero-width/soft-hyphen/control chars & collapse spaces."""
    if not isinstance(raw, str):
        return ""
    text = html.unescape(raw)
    text = re.sub(r"[\u00ad\u200b-\u200f\u2060\ufeff]+", "", text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_issn(raw: str) -> str:
    """Return canonical ``XXXX-XXXX`` ISSN if valid and not an umbrella series ISSN."""
    if not raw:
        return ""
    m = _ISSN_RE.search(raw.strip())
    if not m:
        return ""
    issn = f"{m.group(1)}-{m.group(2).upper()}"
    if issn in UMBRELLA_ISSNS:
        return ""
    return issn


def extract_issns(values: list[str] | tuple[str, ...] | str | None) -> list[str]:
    """Deduplicated canonical ISSNs (excluding umbrella series ISSNs)."""
    if not values:
        return []
    items = [values] if isinstance(values, str) else list(values)
    seen: list[str] = []
    for item in items:
        for m in _ISSN_RE.finditer(item or ""):
            issn = f"{m.group(1)}-{m.group(2).upper()}"
            if issn not in UMBRELLA_ISSNS and issn not in seen:
                seen.append(issn)
    return seen


def extract_dblp_prefix(dblp_key: str) -> str:
    """Extract ``conf/<id>`` or ``journals/<id>`` from a DBLP key or URL."""
    cleaned = clean_venue_text(dblp_key)
    if not cleaned:
        return ""
    m = _DBLP_PREFIX_RE.match(cleaned)
    if not m:
        return ""
    prefix = m.group(1).lower()
    # CoRR is arXiv in DBLP; never treat it as a peer-reviewed journal.
    if prefix == "journals/corr":
        return ""
    return prefix


def is_preprint_venue(venue: str) -> bool:
    """True when ``venue`` names a preprint server rather than a peer-reviewed outlet."""
    cleaned = clean_venue_text(venue)
    if not cleaned:
        return False
    if _OPENREVIEW_UNACCEPTED_RE.search(cleaned):
        return True
    return bool(_PREPRINT_RE.search(cleaned))


def is_umbrella_series(venue: str) -> bool:
    """True when ``venue`` is a generic proceedings book series (LNCS, PMLR, CCIS…)."""
    cleaned = clean_venue_text(venue)
    if not cleaned:
        return False
    # Strip trailing volume numbers like "Lecture Notes in Computer Science (including subseries...)"
    head = re.sub(r"\s*\(.*$", "", cleaned).strip()
    return any(p.search(cleaned) or p.search(head) for p in _UMBRELLA_SERIES_PATTERNS)


def is_strict_satellite_workshop(venue: str) -> bool:
    """True when ``venue`` carries an unambiguous satellite workshop/track marker."""
    cleaned = _HISTORICAL_PAREN_RE.sub(" ", clean_venue_text(venue))
    if not cleaned:
        return False
    return bool(_STRICT_SATELLITE_WORKSHOP_RE.search(cleaned))


def _is_workshop_acronym(token: str) -> bool:
    up = re.sub(r"[^A-Z0-9]", "", token.upper())
    if len(up) < 4 or not up.endswith("W"):
        return False
    base = up[:-1]
    return base in _WORKSHOP_BASE_ACRONYMS


def classify_subtrack(venue: str, *, dblp_key: str = "") -> tuple[str, str]:
    """Classify ``(subtrack, extracted_host_acronym)`` for a venue string.

    ``subtrack`` is one of: ``preprint``, ``findings``, ``workshop``,
    ``companion``, ``main``.
    """
    raw_cleaned = clean_venue_text(venue)
    # Strip historical rename notes like "(was International Workshop on ...)" before subtrack checks!
    cleaned = _HISTORICAL_PAREN_RE.sub(" ", raw_cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    prefix = extract_dblp_prefix(dblp_key)

    if (dblp_key and "journals/corr" in dblp_key.lower()) or (cleaned and is_preprint_venue(cleaned)):
        return "preprint", ""

    for pat in _FINDINGS_PATTERNS:
        m = pat.search(cleaned)
        if m:
            host = (m.group(1) or "ACL").strip().upper()
            return "findings", host

    if _COMPANION_TEXT_RE.search(cleaned):
        return "companion", ""

    # Check if DBLP prefix itself is a workshop stream (e.g. conf/cvprw, conf/iccvw, conf/icdmw)
    if prefix.startswith("conf/"):
        stream = prefix.split("/", 1)[1].upper()
        if _is_workshop_acronym(stream):
            return "workshop", stream[:-1]

    # Check standalone or parenthetical workshop acronym like "CVPRW" or "... (CVPRW 2024)"
    stripped_year = _TRAILING_YEAR_RE.sub("", cleaned).strip()
    if _is_workshop_acronym(stripped_year):
        return "workshop", re.sub(r"[^A-Z0-9]", "", stripped_year.upper())[:-1]
    for grp in _PAREN_GROUP_RE.findall(cleaned):
        tok = _TRAILING_YEAR_RE.sub("", grp).strip()
        if _is_workshop_acronym(tok):
            return "workshop", re.sub(r"[^A-Z0-9]", "", tok.upper())[:-1]

    if _WORKSHOP_TEXT_RE.search(cleaned):
        return "workshop", ""

    return "main", ""


def parse_openreview_status(status_line: str) -> tuple[str, str, str]:
    """Parse an OpenReview venue/acceptance string into ``(venue, subtrack, tier)``.

    Returns ``("", "preprint", "")`` when the submission was not accepted
    (e.g. ``Submitted to ICLR 2025``, ``Withdrawn``, ``ACL ARR ... Submission``).
    """
    cleaned = clean_venue_text(status_line)
    if not cleaned:
        return "", "", ""
    if _OPENREVIEW_UNACCEPTED_RE.search(cleaned):
        return "", "preprint", ""

    subtrack, host = classify_subtrack(cleaned)
    tier = ""
    tm = _OPENREVIEW_TIER_RE.search(cleaned)
    if tm:
        raw_tier = tm.group(1).lower()
        if "oral" in raw_tier or "best" in raw_tier or "notable" in raw_tier:
            tier = "oral"
        elif "spotlight" in raw_tier:
            tier = "spotlight"
        elif "poster" in raw_tier:
            tier = "poster"

    if subtrack == "findings" and host:
        return f"Findings of {host}", "findings", tier

    # Strip presentation tier words, special main-equivalent tracks, and years to recover clean venue
    venue_part = re.sub(
        r"\b(?:accepted\s+(?:by|to|at|in)|oral|spotlight|poster|accept(?:ed)?|"
        r"notable[\s\-]top[\s\-]\d+%?|presentation|main\s+track|conference\s+track|"
        r"datasets?\s+and\s+benchmarks?(?:\s+track)?|"
        r"journal[\s\-]to[\s\-]conference(?:\s+track)?|j2c\s+certification)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    venue_part = re.sub(r"\b(?:19|20)\d{2}\b", " ", venue_part)
    venue_part = re.sub(r"[:\-–,/()]+", " ", venue_part)
    venue_part = re.sub(r"\s+", " ", venue_part).strip()
    return venue_part, subtrack, tier


def normalize_name_key(name: str) -> str:
    """Canonical lowercase lookup key preserving word boundaries (no fuzzy stemming)."""
    cleaned = clean_venue_text(name)
    if not cleaned:
        return ""
    s = cleaned.lower()
    s = s.replace("&", " and ")
    # Remove leading "the " for journals like "The Journal of Machine Learning Research"
    s = re.sub(r"^the\s+", "", s)
    # Keep alphanumeric and CJK characters, turn punctuation into single spaces
    s = re.sub(r"[^\w\u4e00-\u9fff]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def strip_venue_decorations(venue: str) -> tuple[str, list[str]]:
    """Strip proceedings/ordinal/year/volume wrappers and extract parenthetical acronyms.

    Caller must run ``classify_subtrack`` FIRST so workshops and Findings are
    not collapsed into main conferences.
    """
    cleaned = clean_venue_text(venue)
    if not cleaned:
        return "", []

    # Check common abbreviated journal names first
    for pat, replacement in _JOURNAL_ABBREV_MAP:
        if pat.fullmatch(cleaned):
            return replacement, [replacement]

    cleaned_no_hist = _HISTORICAL_PAREN_RE.sub(" ", cleaned)
    cleaned_no_hist = re.sub(r"\s+", " ", cleaned_no_hist).strip()

    acronyms: list[str] = []

    # Extract leading ACM DL / conference tag like "CHI '24: ...", "EuroS&P 2024: ...", "SC24: ..."
    working = cleaned_no_hist
    m_lead = _LEADING_CONF_TAG_RE.match(working)
    if m_lead:
        raw_tag, rest = m_lead.group(1).strip(), m_lead.group(2).strip()
        # Strip glued 2- or 4-digit year suffix like "SC24" -> "SC", "CVPR2024" -> "CVPR"
        tag_clean = re.sub(r"(?:19|20)\d{2}$|(?<=[A-Za-z])\d{2}$", "", raw_tag).strip(" -'")
        if len(tag_clean) >= 2 and tag_clean.lower() not in {"ieee", "acm", "vol", "part"}:
            acronyms.append(tag_clean)
        if len(rest) >= 6:
            working = rest

    for grp in _PAREN_GROUP_RE.findall(working):
        cand = _TRAILING_YEAR_RE.sub("", grp).strip()
        cand = re.sub(r"^(?:19|20)\d{2}\s+", "", cand).strip()
        cand = re.sub(r"(?<=[A-Za-z])\s+\d{2}$", "", cand).strip()
        # Plausible acronym: 2..20 chars, mostly uppercase letters/digits/hyphens/&
        if 2 <= len(cand) <= 20 and re.fullmatch(r"[A-Z0-9][A-Za-z0-9\-&/ ]*[A-Za-z0-9*]", cand):
            if cand.lower() not in {
                "online", "print", "part", "vol", "ieee", "acm", "main",
                "long papers", "short papers", "main track",
            } and cand not in acronyms:
                acronyms.append(cand)

    # Remove DBLP volume suffix like "ACL (1)" -> "ACL", and parenthetical acronyms
    base = _DBLP_VOLUME_SUFFIX_RE.sub("", working).strip()
    base_no_paren = _PAREN_GROUP_RE.sub(" ", base)
    base_no_paren = re.sub(r"\s+", " ", base_no_paren).strip(" ,;:-–/")
    base_no_paren = _TRAILING_PART_VOLUME_RE.sub("", base_no_paren).strip(" ,;:-–/")

    # Iteratively strip leading "Proceedings of the 31st 2023 ..." wrappers
    prev = None
    while prev != base_no_paren:
        prev = base_no_paren
        base_no_paren = _LEADING_PROC_RE.sub("", base_no_paren).strip(" ,;:-–/")
        base_no_paren = _YEAR_ORDINAL_PREFIX_RE.sub("", base_no_paren).strip(" ,;:-–/")

    base_no_paren = _TRAILING_PART_VOLUME_RE.sub("", base_no_paren).strip(" ,;:-–/")
    base_no_paren = _TRAILING_YEAR_RE.sub("", base_no_paren).strip(" ,;:-–/")
    base_no_paren = _TRAILING_PART_VOLUME_RE.sub("", base_no_paren).strip(" ,;:-–/")

    for pat, canonical_acr in _CONFERENCE_CANONICAL_RULES:
        if pat.search(base_no_paren):
            if canonical_acr not in acronyms:
                acronyms.insert(0, canonical_acr)
            break

    return base_no_paren or cleaned, acronyms


def normalize_venue_display(venue: str) -> str:
    """Drop-in replacement for ``sphere_scorer.normalize_venue`` with full guards.

    Maps sprawling conference-proceeding names to canonical acronyms while
    keeping workshops, Findings, and journals distinct.
    """
    cleaned = clean_venue_text(venue)
    if not cleaned:
        return ""
    subtrack, host = classify_subtrack(cleaned)
    if subtrack == "findings":
        return f"Findings of {host}" if host else cleaned
    if subtrack in {"workshop", "companion"}:
        return cleaned
    stripped, acronyms = strip_venue_decorations(cleaned)
    if acronyms:
        return acronyms[0]
    return stripped
