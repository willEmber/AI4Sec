"""URL handling: identity for de-duplication, what may be fetched, and which
URLs are papers rather than web pages.

Papers get their own path. A paper reached through `read_web_page` would be
page-less web text, a weaker duplicate of the full text `download_paper` and
`ensure_paper_parsed` produce with page numbers — so the tool hands back the
identifiers and points the agent at those tools instead.
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

_TRACKING_PARAMS = re.compile(r"^(utm_[a-z]+|fbclid|gclid|mc_[a-z]+|ref|ref_src)$", re.IGNORECASE)
_LOCAL_HOSTS = {"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"}
_LOCAL_SUFFIXES = (".localhost", ".local", ".internal", ".lan", ".home.arpa")

_ARXIV_PATH = re.compile(
    r"^/(?:abs|pdf|html)/((?:\d{4}\.\d{4,5})|(?:[a-z\-]+(?:\.[A-Z]{2})?/\d{7}))(?:v\d+)?(?:\.pdf)?/?$",
    re.IGNORECASE,
)
_DOI = re.compile(r"(10\.\d{4,9}/[^\s?#]+)", re.IGNORECASE)
_ACL_PATH = re.compile(r"^/([A-Z0-9][0-9]{2}-[0-9]{4}|\d{4}\.[a-z\-]+\.\d+)(?:\.pdf)?/?$", re.IGNORECASE)


def domain_of(url: str) -> str:
    host = (urlsplit(url or "").hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def normalize_url(url: str) -> str:
    """One key per page: scheme and `www.` ignored, tracking parameters and
    fragments dropped, trailing slash removed, remaining query sorted."""
    parts = urlsplit((url or "").strip())
    host = domain_of(url)
    if parts.port and parts.port not in (80, 443):
        host = f"{host}:{parts.port}"
    path = parts.path.rstrip("/") or ""
    query = urlencode(
        sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not _TRACKING_PARAMS.match(k))
    )
    return urlunsplit(("", host, path, query, "")).lstrip("/")


def domain_matches(url: str, domains: tuple[str, ...] | list[str]) -> bool:
    """Whether the URL's host is one of `domains` or a subdomain of one."""
    host = domain_of(url)
    for raw in domains:
        d = raw.strip().lower().lstrip(".")
        d = d[4:] if d.startswith("www.") else d
        if d and (host == d or host.endswith("." + d)):
            return True
    return False


def fetchable_reason(url: str) -> str:
    """Why a URL may not be fetched, or `""` when it may.

    Every fetch goes through a third-party API, so none of these can reach our
    own network today; they are refused anyway so that a direct-fetch fallback
    added later cannot inherit an open door.
    """
    try:
        parts = urlsplit((url or "").strip())
    except ValueError:
        return "not a valid URL"
    if parts.scheme.lower() not in ("http", "https"):
        return "only http(s) URLs can be read"
    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        return "the URL has no host"
    if parts.username or parts.password:
        return "URLs carrying credentials are not read"
    if host in _LOCAL_HOSTS or host.endswith(_LOCAL_SUFFIXES) or "." not in host.strip("[]"):
        return "local and intranet hosts are not read"
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return ""
    if not ip.is_global:
        return "private, loopback and reserved addresses are not read"
    return ""


def scholarly_identifiers(url: str) -> dict[str, str]:
    """Paper identifiers a URL carries: `arxiv_id`, `doi`, `openreview_id`,
    `acl_id`. Empty for an ordinary web page."""
    try:
        parts = urlsplit((url or "").strip())
    except ValueError:
        return {}
    host = domain_of(url)
    path = unquote(parts.path or "")

    if host in ("arxiv.org", "export.arxiv.org"):
        m = _ARXIV_PATH.match(path)
        if m:
            return {"arxiv_id": m.group(1)}
    if host in ("doi.org", "dx.doi.org"):
        m = _DOI.search(path)
        if m:
            return {"doi": m.group(1).rstrip("/").lower()}
    if host == "openreview.net" and path.rstrip("/") in ("/forum", "/pdf"):
        forum = dict(parse_qsl(parts.query)).get("id", "")
        if forum:
            return {"openreview_id": forum}
    if host == "aclanthology.org":
        m = _ACL_PATH.match(path)
        if m:
            return {"acl_id": m.group(1)}
    return {}
