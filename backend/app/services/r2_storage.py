"""Minimal Cloudflare R2 (S3-compatible) uploader using AWS SigV4.

Only a single ``PUT object`` is needed (to host report figures so they render in
exported Zotero notes), so rather than pull in boto3 we sign one request by hand
with the stdlib + httpx. R2 speaks the S3 API with ``region = "auto"`` and
path-style addressing (``https://<endpoint>/<bucket>/<key>``).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import logging
from urllib.parse import quote, urlparse

import httpx

from app.config import get_settings

logger = logging.getLogger("scholar.r2")

_REGION = "auto"
_SERVICE = "s3"
_ALGORITHM = "AWS4-HMAC-SHA256"


def is_enabled() -> bool:
    return get_settings().r2_enabled


def public_url(key: str) -> str:
    base = get_settings().r2_public_base_url.rstrip("/")
    return f"{base}/{key.lstrip('/')}"


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _signing_key(secret: str, datestamp: str) -> bytes:
    k_date = _sign(("AWS4" + secret).encode("utf-8"), datestamp)
    k_region = _sign(k_date, _REGION)
    k_service = _sign(k_region, _SERVICE)
    return _sign(k_service, "aws4_request")


def _build_put(key: str, data: bytes, content_type: str) -> tuple[str, dict[str, str]]:
    """Return ``(url, headers)`` for a SigV4-signed ``PUT object`` request."""
    s = get_settings()
    endpoint = s.r2_endpoint.rstrip("/")
    host = urlparse(endpoint).netloc

    now = _dt.datetime.now(_dt.timezone.utc)
    amzdate = now.strftime("%Y%m%dT%H%M%SZ")
    datestamp = now.strftime("%Y%m%d")

    # Path-style canonical URI: /<bucket>/<key> with each segment URI-encoded.
    encoded_key = "/".join(quote(seg, safe="") for seg in key.split("/"))
    canonical_uri = f"/{quote(s.r2_bucket, safe='')}/{encoded_key}"

    payload_hash = hashlib.sha256(data).hexdigest()
    canonical_headers = (
        f"content-type:{content_type}\n"
        f"host:{host}\n"
        f"x-amz-content-sha256:{payload_hash}\n"
        f"x-amz-date:{amzdate}\n"
    )
    signed_headers = "content-type;host;x-amz-content-sha256;x-amz-date"
    canonical_request = "\n".join(
        ["PUT", canonical_uri, "", canonical_headers, signed_headers, payload_hash]
    )

    scope = f"{datestamp}/{_REGION}/{_SERVICE}/aws4_request"
    string_to_sign = "\n".join(
        [
            _ALGORITHM,
            amzdate,
            scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ]
    )
    signature = hmac.new(
        _signing_key(s.r2_secret_access_key, datestamp),
        string_to_sign.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    authorization = (
        f"{_ALGORITHM} Credential={s.r2_access_key_id}/{scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )

    headers = {
        "Authorization": authorization,
        "x-amz-date": amzdate,
        "x-amz-content-sha256": payload_hash,
        "content-type": content_type,
    }
    return f"{endpoint}{canonical_uri}", headers


async def upload_file(key: str, data: bytes, content_type: str) -> str | None:
    """Upload ``data`` to ``key``; return its public URL, or ``None`` on failure.

    Never raises — figure hosting is best-effort, so a failed upload just falls
    back to leaving the figure out of the note rather than breaking the export.
    """
    if not is_enabled():
        return None
    try:
        url, headers = _build_put(key, data, content_type)
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.put(url, content=data, headers=headers)
        if resp.status_code in (200, 201):
            return public_url(key)
        logger.warning(
            "R2 upload failed for %s: HTTP %s %s",
            key,
            resp.status_code,
            resp.text[:200],
        )
    except Exception as e:  # pragma: no cover - network error path
        logger.warning("R2 upload error for %s: %s", key, e)
    return None
