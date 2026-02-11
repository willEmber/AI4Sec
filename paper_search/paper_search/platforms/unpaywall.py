from __future__ import annotations

from ..utils import normalize_doi, normalize_whitespace
from ..http_client import HTTPClient


async def unpaywall_find_pdf_url(
    client: HTTPClient, *, doi: str, email: str
) -> str | None:
    doi_norm = normalize_doi(doi)
    email = (email or "").strip()
    if not doi_norm or not email:
        return None

    url = f"https://api.unpaywall.org/v2/{doi_norm}"
    status, data = await client.get_status_and_json(url, params={"email": email})
    if status == 404 or not isinstance(data, dict):
        return None
    best = data.get("best_oa_location") or {}
    pdf = normalize_whitespace(best.get("url_for_pdf") or "")
    if pdf:
        return pdf

    for loc in data.get("oa_locations") or []:
        pdf = normalize_whitespace((loc or {}).get("url_for_pdf") or "")
        if pdf:
            return pdf

    return None
