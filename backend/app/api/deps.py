"""Request-scoped dependencies for the agent API.

The identity a handler acts on comes from here and nowhere else. A request may
carry a `user_id`, an `owner_token` or any other self-declared field; none of
them is trusted. Only a credential this server signed resolves to a principal
(development plan §6).
"""

from __future__ import annotations

import logging

from fastapi import Header, HTTPException, Response

from app.services import identity

logger = logging.getLogger("scholar.api.deps")


async def require_principal(
    x_agent_token: str = Header(default="", alias=identity.AGENT_TOKEN_HEADER),
) -> str:
    """Resolve the caller's principal, or reject the request.

    Used by every endpoint that reads existing agent data. Endpoints that may
    legitimately be a caller's first contact use `principal_or_new` instead.
    """
    principal_id = await identity.resolve_principal(x_agent_token)
    if principal_id is None:
        raise HTTPException(
            status_code=401,
            detail=f"A valid {identity.AGENT_TOKEN_HEADER} is required.",
        )
    return principal_id


async def principal_or_new(
    response: Response,
    x_agent_token: str = Header(default="", alias=identity.AGENT_TOKEN_HEADER),
) -> str:
    """Resolve the caller's principal, minting one when they have none.

    A freshly minted credential is returned in the response header so the
    client can store it and send it back; it is never put in a response body,
    where it would end up in logs and caches alongside ordinary data.
    """
    principal_id, new_credential = await identity.resolve_or_create_principal(x_agent_token)
    if new_credential:
        response.headers[identity.AGENT_TOKEN_HEADER] = new_credential
        logger.info("Issued a new agent credential for principal %s", principal_id)
    return principal_id
