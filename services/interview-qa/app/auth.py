"""Inbound API-key authentication for this service's own endpoints.

Distinct from `INTERVIEW_QA_GROQ_API_KEY`, which authenticates *us* to
Groq — this authenticates *callers* to us, so an unauthenticated party
can't reach `/api/v1/questions/*` and burn the Groq quota this service is
paying for. There's no shared gateway/auth service in front of Recruitix's
microservices yet, so this is a simple shared-secret header check: enough
to stop casual/accidental abuse of an internet-reachable endpoint, not a
substitute for real per-user auth once one exists upstream.

Off by default (`INTERVIEW_QA_REQUIRE_API_KEY=0`) so local development and
the existing test suite don't need a key configured — but the README and
Dockerfile call out that production deployments must turn this on.
"""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException

from .config import get_settings


def _configured_keys() -> set[str]:
    settings = get_settings()
    return {k.strip() for k in settings.api_keys.split(",") if k.strip()}


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if not settings.require_api_key:
        return

    valid_keys = _configured_keys()
    if not valid_keys:
        # Misconfiguration: auth is required but no keys are set — fail
        # closed (reject everything) rather than silently accepting every
        # request, which is what an empty allowlist would otherwise do.
        raise HTTPException(
            status_code=503,
            detail="API key authentication is required but no keys are configured.",
        )

    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header.")

    # Constant-time comparison against each configured key — a plain `in`
    # check on a set is fine for hashing/lookup, but the per-key equality
    # check under the hood is a regular (timing-variable) string compare;
    # compare_digest closes that side channel.
    if not any(hmac.compare_digest(x_api_key, key) for key in valid_keys):
        raise HTTPException(status_code=401, detail="Invalid API key.")
