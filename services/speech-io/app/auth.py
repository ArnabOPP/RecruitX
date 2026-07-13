"""Inbound API-key authentication for this service's own endpoints.

Distinct from `SPEECH_IO_GROQ_API_KEY`, which authenticates *us* to Groq —
this authenticates *callers* to us, so an unauthenticated party can't reach
`/api/v1/speech/*` and burn the Groq transcription quota this service is
paying for. Simple shared-secret header check, same as interview-qa.

Off by default (`SPEECH_IO_REQUIRE_API_KEY=0`) so local development and the
existing test suite don't need a key configured — but the README and
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
        raise HTTPException(
            status_code=503,
            detail="API key authentication is required but no keys are configured.",
        )

    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header.")

    if not any(hmac.compare_digest(x_api_key, key) for key in valid_keys):
        raise HTTPException(status_code=401, detail="Invalid API key.")
