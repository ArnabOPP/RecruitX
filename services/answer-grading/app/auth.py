"""Inbound API-key authentication for this service's own endpoints.

Unlike interview-qa/speech-io, there's no external provider quota to
protect here — scoring is local compute. This still matters though: an
unauthenticated caller could otherwise hammer /api/v1/grading/score and
exhaust the CPU running the embedding model, degrading the service for
everyone else. Same simple shared-secret header check as the other
services, off by default for local dev/tests.
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
