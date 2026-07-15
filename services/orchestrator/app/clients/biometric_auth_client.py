"""Client for the biometric-auth service: identity verification and
liveness at session creation ("login"). Enrollment is also proxied so
there's a single API surface for the frontend, but is not itself gated —
enrollment is a one-time setup step, not a per-session check."""

from __future__ import annotations

import httpx

from ..config import get_settings
from .base import call_json


def _headers() -> dict:
    settings = get_settings()
    return {"X-API-Key": settings.biometric_auth_api_key} if settings.biometric_auth_api_key else {}


async def enroll(candidate_id: str, files: list[tuple[str, bytes, str]]) -> dict:
    settings = get_settings()
    async with httpx.AsyncClient(
        base_url=settings.biometric_auth_base_url, timeout=settings.downstream_timeout_seconds
    ) as client:
        upload = [("files", (name, data, content_type)) for name, data, content_type in files]
        return await call_json(
            client, "biometric-auth", "POST", "/api/v1/biometric/enroll",
            params={"candidate_id": candidate_id}, files=upload, headers=_headers(),
        )


async def verify(candidate_id: str, files: list[tuple[str, bytes, str]]) -> dict:
    settings = get_settings()
    async with httpx.AsyncClient(
        base_url=settings.biometric_auth_base_url, timeout=settings.downstream_timeout_seconds
    ) as client:
        upload = [("files", (name, data, content_type)) for name, data, content_type in files]
        return await call_json(
            client, "biometric-auth", "POST", "/api/v1/biometric/verify",
            params={"candidate_id": candidate_id}, files=upload, headers=_headers(),
        )


async def check_liveness(files: list[tuple[str, bytes, str]]) -> dict:
    settings = get_settings()
    async with httpx.AsyncClient(
        base_url=settings.biometric_auth_base_url, timeout=settings.downstream_timeout_seconds
    ) as client:
        upload = [("files", (name, data, content_type)) for name, data, content_type in files]
        return await call_json(
            client, "biometric-auth", "POST", "/api/v1/biometric/liveness", files=upload, headers=_headers()
        )


async def is_ready() -> bool:
    settings = get_settings()
    try:
        async with httpx.AsyncClient(base_url=settings.biometric_auth_base_url, timeout=5.0) as client:
            resp = await client.get("/health/live")
            return resp.status_code == 200
    except httpx.RequestError:
        return False
