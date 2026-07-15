"""Client for the proctoring service. The orchestrator's own session_id is
reused as the proctoring service's session_id — there is no separate
mapping to maintain, and it keeps the final report's proctoring summary
trivially joinable to the rest of the session."""

from __future__ import annotations

import httpx

from ..clients.base import DownstreamServiceError, call_json
from ..config import get_settings


def _headers() -> dict:
    settings = get_settings()
    return {"X-API-Key": settings.proctoring_api_key} if settings.proctoring_api_key else {}


async def submit_snapshot(session_id: str, filename: str, image_bytes: bytes, content_type: str) -> dict:
    settings = get_settings()
    async with httpx.AsyncClient(
        base_url=settings.proctoring_base_url, timeout=settings.downstream_timeout_seconds
    ) as client:
        files = {"file": (filename, image_bytes, content_type)}
        return await call_json(
            client, "proctoring", "POST", f"/api/v1/proctoring/{session_id}/snapshot",
            files=files, headers=_headers(),
        )


async def get_summary(session_id: str) -> dict | None:
    """Returns None (rather than raising) when no proctoring snapshots
    were ever submitted for this session — proctoring is optional per
    session, so a missing summary is a normal outcome for the final
    report, not a failure."""
    settings = get_settings()
    async with httpx.AsyncClient(
        base_url=settings.proctoring_base_url, timeout=settings.downstream_timeout_seconds
    ) as client:
        try:
            return await call_json(
                client, "proctoring", "GET", f"/api/v1/proctoring/{session_id}/summary", headers=_headers()
            )
        except DownstreamServiceError as exc:
            if exc.status_code == 404:
                return None
            raise


async def is_ready() -> bool:
    settings = get_settings()
    try:
        async with httpx.AsyncClient(base_url=settings.proctoring_base_url, timeout=5.0) as client:
            resp = await client.get("/health/live")
            return resp.status_code == 200
    except httpx.RequestError:
        return False
