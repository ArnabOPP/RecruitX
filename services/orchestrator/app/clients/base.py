"""Shared HTTP-calling machinery for the five downstream service clients.

Every call raises DownstreamServiceError on failure, carrying which
service failed — the orchestration layer propagates this as a clear
"interview-qa is down" error rather than a generic 500, per the
resilience plan (each downstream service already has its own timeouts
and rate limiting; this layer's job is just attributing failures, not
retrying or circuit-breaking, which would be over-engineering for five
services this size).
"""

from __future__ import annotations

import httpx


class DownstreamServiceError(Exception):
    def __init__(self, service: str, message: str, status_code: int | None = None) -> None:
        super().__init__(f"{service}: {message}")
        self.service = service
        self.status_code = status_code


async def call_json(client: httpx.AsyncClient, service: str, method: str, url: str, **kwargs) -> dict:
    try:
        resp = await client.request(method, url, **kwargs)
    except httpx.RequestError as exc:
        raise DownstreamServiceError(service, f"request failed: {exc}") from exc
    if resp.status_code >= 400:
        raise DownstreamServiceError(
            service, f"HTTP {resp.status_code}: {resp.text[:500]}", status_code=resp.status_code
        )
    return resp.json()


async def call_bytes(client: httpx.AsyncClient, service: str, method: str, url: str, **kwargs) -> bytes:
    try:
        resp = await client.request(method, url, **kwargs)
    except httpx.RequestError as exc:
        raise DownstreamServiceError(service, f"request failed: {exc}") from exc
    if resp.status_code >= 400:
        raise DownstreamServiceError(
            service, f"HTTP {resp.status_code}: {resp.text[:500]}", status_code=resp.status_code
        )
    return resp.content
