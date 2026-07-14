from __future__ import annotations

import httpx

from ..config import get_settings
from .base import call_json


async def parse_resume(file_bytes: bytes, filename: str) -> dict:
    settings = get_settings()
    headers = {"X-API-Key": settings.cv_parser_api_key} if settings.cv_parser_api_key else {}
    async with httpx.AsyncClient(
        base_url=settings.cv_parser_base_url, timeout=settings.downstream_timeout_seconds
    ) as client:
        files = {"file": (filename, file_bytes)}
        return await call_json(client, "cv-parser", "POST", "/api/v1/parse", files=files, headers=headers)


async def is_ready() -> bool:
    settings = get_settings()
    try:
        async with httpx.AsyncClient(base_url=settings.cv_parser_base_url, timeout=5.0) as client:
            resp = await client.get("/health/live")
            return resp.status_code == 200
    except httpx.RequestError:
        return False
