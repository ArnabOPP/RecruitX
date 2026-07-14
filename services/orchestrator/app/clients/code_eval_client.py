from __future__ import annotations

import httpx

from ..config import get_settings
from .base import call_json


async def evaluate(
    language: str, source_code: str, test_cases: list[dict], expected_complexity: str | None
) -> dict:
    settings = get_settings()
    headers = {"X-API-Key": settings.code_eval_api_key} if settings.code_eval_api_key else {}
    payload: dict = {"language": language, "source_code": source_code, "test_cases": test_cases}
    if expected_complexity:
        payload["expected_complexity"] = expected_complexity
    async with httpx.AsyncClient(
        base_url=settings.code_eval_base_url, timeout=settings.downstream_timeout_seconds
    ) as client:
        return await call_json(client, "code-eval", "POST", "/api/v1/code/evaluate", json=payload, headers=headers)


async def is_ready() -> bool:
    settings = get_settings()
    try:
        async with httpx.AsyncClient(base_url=settings.code_eval_base_url, timeout=5.0) as client:
            resp = await client.get("/health/live")
            return resp.status_code == 200
    except httpx.RequestError:
        return False
