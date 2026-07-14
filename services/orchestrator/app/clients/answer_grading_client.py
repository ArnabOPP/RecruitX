from __future__ import annotations

import httpx

from ..config import get_settings
from .base import call_json


async def score(question: str, candidate_answer: str, grounding: dict | None, rubric: dict | None = None) -> dict:
    settings = get_settings()
    headers = {"X-API-Key": settings.answer_grading_api_key} if settings.answer_grading_api_key else {}
    payload: dict = {"question": question, "candidate_answer": candidate_answer}
    if grounding:
        payload["grounding"] = grounding
    if rubric:
        payload["rubric"] = rubric
    async with httpx.AsyncClient(
        base_url=settings.answer_grading_base_url, timeout=settings.downstream_timeout_seconds
    ) as client:
        return await call_json(
            client, "answer-grading", "POST", "/api/v1/grading/score", json=payload, headers=headers
        )


async def is_ready() -> bool:
    settings = get_settings()
    try:
        async with httpx.AsyncClient(base_url=settings.answer_grading_base_url, timeout=5.0) as client:
            resp = await client.get("/health/live")
            return resp.status_code == 200
    except httpx.RequestError:
        return False
