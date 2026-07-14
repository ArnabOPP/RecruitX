from __future__ import annotations

import httpx

from ..config import get_settings
from .base import call_json


async def generate_questions(resume_context: dict, round_type: str, target_company: str | None, count: int) -> dict:
    settings = get_settings()
    headers = {"X-API-Key": settings.interview_qa_api_key} if settings.interview_qa_api_key else {}
    payload: dict = {"resume": resume_context, "round": round_type, "count": count}
    if target_company:
        payload["target_company"] = target_company
    async with httpx.AsyncClient(
        base_url=settings.interview_qa_base_url, timeout=settings.downstream_timeout_seconds
    ) as client:
        return await call_json(
            client, "interview-qa", "POST", "/api/v1/questions/generate", json=payload, headers=headers
        )


async def generate_followup(
    resume_context: dict,
    original_question: str,
    candidate_answer: str,
    round_type: str,
    target_company: str | None,
) -> dict:
    settings = get_settings()
    headers = {"X-API-Key": settings.interview_qa_api_key} if settings.interview_qa_api_key else {}
    payload: dict = {
        "resume": resume_context,
        "original_question": original_question,
        "candidate_answer": candidate_answer,
        "round": round_type,
    }
    if target_company:
        payload["target_company"] = target_company
    async with httpx.AsyncClient(
        base_url=settings.interview_qa_base_url, timeout=settings.downstream_timeout_seconds
    ) as client:
        return await call_json(
            client, "interview-qa", "POST", "/api/v1/questions/followup", json=payload, headers=headers
        )


async def is_ready() -> bool:
    settings = get_settings()
    try:
        async with httpx.AsyncClient(base_url=settings.interview_qa_base_url, timeout=5.0) as client:
            resp = await client.get("/health/live")
            return resp.status_code == 200
    except httpx.RequestError:
        return False
