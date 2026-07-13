"""Follow-up question generation: given a question and the candidate's
answer, produce the natural next question — the same mechanism a real
interviewer uses to probe depth or check an answer against the résumé."""

from __future__ import annotations

import json
import logging

from ..config import get_settings
from ..llm.client import LLMClient, LLMError, get_llm_client
from .prompts import build_followup_prompt
from .schemas import FollowUpRequest, FollowUpResponse

logger = logging.getLogger("interview_qa.followup")


class FollowUpGenerationError(Exception):
    pass


def generate_followup(request: FollowUpRequest, client: LLMClient | None = None) -> FollowUpResponse:
    settings = get_settings()
    client = client or get_llm_client()

    system, user = build_followup_prompt(
        request.resume,
        request.original_question,
        request.candidate_answer,
        request.round,
        request.target_company,
    )

    last_error: Exception | None = None
    for attempt in range(settings.llm_max_retries + 1):
        try:
            raw = client.generate_json(system, user)
            data = json.loads(raw)
            question = data["follow_up_question"]
            if not question or not isinstance(question, str):
                raise ValueError("missing/invalid 'follow_up_question'")
            rationale = data.get("rationale") or ""
            return FollowUpResponse(
                follow_up_question=question.strip(),
                rationale=str(rationale).strip(),
                model_used=client.model_name,
            )
        except (json.JSONDecodeError, KeyError, ValueError, LLMError) as exc:
            last_error = exc
            logger.warning(
                "Follow-up generation attempt %d/%d failed: %s",
                attempt + 1,
                settings.llm_max_retries + 1,
                exc,
            )

    raise FollowUpGenerationError(
        f"Failed to generate a follow-up after {settings.llm_max_retries + 1} attempt(s): {last_error}"
    )
