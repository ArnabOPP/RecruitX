"""Core question-generation orchestration: prompt -> LLM -> validated,
typed output.

The LLM's JSON mode guarantees syntactically valid JSON, not that it matches
our schema — a field might be missing, an enum value might be something the
model invented. Rather than fail the whole request over one bad item,
`_parse_question` validates permissively (falls back to sane defaults for
recoverable fields) and only drops a question outright if its actual text is
missing, surfacing a warning either way so the caller can see what happened.
"""

from __future__ import annotations

import json
import logging

from ..config import get_settings
from ..llm.client import (
    LLMAuthenticationError,
    LLMClient,
    LLMError,
    generate_json_with_backoff,
    get_llm_client,
)
from .prompts import build_generation_prompt
from .schemas import (
    Difficulty,
    GeneratedQuestion,
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
    GroundingEvidence,
    QuestionCategory,
    RoundType,
)

logger = logging.getLogger("interview_qa.generator")


class QuestionGenerationError(Exception):
    """Raised when no valid questions could be produced after all retries."""


def generate_questions(
    request: GenerateQuestionsRequest, client: LLMClient | None = None
) -> GenerateQuestionsResponse:
    settings = get_settings()
    client = client or get_llm_client()
    count = min(max(1, request.count), settings.max_questions_per_request)

    system, user = build_generation_prompt(request.resume, request.round, request.target_company, count)

    last_error: Exception | None = None
    for attempt in range(settings.llm_max_retries + 1):
        try:
            # generate_json_with_backoff already retries internally on rate
            # limits (with backoff) and transient provider errors — this
            # outer loop's job is retrying when the call *succeeded* but
            # produced unusable content (bad JSON, wrong schema).
            raw = generate_json_with_backoff(client, system, user, max_retries=settings.llm_max_retries)
            data = json.loads(raw)
            questions_raw = data["questions"]
            if not isinstance(questions_raw, list):
                raise ValueError("'questions' was not a list")
        except LLMAuthenticationError as exc:
            # Bad credentials won't fix themselves between attempts — fail
            # immediately instead of burning the whole retry budget on
            # something retrying can never resolve.
            raise QuestionGenerationError(f"LLM provider rejected our credentials: {exc}") from exc
        except (json.JSONDecodeError, KeyError, ValueError, LLMError) as exc:
            last_error = exc
            logger.warning("Question generation attempt %d/%d failed: %s", attempt + 1, settings.llm_max_retries + 1, exc)
            continue

        questions: list[GeneratedQuestion] = []
        warnings: list[str] = []
        for item in questions_raw[:count]:
            try:
                questions.append(_parse_question(item, request.round))
            except (KeyError, TypeError, ValueError) as exc:
                warnings.append(f"Skipped a malformed question from the model: {exc}")

        if questions:
            return GenerateQuestionsResponse(questions=questions, model_used=client.model_name, warnings=warnings)
        last_error = QuestionGenerationError("Model returned no valid questions.")
        logger.warning("Question generation attempt %d/%d produced zero valid questions.", attempt + 1, settings.llm_max_retries + 1)

    raise QuestionGenerationError(
        f"Failed to generate questions after {settings.llm_max_retries + 1} attempt(s): {last_error}"
    )


def _parse_question(raw: dict, round_type: RoundType) -> GeneratedQuestion:
    text = raw.get("text")
    if not text or not isinstance(text, str):
        raise ValueError("question is missing 'text'")

    try:
        category = QuestionCategory(raw.get("category", ""))
    except ValueError:
        category = QuestionCategory.PROJECT_DEEP_DIVE

    try:
        difficulty = Difficulty(raw.get("difficulty", ""))
    except ValueError:
        difficulty = Difficulty.MEDIUM

    grounding = None
    g = raw.get("grounding")
    if isinstance(g, dict) and g.get("reference"):
        grounding = GroundingEvidence(
            kind=g.get("kind") or "other",
            reference=str(g["reference"]),
            detail=g.get("detail"),
        )

    return GeneratedQuestion(
        text=text.strip(),
        category=category,
        round=round_type,
        grounding=grounding,
        difficulty=difficulty,
    )
