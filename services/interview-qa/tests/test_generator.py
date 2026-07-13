"""Unit tests for question generation against a fake LLM client — fast,
deterministic, no network. See test_live_groq.py for real-API coverage."""

from __future__ import annotations

import json

import pytest

from app.qa.generator import QuestionGenerationError, generate_questions
from app.qa.schemas import GenerateQuestionsRequest, ResumeContext, RoundType, SkillContext
from tests.fakes import AlwaysFailingLLMClient, FakeLLMClient

GOOD_RESPONSE = json.dumps(
    {
        "questions": [
            {
                "text": "How did you use Python in your Recommender project?",
                "category": "project_deep_dive",
                "grounding": {"kind": "project", "reference": "Recommender", "detail": "used Python"},
                "difficulty": "medium",
            },
            {
                "text": "You listed Rust but it's not used anywhere — tell me about a time you used it.",
                "category": "resume_gap_probe",
                "grounding": {"kind": "skill", "reference": "Rust", "detail": "listed only"},
                "difficulty": "easy",
            },
        ]
    }
)


def _sample_request(count: int = 2) -> GenerateQuestionsRequest:
    resume = ResumeContext(
        full_name="Jordan Lee",
        skills=[
            SkillContext(name="Python", evidenced_in_project=True),
            SkillContext(name="Rust"),
        ],
    )
    return GenerateQuestionsRequest(resume=resume, target_company="Acme", round=RoundType.PERSONAL, count=count)


def test_generate_questions_happy_path():
    client = FakeLLMClient(responses=[GOOD_RESPONSE])
    result = generate_questions(_sample_request(), client=client)

    assert len(result.questions) == 2
    assert result.model_used == "fake-model"
    assert result.warnings == []
    assert result.questions[0].grounding is not None
    assert result.questions[0].grounding.reference == "Recommender"
    assert result.questions[1].category.value == "resume_gap_probe"


def test_generate_questions_retries_on_malformed_json():
    client = FakeLLMClient(responses=["not json at all", GOOD_RESPONSE])
    result = generate_questions(_sample_request(), client=client)

    assert len(result.questions) == 2
    assert len(client.calls) == 2  # confirms it actually retried, not just parsed once


def test_generate_questions_retries_on_llm_error():
    client = FakeLLMClient(responses=[])  # first call raises (no scripted response)
    with pytest.raises(QuestionGenerationError):
        generate_questions(_sample_request(), client=client)


def test_generate_questions_exhausts_retries_and_raises():
    client = FakeLLMClient(responses=["broken", "still broken", "also broken"])
    with pytest.raises(QuestionGenerationError):
        generate_questions(_sample_request(), client=client)


def test_generate_questions_fails_cleanly_on_provider_error():
    with pytest.raises(QuestionGenerationError):
        generate_questions(_sample_request(), client=AlwaysFailingLLMClient())


def test_generate_questions_skips_malformed_individual_questions_but_keeps_good_ones():
    mixed = json.dumps(
        {
            "questions": [
                {"text": "A valid question about Python."},  # missing category/grounding -> defaults
                {"category": "project_deep_dive"},  # missing 'text' entirely -> dropped, not crashed
            ]
        }
    )
    client = FakeLLMClient(responses=[mixed])
    result = generate_questions(_sample_request(), client=client)

    assert len(result.questions) == 1
    assert result.questions[0].text == "A valid question about Python."
    assert len(result.warnings) == 1


def test_generate_questions_ignores_unknown_enum_values_with_sane_defaults():
    raw = json.dumps(
        {
            "questions": [
                {
                    "text": "Question with an invented category",
                    "category": "made_up_category_the_model_invented",
                    "difficulty": "extremely_hard",
                }
            ]
        }
    )
    client = FakeLLMClient(responses=[raw])
    result = generate_questions(_sample_request(count=1), client=client)

    assert len(result.questions) == 1
    assert result.questions[0].category.value == "project_deep_dive"  # fallback default
    assert result.questions[0].difficulty.value == "medium"  # fallback default


def test_count_is_capped_at_max_questions_per_request():
    many_questions = json.dumps(
        {
            "questions": [
                {"text": f"Question {i}", "category": "project_deep_dive"} for i in range(20)
            ]
        }
    )
    client = FakeLLMClient(responses=[many_questions])
    request = _sample_request(count=999)  # will be clamped by the endpoint layer normally;
    # generate_questions itself also clamps against settings.max_questions_per_request.
    result = generate_questions(request, client=client)
    assert len(result.questions) <= 10
