"""Unit tests for follow-up question generation against a fake LLM client."""

from __future__ import annotations

import json

import pytest

from app.qa.followup import FollowUpGenerationError, generate_followup
from app.qa.schemas import FollowUpRequest, ResumeContext, RoundType
from tests.fakes import AlwaysFailingLLMClient, FakeLLMClient

GOOD_RESPONSE = json.dumps(
    {
        "follow_up_question": "You mentioned Kubernetes — can you elaborate on how you set that up?",
        "rationale": "probing a technology mentioned in the answer but not on the resume",
    }
)


def _sample_request() -> FollowUpRequest:
    return FollowUpRequest(
        resume=ResumeContext(full_name="Jordan Lee"),
        original_question="How did you deploy your project?",
        candidate_answer="I used Docker, and also set up a Kubernetes cluster for scaling.",
        round=RoundType.PERSONAL,
    )


def test_followup_happy_path():
    client = FakeLLMClient(responses=[GOOD_RESPONSE])
    result = generate_followup(_sample_request(), client=client)

    assert "Kubernetes" in result.follow_up_question
    assert result.rationale
    assert result.model_used == "fake-model"


def test_followup_retries_on_malformed_json():
    client = FakeLLMClient(responses=["not valid json", GOOD_RESPONSE])
    result = generate_followup(_sample_request(), client=client)
    assert result.follow_up_question
    assert len(client.calls) == 2


def test_followup_raises_after_exhausting_retries():
    client = FakeLLMClient(responses=["broken", "still broken", "also broken"])
    with pytest.raises(FollowUpGenerationError):
        generate_followup(_sample_request(), client=client)


def test_followup_fails_cleanly_on_provider_error():
    with pytest.raises(FollowUpGenerationError):
        generate_followup(_sample_request(), client=AlwaysFailingLLMClient())


def test_followup_raises_on_missing_question_field():
    missing_field = json.dumps({"rationale": "no question given"})
    client = FakeLLMClient(responses=[missing_field, missing_field, missing_field])
    with pytest.raises(FollowUpGenerationError):
        generate_followup(_sample_request(), client=client)
