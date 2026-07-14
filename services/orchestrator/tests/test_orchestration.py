"""Tests for the orchestration state machine — round progression,
question/followup sequencing, and error handling — using mocked
downstream clients (in-process async functions, monkeypatched directly)
for speed and determinism. A real end-to-end run against all five actual
services is in test_live_e2e.py."""

from __future__ import annotations

import pytest

from app import orchestration
from app.orchestration import (
    OrchestrationError,
    create_session,
    get_report,
    submit_answer,
    submit_code,
)
from tests.fakes import FakeSessionStore

SAMPLE_PARSED_RESUME = {
    "contact": {"full_name": {"value": "Jordan Lee"}},
    "summary": "A candidate.",
    "skills": [{"name": "Python", "evidenced_in_project": True, "evidenced_in_experience": False}],
    "education": [],
    "experience": [],
    "projects": [],
    "certifications": [],
}


def _question_response(text="Tell me about your project.", category="project_deep_dive"):
    return {
        "questions": [
            {
                "text": text,
                "category": category,
                "grounding": {"kind": "project", "reference": "TaskTracker", "detail": "used Python"},
                "difficulty": "medium",
            }
        ]
    }


@pytest.fixture(autouse=True)
def _patch_clients(monkeypatch):
    async def fake_parse_resume(file_bytes, filename):
        return SAMPLE_PARSED_RESUME

    async def fake_generate_questions(resume_context, round_type, target_company, count):
        return _question_response(text=f"Question for {round_type} round")

    async def fake_generate_followup(resume_context, original_question, candidate_answer, round_type, target_company):
        return {"follow_up_question": "Can you elaborate?", "rationale": "probing depth"}

    async def fake_score(question, candidate_answer, grounding, rubric=None):
        return {"overall_score": 0.75, "rubric_source": "auto_derived_from_grounding", "criteria_scores": [], "explanation": "..."}

    async def fake_transcribe(audio_bytes, filename):
        return {"text": "transcribed answer", "language": None, "model_used": "whisper-large-v3"}

    async def fake_evaluate(language, source_code, test_cases, expected_complexity):
        return {"correctness": {"passed": 1, "total": 1, "pass_rate": 1.0}, "overall_score": 0.9}

    monkeypatch.setattr(orchestration.cv_parser_client, "parse_resume", fake_parse_resume)
    monkeypatch.setattr(orchestration.interview_qa_client, "generate_questions", fake_generate_questions)
    monkeypatch.setattr(orchestration.interview_qa_client, "generate_followup", fake_generate_followup)
    monkeypatch.setattr(orchestration.answer_grading_client, "score", fake_score)
    monkeypatch.setattr(orchestration.speech_io_client, "transcribe", fake_transcribe)
    monkeypatch.setattr(orchestration.code_eval_client, "evaluate", fake_evaluate)


@pytest.mark.asyncio
async def test_create_session_generates_first_question():
    store = FakeSessionStore()
    session = await create_session(store, b"resume bytes", "resume.pdf", "Acme", 2, 1, True)
    assert session["round"] == "personal"
    assert session["status"] == "in_progress"
    assert session["current_question"]["text"] == "Question for personal round"
    assert session["counts"]["personal_asked"] == 1


@pytest.mark.asyncio
async def test_create_session_zero_personal_questions_skips_straight_to_hr():
    store = FakeSessionStore()
    session = await create_session(store, b"resume bytes", "resume.pdf", None, 0, 1, True)
    assert session["round"] == "hr"
    assert session["current_question"]["text"] == "Question for hr round"


@pytest.mark.asyncio
async def test_create_session_zero_everything_goes_straight_to_awaiting_code():
    store = FakeSessionStore()
    session = await create_session(store, b"resume bytes", "resume.pdf", None, 0, 0, True)
    assert session["status"] == "awaiting_code"
    assert session["round"] == "coding"
    assert session["current_question"] is None


@pytest.mark.asyncio
async def test_submit_answer_with_followups_enabled_asks_one_followup():
    store = FakeSessionStore()
    session = await create_session(store, b"resume", "r.pdf", None, 1, 0, True)
    session_id = session["session_id"]

    result = await submit_answer(store, session_id, "My answer.", None, None)

    assert result["score"]["overall_score"] == 0.75
    assert result["next_question"]["text"] == "Can you elaborate?"
    assert result["next_question"]["category"] == "followup"
    assert result["status"] == "in_progress"

    saved = await store.load(session_id)
    assert saved["stage"] == "followup"
    assert len(saved["history"]) == 1


@pytest.mark.asyncio
async def test_submit_answer_to_followup_advances_to_next_primary_question():
    store = FakeSessionStore()
    session = await create_session(store, b"resume", "r.pdf", None, 2, 0, True)
    session_id = session["session_id"]

    await submit_answer(store, session_id, "First answer.", None, None)  # -> followup
    result = await submit_answer(store, session_id, "Answer to followup.", None, None)  # -> next primary

    assert result["next_question"]["text"] == "Question for personal round"
    saved = await store.load(session_id)
    assert saved["stage"] == "primary"
    assert saved["counts"]["personal_asked"] == 2
    assert len(saved["history"]) == 2


@pytest.mark.asyncio
async def test_submit_answer_without_followups_always_advances_directly():
    store = FakeSessionStore()
    session = await create_session(store, b"resume", "r.pdf", None, 2, 0, False)
    session_id = session["session_id"]

    result = await submit_answer(store, session_id, "My answer.", None, None)

    assert result["next_question"]["text"] == "Question for personal round"
    saved = await store.load(session_id)
    assert saved["counts"]["personal_asked"] == 2


@pytest.mark.asyncio
async def test_submit_answer_transcribes_audio():
    store = FakeSessionStore()
    session = await create_session(store, b"resume", "r.pdf", None, 1, 0, False)
    session_id = session["session_id"]

    result = await submit_answer(store, session_id, None, b"fake audio bytes", "answer.wav")

    saved = await store.load(session_id)
    assert saved["history"][0]["answer"] == "transcribed answer"
    assert result["score"]["overall_score"] == 0.75


@pytest.mark.asyncio
async def test_submit_answer_empty_text_raises():
    store = FakeSessionStore()
    session = await create_session(store, b"resume", "r.pdf", None, 1, 0, False)
    with pytest.raises(OrchestrationError):
        await submit_answer(store, session["session_id"], "   ", None, None)


@pytest.mark.asyncio
async def test_submit_answer_on_completed_session_raises():
    store = FakeSessionStore()
    session = await create_session(store, b"resume", "r.pdf", None, 0, 0, False)  # -> awaiting_code
    with pytest.raises(OrchestrationError):
        await submit_answer(store, session["session_id"], "answer", None, None)


@pytest.mark.asyncio
async def test_round_progresses_personal_to_hr_to_coding():
    store = FakeSessionStore()
    session = await create_session(store, b"resume", "r.pdf", None, 1, 1, False)
    session_id = session["session_id"]
    assert session["round"] == "personal"

    result = await submit_answer(store, session_id, "answer 1", None, None)
    assert result["round"] == "hr"
    assert result["next_question"]["text"] == "Question for hr round"

    result2 = await submit_answer(store, session_id, "answer 2", None, None)
    assert result2["status"] == "awaiting_code"
    assert result2["next_question"] is None


@pytest.mark.asyncio
async def test_submit_code_completes_session():
    store = FakeSessionStore()
    session = await create_session(store, b"resume", "r.pdf", None, 0, 0, False)  # -> awaiting_code
    session_id = session["session_id"]

    result = await submit_code(store, session_id, "python", "print(1)", [{"expected_output": "1"}], None)

    assert result["status"] == "completed"
    assert result["result"]["overall_score"] == 0.9


@pytest.mark.asyncio
async def test_submit_code_before_coding_round_raises():
    store = FakeSessionStore()
    session = await create_session(store, b"resume", "r.pdf", None, 1, 0, False)  # -> in_progress, personal
    with pytest.raises(OrchestrationError):
        await submit_code(store, session["session_id"], "python", "print(1)", [{"expected_output": "1"}], None)


@pytest.mark.asyncio
async def test_get_report_aggregates_scores():
    store = FakeSessionStore()
    session = await create_session(store, b"resume", "r.pdf", None, 2, 0, False)
    session_id = session["session_id"]
    await submit_answer(store, session_id, "answer 1", None, None)
    await submit_answer(store, session_id, "answer 2", None, None)

    report = await get_report(store, session_id)
    assert report["overall_average_score"] == 0.75
    assert len(report["history"]) == 2
