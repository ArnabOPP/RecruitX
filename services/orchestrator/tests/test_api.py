"""API-level tests: HTTP status codes, error schema, and endpoint wiring.

Both the session store and the five downstream clients are mocked at the
points main.py/orchestration.py actually use them, for fast, deterministic
tests. The real, fully-wired flow against all five actual running
services is proven separately in test_live_e2e.py.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import orchestration
from tests.fakes import FakeSessionStore

SAMPLE_PARSED_RESUME = {
    "contact": {"full_name": {"value": "Jordan Lee"}},
    "summary": "A candidate.",
    "skills": [],
    "education": [],
    "experience": [],
    "projects": [],
    "certifications": [],
}


def _question_response(round_type: str):
    return {
        "questions": [
            {
                "text": f"Question for {round_type} round",
                "category": "project_deep_dive",
                "grounding": {"kind": "project", "reference": "TaskTracker", "detail": "used Python"},
                "difficulty": "medium",
            }
        ]
    }


@pytest.fixture
def client(monkeypatch):
    fake_store = FakeSessionStore()
    monkeypatch.setattr("app.main.get_session_store", lambda: fake_store)

    async def fake_parse_resume(file_bytes, filename):
        return SAMPLE_PARSED_RESUME

    async def fake_generate_questions(resume_context, round_type, target_company, count):
        return _question_response(round_type)

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

    from app.main import app

    with TestClient(app) as c:
        yield c


def test_liveness(client):
    resp = client.get("/health/live")
    assert resp.status_code == 200


def test_readiness(client):
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_capabilities(client):
    resp = client.get("/api/v1/capabilities")
    assert resp.status_code == 200
    body = resp.json()
    assert "downstream_services" in body
    assert set(body["downstream_services"]) == {"cv_parser", "interview_qa", "speech_io", "answer_grading", "code_eval"}


def test_metrics_exposed(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200


def test_create_session_and_answer_flow(client):
    create_resp = client.post(
        "/api/v1/sessions",
        files={"file": ("resume.pdf", b"fake pdf bytes", "application/pdf")},
        params={"personal_question_count": 1, "hr_question_count": 0, "enable_followups": False},
    )
    assert create_resp.status_code == 200
    body = create_resp.json()
    session_id = body["session_id"]
    assert body["round"] == "personal"
    assert body["question"]["text"] == "Question for personal round"

    answer_resp = client.post(f"/api/v1/sessions/{session_id}/answer", json={"answer_text": "My answer."})
    assert answer_resp.status_code == 200
    answer_body = answer_resp.json()
    assert answer_body["score"]["overall_score"] == 0.75
    assert answer_body["status"] == "awaiting_code"


def test_create_session_missing_file_is_422(client):
    resp = client.post("/api/v1/sessions", params={"personal_question_count": 1})
    assert resp.status_code == 422


def test_create_session_empty_file_is_400(client):
    resp = client.post("/api/v1/sessions", files={"file": ("resume.pdf", b"", "application/pdf")})
    assert resp.status_code == 400


def test_answer_on_unknown_session_is_404(client):
    resp = client.post("/api/v1/sessions/does-not-exist/answer", json={"answer_text": "hi"})
    assert resp.status_code == 404


def test_answer_empty_text_is_422(client):
    create_resp = client.post(
        "/api/v1/sessions",
        files={"file": ("resume.pdf", b"fake pdf bytes", "application/pdf")},
        params={"personal_question_count": 1, "hr_question_count": 0, "enable_followups": False},
    )
    session_id = create_resp.json()["session_id"]
    resp = client.post(f"/api/v1/sessions/{session_id}/answer", json={"answer_text": "   "})
    assert resp.status_code == 422


def test_answer_audio_transcribes_and_scores(client):
    create_resp = client.post(
        "/api/v1/sessions",
        files={"file": ("resume.pdf", b"fake pdf bytes", "application/pdf")},
        params={"personal_question_count": 1, "hr_question_count": 0, "enable_followups": False},
    )
    session_id = create_resp.json()["session_id"]

    resp = client.post(
        f"/api/v1/sessions/{session_id}/answer/audio", files={"file": ("answer.wav", b"fake audio", "audio/wav")}
    )
    assert resp.status_code == 200
    assert resp.json()["score"]["overall_score"] == 0.75


def test_full_session_reaches_coding_and_report(client):
    create_resp = client.post(
        "/api/v1/sessions",
        files={"file": ("resume.pdf", b"fake pdf bytes", "application/pdf")},
        params={"personal_question_count": 1, "hr_question_count": 0, "enable_followups": False},
    )
    session_id = create_resp.json()["session_id"]

    answer_resp = client.post(f"/api/v1/sessions/{session_id}/answer", json={"answer_text": "answer"})
    assert answer_resp.json()["status"] == "awaiting_code"

    code_resp = client.post(
        f"/api/v1/sessions/{session_id}/code",
        json={"language": "python", "source_code": "print(1)", "test_cases": [{"expected_output": "1"}]},
    )
    assert code_resp.status_code == 200
    assert code_resp.json()["status"] == "completed"

    report_resp = client.get(f"/api/v1/sessions/{session_id}/report")
    assert report_resp.status_code == 200
    report = report_resp.json()
    assert report["status"] == "completed"
    assert len(report["history"]) == 2  # one text answer + one code submission


def test_code_before_awaiting_code_is_409(client):
    create_resp = client.post(
        "/api/v1/sessions",
        files={"file": ("resume.pdf", b"fake pdf bytes", "application/pdf")},
        params={"personal_question_count": 1, "hr_question_count": 0, "enable_followups": False},
    )
    session_id = create_resp.json()["session_id"]

    resp = client.post(
        f"/api/v1/sessions/{session_id}/code",
        json={"language": "python", "source_code": "print(1)", "test_cases": [{"expected_output": "1"}]},
    )
    assert resp.status_code == 409


def test_error_responses_never_leak_tracebacks(client):
    resp = client.post("/api/v1/sessions/does-not-exist/answer", json={"answer_text": "hi"})
    body = resp.json()
    assert "Traceback" not in body["detail"]
    assert 'File "' not in body["detail"]


def test_production_without_required_auth_logs_security_warning(caplog):
    from app.config import Settings
    from app.main import _warn_if_unprotected_in_production

    settings = Settings(environment="production", require_api_key=False)

    with caplog.at_level("WARNING"):
        _warn_if_unprotected_in_production(settings)

    assert any(
        "environment=production" in r.message and "ORCHESTRATOR_REQUIRE_API_KEY" in r.message for r in caplog.records
    )


def test_production_with_required_auth_logs_no_warning(caplog):
    from app.config import Settings
    from app.main import _warn_if_unprotected_in_production

    settings = Settings(environment="production", require_api_key=True, api_keys="some-key")

    with caplog.at_level("WARNING"):
        _warn_if_unprotected_in_production(settings)

    assert not any("environment=production" in r.message for r in caplog.records)


def test_downstream_failure_returns_502(monkeypatch):
    fake_store = FakeSessionStore()
    monkeypatch.setattr("app.main.get_session_store", lambda: fake_store)

    from app.clients.base import DownstreamServiceError

    async def failing_parse_resume(file_bytes, filename):
        raise DownstreamServiceError("cv-parser", "HTTP 500: internal error", status_code=500)

    monkeypatch.setattr(orchestration.cv_parser_client, "parse_resume", failing_parse_resume)

    from app.main import app

    with TestClient(app) as c:
        resp = c.post("/api/v1/sessions", files={"file": ("resume.pdf", b"fake pdf bytes", "application/pdf")})
    assert resp.status_code == 502
    body = resp.json()
    assert body["error"] == "downstream_error"
    assert "cv-parser" in body["detail"]
