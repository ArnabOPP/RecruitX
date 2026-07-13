"""API-level tests: HTTP status codes, error schema, and endpoint wiring.

The LLM client is mocked at the point each module actually imported it
(`app.qa.generator.get_llm_client` / `app.qa.followup.get_llm_client`) —
patching only `app.llm.client.get_llm_client` wouldn't affect those already-
bound references. This keeps these tests fast and network-free; the real
API is exercised separately in test_live_groq.py.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from tests.fakes import FakeLLMClient

GOOD_GENERATE_RESPONSE = json.dumps(
    {
        "questions": [
            {
                "text": "How did you use Python in your project?",
                "category": "project_deep_dive",
                "grounding": {"kind": "project", "reference": "Widget", "detail": "used Python"},
                "difficulty": "medium",
            }
        ]
    }
)

GOOD_FOLLOWUP_RESPONSE = json.dumps(
    {
        "follow_up_question": "Can you elaborate on that?",
        "rationale": "probing depth",
    }
)


@pytest.fixture
def client(monkeypatch):
    fake = FakeLLMClient(responses=[GOOD_GENERATE_RESPONSE, GOOD_FOLLOWUP_RESPONSE] * 10)
    monkeypatch.setattr("app.qa.generator.get_llm_client", lambda: fake)
    monkeypatch.setattr("app.qa.followup.get_llm_client", lambda: fake)
    # main.py's lifespan does its own `from .llm.client import get_llm_client`,
    # a separately-bound name from the two above — patching only those two
    # left the readiness check trying to construct a *real* GroqClient here,
    # which happened to "work" locally only because a real .env file with a
    # real key sits in this directory. That's not true in a clean CI
    # checkout, where there's no .env at all — this was a real bug caught by
    # actually running CI in a clean environment, not by local testing.
    monkeypatch.setattr("app.main.get_llm_client", lambda: fake)

    from app.main import app

    with TestClient(app) as c:
        yield c


def test_liveness(client):
    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readiness(client):
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_capabilities(client):
    resp = client.get("/api/v1/capabilities")
    assert resp.status_code == 200
    body = resp.json()
    assert body["llm_provider"] == "groq"
    assert body["llm_ready"] is True


def test_metrics_exposed(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200


def test_request_id_header_present(client):
    resp = client.get("/health/live")
    assert "x-request-id" in resp.headers


def test_generate_questions_success(client):
    resp = client.post(
        "/api/v1/questions/generate",
        json={
            "resume": {"full_name": "Jordan Lee", "skills": [{"name": "Python"}]},
            "target_company": "Acme",
            "round": "personal",
            "count": 1,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["questions"]) == 1
    assert body["questions"][0]["text"]


def test_generate_questions_invalid_count(client):
    resp = client.post(
        "/api/v1/questions/generate",
        json={"resume": {}, "count": 999},
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "request_error"
    assert "request_id" in body


def test_generate_questions_missing_resume_field(client):
    resp = client.post("/api/v1/questions/generate", json={})
    assert resp.status_code == 422  # FastAPI's own request validation


def test_followup_success(client):
    resp = client.post(
        "/api/v1/questions/followup",
        json={
            "resume": {"full_name": "Jordan Lee"},
            "original_question": "Tell me about your project.",
            "candidate_answer": "I built a web app.",
            "round": "personal",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["follow_up_question"]
    assert body["rationale"]


def test_generate_questions_returns_502_on_persistent_llm_failure(monkeypatch):
    from tests.fakes import AlwaysFailingLLMClient

    monkeypatch.setattr("app.qa.generator.get_llm_client", lambda: AlwaysFailingLLMClient())
    monkeypatch.setattr("app.main.get_llm_client", lambda: AlwaysFailingLLMClient())
    from app.main import app

    with TestClient(app) as c:
        resp = c.post(
            "/api/v1/questions/generate",
            json={"resume": {"full_name": "Jordan Lee"}, "count": 1},
        )
    assert resp.status_code == 502
    assert resp.json()["error"] == "request_error"


def test_error_responses_never_leak_tracebacks(client):
    resp = client.post("/api/v1/questions/generate", json={"resume": {}, "count": -1})
    body = resp.json()
    assert "Traceback" not in body["detail"]
    assert 'File "' not in body["detail"]


def test_production_without_required_auth_logs_security_warning(caplog):
    """environment=production with auth still off must be loud in logs —
    silent misconfiguration here means the Groq quota is wide open."""
    from app.config import Settings
    from app.main import _warn_if_unprotected_in_production

    settings = Settings(environment="production", require_api_key=False)

    with caplog.at_level("WARNING"):
        _warn_if_unprotected_in_production(settings)

    assert any("environment=production" in r.message and "INTERVIEW_QA_REQUIRE_API_KEY" in r.message for r in caplog.records)


def test_production_with_required_auth_logs_no_warning(caplog):
    from app.config import Settings
    from app.main import _warn_if_unprotected_in_production

    settings = Settings(environment="production", require_api_key=True, api_keys="some-key")

    with caplog.at_level("WARNING"):
        _warn_if_unprotected_in_production(settings)

    assert not any("environment=production" in r.message for r in caplog.records)


def test_readiness_reflects_invalid_credentials_not_just_missing(monkeypatch):
    """A configured-but-wrong/revoked key must report not-ready, distinct
    from the "no key at all" case — this is what startup key validation
    (config.validate_key_on_startup) actually buys over just checking the
    key string is non-empty."""
    from tests.fakes import AlwaysUnauthenticatedLLMClient

    monkeypatch.setattr("app.main.get_llm_client", lambda: AlwaysUnauthenticatedLLMClient())
    from app.main import app

    with TestClient(app) as c:
        resp = c.get("/health/ready")
        assert resp.status_code == 503
        assert resp.json()["status"] == "not_ready"

        caps = c.get("/api/v1/capabilities")
        assert caps.json()["llm_ready"] is False
