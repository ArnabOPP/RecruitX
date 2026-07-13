"""API-level tests: HTTP status codes, error schema, and endpoint wiring.

The semantic scorer is mocked at the point main.py actually imports it
(app.main.get_semantic_scorer) for fast, deterministic tests — the real
model's behavior through the full HTTP stack is proven separately by
test_full_stack_scoring_with_real_model below, which doesn't mock anything.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.fakes import AlwaysFailingSemanticScorer, FakeSemanticScorer


@pytest.fixture
def client(monkeypatch):
    fake = FakeSemanticScorer(default=0.5)
    monkeypatch.setattr("app.main.get_semantic_scorer", lambda: fake)

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
    assert body["ready"] is True
    assert "jaccard_weight" in body
    assert "semantic_weight" in body


def test_metrics_exposed(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200


def test_request_id_header_present(client):
    resp = client.get("/health/live")
    assert "x-request-id" in resp.headers


def test_score_success_with_grounding(client):
    resp = client.post(
        "/api/v1/grading/score",
        json={
            "question": "How did you optimize your database?",
            "candidate_answer": "I added indexes to speed up slow queries.",
            "grounding": {"kind": "project", "reference": "TaskTracker", "detail": "PostgreSQL optimization"},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["rubric_source"] == "auto_derived_from_grounding"
    assert len(body["criteria_scores"]) == 1
    assert "overall_score" in body
    assert "explanation" in body


def test_score_success_with_explicit_rubric(client):
    resp = client.post(
        "/api/v1/grading/score",
        json={
            "question": "Tell me about your project.",
            "candidate_answer": "It uses FastAPI and PostgreSQL.",
            "rubric": {
                "criteria": [
                    {"description": "mentions the backend framework", "weight": 2.0, "expected_keywords": ["fastapi"]},
                    {"description": "mentions the database", "weight": 1.0, "expected_keywords": ["postgresql"]},
                ]
            },
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["rubric_source"] == "provided"
    assert len(body["criteria_scores"]) == 2


def test_score_success_with_no_rubric_or_grounding(client):
    resp = client.post(
        "/api/v1/grading/score",
        json={"question": "What is your favorite language?", "candidate_answer": "Python, definitely."},
    )
    assert resp.status_code == 200
    assert resp.json()["rubric_source"] == "auto_derived_from_question"


def test_score_empty_question_is_422(client):
    resp = client.post("/api/v1/grading/score", json={"question": "", "candidate_answer": "something"})
    assert resp.status_code == 422


def test_score_empty_answer_is_422(client):
    resp = client.post("/api/v1/grading/score", json={"question": "something", "candidate_answer": ""})
    assert resp.status_code == 422


def test_score_question_too_long_is_422(client):
    from app.main import settings

    resp = client.post(
        "/api/v1/grading/score",
        json={"question": "x" * (settings.max_question_chars + 1), "candidate_answer": "an answer"},
    )
    assert resp.status_code == 422


def test_score_answer_too_long_is_422(client):
    from app.main import settings

    resp = client.post(
        "/api/v1/grading/score",
        json={"question": "a question", "candidate_answer": "x" * (settings.max_answer_chars + 1)},
    )
    assert resp.status_code == 422


def test_score_rubric_too_many_criteria_is_422(client):
    from app.main import settings

    criteria = [
        {"description": f"c{i}", "weight": 1.0, "expected_keywords": []}
        for i in range(settings.max_rubric_criteria + 1)
    ]
    resp = client.post(
        "/api/v1/grading/score",
        json={"question": "q", "candidate_answer": "a", "rubric": {"criteria": criteria}},
    )
    assert resp.status_code == 422


def test_score_returns_502_on_model_failure(monkeypatch):
    monkeypatch.setattr("app.main.get_semantic_scorer", lambda: AlwaysFailingSemanticScorer())
    from app.main import app

    with TestClient(app) as c:
        resp = c.post("/api/v1/grading/score", json={"question": "q", "candidate_answer": "a"})
    assert resp.status_code == 502
    assert resp.json()["error"] == "request_error"


def test_error_responses_never_leak_tracebacks(client):
    resp = client.post("/api/v1/grading/score", json={"question": "", "candidate_answer": ""})
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
        "environment=production" in r.message and "ANSWER_GRADING_REQUIRE_API_KEY" in r.message
        for r in caplog.records
    )


def test_production_with_required_auth_logs_no_warning(caplog):
    from app.config import Settings
    from app.main import _warn_if_unprotected_in_production

    settings = Settings(environment="production", require_api_key=True, api_keys="some-key")

    with caplog.at_level("WARNING"):
        _warn_if_unprotected_in_production(settings)

    assert not any("environment=production" in r.message for r in caplog.records)


def test_full_stack_scoring_with_real_model():
    """No mocking anywhere in this test — the real embedding model, through
    the real HTTP endpoint, on a genuine paraphrase. This is the proof that
    the whole stack (not just the isolated scorer.py logic) actually works,
    the same role test_live_speech.py/test_live_groq.py play in the other
    services — except nothing here is external, so there's no skip
    condition; it always runs."""
    from app.main import app

    with TestClient(app) as c:
        good_resp = c.post(
            "/api/v1/grading/score",
            json={
                "question": "How did you optimize your database queries?",
                "candidate_answer": "I sped up the database by adding indexes to the slow queries.",
                "grounding": {
                    "kind": "project",
                    "reference": "PostgreSQL query optimization",
                    "detail": "reducing report generation time using indexes",
                },
            },
        )
        bad_resp = c.post(
            "/api/v1/grading/score",
            json={
                "question": "How did you optimize your database queries?",
                "candidate_answer": "I really enjoy playing football on weekends with my friends.",
                "grounding": {
                    "kind": "project",
                    "reference": "PostgreSQL query optimization",
                    "detail": "reducing report generation time using indexes",
                },
            },
        )

    assert good_resp.status_code == 200
    assert bad_resp.status_code == 200
    # The core proof: an on-topic paraphrased answer must score meaningfully
    # higher than a genuinely irrelevant one.
    assert good_resp.json()["overall_score"] > bad_resp.json()["overall_score"]
    assert bad_resp.json()["overall_score"] < 0.15
