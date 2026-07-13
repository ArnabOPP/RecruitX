"""API-level tests: HTTP status codes, error schema, and endpoint wiring.

The STT/TTS clients are mocked at the point main.py actually imports them
(app.main.get_stt_client / app.main.get_tts_client) — these tests are fast
and network-free; the real providers are exercised separately in
test_live_speech.py.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.fakes import AlwaysFailingSTTClient, AlwaysFailingTTSClient, FakeSTTClient, FakeTTSClient


@pytest.fixture
def client(monkeypatch):
    fake_stt = FakeSTTClient(responses=["hello, this is a test answer"] * 10)
    fake_tts = FakeTTSClient(audio=b"fake-mp3-audio-bytes")
    monkeypatch.setattr("app.main.get_stt_client", lambda: fake_stt)
    monkeypatch.setattr("app.main.get_tts_client", lambda: fake_tts)

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
    body = resp.json()
    assert body["status"] == "ready"
    assert body["stt_ready"] is True
    assert body["tts_ready"] is True


def test_capabilities(client):
    resp = client.get("/api/v1/capabilities")
    assert resp.status_code == 200
    body = resp.json()
    assert body["stt_provider"] == "groq"
    assert body["tts_provider"] == "edge"
    assert body["stt_ready"] is True
    assert body["tts_ready"] is True


def test_metrics_exposed(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200


def test_request_id_header_present(client):
    resp = client.get("/health/live")
    assert "x-request-id" in resp.headers


def test_transcribe_success(client):
    resp = client.post(
        "/api/v1/speech/transcribe",
        files={"file": ("answer.wav", b"fake-audio-bytes", "audio/wav")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "hello, this is a test answer"
    assert body["model_used"] == "fake-whisper"


def test_transcribe_missing_file_is_422(client):
    resp = client.post("/api/v1/speech/transcribe")
    assert resp.status_code == 422


def test_transcribe_empty_file_is_400(client):
    resp = client.post(
        "/api/v1/speech/transcribe",
        files={"file": ("answer.wav", b"", "audio/wav")},
    )
    assert resp.status_code == 400


def test_transcribe_returns_502_on_persistent_failure(monkeypatch):
    monkeypatch.setattr("app.main.get_stt_client", lambda: AlwaysFailingSTTClient())
    monkeypatch.setattr("app.main.get_tts_client", lambda: FakeTTSClient())
    from app.main import app

    with TestClient(app) as c:
        resp = c.post(
            "/api/v1/speech/transcribe",
            files={"file": ("answer.wav", b"fake-audio-bytes", "audio/wav")},
        )
    assert resp.status_code == 502
    assert resp.json()["error"] == "request_error"


def test_synthesize_success(client):
    resp = client.post("/api/v1/speech/synthesize", json={"text": "Tell me about your project."})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    assert resp.content == b"fake-mp3-audio-bytes"


def test_synthesize_empty_text_is_422(client):
    resp = client.post("/api/v1/speech/synthesize", json={"text": ""})
    assert resp.status_code == 422


def test_synthesize_text_too_long_is_422(client):
    # Settings are resolved once at module import (like every config value
    # in this app), so this exercises the actual configured default limit
    # rather than monkeypatching an env var the already-built app can't see.
    from app.main import settings

    resp = client.post("/api/v1/speech/synthesize", json={"text": "x" * (settings.max_synthesize_text_chars + 1)})
    assert resp.status_code == 422


def test_synthesize_returns_502_on_persistent_failure(monkeypatch):
    monkeypatch.setattr("app.main.get_stt_client", lambda: FakeSTTClient())
    monkeypatch.setattr("app.main.get_tts_client", lambda: AlwaysFailingTTSClient())
    from app.main import app

    with TestClient(app) as c:
        resp = c.post("/api/v1/speech/synthesize", json={"text": "hello"})
    assert resp.status_code == 502
    assert resp.json()["error"] == "request_error"


def test_error_responses_never_leak_tracebacks(client):
    resp = client.post("/api/v1/speech/synthesize", json={"text": ""})
    body = resp.json()
    assert "Traceback" not in body["detail"]
    assert 'File "' not in body["detail"]


def test_readiness_reflects_invalid_stt_credentials(monkeypatch):
    from tests.fakes import AlwaysUnauthenticatedSTTClient

    monkeypatch.setattr("app.main.get_stt_client", lambda: AlwaysUnauthenticatedSTTClient())
    monkeypatch.setattr("app.main.get_tts_client", lambda: FakeTTSClient())
    from app.main import app

    with TestClient(app) as c:
        resp = c.get("/health/ready")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "not_ready"
        assert body["stt_ready"] is False
        assert body["tts_ready"] is True


def test_production_without_required_auth_logs_security_warning(caplog):
    from app.config import Settings
    from app.main import _warn_if_unprotected_in_production

    settings = Settings(environment="production", require_api_key=False)

    with caplog.at_level("WARNING"):
        _warn_if_unprotected_in_production(settings)

    assert any(
        "environment=production" in r.message and "SPEECH_IO_REQUIRE_API_KEY" in r.message
        for r in caplog.records
    )


def test_production_with_required_auth_logs_no_warning(caplog):
    from app.config import Settings
    from app.main import _warn_if_unprotected_in_production

    settings = Settings(environment="production", require_api_key=True, api_keys="some-key")

    with caplog.at_level("WARNING"):
        _warn_if_unprotected_in_production(settings)

    assert not any("environment=production" in r.message for r in caplog.records)
