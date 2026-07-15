"""API-level tests against the real HTTP stack and the real MediaPipe
vision pipeline. The session store is swapped for an in-memory fake (see
tests/fakes.py) so these tests don't need a real Redis — session-store
correctness itself is proven separately in test_session_store_live.py."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from tests.fakes import FakeProctoringSessionStore


@pytest.fixture
def client(monkeypatch):
    fake_store = FakeProctoringSessionStore()
    monkeypatch.setattr("app.main.get_session_store", lambda: fake_store)

    from app.main import app

    with TestClient(app) as c:
        yield c


def _snapshot_file(image_bytes: bytes) -> dict:
    return {"file": ("frame.jpg", io.BytesIO(image_bytes), "image/jpeg")}


# --- Health / meta -----------------------------------------------------------


def test_liveness_probe(client: TestClient):
    resp = client.get("/health/live")
    assert resp.status_code == 200


def test_readiness_probe_reports_ready(client: TestClient):
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_capabilities(client: TestClient):
    resp = client.get("/api/v1/capabilities")
    assert resp.status_code == 200
    assert "head_turn_threshold_degrees" in resp.json()


def test_metrics_exposed(client: TestClient):
    resp = client.get("/metrics")
    assert resp.status_code == 200


def test_request_id_header_present(client: TestClient):
    resp = client.get("/health/live")
    assert "x-request-id" in resp.headers


# --- Snapshot / summary -----------------------------------------------------------


def test_snapshot_with_frontal_face_reports_no_events(client: TestClient, face_image_bytes: bytes):
    resp = client.post("/api/v1/proctoring/session-1/snapshot", files=_snapshot_file(face_image_bytes))
    assert resp.status_code == 200
    body = resp.json()
    assert body["faces_detected"] == 1
    assert body["events_recorded"] == []
    assert body["integrity_score"] == 100.0
    assert body["frames_processed"] == 1


def test_snapshot_blank_image_reports_zero_faces(client: TestClient, blank_image_bytes: bytes):
    resp = client.post("/api/v1/proctoring/session-blank/snapshot", files=_snapshot_file(blank_image_bytes))
    assert resp.status_code == 200
    assert resp.json()["faces_detected"] == 0


def test_snapshot_two_faces_flags_multiple_faces_this_frame(client: TestClient, two_faces_image_bytes: bytes):
    resp = client.post("/api/v1/proctoring/session-two/snapshot", files=_snapshot_file(two_faces_image_bytes))
    assert resp.status_code == 200
    body = resp.json()
    assert body["faces_detected"] == 2
    assert "multiple_faces" in body["flagged_this_frame"]


def test_sustained_no_face_lowers_integrity_score_and_records_event(client: TestClient, blank_image_bytes: bytes):
    session_id = "session-sustained-no-face"
    last_body = None
    for _ in range(5):
        resp = client.post(f"/api/v1/proctoring/{session_id}/snapshot", files=_snapshot_file(blank_image_bytes))
        assert resp.status_code == 200
        last_body = resp.json()

    assert last_body["integrity_score"] < 100.0
    assert "no_face" in last_body["events_recorded"] or last_body["frames_processed"] == 5


def test_summary_reflects_processed_frames(client: TestClient, face_image_bytes: bytes):
    session_id = "session-summary"
    client.post(f"/api/v1/proctoring/{session_id}/snapshot", files=_snapshot_file(face_image_bytes))
    client.post(f"/api/v1/proctoring/{session_id}/snapshot", files=_snapshot_file(face_image_bytes))

    resp = client.get(f"/api/v1/proctoring/{session_id}/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["frames_processed"] == 2
    assert body["integrity_score"] == 100.0


def test_summary_for_unknown_session_is_404(client: TestClient):
    resp = client.get("/api/v1/proctoring/never-seen-session/summary")
    assert resp.status_code == 404


def test_delete_session(client: TestClient, face_image_bytes: bytes):
    session_id = "session-to-delete"
    client.post(f"/api/v1/proctoring/{session_id}/snapshot", files=_snapshot_file(face_image_bytes))

    delete_resp = client.delete(f"/api/v1/proctoring/{session_id}")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["deleted"] is True

    summary_resp = client.get(f"/api/v1/proctoring/{session_id}/summary")
    assert summary_resp.status_code == 404


def test_snapshot_empty_image_is_422(client: TestClient):
    resp = client.post(
        "/api/v1/proctoring/session-empty/snapshot",
        files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
    )
    assert resp.status_code == 422


def test_snapshot_undecodable_image_is_422(client: TestClient):
    resp = client.post(
        "/api/v1/proctoring/session-bad/snapshot",
        files={"file": ("bad.jpg", io.BytesIO(b"not a real image"), "image/jpeg")},
    )
    assert resp.status_code == 422


# --- Error hygiene -----------------------------------------------------------


def test_error_responses_never_leak_tracebacks(client: TestClient):
    resp = client.get("/api/v1/proctoring/never-seen-session/summary")
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
        "environment=production" in r.message and "PROCTORING_REQUIRE_API_KEY" in r.message
        for r in caplog.records
    )


def test_production_with_required_auth_logs_no_warning(caplog):
    from app.config import Settings
    from app.main import _warn_if_unprotected_in_production

    settings = Settings(environment="production", require_api_key=True, api_keys="some-key")

    with caplog.at_level("WARNING"):
        _warn_if_unprotected_in_production(settings)

    assert not any("environment=production" in r.message for r in caplog.records)
