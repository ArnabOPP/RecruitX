"""API-level tests against the real HTTP stack. No mocking of the vision
models anywhere here — enroll/verify/liveness only mean something if
they're proven against the real MediaPipe + ArcFace pipeline end to end,
matching every other Recruitix service's "prove the real stack works"
philosophy. Model loading is expensive, so the TestClient is module-scoped
rather than rebuilt per test."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


def _files(image_bytes: bytes, count: int, field: str = "files") -> list[tuple[str, tuple[str, io.BytesIO, str]]]:
    return [(field, (f"frame{i}.jpg", io.BytesIO(image_bytes), "image/jpeg")) for i in range(count)]


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
    body = resp.json()
    assert body["ready"] is True
    assert "match_similarity_threshold" in body


def test_metrics_exposed(client: TestClient):
    resp = client.get("/metrics")
    assert resp.status_code == 200


def test_request_id_header_present(client: TestClient):
    resp = client.get("/health/live")
    assert "x-request-id" in resp.headers


# --- Enroll / verify (real vision + embedding pipeline) ----------------------


def test_enroll_then_verify_same_person_matches(client: TestClient, face_image_bytes: bytes):
    candidate_id = "candidate-enroll-verify-match"
    enroll_resp = client.post(
        "/api/v1/biometric/enroll",
        params={"candidate_id": candidate_id},
        files=_files(face_image_bytes, 3),
    )
    assert enroll_resp.status_code == 200
    body = enroll_resp.json()
    assert body["enrolled"] is True
    assert body["images_used"] == 3

    verify_resp = client.post(
        "/api/v1/biometric/verify",
        params={"candidate_id": candidate_id},
        files=_files(face_image_bytes, 1),
    )
    assert verify_resp.status_code == 200
    verify_body = verify_resp.json()
    assert verify_body["match"] is True
    assert verify_body["similarity"] > 0.9


def test_verify_unknown_candidate_is_404(client: TestClient, face_image_bytes: bytes):
    resp = client.post(
        "/api/v1/biometric/verify",
        params={"candidate_id": "never-enrolled-candidate"},
        files=_files(face_image_bytes, 1),
    )
    assert resp.status_code == 404


def test_enroll_below_minimum_images_is_422(client: TestClient, face_image_bytes: bytes):
    resp = client.post(
        "/api/v1/biometric/enroll",
        params={"candidate_id": "candidate-too-few-images"},
        files=_files(face_image_bytes, 1),
    )
    assert resp.status_code == 422


def test_enroll_empty_candidate_id_is_422(client: TestClient, face_image_bytes: bytes):
    resp = client.post(
        "/api/v1/biometric/enroll",
        params={"candidate_id": "  "},
        files=_files(face_image_bytes, 3),
    )
    assert resp.status_code == 422


def test_enroll_blank_image_yields_no_face_422(client: TestClient, blank_image_bytes: bytes):
    resp = client.post(
        "/api/v1/biometric/enroll",
        params={"candidate_id": "candidate-blank-image"},
        files=_files(blank_image_bytes, 3),
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "no_face_detected"


def test_enroll_two_face_image_is_rejected(client: TestClient, two_faces_image_bytes: bytes):
    resp = client.post(
        "/api/v1/biometric/enroll",
        params={"candidate_id": "candidate-two-faces"},
        files=_files(two_faces_image_bytes, 3),
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "multiple_faces_detected"


def test_enroll_too_many_images_is_422(client: TestClient, face_image_bytes: bytes):
    from app.config import get_settings

    limit = get_settings().max_images_per_request
    resp = client.post(
        "/api/v1/biometric/enroll",
        params={"candidate_id": "candidate-too-many"},
        files=_files(face_image_bytes, limit + 1),
    )
    assert resp.status_code == 422


def test_delete_enrollment(client: TestClient, face_image_bytes: bytes):
    candidate_id = "candidate-to-delete"
    client.post(
        "/api/v1/biometric/enroll",
        params={"candidate_id": candidate_id},
        files=_files(face_image_bytes, 3),
    )
    delete_resp = client.delete(f"/api/v1/biometric/enroll/{candidate_id}")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["deleted"] is True

    verify_resp = client.post(
        "/api/v1/biometric/verify",
        params={"candidate_id": candidate_id},
        files=_files(face_image_bytes, 1),
    )
    assert verify_resp.status_code == 404


# --- Liveness -----------------------------------------------------------


def test_liveness_static_replay_fails_on_blink_not_bogus_head_pose(client: TestClient, face_image_bytes: bytes):
    """Regression test for the solvePnP bug fixed in liveness.py: a static
    photo replayed as every "frame" must fail because no blink occurred —
    not because of an impossible head-pose deviation value."""
    from app.config import get_settings

    frame_count = get_settings().liveness_min_frames
    resp = client.post("/api/v1/biometric/liveness", files=_files(face_image_bytes, frame_count))
    assert resp.status_code == 200
    body = resp.json()
    assert body["live"] is False
    assert body["max_head_pose_deviation_degrees"] < 35.0
    assert "blink" in body["reason"].lower()


def test_liveness_below_minimum_frames_is_422(client: TestClient, face_image_bytes: bytes):
    resp = client.post("/api/v1/biometric/liveness", files=_files(face_image_bytes, 2))
    assert resp.status_code == 422


def test_liveness_blank_frames_report_zero_frames_analyzed(client: TestClient, blank_image_bytes: bytes):
    from app.config import get_settings

    frame_count = get_settings().liveness_min_frames
    resp = client.post("/api/v1/biometric/liveness", files=_files(blank_image_bytes, frame_count))
    assert resp.status_code == 200
    body = resp.json()
    assert body["live"] is False
    assert body["frames_analyzed"] == 0


# --- Error hygiene -----------------------------------------------------------


def test_error_responses_never_leak_tracebacks(client: TestClient, blank_image_bytes: bytes):
    resp = client.post(
        "/api/v1/biometric/enroll",
        params={"candidate_id": "candidate-traceback-check"},
        files=_files(blank_image_bytes, 3),
    )
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
        "environment=production" in r.message and "BIOMETRIC_AUTH_REQUIRE_API_KEY" in r.message
        for r in caplog.records
    )


def test_production_with_required_auth_logs_no_warning(caplog):
    from app.config import Settings
    from app.main import _warn_if_unprotected_in_production

    settings = Settings(environment="production", require_api_key=True, api_keys="some-key")

    with caplog.at_level("WARNING"):
        _warn_if_unprotected_in_production(settings)

    assert not any("environment=production" in r.message for r in caplog.records)
