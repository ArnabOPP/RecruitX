"""API-level tests: HTTP status codes, error schema, and endpoint wiring.

Uses TestClient as a context manager so FastAPI's lifespan (model preload)
actually runs — otherwise /health/ready would never flip to "ready" and
/api/v1/parse would hit a cold, un-preloaded pipeline like a real deploy
never would.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_liveness(client):
    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readiness_after_startup(client):
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_legacy_health_alias(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_capabilities(client):
    resp = client.get("/api/v1/capabilities")
    assert resp.status_code == 200
    body = resp.json()
    assert body["supported_file_types"] == [".docx", ".pdf", ".txt"]
    assert "max_upload_bytes" in body


def test_metrics_exposed(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "python_gc_objects_collected_total" in resp.text


def test_request_id_header_present(client):
    resp = client.get("/health/live")
    assert "x-request-id" in resp.headers


def test_request_id_echoed_when_supplied(client):
    resp = client.get("/health/live", headers={"X-Request-ID": "test-fixed-id"})
    assert resp.headers["x-request-id"] == "test-fixed-id"


def test_parse_success(client):
    with open(FIXTURES / "sample_resume.txt", "rb") as f:
        resp = client.post(
            "/api/v1/parse",
            files={"file": ("sample_resume.txt", f, "text/plain")},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["contact"]["full_name"]["value"] == "Aarav Sharma"
    assert len(body["skills"]) > 0


def test_parse_unsupported_extension(client):
    resp = client.post(
        "/api/v1/parse",
        files={"file": ("resume.xyz", b"hello world", "application/octet-stream")},
    )
    assert resp.status_code == 415
    body = resp.json()
    assert body["error"] == "request_error"
    assert "request_id" in body


def test_parse_spoofed_pdf_signature(client):
    resp = client.post(
        "/api/v1/parse",
        files={"file": ("resume.pdf", b"not a real pdf", "application/pdf")},
    )
    assert resp.status_code == 422


def test_parse_empty_file(client):
    resp = client.post(
        "/api/v1/parse",
        files={"file": ("resume.txt", b"", "text/plain")},
    )
    assert resp.status_code == 400


def test_parse_missing_file(client):
    resp = client.post("/api/v1/parse")
    assert resp.status_code == 422  # FastAPI's own request-validation error


def test_parse_too_large(client):
    huge = b"a" * (11 * 1024 * 1024)  # over the 10 MB default limit
    resp = client.post(
        "/api/v1/parse",
        files={"file": ("resume.txt", huge, "text/plain")},
    )
    assert resp.status_code == 413


def test_error_responses_never_leak_tracebacks(client):
    resp = client.post(
        "/api/v1/parse",
        files={"file": ("resume.xyz", b"hello", "application/octet-stream")},
    )
    body = resp.json()
    assert "Traceback" not in body["detail"]
    assert "File \"" not in body["detail"]
