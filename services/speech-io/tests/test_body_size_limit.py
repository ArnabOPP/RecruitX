"""Tests for MaxBodySizeMiddleware — same purpose as interview-qa's version,
plus the exempt_paths mechanism this service needs since /transcribe
legitimately carries much larger bodies (audio) than /synthesize (JSON
text), so it can't share one blanket limit.

Tested against a minimal standalone app rather than the real app.main.app
— reloading that shared module to vary max_bytes per test would pollute
module-level state for every other test file that imports it.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware import MaxBodySizeMiddleware


def _make_app(max_bytes: int, exempt_paths: frozenset[str] = frozenset()) -> FastAPI:
    app = FastAPI()
    app.add_middleware(MaxBodySizeMiddleware, max_bytes=max_bytes, exempt_paths=exempt_paths)

    @app.post("/echo")
    def echo(payload: dict) -> dict:
        return {"received": True}

    @app.post("/upload")
    def upload() -> dict:
        return {"received": True}

    @app.get("/ping")
    def ping() -> dict:
        return {"status": "ok"}

    return app


def test_oversized_body_rejected_with_413():
    client = TestClient(_make_app(max_bytes=1000))
    huge_payload = {"data": "x" * 50_000}

    resp = client.post("/echo", json=huge_payload)

    assert resp.status_code == 413
    body = resp.json()
    assert body["error"] == "request_error"
    assert "request_id" in body


def test_body_under_limit_is_not_rejected():
    client = TestClient(_make_app(max_bytes=1_000_000))

    resp = client.post("/echo", json={"data": "small"})

    assert resp.status_code == 200


def test_get_requests_are_never_size_checked():
    client = TestClient(_make_app(max_bytes=1))

    resp = client.get("/ping")

    assert resp.status_code == 200


def test_missing_content_length_is_rejected():
    client = TestClient(_make_app(max_bytes=1_000_000))

    def fake_generator():
        yield b'{"data": "small"}'

    resp = client.post("/echo", content=fake_generator())

    assert resp.status_code == 411


def test_exempt_path_bypasses_the_size_check():
    """/transcribe is exempted here because it enforces its own, larger,
    audio-specific limit inside the endpoint itself (mirroring cv-parser's
    file upload check) — it must not also be capped by the small JSON-body
    limit meant for /synthesize."""
    client = TestClient(_make_app(max_bytes=10, exempt_paths=frozenset({"/upload"})))

    resp = client.post("/upload", content=b"x" * 5000, headers={"content-length": "5000"})

    assert resp.status_code == 200


def test_non_exempt_path_still_enforces_the_limit():
    client = TestClient(_make_app(max_bytes=10, exempt_paths=frozenset({"/upload"})))

    resp = client.post("/echo", json={"data": "this is definitely over 10 bytes"})

    assert resp.status_code == 413
