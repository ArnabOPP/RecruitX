"""Tests for MaxBodySizeMiddleware — same pattern as interview-qa/speech-io.
Tested against a minimal standalone app rather than the real app.main.app
to avoid polluting shared module state across test files."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware import MaxBodySizeMiddleware


def _make_app(max_bytes: int) -> FastAPI:
    app = FastAPI()
    app.add_middleware(MaxBodySizeMiddleware, max_bytes=max_bytes)

    @app.post("/echo")
    def echo(payload: dict) -> dict:
        return {"received": True}

    @app.get("/ping")
    def ping() -> dict:
        return {"status": "ok"}

    return app


def test_oversized_body_rejected_with_413():
    client = TestClient(_make_app(max_bytes=1000))
    resp = client.post("/echo", json={"data": "x" * 50_000})
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
