"""Tests for MaxBodySizeMiddleware, including the exempt_path_suffixes
mechanism needed for routes with a dynamic segment before the exempted
suffix (e.g. /sessions/{id}/answer/audio)."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.middleware import MaxBodySizeMiddleware


def _make_app(max_bytes: int, exempt_paths=frozenset(), exempt_path_suffixes=()) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        MaxBodySizeMiddleware, max_bytes=max_bytes, exempt_paths=exempt_paths, exempt_path_suffixes=exempt_path_suffixes
    )

    @app.post("/echo")
    def echo(payload: dict) -> dict:
        return {"received": True}

    @app.post("/sessions/{session_id}/answer/audio")
    def audio(session_id: str) -> dict:
        return {"received": True}

    @app.get("/ping")
    def ping() -> dict:
        return {"status": "ok"}

    return app


def test_oversized_body_rejected_with_413():
    client = TestClient(_make_app(max_bytes=1000))
    resp = client.post("/echo", json={"data": "x" * 50_000})
    assert resp.status_code == 413


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


def test_dynamic_segment_route_exempted_by_suffix():
    """/sessions/{id}/answer/audio has a different session_id on every
    call, so an exact-match exempt_paths set can't express "exempt this
    route for any id" — the suffix check must."""
    client = TestClient(_make_app(max_bytes=10, exempt_path_suffixes=("/answer/audio",)))
    resp = client.post("/sessions/abc123/answer/audio", content=b"x" * 5000, headers={"content-length": "5000"})
    assert resp.status_code == 200


def test_non_exempt_route_still_enforces_limit_despite_suffix_config():
    client = TestClient(_make_app(max_bytes=10, exempt_path_suffixes=("/answer/audio",)))
    resp = client.post("/echo", json={"data": "this is definitely over 10 bytes"})
    assert resp.status_code == 413
