"""Tests for inbound API-key authentication — protects host compute
resources (spawning sandboxed containers) from unauthenticated abuse."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.auth import verify_api_key
from app.config import get_settings


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_auth_disabled_by_default_allows_any_request(monkeypatch):
    monkeypatch.delenv("CODE_EVAL_REQUIRE_API_KEY", raising=False)
    get_settings.cache_clear()
    verify_api_key(x_api_key=None)


def test_auth_required_rejects_missing_header(monkeypatch):
    monkeypatch.setenv("CODE_EVAL_REQUIRE_API_KEY", "1")
    monkeypatch.setenv("CODE_EVAL_API_KEYS", "secret-key-one")
    get_settings.cache_clear()

    with pytest.raises(HTTPException) as exc_info:
        verify_api_key(x_api_key=None)
    assert exc_info.value.status_code == 401


def test_auth_required_rejects_wrong_key(monkeypatch):
    monkeypatch.setenv("CODE_EVAL_REQUIRE_API_KEY", "1")
    monkeypatch.setenv("CODE_EVAL_API_KEYS", "secret-key-one")
    get_settings.cache_clear()

    with pytest.raises(HTTPException) as exc_info:
        verify_api_key(x_api_key="totally-wrong-key")
    assert exc_info.value.status_code == 401


def test_auth_required_accepts_correct_key(monkeypatch):
    monkeypatch.setenv("CODE_EVAL_REQUIRE_API_KEY", "1")
    monkeypatch.setenv("CODE_EVAL_API_KEYS", "secret-key-one")
    get_settings.cache_clear()

    verify_api_key(x_api_key="secret-key-one")


def test_auth_accepts_any_of_multiple_configured_keys(monkeypatch):
    monkeypatch.setenv("CODE_EVAL_REQUIRE_API_KEY", "1")
    monkeypatch.setenv("CODE_EVAL_API_KEYS", "key-a,key-b, key-c ")
    get_settings.cache_clear()

    verify_api_key(x_api_key="key-a")
    verify_api_key(x_api_key="key-b")
    verify_api_key(x_api_key="key-c")
    with pytest.raises(HTTPException):
        verify_api_key(x_api_key="key-d")


def test_auth_required_but_misconfigured_fails_closed(monkeypatch):
    monkeypatch.setenv("CODE_EVAL_REQUIRE_API_KEY", "1")
    monkeypatch.setenv("CODE_EVAL_API_KEYS", "")
    get_settings.cache_clear()

    with pytest.raises(HTTPException) as exc_info:
        verify_api_key(x_api_key="anything")
    assert exc_info.value.status_code == 503
