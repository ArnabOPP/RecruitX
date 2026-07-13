"""Tests for the STT provider-level resilience layer: rate-limit backoff and
fail-fast-on-bad-credentials — mirrors interview-qa's llm/client.py tests
since Groq's failure modes are the same across its chat and audio APIs."""

from __future__ import annotations

import time

import pytest

from app.stt.client import (
    STTAuthenticationError,
    STTRateLimitError,
    _parse_retry_after,
    transcribe_with_backoff,
)
from tests.fakes import AlwaysRateLimitedSTTClient, AlwaysUnauthenticatedSTTClient, FakeSTTClient


def test_backoff_retries_rate_limit_and_eventually_raises():
    client = AlwaysRateLimitedSTTClient(retry_after=0.01)
    with pytest.raises(STTRateLimitError):
        transcribe_with_backoff(client, b"audio", "a.wav", language=None, max_retries=2)
    assert client.call_count == 3


def test_backoff_actually_waits_between_rate_limit_retries():
    client = AlwaysRateLimitedSTTClient(retry_after=0.05)
    start = time.monotonic()
    with pytest.raises(STTRateLimitError):
        transcribe_with_backoff(client, b"audio", "a.wav", language=None, max_retries=2)
    elapsed = time.monotonic() - start
    assert elapsed >= 0.09


def test_backoff_succeeds_after_transient_rate_limit():
    client = FakeSTTClient(responses=["hello world"])
    call_count = {"n": 0}
    original_transcribe = client.transcribe

    def flaky_transcribe(audio_bytes, filename, language=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise STTRateLimitError("first call rate limited", retry_after=0.01)
        return original_transcribe(audio_bytes, filename, language)

    client.transcribe = flaky_transcribe  # type: ignore[method-assign]
    result = transcribe_with_backoff(client, b"audio", "a.wav", language=None, max_retries=2)
    assert result == "hello world"
    assert call_count["n"] == 2


def test_backoff_fails_fast_on_authentication_error_without_wasting_retries():
    client = AlwaysUnauthenticatedSTTClient()
    with pytest.raises(STTAuthenticationError):
        transcribe_with_backoff(client, b"audio", "a.wav", language=None, max_retries=5)
    assert client.call_count == 1


def test_backoff_retries_generic_stt_error_without_delay():
    client = FakeSTTClient(responses=[])  # every call raises a plain STTError
    start = time.monotonic()
    with pytest.raises(Exception):  # noqa: B017 - FakeSTTClient raises STTError
        transcribe_with_backoff(client, b"audio", "a.wav", language=None, max_retries=2)
    elapsed = time.monotonic() - start
    assert elapsed < 0.5


def test_parse_retry_after_reads_header():
    class FakeResponse:
        headers = {"retry-after": "3.5"}

    class FakeExc(Exception):
        response = FakeResponse()

    assert _parse_retry_after(FakeExc()) == 3.5


def test_parse_retry_after_handles_missing_header():
    class FakeResponse:
        headers: dict = {}

    class FakeExc(Exception):
        response = FakeResponse()

    assert _parse_retry_after(FakeExc()) is None


def test_parse_retry_after_handles_no_response_attribute():
    assert _parse_retry_after(Exception("no response attr")) is None


def test_parse_retry_after_handles_garbage_header_value():
    class FakeResponse:
        headers = {"retry-after": "not-a-number"}

    class FakeExc(Exception):
        response = FakeResponse()

    assert _parse_retry_after(FakeExc()) is None
