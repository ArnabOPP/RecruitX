"""Tests for the provider-level resilience layer: rate-limit backoff and
fail-fast-on-bad-credentials — the two gaps identified when this service was
reviewed against cv-parser's production bar (that review had only checked
the parsing/validation retry loop, not the provider-call layer beneath it).
"""

from __future__ import annotations

import time

import pytest

from app.llm.client import (
    LLMAuthenticationError,
    LLMRateLimitError,
    _parse_retry_after,
    generate_json_with_backoff,
)
from tests.fakes import AlwaysRateLimitedLLMClient, AlwaysUnauthenticatedLLMClient, FakeLLMClient


def test_backoff_retries_rate_limit_and_eventually_raises():
    client = AlwaysRateLimitedLLMClient(retry_after=0.01)
    with pytest.raises(LLMRateLimitError):
        generate_json_with_backoff(client, "sys", "user", max_retries=2)
    # max_retries=2 -> 3 total attempts (1 initial + 2 retries)
    assert client.call_count == 3


def test_backoff_actually_waits_between_rate_limit_retries():
    client = AlwaysRateLimitedLLMClient(retry_after=0.05)
    start = time.monotonic()
    with pytest.raises(LLMRateLimitError):
        generate_json_with_backoff(client, "sys", "user", max_retries=2)
    elapsed = time.monotonic() - start
    # 2 backoff sleeps of ~0.05s each should add up to at least ~0.1s —
    # confirms it's genuinely sleeping, not just looping instantly.
    assert elapsed >= 0.09


def test_backoff_succeeds_after_transient_rate_limit():
    client = FakeLLMClient(responses=['{"ok": true}'])
    # Wrap so the first call raises a rate limit, second succeeds — reuse
    # FakeLLMClient's scripted-response mechanism by monkeypatching a single
    # failure in front of it.
    call_count = {"n": 0}
    original_generate = client.generate_json

    def flaky_generate(system, user, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise LLMRateLimitError("first call rate limited", retry_after=0.01)
        return original_generate(system, user, **kwargs)

    client.generate_json = flaky_generate  # type: ignore[method-assign]
    result = generate_json_with_backoff(client, "sys", "user", max_retries=2)
    assert result == '{"ok": true}'
    assert call_count["n"] == 2


def test_backoff_fails_fast_on_authentication_error_without_wasting_retries():
    client = AlwaysUnauthenticatedLLMClient()
    with pytest.raises(LLMAuthenticationError):
        generate_json_with_backoff(client, "sys", "user", max_retries=5)
    # Must not have burned the whole retry budget on something that will
    # never succeed no matter how many times it's retried.
    assert client.call_count == 1


def test_backoff_retries_generic_llm_error_without_delay():
    client = FakeLLMClient(responses=[])  # every call raises a plain LLMError
    start = time.monotonic()
    with pytest.raises(Exception):  # noqa: B017 - FakeLLMClient raises LLMError
        generate_json_with_backoff(client, "sys", "user", max_retries=2)
    elapsed = time.monotonic() - start
    # Generic (non-rate-limit) errors retry immediately — this should be
    # near-instant, not accumulate backoff delay.
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
