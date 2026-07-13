"""A fake LLM client for fast, deterministic unit tests that don't hit the
real Groq API — see test_live_groq.py for the tests that do."""

from __future__ import annotations

from app.llm.client import LLMAuthenticationError, LLMError, LLMRateLimitError


class FakeLLMClient:
    """Returns a scripted sequence of responses, one per call — lets a test
    simulate "malformed JSON, then a good response" to exercise retry logic,
    or "always malformed" to exercise the failure path."""

    def __init__(self, responses: list[str] | None = None, model_name: str = "fake-model"):
        self._responses = list(responses or [])
        self.model_name = model_name
        self.calls: list[tuple[str, str]] = []

    def generate_json(self, system: str, user: str, *, max_tokens=None, temperature=None) -> str:
        self.calls.append((system, user))
        if not self._responses:
            raise LLMError("FakeLLMClient has no scripted responses left.")
        return self._responses.pop(0)

    def validate(self) -> None:
        pass


class AlwaysFailingLLMClient:
    model_name = "fake-model"

    def generate_json(self, system: str, user: str, *, max_tokens=None, temperature=None) -> str:
        raise LLMError("simulated provider failure")

    def validate(self) -> None:
        pass


class AlwaysRateLimitedLLMClient:
    """Simulates a provider that's always rate-limited — used to verify the
    backoff/retry mechanism actually sleeps and eventually gives up rather
    than retrying forever or hammering immediately."""

    model_name = "fake-model"

    def __init__(self, retry_after: float | None = 0.01) -> None:
        self.retry_after = retry_after
        self.call_count = 0

    def generate_json(self, system: str, user: str, *, max_tokens=None, temperature=None) -> str:
        self.call_count += 1
        raise LLMRateLimitError("simulated rate limit", retry_after=self.retry_after)

    def validate(self) -> None:
        pass


class AlwaysUnauthenticatedLLMClient:
    """Simulates a provider rejecting the API key outright — used to verify
    this fails fast without wasting the retry budget."""

    model_name = "fake-model"

    def __init__(self) -> None:
        self.call_count = 0

    def generate_json(self, system: str, user: str, *, max_tokens=None, temperature=None) -> str:
        self.call_count += 1
        raise LLMAuthenticationError("simulated bad credentials")

    def validate(self) -> None:
        raise LLMAuthenticationError("simulated bad credentials")
