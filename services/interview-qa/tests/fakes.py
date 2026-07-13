"""A fake LLM client for fast, deterministic unit tests that don't hit the
real Groq API — see test_live_groq.py for the tests that do."""

from __future__ import annotations

from app.llm.client import LLMError


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


class AlwaysFailingLLMClient:
    model_name = "fake-model"

    def generate_json(self, system: str, user: str, *, max_tokens=None, temperature=None) -> str:
        raise LLMError("simulated provider failure")
