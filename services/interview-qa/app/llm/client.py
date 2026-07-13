"""LLM provider abstraction.

The BRD names "GPT-4-class / Llama-3 / Gemini-class" as interchangeable
options for this role — nothing about the design should hardcode one
provider. `LLMClient` is the seam: callers depend only on `generate_json`,
never on a provider SDK directly, so adding Ollama, Anthropic, or OpenAI
later means registering a new class in `get_llm_client()`, not touching the
question-generation logic that calls it.

Groq is the concrete choice today because it has a genuinely usable free
tier with hosted (not self-managed) inference — see the interview-qa
README for the reasoning. It speaks an OpenAI-compatible chat-completions
API, which is what most alternatives (including a local Ollama server) also
expose, so the concrete client below is a reasonable template for the next
provider too.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Protocol

from ..config import get_settings

logger = logging.getLogger("interview_qa.llm")


class LLMError(Exception):
    """Raised for any provider-level failure (auth, rate limit, network,
    malformed response) — callers handle one exception type regardless of
    which provider is configured."""


class LLMClient(Protocol):
    model_name: str

    def generate_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Returns the raw JSON text of the model's response. Raises
        LLMError on any failure — network, auth, rate limit, or the
        provider refusing to produce a response at all."""
        ...


class GroqClient:
    def __init__(self, api_key: str, model: str, timeout_seconds: float) -> None:
        if not api_key:
            raise LLMError(
                "No Groq API key configured — set INTERVIEW_QA_GROQ_API_KEY."
            )
        from groq import Groq

        self._client = Groq(api_key=api_key, timeout=timeout_seconds)
        self.model_name = model

    def generate_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        settings = get_settings()
        from groq import APIError

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature if temperature is not None else settings.llm_temperature,
                max_tokens=max_tokens or settings.llm_max_tokens,
                response_format={"type": "json_object"},
            )
        except APIError as exc:
            raise LLMError(f"Groq API error: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - network/timeout/etc.
            raise LLMError(f"Groq request failed: {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise LLMError("Groq returned an empty response.")
        return content


def _build_client(settings) -> LLMClient:  # noqa: ANN001
    if settings.llm_provider == "groq":
        return GroqClient(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            timeout_seconds=settings.llm_request_timeout_seconds,
        )
    raise LLMError(f"Unsupported LLM provider: {settings.llm_provider!r}")


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    return _build_client(get_settings())
