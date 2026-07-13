"""LLM provider abstraction.

The BRD names "GPT-4-class / Llama-3 / Gemini-class" as interchangeable
options for this role — nothing about the design should hardcode one
provider. `LLMClient` is the seam: callers depend only on `generate_json`,
`validate`, and `generate_json_with_backoff`, never on a provider SDK
directly, so adding Ollama, Anthropic, or OpenAI later means registering a
new class in `get_llm_client()`, not touching the question-generation logic
that calls it.

Groq is the concrete choice today because it has a genuinely usable free
tier with hosted (not self-managed) inference — see the interview-qa
README for the reasoning. It speaks an OpenAI-compatible chat-completions
API, which is what most alternatives (including a local Ollama server) also
expose, so the concrete client below is a reasonable template for the next
provider too.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Protocol

from ..config import get_settings

logger = logging.getLogger("interview_qa.llm")


class LLMError(Exception):
    """Raised for any provider-level failure (auth, network, malformed
    response) — callers handle one exception type regardless of which
    provider is configured."""


class LLMRateLimitError(LLMError):
    """A specific LLMError subtype for 429s, carrying how long the provider
    says to wait before retrying — so callers can back off intelligently
    instead of hammering a rate-limited endpoint with immediate retries,
    which just extends the outage."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMAuthenticationError(LLMError):
    """The provider rejected the credentials outright (bad/revoked key) —
    distinct from a rate limit or a transient network issue, since retrying
    this one is pointless without a config change."""


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
        """Returns the raw JSON text of the model's response for a single
        attempt. Raises LLMRateLimitError, LLMAuthenticationError, or the
        base LLMError on any other failure."""
        ...

    def validate(self) -> None:
        """Confirms the configured credentials are actually accepted by the
        provider — cheap, must not consume completion tokens. Raises
        LLMAuthenticationError (or LLMError) if they aren't."""
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

    def validate(self) -> None:
        from groq import AuthenticationError

        try:
            # Listing models is metadata-only — costs no completion tokens,
            # unlike a real chat.completions.create() call, which is why
            # this (not a throwaway generation) is what startup validation
            # and the rate-limit-aware retry's "should I even bother
            # retrying" check both use.
            self._client.models.list()
        except AuthenticationError as exc:
            raise LLMAuthenticationError(f"Groq rejected the configured API key: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Could not validate Groq credentials: {exc}") from exc

    def generate_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        settings = get_settings()
        from groq import APIError, AuthenticationError, RateLimitError

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
        except RateLimitError as exc:
            retry_after = _parse_retry_after(exc)
            raise LLMRateLimitError(f"Groq rate limit hit: {exc}", retry_after=retry_after) from exc
        except AuthenticationError as exc:
            raise LLMAuthenticationError(f"Groq rejected the configured API key: {exc}") from exc
        except APIError as exc:
            raise LLMError(f"Groq API error: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - network/timeout/etc.
            raise LLMError(f"Groq request failed: {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise LLMError("Groq returned an empty response.")
        return content


def _parse_retry_after(exc: Exception) -> float | None:
    """Pull the provider's own suggested wait time off the HTTP response,
    when it supplies one, rather than guessing at a backoff duration that
    might be too short (retries again immediately) or too long (wastes
    time the provider didn't actually need)."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    header = response.headers.get("retry-after")
    if not header:
        return None
    try:
        return float(header)
    except (TypeError, ValueError):
        return None


def generate_json_with_backoff(
    client: LLMClient,
    system: str,
    user: str,
    *,
    max_retries: int,
) -> str:
    """Calls `client.generate_json` with provider-level resilience: a 429
    backs off (respecting the provider's Retry-After when given, else
    exponential) before retrying; any other LLMError retries immediately,
    since it's likely transient (network blip) rather than something waiting
    helps with. An LLMAuthenticationError is never worth retrying — the
    credentials aren't going to become valid between attempts — so it's
    raised immediately instead of burning the retry budget on it.

    This is deliberately separate from qa/generator.py's and
    qa/followup.py's own retry loops, which handle a different failure mode
    (the call succeeded but returned malformed/schema-invalid JSON) — this
    function only concerns itself with the provider call itself failing.
    """
    last_error: LLMError | None = None
    for attempt in range(max_retries + 1):
        try:
            return client.generate_json(system, user)
        except LLMAuthenticationError:
            raise
        except LLMRateLimitError as exc:
            last_error = exc
            if attempt < max_retries:
                delay = exc.retry_after if exc.retry_after is not None else min(2**attempt, 8.0)
                logger.warning(
                    "Rate limited by LLM provider; backing off %.1fs before retry %d/%d.",
                    delay,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(delay)
        except LLMError as exc:
            last_error = exc
            logger.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, max_retries + 1, exc)

    assert last_error is not None  # loop always executes at least once
    raise last_error


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
