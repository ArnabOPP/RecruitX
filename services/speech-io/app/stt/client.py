"""STT (speech-to-text) provider abstraction.

The BRD names "Whisper-class STT" as the requirement — nothing about the
design should hardcode one provider or deployment mode. `STTClient` is the
seam: callers depend only on `transcribe`, `validate`, and
`transcribe_with_backoff`, never on a provider SDK directly, so adding a
self-hosted Whisper fallback later means registering a new class in
`get_stt_client()`, not touching the endpoint that calls it.

Groq is the concrete choice today: it hosts whisper-large-v3, is fast, and
reuses the same API key already configured for interview-qa's LLM calls —
one provider account for the whole voice pipeline. This mirrors
interview-qa's llm/client.py almost exactly, including the rate-limit
backoff and auth-vs-transient error distinction, since Groq's failure modes
are the same across its APIs.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Protocol

from ..config import get_settings

logger = logging.getLogger("speech_io.stt")


class STTError(Exception):
    """Raised for any provider-level failure (auth, network, malformed
    response) — callers handle one exception type regardless of which
    provider is configured."""


class STTRateLimitError(STTError):
    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class STTAuthenticationError(STTError):
    """The provider rejected the credentials outright — retrying is
    pointless without a config change."""


class STTClient(Protocol):
    model_name: str

    def transcribe(self, audio_bytes: bytes, filename: str, language: str | None = None) -> str:
        """Returns the transcribed text for a single attempt. Raises
        STTRateLimitError, STTAuthenticationError, or the base STTError on
        any other failure."""
        ...

    def validate(self) -> None:
        """Confirms the configured credentials are actually accepted by the
        provider — cheap, must not consume real transcription quota."""
        ...


class GroqWhisperClient:
    def __init__(self, api_key: str, model: str, timeout_seconds: float) -> None:
        if not api_key:
            raise STTError("No Groq API key configured — set SPEECH_IO_GROQ_API_KEY.")
        from groq import Groq

        self._client = Groq(api_key=api_key, timeout=timeout_seconds)
        self.model_name = model

    def validate(self) -> None:
        from groq import AuthenticationError

        try:
            # Same trick as interview-qa: listing models is metadata-only,
            # costs nothing, and confirms the key actually works.
            self._client.models.list()
        except AuthenticationError as exc:
            raise STTAuthenticationError(f"Groq rejected the configured API key: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise STTError(f"Could not validate Groq credentials: {exc}") from exc

    def transcribe(self, audio_bytes: bytes, filename: str, language: str | None = None) -> str:
        from groq import APIError, AuthenticationError, RateLimitError

        try:
            kwargs: dict = {
                "model": self.model_name,
                "file": (filename, audio_bytes),
                "response_format": "text",
            }
            if language:
                kwargs["language"] = language
            result = self._client.audio.transcriptions.create(**kwargs)
        except RateLimitError as exc:
            retry_after = _parse_retry_after(exc)
            raise STTRateLimitError(f"Groq rate limit hit: {exc}", retry_after=retry_after) from exc
        except AuthenticationError as exc:
            raise STTAuthenticationError(f"Groq rejected the configured API key: {exc}") from exc
        except APIError as exc:
            raise STTError(f"Groq API error: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - network/timeout/etc.
            raise STTError(f"Groq transcription request failed: {exc}") from exc

        # response_format="text" returns a plain string; the SDK still
        # wraps some responses in an object depending on version, so handle
        # both shapes defensively.
        text = result if isinstance(result, str) else getattr(result, "text", None)
        if not text or not text.strip():
            raise STTError("Groq returned an empty transcription.")
        return text.strip()


def _parse_retry_after(exc: Exception) -> float | None:
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


def transcribe_with_backoff(
    client: STTClient,
    audio_bytes: bytes,
    filename: str,
    *,
    language: str | None,
    max_retries: int,
) -> str:
    """Same resilience pattern as interview-qa's generate_json_with_backoff:
    a 429 backs off (respecting Retry-After when given, else exponential)
    before retrying; any other STTError retries immediately since it's
    likely transient; an STTAuthenticationError is never retried."""
    last_error: STTError | None = None
    for attempt in range(max_retries + 1):
        try:
            return client.transcribe(audio_bytes, filename, language)
        except STTAuthenticationError:
            raise
        except STTRateLimitError as exc:
            last_error = exc
            if attempt < max_retries:
                delay = exc.retry_after if exc.retry_after is not None else min(2**attempt, 8.0)
                logger.warning(
                    "Rate limited by STT provider; backing off %.1fs before retry %d/%d.",
                    delay,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(delay)
        except STTError as exc:
            last_error = exc
            logger.warning("STT call failed (attempt %d/%d): %s", attempt + 1, max_retries + 1, exc)

    assert last_error is not None
    raise last_error


def _build_client(settings) -> STTClient:  # noqa: ANN001
    if settings.stt_provider == "groq":
        return GroqWhisperClient(
            api_key=settings.groq_api_key,
            model=settings.stt_model,
            timeout_seconds=settings.stt_request_timeout_seconds,
        )
    raise STTError(f"Unsupported STT provider: {settings.stt_provider!r}")


@lru_cache(maxsize=1)
def get_stt_client() -> STTClient:
    return _build_client(get_settings())
