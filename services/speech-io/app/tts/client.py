"""TTS (text-to-speech) provider abstraction.

The BRD names "neural TTS (Azure / ElevenLabs / Coqui)" as the requirement.
`TTSClient` is the seam: callers depend only on `synthesize` and
`validate`, never on a provider SDK directly, so Azure Neural TTS or Coqui
can be registered in `get_tts_client()`'s factory later without touching
the endpoint that calls it.

edge-tts is the concrete choice today: it's an open-source wrapper around
the same neural voices Microsoft Edge's "Read Aloud" feature uses, free,
requires no API key or account, and has no published rate limit. It is
*unofficial* — a reverse-engineered client against a Microsoft service, not
a published/supported API — which is exactly why this is built as a
swappable provider from day one rather than called directly from the
endpoint: if Microsoft ever changes something that breaks it, swapping to
Azure Neural TTS (the officially-supported version of the same voice
engine) is a new class here, not a rewrite of the service.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Protocol

from ..config import get_settings

logger = logging.getLogger("speech_io.tts")


class TTSError(Exception):
    """Raised for any provider-level failure (network, invalid voice,
    malformed response) — callers handle one exception type regardless of
    which provider is configured."""


class TTSClient(Protocol):
    def synthesize(self, text: str, voice: str | None = None) -> bytes:
        """Returns synthesized audio bytes (mp3) for the given text. Raises
        TTSError on any failure."""
        ...

    def validate(self) -> None:
        """Confirms the provider is reachable — cheap. Raises TTSError if
        it isn't."""
        ...


class EdgeTTSClient:
    def __init__(self, default_voice: str) -> None:
        self.default_voice = default_voice

    def synthesize(self, text: str, voice: str | None = None) -> bytes:
        import edge_tts

        try:
            communicate = edge_tts.Communicate(text, voice or self.default_voice)
            chunks = bytearray()
            for chunk in communicate.stream_sync():
                if chunk["type"] == "audio":
                    chunks.extend(chunk["data"])
        except Exception as exc:  # noqa: BLE001 - network/protocol errors from the unofficial API
            raise TTSError(f"edge-tts synthesis failed: {exc}") from exc

        if not chunks:
            raise TTSError("edge-tts returned no audio data.")
        return bytes(chunks)

    def validate(self) -> None:
        # A cheap, real synthesis call is the only reliable way to confirm
        # this unofficial API is currently reachable — there's no
        # lightweight metadata endpoint like Groq's models.list(). The text
        # is intentionally tiny to keep this fast.
        try:
            self.synthesize("test", self.default_voice)
        except TTSError as exc:
            raise TTSError(f"Could not validate edge-tts reachability: {exc}") from exc


def _build_client(settings) -> TTSClient:  # noqa: ANN001
    if settings.tts_provider == "edge":
        return EdgeTTSClient(default_voice=settings.tts_default_voice)
    raise TTSError(f"Unsupported TTS provider: {settings.tts_provider!r}")


@lru_cache(maxsize=1)
def get_tts_client() -> TTSClient:
    return _build_client(get_settings())
