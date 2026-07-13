"""Fake STT/TTS clients for fast, deterministic unit tests that don't hit
real providers — see test_live_speech.py for the tests that do."""

from __future__ import annotations

from app.stt.client import STTAuthenticationError, STTError, STTRateLimitError
from app.tts.client import TTSError


class FakeSTTClient:
    model_name = "fake-whisper"

    def __init__(self, responses: list[str] | None = None):
        self._responses = list(responses or [])
        self.calls: list[tuple[bytes, str]] = []

    def transcribe(self, audio_bytes: bytes, filename: str, language: str | None = None) -> str:
        self.calls.append((audio_bytes, filename))
        if not self._responses:
            raise STTError("FakeSTTClient has no scripted responses left.")
        return self._responses.pop(0)

    def validate(self) -> None:
        pass


class AlwaysFailingSTTClient:
    model_name = "fake-whisper"

    def transcribe(self, audio_bytes: bytes, filename: str, language: str | None = None) -> str:
        raise STTError("simulated provider failure")

    def validate(self) -> None:
        pass


class AlwaysRateLimitedSTTClient:
    model_name = "fake-whisper"

    def __init__(self, retry_after: float | None = 0.01) -> None:
        self.retry_after = retry_after
        self.call_count = 0

    def transcribe(self, audio_bytes: bytes, filename: str, language: str | None = None) -> str:
        self.call_count += 1
        raise STTRateLimitError("simulated rate limit", retry_after=self.retry_after)

    def validate(self) -> None:
        pass


class AlwaysUnauthenticatedSTTClient:
    model_name = "fake-whisper"

    def __init__(self) -> None:
        self.call_count = 0

    def transcribe(self, audio_bytes: bytes, filename: str, language: str | None = None) -> str:
        self.call_count += 1
        raise STTAuthenticationError("simulated bad credentials")

    def validate(self) -> None:
        raise STTAuthenticationError("simulated bad credentials")


class FakeTTSClient:
    def __init__(self, audio: bytes = b"fake-mp3-bytes"):
        self._audio = audio
        self.calls: list[tuple[str, str | None]] = []

    def synthesize(self, text: str, voice: str | None = None) -> bytes:
        self.calls.append((text, voice))
        return self._audio

    def validate(self) -> None:
        pass


class AlwaysFailingTTSClient:
    def synthesize(self, text: str, voice: str | None = None) -> bytes:
        raise TTSError("simulated provider failure")

    def validate(self) -> None:
        raise TTSError("simulated provider failure")
