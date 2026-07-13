"""Tests for the TTS client abstraction — the fakes exercise the same
Protocol surface real EdgeTTSClient does, so callers (main.py) can't tell
the difference. EdgeTTSClient itself is only exercised against the real
network in test_live_speech.py, since there's nothing meaningful to unit
test about a thin wrapper over an external, unofficial API."""

from __future__ import annotations

import pytest

from app.tts.client import TTSError
from tests.fakes import AlwaysFailingTTSClient, FakeTTSClient


def test_fake_tts_client_returns_scripted_audio():
    client = FakeTTSClient(audio=b"some-mp3-bytes")
    result = client.synthesize("hello", voice="en-US-AriaNeural")
    assert result == b"some-mp3-bytes"
    assert client.calls == [("hello", "en-US-AriaNeural")]


def test_failing_tts_client_raises_tts_error():
    client = AlwaysFailingTTSClient()
    with pytest.raises(TTSError):
        client.synthesize("hello")
