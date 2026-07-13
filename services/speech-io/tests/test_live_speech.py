"""Real end-to-end tests against the actual Groq Whisper API and the real
edge-tts service — not mocked. A mocked test can prove our request/response
plumbing works, but can never prove real audio round-trips correctly
through synthesis and transcription. Skips the Groq-dependent tests cleanly
if no API key is configured (e.g. in CI); the edge-tts tests need no
credential but do need real network access.
"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.stt.client import GroqWhisperClient
from app.tts.client import EdgeTTSClient


def _has_groq_key() -> bool:
    return bool(get_settings().groq_api_key)


def test_live_tts_produces_real_audio():
    client = EdgeTTSClient(default_voice="en-US-AriaNeural")
    audio = client.synthesize("This is a test of the speech synthesis system.")
    assert len(audio) > 1000
    # MP3 frames either start with an ID3 tag or an MPEG sync word (0xFFEx).
    assert audio[:3] == b"ID3" or (audio[0] == 0xFF and (audio[1] & 0xE0) == 0xE0)


def test_live_tts_validate_succeeds():
    client = EdgeTTSClient(default_voice="en-US-AriaNeural")
    client.validate()  # must not raise


@pytest.mark.skipif(not _has_groq_key(), reason="No SPEECH_IO_GROQ_API_KEY configured")
def test_live_stt_validate_succeeds_with_real_key():
    settings = get_settings()
    client = GroqWhisperClient(
        api_key=settings.groq_api_key, model=settings.stt_model, timeout_seconds=settings.stt_request_timeout_seconds
    )
    client.validate()  # must not raise


@pytest.mark.skipif(not _has_groq_key(), reason="No SPEECH_IO_GROQ_API_KEY configured")
def test_live_round_trip_synthesize_then_transcribe_matches_original_text():
    """The core proof: text -> real neural speech -> real Whisper
    transcription should recover (approximately) the original words. This
    is the actual mechanism the voice-based HR round depends on."""
    settings = get_settings()
    tts_client = EdgeTTSClient(default_voice="en-US-AriaNeural")
    stt_client = GroqWhisperClient(
        api_key=settings.groq_api_key, model=settings.stt_model, timeout_seconds=settings.stt_request_timeout_seconds
    )

    original_text = "The quick brown fox jumps over the lazy dog near the riverbank."
    audio = tts_client.synthesize(original_text)
    transcribed = stt_client.transcribe(audio, "speech.mp3")

    # Whisper won't reproduce punctuation/casing identically, so compare on
    # a normalized bag of significant words rather than exact string match.
    original_words = {w.strip(".,!?").lower() for w in original_text.split()}
    transcribed_words = {w.strip(".,!?").lower() for w in transcribed.split()}
    overlap = original_words & transcribed_words
    assert len(overlap) >= len(original_words) * 0.8, (
        f"expected most words to survive the round trip; original={original_words} transcribed={transcribed_words}"
    )
