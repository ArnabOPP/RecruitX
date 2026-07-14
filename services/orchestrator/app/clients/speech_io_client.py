from __future__ import annotations

import httpx

from ..config import get_settings
from .base import call_bytes, call_json


async def synthesize(text: str, voice: str | None = None) -> bytes:
    settings = get_settings()
    headers = {"X-API-Key": settings.speech_io_api_key} if settings.speech_io_api_key else {}
    payload: dict = {"text": text}
    if voice:
        payload["voice"] = voice
    async with httpx.AsyncClient(
        base_url=settings.speech_io_base_url, timeout=settings.downstream_timeout_seconds
    ) as client:
        return await call_bytes(
            client, "speech-io", "POST", "/api/v1/speech/synthesize", json=payload, headers=headers
        )


async def transcribe(audio_bytes: bytes, filename: str) -> dict:
    settings = get_settings()
    headers = {"X-API-Key": settings.speech_io_api_key} if settings.speech_io_api_key else {}
    async with httpx.AsyncClient(
        base_url=settings.speech_io_base_url, timeout=settings.downstream_timeout_seconds
    ) as client:
        files = {"file": (filename, audio_bytes)}
        return await call_json(
            client, "speech-io", "POST", "/api/v1/speech/transcribe", files=files, headers=headers
        )


async def is_ready() -> bool:
    settings = get_settings()
    try:
        async with httpx.AsyncClient(base_url=settings.speech_io_base_url, timeout=5.0) as client:
            resp = await client.get("/health/live")
            return resp.status_code == 200
    except httpx.RequestError:
        return False
