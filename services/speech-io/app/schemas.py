"""Data contracts for the speech-io API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TranscribeResponse(BaseModel):
    text: str
    language: str | None = None
    model_used: str


class SynthesizeRequest(BaseModel):
    text: str
    voice: str | None = Field(
        default=None,
        description="Edge neural voice name, e.g. 'en-US-AriaNeural'. Falls back to the service default if omitted.",
    )
