"""Consistent error response contract for the public API. Same shape as
cv-parser/interview-qa/speech-io for consistency across Recruitix services."""

from __future__ import annotations

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str | None = None
