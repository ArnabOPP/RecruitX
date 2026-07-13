"""Consistent error response contract for the public API. See cv-parser's
errors.py for the full rationale — same shape here for consistency across
Recruitix services."""

from __future__ import annotations

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str | None = None
