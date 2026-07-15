"""Consistent error response contract for the public API. Same shape as
the other Recruitix services."""

from __future__ import annotations

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str | None = None
