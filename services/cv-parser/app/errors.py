"""Consistent error response contract for the public API.

Every error the API returns — validation failure, unsupported file, internal
exception — comes back in this shape, so a client never has to special-case
FastAPI's default `{"detail": ...}` for some errors and something else for
others. Internal exception messages/tracebacks are never put in `detail`;
they go to the (non-PII) logs, keyed by `request_id`, and the client gets a
generic message plus the ID to quote when filing a support request.
"""

from __future__ import annotations

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: str | None = None
