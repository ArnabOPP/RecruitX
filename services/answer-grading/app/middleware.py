"""ASGI middleware enforcing a hard cap on inbound request body size.

Same purpose as interview-qa/speech-io's version: rejects an oversized
body via Content-Length before FastAPI ever reads/parses it. Every
endpoint here is JSON-only (no file uploads), so this applies uniformly —
unlike speech-io there's no exempt_paths need.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from .errors import ErrorResponse
from .logging_config import request_id_var

_BODY_BEARING_METHODS = {"POST", "PUT", "PATCH"}


def _size_error(status_code: int, detail: str) -> JSONResponse:
    body = ErrorResponse(error="request_error", detail=detail, request_id=request_id_var.get())
    return JSONResponse(status_code=status_code, content=body.model_dump())


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_bytes: int) -> None:  # noqa: ANN001
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        if request.method in _BODY_BEARING_METHODS:
            content_length = request.headers.get("content-length")
            if content_length is None:
                return _size_error(411, "Content-Length header is required.")
            try:
                length = int(content_length)
            except ValueError:
                return _size_error(400, "Invalid Content-Length header.")
            if length > self.max_bytes:
                return _size_error(413, f"Request body exceeds the {self.max_bytes}-byte limit.")
        return await call_next(request)
