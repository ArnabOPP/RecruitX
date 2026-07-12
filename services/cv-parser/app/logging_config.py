"""Structured (JSON) logging with per-request correlation IDs.

Two things production log aggregation (CloudWatch, Datadog, Loki, ...) needs
that Python's default logging doesn't give you for free: machine-parseable
output, and a way to tie every log line from one HTTP request together. We
solve the second with a contextvar the request-ID middleware populates, and
a logging filter that injects it into every record — no need to thread a
request ID through every function call by hand.

Résumés are personal data, so request/response bodies are deliberately never
logged here — only metadata (filename, size, status code, latency).
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_RequestIdFilter())
    if json_logs:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"
            )
        )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [handler]

    # Keep noisy third-party libraries at WARNING unless we're in DEBUG.
    if level != "DEBUG":
        for noisy in ("uvicorn.access", "httpx", "transformers", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
