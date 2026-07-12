"""Recruitix CV Parsing Service — FastAPI entrypoint.

Implements FR-08 ("Users upload a CV; the system parses it into structured
fields") from the Recruitix BRD. Output feeds the Round-2 CV-grounded
personal-interview question generator (FR-09).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware

from .config import get_settings
from .errors import ErrorResponse
from .logging_config import configure_logging, request_id_var
from .parser.extractor import (
    SUPPORTED_EXTENSIONS,
    EmptyDocumentError,
    FileSignatureMismatchError,
    UnsupportedFileTypeError,
)
from .parser.pipeline import parse_resume
from .parser.schemas import ParsedResume

settings = get_settings()
configure_logging(level=settings.log_level, json_logs=settings.log_json)
logger = logging.getLogger("cv_parser.api")

limiter = Limiter(key_func=get_remote_address, enabled=settings.rate_limit_enabled)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Preloading NER models (spaCy=%s, transformer=%s)...", settings.spacy_model, settings.transformer_model)
    from .parser.ner import get_spacy_pipeline, get_transformer_ner

    get_spacy_pipeline()
    transformer = get_transformer_ner()
    transformer._ensure_loaded()  # noqa: SLF001 - deliberate eager load at startup, not on first request
    app.state.ready = True
    app.state.transformer_available = transformer.available
    logger.info("Models loaded (transformer_available=%s). Service ready.", transformer.available)
    yield
    logger.info("Shutting down cv-parser service.")


app = FastAPI(
    title="Recruitix CV Parsing Service",
    description=(
        "spaCy NER + transformer (BERT-class) résumé parser. Extracts "
        "contact info, skills, education, experience, projects, and "
        "certifications from an uploaded CV."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.state.ready = False
app.state.transformer_available = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attaches a correlation ID to every request: generated if the caller
    didn't supply one, echoed back in the response, and made available to
    every log line emitted while handling the request (see logging_config).
    """

    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        token = request_id_var.set(req_id)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = req_id
        return response


app.add_middleware(RequestIdMiddleware)

if settings.metrics_enabled:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


# --- Error handling -------------------------------------------------------


def _error_response(status_code: int, error: str, detail: str) -> JSONResponse:
    body = ErrorResponse(error=error, detail=detail, request_id=request_id_var.get())
    return JSONResponse(status_code=status_code, content=body.model_dump())


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return _error_response(429, "rate_limited", f"Rate limit exceeded: {exc.detail}")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return _error_response(exc.status_code, "request_error", str(exc.detail))


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception for %s %s", request.method, request.url.path)
    return _error_response(500, "internal_error", "Internal server error.")


# --- Health / readiness ----------------------------------------------------


@app.get("/health/live", tags=["health"])
def liveness() -> dict:
    """Process is up and serving HTTP. Does not imply models are loaded."""
    return {"status": "ok"}


@app.get("/health/ready", tags=["health"])
def readiness(request: Request) -> JSONResponse:
    """Models are loaded and the service can actually parse a résumé."""
    ready = bool(request.app.state.ready)
    return JSONResponse(status_code=200 if ready else 503, content={"status": "ready" if ready else "loading"})


@app.get("/health", tags=["health"])
def health() -> dict:
    """Back-compat alias for /health/live."""
    return {"status": "ok"}


@app.get("/api/v1/capabilities", tags=["meta"])
def capabilities(request: Request) -> dict:
    return {
        "supported_file_types": sorted(SUPPORTED_EXTENSIONS),
        "spacy_model": settings.spacy_model,
        "transformer_model": settings.transformer_model,
        "transformer_enabled": settings.enable_transformer,
        "transformer_available": bool(request.app.state.transformer_available),
        "max_upload_bytes": settings.max_upload_bytes,
    }


# --- Parsing ----------------------------------------------------------------


@app.post(
    "/api/v1/parse",
    response_model=ParsedResume,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
    tags=["parsing"],
)
@limiter.limit(settings.rate_limit_parse)
async def parse_cv(request: Request, file: UploadFile = File(...)) -> ParsedResume:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({content_length} bytes). Limit is {settings.max_upload_bytes} bytes.",
        )

    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(data)} bytes). Limit is {settings.max_upload_bytes} bytes.",
        )
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    filename = file.filename
    try:
        # NER inference is CPU-bound; run it off the event loop so one slow
        # parse doesn't stall every other in-flight request on this worker.
        result = await asyncio.wait_for(
            run_in_threadpool(parse_resume, filename, data),
            timeout=settings.request_timeout_seconds,
        )
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except FileSignatureMismatchError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except EmptyDocumentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except TimeoutError as exc:
        logger.warning("Parse timed out after %ss for %s", settings.request_timeout_seconds, filename)
        raise HTTPException(status_code=504, detail="Parsing timed out.") from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error parsing %s", filename)
        raise HTTPException(status_code=500, detail="Internal parsing error.") from exc

    return result
