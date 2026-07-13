"""Recruitix Interview Q&A Generation Service — FastAPI entrypoint.

Implements the "Interview Q&A generation" row of the Recruitix BRD's AI/ML
model table: an instruction-tuned LLM that generates CV-grounded personal &
HR questions and follow-ups, consuming cv-parser's résumé output. Grading is
explicitly out of scope here — per the BRD's design principle, "LLMs
propose and converse; the deterministic engine decides the score."
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware

from .auth import verify_api_key
from .config import get_settings
from .errors import ErrorResponse
from .llm.client import LLMAuthenticationError, LLMError, get_llm_client
from .logging_config import configure_logging, request_id_var
from .qa.followup import FollowUpGenerationError, generate_followup
from .qa.generator import QuestionGenerationError, generate_questions
from .qa.schemas import (
    FollowUpRequest,
    FollowUpResponse,
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
)

settings = get_settings()
configure_logging(level=settings.log_level, json_logs=settings.log_json)
logger = logging.getLogger("interview_qa.api")


def _redact_uri(uri: str) -> str:
    return re.sub(r"//[^@/]*@", "//***@", uri)


limiter = Limiter(
    key_func=get_remote_address,
    enabled=settings.rate_limit_enabled,
    storage_uri=settings.rate_limit_storage_uri,
    in_memory_fallback_enabled=True,
)
if settings.rate_limit_storage_uri:
    logger.info("Rate limiter using shared storage: %s", _redact_uri(settings.rate_limit_storage_uri))
else:
    logger.info("Rate limiter using in-memory storage (single-instance only).")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Validating LLM provider configuration (provider=%s)...", settings.llm_provider)
    try:
        client = get_llm_client()
        if settings.validate_key_on_startup:
            # A non-empty key isn't the same as a *working* key — this makes
            # one cheap, token-free call to actually confirm the provider
            # accepts it, so readiness means "will actually work" rather
            # than just "something is configured".
            client.validate()
            logger.info("LLM credentials confirmed valid by provider.")
        app.state.ready = True
        app.state.llm_model = client.model_name
        logger.info("LLM client ready (model=%s). Service ready.", client.model_name)
    except LLMAuthenticationError as exc:
        app.state.ready = False
        app.state.llm_model = None
        logger.error("LLM provider rejected the configured credentials: %s", exc)
    except LLMError as exc:
        # Fail loud in logs but don't crash the process — /health/ready
        # will correctly report 503 until this is fixed (e.g. a missing
        # API key gets set via a config reload/restart), which is more
        # operable than the container immediately crash-looping.
        app.state.ready = False
        app.state.llm_model = None
        logger.error("LLM client failed to initialize: %s", exc)
    yield
    logger.info("Shutting down interview-qa service.")


app = FastAPI(
    title="Recruitix Interview Q&A Generation Service",
    description=(
        "Generates CV-grounded personal & HR interview questions and "
        "follow-ups using an instruction-tuned LLM. Consumes cv-parser's "
        "résumé output; never grades answers."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.state.ready = False
app.state.llm_model = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class RequestIdMiddleware(BaseHTTPMiddleware):
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


# --- Error handling -----------------------------------------------------------


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


# --- Health / readiness --------------------------------------------------------


@app.get("/health/live", tags=["health"])
def liveness() -> dict:
    return {"status": "ok"}


@app.get("/health/ready", tags=["health"])
def readiness(request: Request) -> JSONResponse:
    ready = bool(request.app.state.ready)
    return JSONResponse(status_code=200 if ready else 503, content={"status": "ready" if ready else "not_ready"})


@app.get("/health", tags=["health"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/v1/capabilities", tags=["meta"])
def capabilities(request: Request) -> dict:
    return {
        "llm_provider": settings.llm_provider,
        "llm_model": request.app.state.llm_model,
        "llm_ready": bool(request.app.state.ready),
        "max_questions_per_request": settings.max_questions_per_request,
        "default_questions_per_request": settings.default_questions_per_request,
    }


# --- Question generation -----------------------------------------------------


@app.post(
    "/api/v1/questions/generate",
    response_model=GenerateQuestionsResponse,
    responses={
        401: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
    tags=["questions"],
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(settings.rate_limit_generate)
async def generate_questions_endpoint(request: Request, body: GenerateQuestionsRequest) -> GenerateQuestionsResponse:
    if body.count < 1 or body.count > settings.max_questions_per_request:
        raise HTTPException(
            status_code=422,
            detail=f"count must be between 1 and {settings.max_questions_per_request}.",
        )
    try:
        return await asyncio.wait_for(
            run_in_threadpool(generate_questions, body),
            timeout=settings.request_timeout_seconds,
        )
    except QuestionGenerationError as exc:
        logger.warning("Question generation failed: %s", exc)
        raise HTTPException(status_code=502, detail="The question generator could not produce a valid response.") from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Question generation timed out.") from exc


@app.post(
    "/api/v1/questions/followup",
    response_model=FollowUpResponse,
    responses={
        401: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
    tags=["questions"],
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(settings.rate_limit_generate)
async def generate_followup_endpoint(request: Request, body: FollowUpRequest) -> FollowUpResponse:
    try:
        return await asyncio.wait_for(
            run_in_threadpool(generate_followup, body),
            timeout=settings.request_timeout_seconds,
        )
    except FollowUpGenerationError as exc:
        logger.warning("Follow-up generation failed: %s", exc)
        raise HTTPException(status_code=502, detail="The follow-up generator could not produce a valid response.") from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Follow-up generation timed out.") from exc
