"""Recruitix Code Evaluation Service — FastAPI entrypoint.

Implements the "Code evaluation" row of the Recruitix BRD's AI/ML model
table: a sandboxed test-case runner plus static/complexity analysis,
grading coding-round submissions on correctness and efficiency. This is
the highest-risk service in Recruitix so far — it executes arbitrary
candidate-submitted code — so the sandbox isolation model is documented in
detail in sandbox/docker_runner.py and was verified empirically (memory
bombs get OOM-killed, network access is unreachable, filesystem writes
outside /tmp are blocked) before anything was built around it.
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
from .grading import grade_submission
from .logging_config import configure_logging, request_id_var
from .middleware import MaxBodySizeMiddleware
from .sandbox.docker_runner import SUPPORTED_LANGUAGES, SandboxError, get_sandbox_runner
from .schemas import EvaluateRequest, EvaluateResponse

settings = get_settings()
configure_logging(level=settings.log_level, json_logs=settings.log_json)
logger = logging.getLogger("code_eval.api")


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


def _warn_if_unprotected_in_production(settings) -> None:  # noqa: ANN001
    if settings.is_production and not settings.require_api_key:
        logger.warning(
            "SECURITY: environment=production but CODE_EVAL_REQUIRE_API_KEY is not set. "
            "This service executes arbitrary submitted code in sandboxed containers — leaving "
            "it unauthenticated exposes host compute resources to any caller who can reach it. "
            "Set CODE_EVAL_REQUIRE_API_KEY=1 and CODE_EVAL_API_KEYS before deploying this "
            "internet-reachable."
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _warn_if_unprotected_in_production(settings)

    logger.info("Validating sandbox (Docker reachable, language images present)...")
    try:
        runner = get_sandbox_runner()
        if settings.validate_docker_on_startup:
            await run_in_threadpool(runner.validate)
        app.state.ready = True
        logger.info("Sandbox ready. Service ready.")
    except SandboxError as exc:
        app.state.ready = False
        logger.error("Sandbox validation failed: %s", exc)
    yield
    logger.info("Shutting down code-eval service.")


app = FastAPI(
    title="Recruitix Code Evaluation Service",
    description=(
        "Sandboxed test-case runner plus static/complexity analysis, grading "
        "coding-round submissions on correctness and efficiency."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.state.ready = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(MaxBodySizeMiddleware, max_bytes=settings.max_request_body_bytes)


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
        "ready": bool(request.app.state.ready),
        "supported_languages": sorted(SUPPORTED_LANGUAGES),
        "sandbox_timeout_seconds": settings.sandbox_timeout_seconds,
        "sandbox_memory_mb": settings.sandbox_memory_mb,
        "max_test_cases": settings.max_test_cases,
    }


# --- Evaluation -----------------------------------------------------------


@app.post(
    "/api/v1/code/evaluate",
    response_model=EvaluateResponse,
    responses={
        401: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
    tags=["evaluation"],
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(settings.rate_limit_evaluate)
async def evaluate_endpoint(request: Request, body: EvaluateRequest) -> EvaluateResponse:
    if body.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported language {body.language!r}. Supported: {sorted(SUPPORTED_LANGUAGES)}.",
        )
    if not body.source_code.strip():
        raise HTTPException(status_code=422, detail="source_code must not be empty.")
    if len(body.source_code) > settings.max_source_code_chars:
        raise HTTPException(
            status_code=422, detail=f"source_code exceeds the {settings.max_source_code_chars}-character limit."
        )
    if len(body.test_cases) > settings.max_test_cases:
        raise HTTPException(status_code=422, detail=f"test_cases exceeds the {settings.max_test_cases}-case limit.")
    for tc in body.test_cases:
        if len(tc.input) > settings.max_stdin_chars or len(tc.expected_output) > settings.max_stdin_chars:
            raise HTTPException(
                status_code=422, detail=f"a test case's input/expected_output exceeds the {settings.max_stdin_chars}-character limit."
            )

    runner = get_sandbox_runner()
    try:
        # A submission with N test cases makes N+1 sequential sandboxed
        # runs (plus one baseline measurement when efficiency estimation
        # applies) — run_in_threadpool offloads the whole blocking chain so
        # one slow/timing-out submission doesn't stall the event loop for
        # every other in-flight request.
        result = await asyncio.wait_for(
            run_in_threadpool(grade_submission, body, runner),
            timeout=settings.request_timeout_seconds,
        )
    except SandboxError as exc:
        logger.error("Sandbox failure during evaluation: %s", exc)
        raise HTTPException(status_code=502, detail="The sandbox could not execute this submission.") from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Evaluation timed out.") from exc

    return result
