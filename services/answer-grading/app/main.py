"""Recruitix Answer Grading Service — FastAPI entrypoint.

Implements the "Answer grading" row of the Recruitix BRD's AI/ML model
table: a deterministic engine (Jaccard / semantic similarity + keyword
weighting + rubric) producing reproducible, auditable scores for open text
and transcripts. This is the counterpart to interview-qa's own design
principle — "LLMs propose and converse; the deterministic engine decides
the score" — nothing in this service calls an LLM. It consumes a question
(optionally with interview-qa's `grounding` evidence) and a candidate
answer (typed or transcribed via speech-io — this service doesn't care
which), and returns a score with a full breakdown of why.
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
from .grading.scorer import score_answer
from .grading.semantic import SemanticSimilarityError, get_semantic_scorer
from .logging_config import configure_logging, request_id_var
from .middleware import MaxBodySizeMiddleware
from .schemas import ScoreRequest, ScoreResponse

settings = get_settings()
configure_logging(level=settings.log_level, json_logs=settings.log_json)
logger = logging.getLogger("answer_grading.api")


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
            "SECURITY: environment=production but ANSWER_GRADING_REQUIRE_API_KEY is not set. "
            "This service's endpoints are unauthenticated and its CPU resources are exposed to "
            "any caller who can reach them. Set ANSWER_GRADING_REQUIRE_API_KEY=1 and "
            "ANSWER_GRADING_API_KEYS before deploying this internet-reachable."
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _warn_if_unprotected_in_production(settings)

    logger.info("Loading semantic similarity model (%s)...", settings.semantic_model_name)
    try:
        scorer = get_semantic_scorer()
        if settings.validate_model_on_startup:
            # Runs one real (tiny) inference — confirms the model actually
            # works, not just that it loaded, and warms it up so the first
            # real request isn't the one paying model-warmup latency.
            scorer.validate()
        app.state.ready = True
        logger.info("Semantic model ready (%s). Service ready.", scorer.model_name)
    except SemanticSimilarityError as exc:
        app.state.ready = False
        logger.error("Semantic model failed to load: %s", exc)
    yield
    logger.info("Shutting down answer-grading service.")


app = FastAPI(
    title="Recruitix Answer Grading Service",
    description=(
        "Deterministic, reproducible scoring of open-text interview answers "
        "and transcripts: Jaccard + keyword weighting + semantic similarity "
        "against a rubric. Never calls an LLM — the score is always the "
        "same for the same input."
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
        "semantic_model": settings.semantic_model_name,
        "ready": bool(request.app.state.ready),
        "jaccard_weight": settings.jaccard_weight,
        "semantic_weight": settings.semantic_weight,
        "max_question_chars": settings.max_question_chars,
        "max_answer_chars": settings.max_answer_chars,
    }


# --- Grading -----------------------------------------------------------


@app.post(
    "/api/v1/grading/score",
    response_model=ScoreResponse,
    responses={
        401: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
    tags=["grading"],
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(settings.rate_limit_score)
async def score_endpoint(request: Request, body: ScoreRequest) -> ScoreResponse:
    if not body.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty.")
    if len(body.question) > settings.max_question_chars:
        raise HTTPException(status_code=422, detail=f"question exceeds the {settings.max_question_chars}-character limit.")
    if not body.candidate_answer.strip():
        raise HTTPException(status_code=422, detail="candidate_answer must not be empty.")
    if len(body.candidate_answer) > settings.max_answer_chars:
        raise HTTPException(
            status_code=422, detail=f"candidate_answer exceeds the {settings.max_answer_chars}-character limit."
        )
    if body.rubric is not None:
        if len(body.rubric.criteria) > settings.max_rubric_criteria:
            raise HTTPException(
                status_code=422, detail=f"rubric exceeds the {settings.max_rubric_criteria}-criterion limit."
            )
        for criterion in body.rubric.criteria:
            if len(criterion.expected_keywords) > settings.max_keywords_per_criterion:
                raise HTTPException(
                    status_code=422,
                    detail=f"a rubric criterion exceeds the {settings.max_keywords_per_criterion}-keyword limit.",
                )

    scorer = get_semantic_scorer()
    try:
        result = await asyncio.wait_for(
            run_in_threadpool(score_answer, body.question, body.candidate_answer, body.grounding, body.rubric, scorer),
            timeout=settings.request_timeout_seconds,
        )
    except SemanticSimilarityError as exc:
        logger.warning("Scoring failed: %s", exc)
        raise HTTPException(status_code=502, detail="The semantic similarity model could not score this answer.") from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Scoring timed out.") from exc

    return result
