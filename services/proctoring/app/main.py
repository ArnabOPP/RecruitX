"""Recruitix Proctoring Service — FastAPI entrypoint.

Implements the "Proctoring" row of the Recruitix BRD's AI/ML model table:
MediaPipe gaze/head-pose + event-driven severity scoring, detecting
integrity events during an interview session and feeding an integrity
summary (for the orchestrator to fold into the final report).

Same core trust principle as biometric-auth: the client never gets to
self-report an integrity event. Every snapshot frame is a real image,
re-detected and re-measured by this service — never a client-asserted
"looking_away: true" boolean.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Path, Request, UploadFile
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
from .logging_config import configure_logging, request_id_var
from .middleware import MaxBodySizeMiddleware
from .schemas import DeleteSessionResponse, SnapshotResponse, SummaryResponse
from .session_store import SessionNotFoundError, get_session_store
from .severity import analyze_frame
from .vision.detector import get_face_detector

settings = get_settings()
configure_logging(level=settings.log_level, json_logs=settings.log_json)
logger = logging.getLogger("proctoring.api")


def _redact_uri(uri: str) -> str:
    return re.sub(r"//[^@/]*@", "//***@", uri)


limiter = Limiter(
    key_func=get_remote_address,
    enabled=settings.rate_limit_enabled,
    storage_uri=settings.rate_limit_storage_uri,
    in_memory_fallback_enabled=True,
)


def _warn_if_unprotected_in_production(settings) -> None:  # noqa: ANN001
    if settings.is_production and not settings.require_api_key:
        logger.warning(
            "SECURITY: environment=production but PROCTORING_REQUIRE_API_KEY is not set. "
            "This service holds per-candidate integrity scores and event history — an "
            "unauthenticated summary endpoint would let anyone read another candidate's "
            "proctoring record. Set PROCTORING_REQUIRE_API_KEY=1 and PROCTORING_API_KEYS "
            "before deploying this internet-reachable."
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _warn_if_unprotected_in_production(settings)

    logger.info("Loading vision model (FaceLandmarker) and validating Redis reachability...")
    try:
        get_face_detector()
        store = get_session_store()
        await store.ping()
        app.state.ready = True
        logger.info("Model loaded, Redis reachable. Service ready.")
    except Exception as exc:  # noqa: BLE001
        app.state.ready = False
        logger.error("Startup validation failed: %s", exc)
    yield
    logger.info("Shutting down proctoring service.")


app = FastAPI(
    title="Recruitix Proctoring Service",
    description=(
        "MediaPipe gaze/head-pose + event-driven severity scoring for continuous "
        "interview proctoring. Every integrity event is computed server-side from "
        "a real submitted snapshot frame — never a client-reported boolean."
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
    allow_methods=["GET", "POST", "DELETE"],
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


@app.exception_handler(SessionNotFoundError)
async def session_not_found_handler(request: Request, exc: SessionNotFoundError) -> JSONResponse:
    return _error_response(404, "not_found", f"No proctoring session found with id {exc}.")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception for %s %s", request.method, request.url.path)
    return _error_response(500, "internal_error", "Internal server error.")


# --- Health / readiness --------------------------------------------------------


@app.get("/health/live", tags=["health"])
def liveness_probe() -> dict:
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
        "head_turn_threshold_degrees": settings.head_turn_threshold_degrees,
        "gaze_offset_threshold": settings.gaze_offset_threshold,
        "consecutive_frames_to_flag": settings.consecutive_frames_to_flag,
    }


# --- Helpers -----------------------------------------------------------


def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=422, detail="Snapshot image could not be decoded — not a valid image file.")
    return image


async def _read_and_validate_image(file: UploadFile) -> np.ndarray:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="Snapshot image is empty.")
    if len(data) > settings.max_image_bytes:
        raise HTTPException(status_code=413, detail=f"Snapshot image exceeds the {settings.max_image_bytes}-byte limit.")
    return _decode_image(data)


# --- Proctoring -----------------------------------------------------------


@app.post(
    "/api/v1/proctoring/{session_id}/snapshot",
    response_model=SnapshotResponse,
    responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
    tags=["proctoring"],
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(settings.rate_limit_default)
async def snapshot_endpoint(request: Request, session_id: str = Path(...), file: UploadFile = File(...)) -> SnapshotResponse:
    image = await _read_and_validate_image(file)

    store = get_session_store()
    state = await store.get_or_create(session_id)

    detector = get_face_detector()
    analysis = await run_in_threadpool(analyze_frame, detector, image, state)

    await store.save(session_id, state)

    return SnapshotResponse(
        session_id=session_id,
        faces_detected=analysis.faces_detected,
        head_pose_deviation_degrees=(
            round(analysis.head_pose_deviation_degrees, 2) if analysis.head_pose_deviation_degrees is not None else None
        ),
        gaze_offset=(round(analysis.gaze_offset, 3) if analysis.gaze_offset is not None else None),
        flagged_this_frame=analysis.flagged_this_frame,
        events_recorded=analysis.events_recorded,
        integrity_score=analysis.integrity_score,
        frames_processed=state["frames_processed"],
    )


@app.get(
    "/api/v1/proctoring/{session_id}/summary",
    response_model=SummaryResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    tags=["proctoring"],
    dependencies=[Depends(verify_api_key)],
)
async def summary_endpoint(session_id: str = Path(...)) -> SummaryResponse:
    store = get_session_store()
    state = await store.load(session_id)
    return SummaryResponse(
        session_id=session_id,
        frames_processed=state["frames_processed"],
        integrity_score=state["integrity_score"],
        event_counts=state["event_counts"],
        events=state["events"],
    )


@app.delete(
    "/api/v1/proctoring/{session_id}",
    response_model=DeleteSessionResponse,
    responses={401: {"model": ErrorResponse}},
    tags=["proctoring"],
    dependencies=[Depends(verify_api_key)],
)
async def delete_session_endpoint(session_id: str = Path(...)) -> DeleteSessionResponse:
    store = get_session_store()
    await store.delete(session_id)
    return DeleteSessionResponse(session_id=session_id, deleted=True)
