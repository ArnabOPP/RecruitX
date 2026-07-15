"""Recruitix Biometric Auth Service — FastAPI entrypoint.

Implements three rows of the BRD's AI/ML model table: face detection &
landmarks (MediaPipe FaceMesh + OpenCV alignment), face recognition
(ArcFace embeddings, cosine-similarity matching), and liveness/anti-
spoofing (EAR blink detection + head-pose PnP heuristics). All three are
grouped into one service because they're used together at the same two
checkpoints — enrollment and login — and share the same detection code
(see app/vision/).

Core design principle, carried over from the rest of Recruitix's "the
deterministic engine decides" philosophy: the client never gets to
self-report a result that gates a real decision. Every endpoint here
receives actual image frames and computes its own verdict — never a
client-asserted "match: true" or "isLive: true" boolean, since either
would be trivially forgeable with a raw HTTP request that never touched a
camera.
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
from .db import EnrollmentNotFoundError, get_embedding_store
from .errors import ErrorResponse
from .logging_config import configure_logging, request_id_var
from .middleware import MaxBodySizeMiddleware
from .schemas import DeleteEnrollmentResponse, EnrollResponse, LivenessResponse, VerifyResponse
from .vision.detector import (
    FaceDetector,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
    get_face_detector,
)
from .vision.embedding import ArcFaceEmbedder, EmbeddingError, cosine_similarity, get_embedder
from .vision.liveness import analyze_liveness

settings = get_settings()
configure_logging(level=settings.log_level, json_logs=settings.log_json)
logger = logging.getLogger("biometric_auth.api")


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
            "SECURITY: environment=production but BIOMETRIC_AUTH_REQUIRE_API_KEY is not set. "
            "This service holds enrolled biometric reference data and an unauthenticated "
            "/verify endpoint would let anyone probe whether a face matches a candidate_id. "
            "Set BIOMETRIC_AUTH_REQUIRE_API_KEY=1 and BIOMETRIC_AUTH_API_KEYS before deploying "
            "this internet-reachable."
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _warn_if_unprotected_in_production(settings)

    logger.info("Loading vision models (FaceLandmarker, ArcFace) and opening the embedding store...")
    try:
        get_face_detector()
        embedder = get_embedder()
        store = get_embedding_store()
        if settings.validate_models_on_startup:
            await run_in_threadpool(embedder.validate)
        store.ping()
        app.state.ready = True
        logger.info("Models loaded, store reachable. Service ready.")
    except Exception as exc:  # noqa: BLE001
        app.state.ready = False
        logger.error("Startup validation failed: %s", exc)
    yield
    logger.info("Shutting down biometric-auth service.")


app = FastAPI(
    title="Recruitix Biometric Auth Service",
    description=(
        "Face detection/landmarks, recognition, and liveness for Recruitix "
        "enrollment and login. Every verdict is computed server-side from "
        "real submitted frames — never a client-reported boolean."
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


@app.exception_handler(NoFaceDetectedError)
async def no_face_handler(request: Request, exc: NoFaceDetectedError) -> JSONResponse:
    return _error_response(422, "no_face_detected", str(exc))


@app.exception_handler(MultipleFacesDetectedError)
async def multiple_faces_handler(request: Request, exc: MultipleFacesDetectedError) -> JSONResponse:
    return _error_response(422, "multiple_faces_detected", str(exc))


@app.exception_handler(EnrollmentNotFoundError)
async def enrollment_not_found_handler(request: Request, exc: EnrollmentNotFoundError) -> JSONResponse:
    return _error_response(404, "not_found", f"No enrollment found for candidate_id {exc}.")


@app.exception_handler(EmbeddingError)
async def embedding_error_handler(request: Request, exc: EmbeddingError) -> JSONResponse:
    return _error_response(422, "embedding_error", str(exc))


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
        "arcface_model": settings.arcface_model_name,
        "match_similarity_threshold": settings.match_similarity_threshold,
        "min_enrollment_images": settings.min_enrollment_images,
        "liveness_min_frames": settings.liveness_min_frames,
    }


# --- Helpers -----------------------------------------------------------


def _decode_image(data: bytes, index: int) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=422, detail=f"Image {index} could not be decoded — not a valid image file.")
    return image


async def _read_and_validate_images(files: list[UploadFile], *, min_count: int = 1) -> list[np.ndarray]:
    if len(files) > settings.max_images_per_request:
        raise HTTPException(
            status_code=422, detail=f"Too many images: {len(files)} exceeds the {settings.max_images_per_request}-image limit."
        )
    if len(files) < min_count:
        raise HTTPException(status_code=422, detail=f"At least {min_count} image(s) are required.")

    images = []
    for i, file in enumerate(files):
        data = await file.read()
        if not data:
            raise HTTPException(status_code=422, detail=f"Image {i} is empty.")
        if len(data) > settings.max_image_bytes:
            raise HTTPException(status_code=413, detail=f"Image {i} exceeds the {settings.max_image_bytes}-byte limit.")
        images.append(_decode_image(data, i))
    return images


def _average_embedding(detector: FaceDetector, embedder: ArcFaceEmbedder, images: list[np.ndarray]) -> list[float]:
    embeddings = []
    for image in images:
        detector.detect_single(image)  # validates exactly one clear face; raises otherwise
        embeddings.append(embedder.compute_embedding(image))
    matrix = np.array(embeddings)
    mean = matrix.mean(axis=0)
    norm = np.linalg.norm(mean)
    return [float(x) for x in (mean / norm if norm > 0 else mean)]


# --- Enrollment / verification -----------------------------------------------------------


@app.post(
    "/api/v1/biometric/enroll",
    response_model=EnrollResponse,
    responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
    tags=["biometric"],
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(settings.rate_limit_default)
async def enroll_endpoint(
    request: Request, candidate_id: str, files: list[UploadFile] = File(...)
) -> EnrollResponse:
    if not candidate_id.strip():
        raise HTTPException(status_code=422, detail="candidate_id must not be empty.")

    images = await _read_and_validate_images(files, min_count=settings.min_enrollment_images)

    detector = get_face_detector()
    embedder = get_embedder()
    embedding = await run_in_threadpool(_average_embedding, detector, embedder, images)

    store = get_embedding_store()
    await run_in_threadpool(store.enroll, candidate_id, embedding, embedder.model_name)

    return EnrollResponse(candidate_id=candidate_id, enrolled=True, images_used=len(images), model_used=embedder.model_name)


@app.post(
    "/api/v1/biometric/verify",
    response_model=VerifyResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
    tags=["biometric"],
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(settings.rate_limit_default)
async def verify_endpoint(request: Request, candidate_id: str, files: list[UploadFile] = File(...)) -> VerifyResponse:
    if not candidate_id.strip():
        raise HTTPException(status_code=422, detail="candidate_id must not be empty.")

    images = await _read_and_validate_images(files, min_count=1)

    store = get_embedding_store()
    enrolled_embedding = await run_in_threadpool(store.get_embedding, candidate_id)

    detector = get_face_detector()
    embedder = get_embedder()
    fresh_embedding = await run_in_threadpool(_average_embedding, detector, embedder, images)

    similarity = cosine_similarity(fresh_embedding, enrolled_embedding)
    match = similarity >= settings.match_similarity_threshold

    return VerifyResponse(
        candidate_id=candidate_id, match=match, similarity=round(similarity, 4), threshold=settings.match_similarity_threshold
    )


@app.delete(
    "/api/v1/biometric/enroll/{candidate_id}",
    response_model=DeleteEnrollmentResponse,
    responses={401: {"model": ErrorResponse}},
    tags=["biometric"],
    dependencies=[Depends(verify_api_key)],
)
async def delete_enrollment_endpoint(candidate_id: str = Path(...)) -> DeleteEnrollmentResponse:
    store = get_embedding_store()
    await run_in_threadpool(store.delete, candidate_id)
    return DeleteEnrollmentResponse(candidate_id=candidate_id, deleted=True)


# --- Liveness -----------------------------------------------------------


@app.post(
    "/api/v1/biometric/liveness",
    response_model=LivenessResponse,
    responses={401: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 429: {"model": ErrorResponse}},
    tags=["biometric"],
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(settings.rate_limit_default)
async def liveness_endpoint(request: Request, files: list[UploadFile] = File(...)) -> LivenessResponse:
    images = await _read_and_validate_images(files, min_count=settings.liveness_min_frames)

    detector = get_face_detector()
    result = await run_in_threadpool(
        analyze_liveness,
        detector,
        images,
        ear_threshold=settings.liveness_ear_threshold,
        min_blinks_required=settings.liveness_min_blinks_required,
        max_head_pose_deviation_degrees=settings.liveness_max_head_pose_deviation_degrees,
    )

    return LivenessResponse(
        live=result.live,
        blink_count=result.blink_count,
        max_head_pose_deviation_degrees=round(result.max_head_pose_deviation_degrees, 2),
        frames_analyzed=result.frames_analyzed,
        reason=result.reason,
    )
