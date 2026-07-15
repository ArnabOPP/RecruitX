"""Recruitix Orchestrator Service — FastAPI entrypoint.

Wires the five independent Recruitix AI/ML services (cv-parser,
interview-qa, speech-io, answer-grading, code-eval) into one coherent
interview session: résumé upload -> personal round -> HR round ->
coding round -> report. None of the five services know about each other;
this is what turns them into a product flow, owning session state (in
Redis — see session_store.py) and the schema-translation glue between
them (see mapping.py).
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, Path, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from .auth import verify_api_key
from .clients import proctoring_client
from .clients.base import DownstreamServiceError
from .config import get_settings
from .errors import ErrorResponse
from .logging_config import configure_logging, request_id_var
from .middleware import MaxBodySizeMiddleware
from .orchestration import (
    IdentityVerificationError,
    OrchestrationError,
    create_session,
    get_report,
    submit_answer,
    submit_code,
)
from .schemas import (
    AnswerResponse,
    AnswerTextRequest,
    CodeSubmissionRequest,
    CodeSubmissionResponse,
    CreateSessionResponse,
    ProctoringSnapshotResponse,
    QuestionOut,
    SessionReport,
)
from .session_store import SessionNotFoundError, get_session_store

settings = get_settings()
configure_logging(level=settings.log_level, json_logs=settings.log_json)
logger = logging.getLogger("orchestrator.api")


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
            "SECURITY: environment=production but ORCHESTRATOR_REQUIRE_API_KEY is not set. "
            "This service sits in front of all five downstream services — leaving it "
            "unauthenticated is strictly worse than leaving any single one open. Set "
            "ORCHESTRATOR_REQUIRE_API_KEY=1 and ORCHESTRATOR_API_KEYS before deploying this "
            "internet-reachable."
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _warn_if_unprotected_in_production(settings)

    logger.info("Validating session store (Redis reachable)...")
    try:
        store = get_session_store()
        await store.ping()
        app.state.ready = True
        logger.info("Redis reachable. Service ready.")
    except Exception as exc:  # noqa: BLE001
        # Readiness gates only on Redis — the session store this service
        # cannot function without at all — not on all five downstream
        # services also being up. Gating on every downstream would make
        # this service's own uptime hostage to any one of theirs; a
        # downstream being down instead surfaces as a clear per-call 502
        # naming which service failed (DownstreamServiceError), not a
        # blanket "orchestrator is down".
        app.state.ready = False
        logger.error("Redis is not reachable: %s", exc)
    yield
    logger.info("Shutting down orchestrator service.")


app = FastAPI(
    title="Recruitix Orchestrator Service",
    description=(
        "Wires cv-parser, interview-qa, speech-io, answer-grading, and code-eval "
        "into one interview session: résumé upload -> personal round -> HR round "
        "-> coding round -> report."
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
# /sessions (résumé + optional face_files upload), /answer/audio, and
# /proctoring/snapshot carry file uploads with their own, larger,
# file-specific limits enforced inside the endpoints — exempted here the
# same way speech-io exempts /transcribe.
app.add_middleware(
    MaxBodySizeMiddleware,
    max_bytes=settings.max_request_body_bytes,
    exempt_paths=frozenset({"/api/v1/sessions"}),
    exempt_path_suffixes=("/answer/audio", "/proctoring/snapshot"),
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


@app.exception_handler(SessionNotFoundError)
async def session_not_found_handler(request: Request, exc: SessionNotFoundError) -> JSONResponse:
    return _error_response(404, "not_found", f"No session found with id {exc}.")


@app.exception_handler(IdentityVerificationError)
async def identity_verification_error_handler(request: Request, exc: IdentityVerificationError) -> JSONResponse:
    return _error_response(403, "identity_verification_failed", str(exc))


@app.exception_handler(OrchestrationError)
async def orchestration_error_handler(request: Request, exc: OrchestrationError) -> JSONResponse:
    return _error_response(409, "conflict", str(exc))


@app.exception_handler(DownstreamServiceError)
async def downstream_error_handler(request: Request, exc: DownstreamServiceError) -> JSONResponse:
    logger.warning("Downstream service failure: %s", exc)
    return _error_response(502, "downstream_error", f"{exc.service} failed: {exc}")


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
        "downstream_services": {
            "cv_parser": settings.cv_parser_base_url,
            "interview_qa": settings.interview_qa_base_url,
            "speech_io": settings.speech_io_base_url,
            "answer_grading": settings.answer_grading_base_url,
            "code_eval": settings.code_eval_base_url,
            "biometric_auth": settings.biometric_auth_base_url,
            "proctoring": settings.proctoring_base_url,
        },
        "default_personal_question_count": settings.default_personal_question_count,
        "default_hr_question_count": settings.default_hr_question_count,
        "default_enable_followups": settings.default_enable_followups,
        "require_biometric_verification": settings.require_biometric_verification,
    }


# --- Sessions -----------------------------------------------------------


def _question_out(session_data: dict) -> QuestionOut | None:
    q = session_data.get("current_question")
    if q is None:
        return None
    return QuestionOut(text=q["text"], category=q.get("category"), grounding=q.get("grounding"), difficulty=q.get("difficulty"), round=q["round"], stage=session_data.get("stage"))


@app.post(
    "/api/v1/sessions",
    response_model=CreateSessionResponse,
    responses={
        400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}, 403: {"model": ErrorResponse},
        413: {"model": ErrorResponse}, 502: {"model": ErrorResponse},
    },
    tags=["sessions"],
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(settings.rate_limit_default)
async def create_session_endpoint(
    request: Request,
    file: UploadFile = File(...),
    target_company: str | None = None,
    personal_question_count: int | None = None,
    hr_question_count: int | None = None,
    enable_followups: bool | None = None,
    candidate_id: str | None = None,
    face_files: list[UploadFile] | None = None,
) -> CreateSessionResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.max_resume_upload_bytes:
        raise HTTPException(status_code=413, detail=f"File too large. Limit is {settings.max_resume_upload_bytes} bytes.")

    data = await file.read()
    if len(data) > settings.max_resume_upload_bytes:
        raise HTTPException(status_code=413, detail=f"File too large. Limit is {settings.max_resume_upload_bytes} bytes.")
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if settings.require_biometric_verification and not (candidate_id and face_files):
        raise HTTPException(
            status_code=400,
            detail="This deployment requires candidate_id and face_files for identity verification.",
        )

    face_upload: list[tuple[str, bytes, str]] | None = None
    if candidate_id and face_files:
        if len(face_files) > settings.max_face_images_per_request:
            raise HTTPException(
                status_code=422,
                detail=f"Too many face images: {len(face_files)} exceeds the {settings.max_face_images_per_request}-image limit.",
            )
        face_upload = []
        for face_file in face_files:
            face_data = await face_file.read()
            if not face_data:
                raise HTTPException(status_code=422, detail="A face_files image is empty.")
            if len(face_data) > settings.max_face_image_bytes:
                raise HTTPException(
                    status_code=413, detail=f"A face image exceeds the {settings.max_face_image_bytes}-byte limit."
                )
            face_upload.append((face_file.filename or "face.jpg", face_data, face_file.content_type or "image/jpeg"))

    store = get_session_store()
    session_data = await create_session(
        store,
        data,
        file.filename,
        target_company,
        personal_question_count if personal_question_count is not None else settings.default_personal_question_count,
        hr_question_count if hr_question_count is not None else settings.default_hr_question_count,
        enable_followups if enable_followups is not None else settings.default_enable_followups,
        candidate_id,
        face_upload,
    )

    return CreateSessionResponse(
        session_id=session_data["session_id"],
        status=session_data["status"],
        round=session_data["round"],
        question=_question_out(session_data),
    )


@app.post(
    "/api/v1/sessions/{session_id}/answer",
    response_model=AnswerResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
    tags=["sessions"],
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(settings.rate_limit_default)
async def submit_answer_text_endpoint(
    request: Request, body: AnswerTextRequest, session_id: str = Path(...)
) -> AnswerResponse:
    if not body.answer_text.strip():
        raise HTTPException(status_code=422, detail="answer_text must not be empty.")
    if len(body.answer_text) > settings.max_answer_text_chars:
        raise HTTPException(status_code=422, detail=f"answer_text exceeds the {settings.max_answer_text_chars}-character limit.")

    store = get_session_store()
    result = await submit_answer(store, session_id, body.answer_text, None, None)
    return AnswerResponse(
        score=result["score"],
        round=result["round"],
        status=result["status"],
        next_question=QuestionOut(**result["next_question"]) if result["next_question"] else None,
    )


@app.post(
    "/api/v1/sessions/{session_id}/answer/audio",
    response_model=AnswerResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
    tags=["sessions"],
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(settings.rate_limit_default)
async def submit_answer_audio_endpoint(
    request: Request, session_id: str = Path(...), file: UploadFile = File(...)
) -> AnswerResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.max_resume_upload_bytes:
        raise HTTPException(status_code=413, detail=f"File too large. Limit is {settings.max_resume_upload_bytes} bytes.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    store = get_session_store()
    result = await submit_answer(store, session_id, None, data, file.filename)
    return AnswerResponse(
        score=result["score"],
        round=result["round"],
        status=result["status"],
        next_question=QuestionOut(**result["next_question"]) if result["next_question"] else None,
    )


@app.post(
    "/api/v1/sessions/{session_id}/code",
    response_model=CodeSubmissionResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
    tags=["sessions"],
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(settings.rate_limit_default)
async def submit_code_endpoint(
    request: Request, body: CodeSubmissionRequest, session_id: str = Path(...)
) -> CodeSubmissionResponse:
    store = get_session_store()
    result = await submit_code(store, session_id, body.language, body.source_code, body.test_cases, body.expected_complexity)
    return CodeSubmissionResponse(result=result["result"], status=result["status"])


@app.post(
    "/api/v1/sessions/{session_id}/proctoring/snapshot",
    response_model=ProctoringSnapshotResponse,
    responses={
        401: {"model": ErrorResponse}, 404: {"model": ErrorResponse},
        413: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 502: {"model": ErrorResponse},
    },
    tags=["sessions"],
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(settings.rate_limit_default)
async def proctoring_snapshot_endpoint(
    request: Request, session_id: str = Path(...), file: UploadFile = File(...)
) -> ProctoringSnapshotResponse:
    store = get_session_store()
    await store.load(session_id)  # 404s via SessionNotFoundError if the session doesn't exist

    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="Snapshot image is empty.")
    if len(data) > settings.max_snapshot_image_bytes:
        raise HTTPException(status_code=413, detail=f"Snapshot image exceeds the {settings.max_snapshot_image_bytes}-byte limit.")

    result = await proctoring_client.submit_snapshot(
        session_id, file.filename or "snapshot.jpg", data, file.content_type or "image/jpeg"
    )
    return ProctoringSnapshotResponse(**result)


@app.get(
    "/api/v1/sessions/{session_id}/report",
    response_model=SessionReport,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
    tags=["sessions"],
    dependencies=[Depends(verify_api_key)],
)
async def get_report_endpoint(session_id: str = Path(...)) -> SessionReport:
    store = get_session_store()
    report = await get_report(store, session_id)
    return SessionReport(**report)
