"""Recruitix Speech I/O Service — FastAPI entrypoint.

Implements the "Speech in / out" row of the Recruitix BRD's AI/ML model
table: Whisper-class STT + neural TTS, powering the voice-based HR round —
spoken questions (synthesized from interview-qa's generated text) and
transcribed answers (fed back into interview-qa's /followup as
candidate_answer). This service owns neither question generation nor
grading — it's purely audio-in/audio-out, consumed by whatever orchestrates
a candidate's session alongside cv-parser and interview-qa.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
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
from .schemas import SynthesizeRequest, TranscribeResponse
from .stt.client import STTAuthenticationError, STTError, get_stt_client, transcribe_with_backoff
from .tts.client import TTSError, get_tts_client

settings = get_settings()
configure_logging(level=settings.log_level, json_logs=settings.log_json)
logger = logging.getLogger("speech_io.api")


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
    """Same reasoning as interview-qa: auth is opt-in, so a production
    deploy that forgets to enable it silently exposes the Groq
    transcription quota to anyone who can reach this service."""
    if settings.is_production and not settings.require_api_key:
        logger.warning(
            "SECURITY: environment=production but SPEECH_IO_REQUIRE_API_KEY is not set. "
            "This service's endpoints are unauthenticated and its Groq quota is exposed to "
            "any caller who can reach them. Set SPEECH_IO_REQUIRE_API_KEY=1 and "
            "SPEECH_IO_API_KEYS before deploying this internet-reachable."
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _warn_if_unprotected_in_production(settings)

    stt_ready = False
    tts_ready = False

    logger.info("Validating STT provider configuration (provider=%s)...", settings.stt_provider)
    try:
        stt_client = get_stt_client()
        if settings.validate_key_on_startup:
            stt_client.validate()
            logger.info("STT credentials confirmed valid by provider.")
        stt_ready = True
        app.state.stt_model = stt_client.model_name
    except STTAuthenticationError as exc:
        app.state.stt_model = None
        logger.error("STT provider rejected the configured credentials: %s", exc)
    except STTError as exc:
        app.state.stt_model = None
        logger.error("STT client failed to initialize: %s", exc)

    logger.info("Validating TTS provider configuration (provider=%s)...", settings.tts_provider)
    try:
        tts_client = get_tts_client()
        if settings.validate_tts_on_startup:
            await run_in_threadpool(tts_client.validate)
            logger.info("TTS provider confirmed reachable.")
        tts_ready = True
    except TTSError as exc:
        logger.error("TTS client failed to initialize: %s", exc)

    app.state.ready = stt_ready and tts_ready
    app.state.stt_ready = stt_ready
    app.state.tts_ready = tts_ready
    logger.info(
        "Startup complete. stt_ready=%s tts_ready=%s overall_ready=%s",
        stt_ready,
        tts_ready,
        app.state.ready,
    )
    yield
    logger.info("Shutting down speech-io service.")


app = FastAPI(
    title="Recruitix Speech I/O Service",
    description=(
        "Whisper-class speech-to-text and neural text-to-speech powering the "
        "voice-based HR round: spoken questions and transcribed answers."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.state.ready = False
app.state.stt_ready = False
app.state.tts_ready = False
app.state.stt_model = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# /transcribe carries large audio uploads with their own, separate limit
# (checked inside the endpoint, like cv-parser's file uploads) — the
# generic body-size middleware only needs to guard the small JSON bodies
# on every other POST endpoint (i.e. /synthesize).
app.add_middleware(
    MaxBodySizeMiddleware,
    max_bytes=settings.max_request_body_bytes,
    exempt_paths=frozenset({"/api/v1/speech/transcribe"}),
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


# Added last so it's outermost — request_id_var must be set before the
# body-size middleware runs, so its error responses carry a request_id.
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
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ready" if ready else "not_ready",
            "stt_ready": bool(request.app.state.stt_ready),
            "tts_ready": bool(request.app.state.tts_ready),
        },
    )


@app.get("/health", tags=["health"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/v1/capabilities", tags=["meta"])
def capabilities(request: Request) -> dict:
    return {
        "stt_provider": settings.stt_provider,
        "stt_model": request.app.state.stt_model,
        "stt_ready": bool(request.app.state.stt_ready),
        "tts_provider": settings.tts_provider,
        "tts_default_voice": settings.tts_default_voice,
        "tts_ready": bool(request.app.state.tts_ready),
        "max_audio_upload_bytes": settings.max_audio_upload_bytes,
        "max_synthesize_text_chars": settings.max_synthesize_text_chars,
    }


# --- Speech-to-text -----------------------------------------------------------


@app.post(
    "/api/v1/speech/transcribe",
    response_model=TranscribeResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
    tags=["speech"],
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(settings.rate_limit_transcribe)
async def transcribe_endpoint(
    request: Request, file: UploadFile = File(...), language: str | None = None
) -> TranscribeResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.max_audio_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large ({content_length} bytes). Limit is {settings.max_audio_upload_bytes} bytes.",
        )

    data = await file.read()
    if len(data) > settings.max_audio_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large ({len(data)} bytes). Limit is {settings.max_audio_upload_bytes} bytes.",
        )
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    client = get_stt_client()
    try:
        text = await asyncio.wait_for(
            run_in_threadpool(
                transcribe_with_backoff,
                client,
                data,
                file.filename,
                language=language,
                max_retries=settings.stt_max_retries,
            ),
            timeout=settings.request_timeout_seconds,
        )
    except STTAuthenticationError as exc:
        logger.error("STT provider rejected credentials mid-request: %s", exc)
        raise HTTPException(status_code=502, detail="The speech-to-text provider rejected our credentials.") from exc
    except STTError as exc:
        logger.warning("Transcription failed: %s", exc)
        raise HTTPException(status_code=502, detail="The speech-to-text provider could not transcribe this audio.") from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Transcription timed out.") from exc

    return TranscribeResponse(text=text, language=language, model_used=client.model_name)


# --- Text-to-speech -----------------------------------------------------------


@app.post(
    "/api/v1/speech/synthesize",
    responses={
        401: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
    tags=["speech"],
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit(settings.rate_limit_synthesize)
async def synthesize_endpoint(request: Request, body: SynthesizeRequest) -> Response:
    if not body.text or not body.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty.")
    if len(body.text) > settings.max_synthesize_text_chars:
        raise HTTPException(
            status_code=422,
            detail=f"text exceeds the {settings.max_synthesize_text_chars}-character limit.",
        )

    client = get_tts_client()
    try:
        audio_bytes = await asyncio.wait_for(
            run_in_threadpool(client.synthesize, body.text, body.voice),
            timeout=settings.request_timeout_seconds,
        )
    except TTSError as exc:
        logger.warning("Synthesis failed: %s", exc)
        raise HTTPException(status_code=502, detail="The text-to-speech provider could not synthesize this text.") from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Synthesis timed out.") from exc

    return Response(content=audio_bytes, media_type="audio/mpeg")
