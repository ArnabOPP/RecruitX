"""Centralized, typed application configuration.

Mirrors biometric-auth's config.py (same MediaPipe vision-model settings)
crossed with orchestrator's config.py (Redis *is* the persistence layer
here — proctoring session state is ephemeral, tied to one interview
sitting, not durable reference data like an enrolled face embedding).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PROCTORING_", env_file=".env", extra="ignore")

    # --- Vision model ---------------------------------------------------
    face_landmarker_model_path: str = "models/face_landmarker.task"
    min_face_detection_confidence: float = 0.5
    validate_models_on_startup: bool = True

    # --- Event thresholds -------------------------------------------------
    # Head-pose deviation (pitch/yaw/roll, degrees) beyond which a frame is
    # flagged as "head turned away from the screen". Looser than
    # biometric-auth's liveness threshold (35 degrees) — proctoring cares
    # about a candidate looking away, not the tighter spoofing-detection
    # tolerance liveness needs.
    head_turn_threshold_degrees: float = 30.0
    # Horizontal/vertical iris-offset ratio (0 = iris centered in the eye)
    # beyond which a frame is flagged as "gaze averted".
    gaze_offset_threshold: float = 0.35
    # A single flagged frame is just as consistent with a brief natural
    # glance as with genuine inattention — an event is only recorded once
    # this many *consecutive* flagged frames accumulate for that event
    # type, matching liveness's "a transition, not one frame" philosophy.
    consecutive_frames_to_flag: int = 3

    # --- Severity scoring -------------------------------------------------
    # Points subtracted from a session's integrity_score (starts at 100,
    # floored at 0) each time a debounced event fires. Multiple faces in
    # frame is weighted heaviest — the strongest signal of collusion.
    severity_no_face: float = 5.0
    severity_multiple_faces: float = 15.0
    severity_looking_away: float = 3.0
    severity_head_turned: float = 3.0
    # Bounded so a long session's event log can't grow without limit.
    max_events_retained: int = 500

    # --- Input limits -------------------------------------------------------
    max_image_bytes: int = 5 * 1024 * 1024
    max_request_body_bytes: int = 8 * 1024 * 1024  # a single snapshot frame
    request_timeout_seconds: float = 30.0

    # --- Session store (Redis) -------------------------------------------
    # Proctoring state is ephemeral, scoped to one interview sitting — not
    # durable reference data the way biometric-auth's enrolled embeddings
    # are, so Redis (already relied on by every service for rate limiting)
    # is the right store here, same role it plays for the orchestrator.
    redis_uri: str = "redis://localhost:6379"
    session_ttl_seconds: int = 3600 * 4  # 4 hours — long enough for one sitting

    # --- Inbound auth --------------------------------------------------
    require_api_key: bool = False
    api_keys: str = ""

    # --- CORS -----------------------------------------------------------
    cors_allow_origins: str = ""

    # --- Rate limiting ----------------------------------------------------
    rate_limit_enabled: bool = True
    rate_limit_default: str = "30/minute"
    rate_limit_storage_uri: str | None = None

    # --- Logging ------------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = True

    # --- Metrics --------------------------------------------------------
    metrics_enabled: bool = True

    # --- Environment ------------------------------------------------------
    environment: str = "development"

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        v = v.upper()
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v not in valid:
            raise ValueError(f"log_level must be one of {valid}, got {v!r}")
        return v

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
