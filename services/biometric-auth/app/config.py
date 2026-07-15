"""Centralized, typed application configuration.

Mirrors the other Recruitix services' config.py. Like answer-grading and
code-eval, this service calls no external API — everything is local
compute (self-hosted MediaPipe + ArcFace models) — so auth/rate-limiting
protect compute resources, not a paid quota. Unlike every other service,
this one also owns durable storage (enrolled face embeddings survive
indefinitely, not a Redis TTL).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BIOMETRIC_AUTH_", env_file=".env", extra="ignore")

    # --- Vision models -------------------------------------------------------
    face_landmarker_model_path: str = "models/face_landmarker.task"
    arcface_model_name: str = "buffalo_l"
    min_face_detection_confidence: float = 0.5
    validate_models_on_startup: bool = True

    # --- Matching -------------------------------------------------------
    # Cosine similarity threshold above which two embeddings are considered
    # the same person. Tuned empirically (see README) — ArcFace/buffalo_l
    # embeddings are L2-normalized, so this lives in [-1, 1]; ~0.35-0.45
    # is the commonly cited operating point balancing false-accept vs
    # false-reject for buffalo_l specifically.
    match_similarity_threshold: float = 0.38

    # --- Liveness -------------------------------------------------------
    # Eye-Aspect-Ratio below this is "eyes closed" — the widely-cited
    # value from Soukupova & Cech's original EAR paper, applicable across
    # face geometries since EAR is a ratio, not an absolute distance.
    liveness_ear_threshold: float = 0.21
    liveness_min_blinks_required: int = 1
    # Natural blink rate is roughly one every 3-4 seconds — a short burst
    # (e.g. 5 frames over under a second) has a real chance of catching
    # zero blinks from a genuinely live person, producing false rejections
    # rather than catching spoofing. 15 frames, captured by the client at
    # ~5-8fps over roughly 2-3 seconds, gives good odds of a natural blink
    # actually falling inside the window.
    liveness_min_frames: int = 15
    liveness_max_head_pose_deviation_degrees: float = 35.0

    # --- Input limits -------------------------------------------------------
    # Must comfortably exceed liveness_min_frames below — a caller
    # submitting a full liveness frame burst shouldn't be rejected by a
    # blanket image-count cap sized only for enrollment/verify.
    max_images_per_request: int = 20
    max_image_bytes: int = 5 * 1024 * 1024
    min_enrollment_images: int = 3
    max_request_body_bytes: int = 30 * 1024 * 1024  # a handful of JPEGs
    request_timeout_seconds: float = 30.0

    # --- Storage -------------------------------------------------------
    # A single SQLite file — the first *durable* store in Recruitix (every
    # other service is either stateless or Redis-TTL'd ephemeral session
    # state). Enrolled face embeddings are reference data that must
    # survive indefinitely, not expire with a session.
    db_path: str = "data/biometric.db"

    # --- Inbound auth --------------------------------------------------
    require_api_key: bool = False
    api_keys: str = ""

    # --- CORS -----------------------------------------------------------
    cors_allow_origins: str = ""

    # --- Rate limiting ----------------------------------------------------
    rate_limit_enabled: bool = True
    rate_limit_default: str = "10/minute"
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
