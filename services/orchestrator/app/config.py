"""Centralized, typed application configuration.

Mirrors the other five services' config.py. Unlike them, this service's
entire job is calling the *other five* — so its configuration surface is
mostly base URLs and API keys for each, plus the Redis connection every
other service already depends on for rate limiting, reused here as the
actual session store (not just a rate-limit backend).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ORCHESTRATOR_", env_file=".env", extra="ignore")

    # --- Downstream services --------------------------------------------
    cv_parser_base_url: str = "http://localhost:8100"
    interview_qa_base_url: str = "http://localhost:8000"
    speech_io_base_url: str = "http://localhost:8001"
    answer_grading_base_url: str = "http://localhost:8002"
    code_eval_base_url: str = "http://localhost:8003"
    biometric_auth_base_url: str = "http://localhost:8005"
    proctoring_base_url: str = "http://localhost:8006"

    # Shared-secret keys this service presents to each downstream service
    # if that service has its own require_api_key turned on. Empty is
    # fine when calling a downstream service that has auth disabled
    # (e.g. local dev).
    cv_parser_api_key: str = ""
    interview_qa_api_key: str = ""
    speech_io_api_key: str = ""
    answer_grading_api_key: str = ""
    code_eval_api_key: str = ""
    biometric_auth_api_key: str = ""
    proctoring_api_key: str = ""

    downstream_timeout_seconds: float = 45.0

    # --- Biometric verification gate -----------------------------------
    # Off by default so existing behavior (and every prior test) is
    # unaffected — a deployment opts in once biometric-auth is actually
    # enrolled/reachable. When on, /sessions requires candidate_id +
    # face_files and rejects session creation on a non-match, the same
    # "the deterministic engine decides, not the client" principle as
    # biometric-auth's own /verify.
    require_biometric_verification: bool = False
    max_face_image_bytes: int = 5 * 1024 * 1024
    max_face_images_per_request: int = 10
    max_snapshot_image_bytes: int = 5 * 1024 * 1024

    # --- Session store (Redis) -------------------------------------------
    # Unlike the other services (where Redis is optional, only needed for
    # multi-replica rate limiting), this service's session state *is*
    # Redis — there is no in-memory fallback, since session data must
    # survive a single request and be visible across replicas.
    redis_uri: str = "redis://localhost:6379"
    session_ttl_seconds: int = 3600 * 4  # 4 hours — long enough for one sitting

    # --- Round configuration (defaults, overridable per session) -----------
    default_personal_question_count: int = 3
    default_hr_question_count: int = 2
    default_enable_followups: bool = True

    # --- Input limits -------------------------------------------------------
    max_resume_upload_bytes: int = 10 * 1024 * 1024
    max_answer_text_chars: int = 4_000
    max_request_body_bytes: int = 2_000_000
    request_timeout_seconds: float = 90.0

    # --- Inbound auth --------------------------------------------------
    require_api_key: bool = False
    api_keys: str = ""

    # --- CORS -----------------------------------------------------------
    cors_allow_origins: str = ""

    # --- Rate limiting ----------------------------------------------------
    rate_limit_enabled: bool = True
    rate_limit_default: str = "20/minute"
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
