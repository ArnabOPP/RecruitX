"""Centralized, typed application configuration.

Every environment-driven knob in the service is declared here rather than
read ad hoc via `os.environ` at the point of use — that makes the full
configuration surface discoverable in one place, gives free validation
(wrong types fail at startup, not mid-request), and makes `.env` files and
`vercel env` / container env-var injection interchangeable.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CV_PARSER_", env_file=".env", extra="ignore")

    # --- NER models -------------------------------------------------------
    spacy_model: str = "en_core_web_sm"
    transformer_model: str = "dslim/bert-base-NER"
    enable_transformer: bool = True

    # --- Upload limits ------------------------------------------------------
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB
    request_timeout_seconds: float = 30.0

    # --- CORS -----------------------------------------------------------
    # Comma-separated origins, e.g. "https://app.recruitix.example,https://staging.recruitix.example".
    # Empty by default (deny all cross-origin) — explicit opt-in for production.
    cors_allow_origins: str = ""

    # --- Rate limiting ----------------------------------------------------
    rate_limit_enabled: bool = True
    rate_limit_parse: str = "20/minute"

    # --- Logging ------------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = True

    # --- Metrics --------------------------------------------------------
    metrics_enabled: bool = True

    # --- Environment ------------------------------------------------------
    environment: str = "development"  # development | staging | production

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
