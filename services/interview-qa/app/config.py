"""Centralized, typed application configuration.

Mirrors the cv-parser service's config.py: every environment-driven knob is
declared here, not read ad hoc via `os.environ`, so wrong types fail fast at
startup rather than mid-request, and the full configuration surface is
discoverable in one place.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INTERVIEW_QA_", env_file=".env", extra="ignore")

    # --- LLM provider -------------------------------------------------------
    # "groq" is the only implementation today; the provider is still an
    # explicit setting (not hardcoded into the client) so adding a second
    # implementation (Ollama, Anthropic, OpenAI) later is a matter of
    # registering it in llm/client.py's factory, not restructuring callers.
    llm_provider: str = "groq"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.6
    llm_max_tokens: int = 2000
    # The LLM occasionally returns malformed JSON despite response_format
    # constraints; retrying the call (not just re-parsing) is what actually
    # recovers, since a fresh sample is usually well-formed.
    llm_max_retries: int = 2
    llm_request_timeout_seconds: float = 30.0

    # --- Question generation ------------------------------------------------
    max_questions_per_request: int = 10
    default_questions_per_request: int = 5

    # --- Upload/request limits -----------------------------------------------
    max_resume_context_chars: int = 20_000
    request_timeout_seconds: float = 45.0

    # --- CORS -----------------------------------------------------------
    cors_allow_origins: str = ""

    # --- Rate limiting ----------------------------------------------------
    rate_limit_enabled: bool = True
    rate_limit_generate: str = "20/minute"
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

    @field_validator("llm_provider")
    @classmethod
    def _validate_provider(cls, v: str) -> str:
        v = v.lower()
        valid = {"groq"}
        if v not in valid:
            raise ValueError(f"llm_provider must be one of {valid}, got {v!r}")
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
