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
    # A missing key is obviously invalid, but a *present but wrong/revoked*
    # key would previously only surface on the first real request — this
    # makes startup confirm the key actually works with one cheap,
    # token-free call, so /health/ready means "will actually work", not
    # just "a key is configured". Costs one lightweight request per
    # process start; disable if that's undesirable (e.g. very frequent
    # restarts against a rate-limited provider).
    validate_key_on_startup: bool = True

    # --- Question generation ------------------------------------------------
    max_questions_per_request: int = 10
    default_questions_per_request: int = 5

    # --- Upload/request limits -----------------------------------------------
    max_resume_context_chars: int = 20_000
    # original_question / candidate_answer are free-text, directly
    # attacker-controlled input with no résumé-parsing step in between —
    # same prompt-cost/DoS exposure as an oversized résumé, capped
    # separately since a spoken interview answer is naturally much shorter
    # than a full résumé.
    max_followup_field_chars: int = 4_000
    # Field-level truncation above only bounds what reaches the LLM prompt
    # — by the time that code runs, FastAPI has already fully parsed the
    # JSON body into memory (e.g. a résumé with 500,000 tiny skill entries
    # costs real CPU/memory regardless of the final string being
    # truncated). This caps the raw request body itself, rejected before
    # parsing even starts.
    max_request_body_bytes: int = 512_000
    request_timeout_seconds: float = 45.0

    # --- Inbound auth (protects the Groq quota this service spends) --------
    # Off by default for local dev / tests; the README and Dockerfile call
    # out that this must be turned on before any internet-reachable deploy.
    require_api_key: bool = False
    api_keys: str = ""  # comma-separated shared secrets, checked against X-API-Key

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
