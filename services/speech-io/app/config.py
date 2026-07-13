"""Centralized, typed application configuration.

Mirrors cv-parser/interview-qa's config.py: every environment-driven knob is
declared here, not read ad hoc via `os.environ`, so wrong types fail fast at
startup rather than mid-request.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SPEECH_IO_", env_file=".env", extra="ignore")

    # --- STT provider (speech-to-text) ---------------------------------------
    # "groq" is the only implementation today; kept as an explicit setting
    # (not hardcoded) so a self-hosted Whisper fallback can be registered in
    # stt/client.py's factory later without restructuring callers.
    stt_provider: str = "groq"
    groq_api_key: str = ""
    stt_model: str = "whisper-large-v3"
    stt_max_retries: int = 2
    stt_request_timeout_seconds: float = 30.0
    validate_key_on_startup: bool = True

    # --- TTS provider (text-to-speech) ---------------------------------------
    # edge-tts needs no API key (it's an unofficial free wrapper around
    # Microsoft Edge's neural voices) — kept swappable the same way so Azure
    # Neural TTS or Coqui can replace it later without touching callers, if
    # this unofficial API ever breaks.
    tts_provider: str = "edge"
    tts_default_voice: str = "en-US-AriaNeural"
    tts_max_retries: int = 2
    # Unlike Groq's cheap models.list() check, edge-tts has no lightweight
    # metadata endpoint — validating it means a real (tiny) synthesis call,
    # which costs a couple hundred ms on every process start. Worth it for
    # /health/ready to mean "will actually work", but toggleable in case
    # that startup cost/flakiness against an unofficial API is undesirable.
    validate_tts_on_startup: bool = True

    # --- Upload/request limits -----------------------------------------------
    max_audio_upload_bytes: int = 10 * 1024 * 1024  # 10 MB — a few minutes of speech
    max_synthesize_text_chars: int = 2_000  # a spoken interview question, not an essay
    max_request_body_bytes: int = 512_000  # non-audio JSON bodies (synthesize)
    request_timeout_seconds: float = 45.0

    # --- Inbound auth (protects the Groq quota this service spends) --------
    require_api_key: bool = False
    api_keys: str = ""  # comma-separated shared secrets, checked against X-API-Key

    # --- CORS -----------------------------------------------------------
    cors_allow_origins: str = ""

    # --- Rate limiting ----------------------------------------------------
    rate_limit_enabled: bool = True
    rate_limit_transcribe: str = "20/minute"
    rate_limit_synthesize: str = "30/minute"
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

    @field_validator("stt_provider")
    @classmethod
    def _validate_stt_provider(cls, v: str) -> str:
        v = v.lower()
        valid = {"groq"}
        if v not in valid:
            raise ValueError(f"stt_provider must be one of {valid}, got {v!r}")
        return v

    @field_validator("tts_provider")
    @classmethod
    def _validate_tts_provider(cls, v: str) -> str:
        v = v.lower()
        valid = {"edge"}
        if v not in valid:
            raise ValueError(f"tts_provider must be one of {valid}, got {v!r}")
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
