"""Centralized, typed application configuration.

Mirrors cv-parser/interview-qa/speech-io's config.py. Unlike those three,
this service calls no external API at all — scoring is 100% local compute
(Jaccard + a self-hosted embedding model) — so there's no provider API key
here. Auth/rate-limiting still exist to protect CPU resources from abuse,
not a paid quota.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ANSWER_GRADING_", env_file=".env", extra="ignore")

    # --- Semantic similarity model --------------------------------------
    # A small, well-established sentence-embedding model — same self-hosted
    # pattern as cv-parser's spaCy/BERT pipeline. Embeddings are a
    # deterministic function of the model weights + input text (no
    # sampling), which is why this is safe to use in a "reproducible"
    # scoring engine, unlike an LLM chat completion.
    semantic_model_name: str = "all-MiniLM-L6-v2"
    validate_model_on_startup: bool = True

    # --- Score combination weights ----------------------------------------
    # Within each rubric criterion, the criterion's own score is a weighted
    # blend of Jaccard (exact/near-exact term overlap) and semantic
    # similarity (paraphrase-tolerant). Semantic gets more weight by
    # default since it's strictly more informative when it disagrees with
    # Jaccard (Jaccard can't recognize a paraphrase at all).
    jaccard_weight: float = 0.4
    semantic_weight: float = 0.6

    # --- Input limits -------------------------------------------------------
    max_question_chars: int = 2_000
    max_answer_chars: int = 4_000
    max_rubric_criteria: int = 10
    max_keywords_per_criterion: int = 20
    max_request_body_bytes: int = 512_000
    request_timeout_seconds: float = 30.0

    # --- Inbound auth (protects CPU resources from abuse) ------------------
    require_api_key: bool = False
    api_keys: str = ""  # comma-separated shared secrets, checked against X-API-Key

    # --- CORS -----------------------------------------------------------
    cors_allow_origins: str = ""

    # --- Rate limiting ----------------------------------------------------
    rate_limit_enabled: bool = True
    rate_limit_score: str = "30/minute"
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
