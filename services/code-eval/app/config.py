"""Centralized, typed application configuration.

Mirrors the other four services' config.py. Like answer-grading, this
service calls no external API — grading is local (sandboxed Docker
execution + static analysis) — so auth/rate-limiting protect compute and
host resources from abuse, not a paid quota.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CODE_EVAL_", env_file=".env", extra="ignore")

    # --- Sandbox execution ---------------------------------------------------
    # Deliberately plain `docker run` isolation (--network none, resource
    # limits, --cap-drop=ALL, --read-only, non-root) rather than a nested
    # privileged sandbox engine (e.g. Piston/nsjail) — see sandbox/docker_runner.py
    # for the full reasoning. Image tags are configurable so a deploy can
    # pin an exact digest.
    python_image: str = "python:3.11-slim"
    javascript_image: str = "node:20-slim"
    sandbox_timeout_seconds: float = 10.0
    sandbox_memory_mb: int = 128
    sandbox_cpus: float = 0.5
    sandbox_pids_limit: int = 64
    validate_docker_on_startup: bool = True

    # --- Input limits -------------------------------------------------------
    max_source_code_chars: int = 20_000
    # Generous enough for genuine efficiency-probing test cases — an
    # array of a few thousand integers (a realistic size for actually
    # distinguishing O(n) from O(n^2) empirically) can easily run past a
    # few thousand characters once serialized as space-separated stdin.
    max_stdin_chars: int = 50_000
    max_test_cases: int = 20
    max_request_body_bytes: int = 1_000_000
    request_timeout_seconds: float = 60.0

    # --- Grading weights ------------------------------------------------------
    # Overall score blends correctness (test pass rate) with efficiency
    # (how the empirically-estimated complexity compares to a target) and a
    # small static-analysis quality signal. Correctness dominates since a
    # wrong answer that happens to run fast isn't a good submission.
    correctness_weight: float = 0.7
    efficiency_weight: float = 0.2
    static_quality_weight: float = 0.1

    # --- Inbound auth (protects host/compute resources from abuse) --------
    require_api_key: bool = False
    api_keys: str = ""

    # --- CORS -----------------------------------------------------------
    cors_allow_origins: str = ""

    # --- Rate limiting ----------------------------------------------------
    rate_limit_enabled: bool = True
    rate_limit_evaluate: str = "10/minute"
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
