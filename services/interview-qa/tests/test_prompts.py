"""Tests for prompt construction, in particular the résumé-context size cap
— the gap where an arbitrarily large résumé payload had no server-side
limit despite `INTERVIEW_QA_MAX_RESUME_CONTEXT_CHARS` being configurable."""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.qa.prompts import build_generation_prompt
from app.qa.schemas import ProjectContext, ResumeContext, RoundType


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_resume_context_is_truncated_to_configured_limit(monkeypatch):
    monkeypatch.setenv("INTERVIEW_QA_MAX_RESUME_CONTEXT_CHARS", "500")
    get_settings.cache_clear()

    huge_description = "x" * 50_000
    resume = ResumeContext(
        full_name="Jordan Lee",
        projects=[ProjectContext(title="Huge Project", description=huge_description)],
    )

    _system, user = build_generation_prompt(resume, RoundType.PERSONAL, None, 3)

    # The user prompt contains other text (round guidance, instructions)
    # besides the résumé context, so we can't assert on len(user) directly —
    # what matters is that the huge field didn't make it through whole.
    assert huge_description not in user
    assert "[résumé context truncated]" in user


def test_resume_context_under_limit_is_not_truncated(monkeypatch):
    monkeypatch.setenv("INTERVIEW_QA_MAX_RESUME_CONTEXT_CHARS", "20000")
    get_settings.cache_clear()

    resume = ResumeContext(full_name="Jordan Lee", projects=[ProjectContext(title="Small Project", description="A small app.")])

    _system, user = build_generation_prompt(resume, RoundType.PERSONAL, None, 3)

    assert "[résumé context truncated]" not in user
    assert "Small Project" in user
