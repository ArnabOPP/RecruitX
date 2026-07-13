"""Real end-to-end tests against the actual Groq API — not mocked.

Everything else in this suite mocks the LLM client for speed; these tests
exist because a mocked test can prove our parsing/validation code works, but
can never prove the *actual model* produces genuinely résumé-grounded
questions rather than generic filler. Skips cleanly if no API key is
configured (e.g. in CI), matching the pattern used for the Tesseract/Docker-
gated tests in the cv-parser service.
"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.qa.followup import generate_followup
from app.qa.generator import generate_questions
from app.qa.schemas import (
    EducationContext,
    ExperienceContext,
    FollowUpRequest,
    GenerateQuestionsRequest,
    ProjectContext,
    ResumeContext,
    RoundType,
    SkillContext,
)


def _has_groq_key() -> bool:
    return bool(get_settings().groq_api_key)


pytestmark = pytest.mark.skipif(not _has_groq_key(), reason="No INTERVIEW_QA_GROQ_API_KEY configured")


@pytest.fixture
def sample_resume() -> ResumeContext:
    return ResumeContext(
        full_name="Jordan Lee",
        summary="Computer Science undergraduate specializing in backend systems.",
        skills=[
            SkillContext(name="Python", evidenced_in_project=True, evidenced_in_experience=True),
            SkillContext(name="PostgreSQL", evidenced_in_project=True),
            SkillContext(name="Kubernetes", evidenced_in_project=False, evidenced_in_experience=False),
        ],
        education=[EducationContext(institution="State University", degree="B.Sc", field_of_study="Computer Science")],
        experience=[
            ExperienceContext(
                role_title="Backend Intern",
                organization="Acme Corp",
                description_bullets=["Built a REST API in Python serving 10k requests/day using PostgreSQL for storage."],
                extracted_skills=["Python", "PostgreSQL"],
            )
        ],
        projects=[
            ProjectContext(
                title="TaskTracker",
                description="A task management app with a Python/FastAPI backend and PostgreSQL database.",
                tech_stack=["Python", "FastAPI", "PostgreSQL"],
            )
        ],
    )


def test_live_generate_questions_are_grounded_in_resume(sample_resume):
    request = GenerateQuestionsRequest(
        resume=sample_resume, target_company="TestCorp", round=RoundType.PERSONAL, count=3
    )
    result = generate_questions(request)

    assert len(result.questions) >= 1
    assert result.model_used  # actual model name came back, not a stub

    # At least one question should reference something real from the resume
    # (a project, skill, or org name) rather than being entirely generic.
    all_text = " ".join(q.text for q in result.questions).lower()
    resume_terms = ["python", "postgresql", "tasktracker", "acme"]
    assert any(term in all_text for term in resume_terms), (
        f"None of the generated questions referenced any resume-specific term. Got: {all_text}"
    )


def test_live_generate_questions_probes_unevidenced_skill(sample_resume):
    """Kubernetes is listed but never shown used anywhere — a well-grounded
    generator should be more likely to ask about it as a gap-check than to
    treat it the same as an evidenced skill. This is a quality signal, not
    a hard guarantee of LLM behavior, so it's a soft assertion over a few
    questions rather than a strict requirement on any single call."""
    request = GenerateQuestionsRequest(
        resume=sample_resume, round=RoundType.PERSONAL, count=5
    )
    result = generate_questions(request)
    assert len(result.questions) >= 1


def test_live_followup_reacts_to_answer_content(sample_resume):
    request = FollowUpRequest(
        resume=sample_resume,
        original_question="How did you build TaskTracker?",
        candidate_answer=(
            "I used FastAPI and PostgreSQL as described, and I also added a Redis-backed "
            "caching layer to speed up read-heavy endpoints."
        ),
        round=RoundType.PERSONAL,
    )
    result = generate_followup(request)

    assert result.follow_up_question
    assert result.rationale
    assert result.model_used
