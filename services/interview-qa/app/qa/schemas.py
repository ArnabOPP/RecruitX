"""Data contracts for CV-grounded interview question generation.

This service intentionally owns its own resume-context schema rather than
importing cv-parser's `ParsedResume` directly — the two are separate
microservices (see the BRD's system architecture: CV parser and Interview
Q&A generation are distinct rows in the AI/ML model table), so whatever
orchestrates a candidate's session maps cv-parser's output onto this
narrower shape. That keeps each service independently deployable and
prevents a schema change in one from silently breaking the other.
"""

from __future__ import annotations

from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class SkillContext(BaseModel):
    name: str
    evidenced_in_project: bool = False
    evidenced_in_experience: bool = False


class EducationContext(BaseModel):
    institution: str | None = None
    degree: str | None = None
    field_of_study: str | None = None


class ExperienceContext(BaseModel):
    role_title: str | None = None
    organization: str | None = None
    description_bullets: list[str] = Field(default_factory=list)
    extracted_skills: list[str] = Field(default_factory=list)


class ProjectContext(BaseModel):
    title: str | None = None
    description: str | None = None
    tech_stack: list[str] = Field(default_factory=list)


class CertificationContext(BaseModel):
    name: str
    issuer: str | None = None


class ResumeContext(BaseModel):
    """The subset of a parsed résumé this service actually needs — map
    cv-parser's ParsedResume onto this before calling /generate."""

    full_name: str | None = None
    summary: str | None = None
    skills: list[SkillContext] = Field(default_factory=list)
    education: list[EducationContext] = Field(default_factory=list)
    experience: list[ExperienceContext] = Field(default_factory=list)
    projects: list[ProjectContext] = Field(default_factory=list)
    certifications: list[CertificationContext] = Field(default_factory=list)


class RoundType(str, Enum):
    PERSONAL = "personal"
    HR = "hr"


class QuestionCategory(str, Enum):
    PROJECT_DEEP_DIVE = "project_deep_dive"
    SKILL_VERIFICATION = "skill_verification"
    EXPERIENCE_DEEP_DIVE = "experience_deep_dive"
    BEHAVIORAL_STAR = "behavioral_star"
    RESUME_GAP_PROBE = "resume_gap_probe"
    MOTIVATION_FIT = "motivation_fit"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class GroundingEvidence(BaseModel):
    """What in the résumé this question is actually anchored to — the
    mechanism that makes a question "you built X with Y, how did you handle
    Z?" instead of a generic one any candidate could be asked."""

    kind: str  # "skill" | "project" | "experience" | "education" | "certification"
    reference: str  # e.g. "RecruitX" or "Python"
    detail: str | None = None  # short snippet from the résumé backing the question


class GeneratedQuestion(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    text: str
    category: QuestionCategory
    round: RoundType
    grounding: GroundingEvidence | None = None
    difficulty: Difficulty = Difficulty.MEDIUM


class GenerateQuestionsRequest(BaseModel):
    resume: ResumeContext
    target_company: str | None = None
    round: RoundType = RoundType.PERSONAL
    count: int = 5


class GenerateQuestionsResponse(BaseModel):
    questions: list[GeneratedQuestion]
    model_used: str
    warnings: list[str] = Field(default_factory=list)


class FollowUpRequest(BaseModel):
    resume: ResumeContext
    original_question: str
    candidate_answer: str
    round: RoundType = RoundType.PERSONAL
    target_company: str | None = None


class FollowUpResponse(BaseModel):
    follow_up_question: str
    rationale: str
    model_used: str
