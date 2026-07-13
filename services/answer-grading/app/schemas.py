"""Data contracts for the answer-grading API.

`GroundingEvidence` mirrors interview-qa's own schema of the same name
(kind/reference/detail) — this service's rubric auto-derivation is built
to consume that shape directly, since interview-qa's generated questions
already carry it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GroundingEvidence(BaseModel):
    kind: str  # "skill" | "project" | "experience" | "education" | "certification"
    reference: str
    detail: str | None = None


class RubricCriterion(BaseModel):
    description: str
    weight: float = Field(gt=0)
    expected_keywords: list[str] = Field(default_factory=list)


class Rubric(BaseModel):
    criteria: list[RubricCriterion] = Field(min_length=1)


class ScoreRequest(BaseModel):
    question: str
    candidate_answer: str
    grounding: GroundingEvidence | None = None
    rubric: Rubric | None = None


class CriterionScore(BaseModel):
    description: str
    weight: float
    score: float
    jaccard: float
    semantic: float
    matched_keywords: list[str]
    missing_keywords: list[str]


class ScoreResponse(BaseModel):
    overall_score: float
    rubric_source: str  # "provided" | "auto_derived_from_grounding" | "auto_derived_from_question"
    criteria_scores: list[CriterionScore]
    explanation: str
