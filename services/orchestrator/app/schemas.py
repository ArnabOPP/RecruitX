"""Data contracts for the orchestrator API.

Fields owned by a downstream service (grounding, score breakdowns,
code-eval results) are passed through as plain dicts rather than
re-declared here — they're already validated by their owning service;
re-typing them here would just be a second copy to keep in sync.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QuestionOut(BaseModel):
    text: str
    category: str | None = None
    grounding: dict | None = None
    difficulty: str | None = None
    round: str
    stage: str | None = None


class CreateSessionResponse(BaseModel):
    session_id: str
    status: str
    round: str
    question: QuestionOut | None = None


class AnswerTextRequest(BaseModel):
    answer_text: str


class AnswerResponse(BaseModel):
    score: dict
    round: str
    status: str
    next_question: QuestionOut | None = None


class CodeSubmissionRequest(BaseModel):
    language: str
    source_code: str
    test_cases: list[dict] = Field(min_length=1)
    expected_complexity: str | None = None


class CodeSubmissionResponse(BaseModel):
    result: dict
    status: str


class SessionReport(BaseModel):
    session_id: str
    status: str
    round: str
    history: list[dict]
    overall_average_score: float | None = None
    proctoring_summary: dict | None = None


class ProctoringSnapshotResponse(BaseModel):
    session_id: str
    faces_detected: int
    head_pose_deviation_degrees: float | None
    gaze_offset: float | None
    flagged_this_frame: list[str]
    events_recorded: list[str]
    integrity_score: float
    frames_processed: int
