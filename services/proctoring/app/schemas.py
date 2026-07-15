"""Data contracts for the proctoring API."""

from __future__ import annotations

from pydantic import BaseModel


class SnapshotResponse(BaseModel):
    session_id: str
    faces_detected: int
    head_pose_deviation_degrees: float | None
    gaze_offset: float | None
    flagged_this_frame: list[str]
    events_recorded: list[str]
    integrity_score: float
    frames_processed: int


class ProctoringEvent(BaseModel):
    type: str
    timestamp: float
    severity: float


class SummaryResponse(BaseModel):
    session_id: str
    frames_processed: int
    integrity_score: float
    event_counts: dict[str, int]
    events: list[ProctoringEvent]


class DeleteSessionResponse(BaseModel):
    session_id: str
    deleted: bool
