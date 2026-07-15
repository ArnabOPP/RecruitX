"""Data contracts for the biometric-auth API."""

from __future__ import annotations

from pydantic import BaseModel


class EnrollResponse(BaseModel):
    candidate_id: str
    enrolled: bool
    images_used: int
    model_used: str


class VerifyResponse(BaseModel):
    candidate_id: str
    match: bool
    similarity: float
    threshold: float


class LivenessResponse(BaseModel):
    live: bool
    blink_count: int
    max_head_pose_deviation_degrees: float
    frames_analyzed: int
    reason: str


class DeleteEnrollmentResponse(BaseModel):
    candidate_id: str
    deleted: bool
