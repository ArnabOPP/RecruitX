"""Event-driven severity scoring: classifies one snapshot frame's
detection result into zero or more integrity events, then folds that
classification into a session's running state (consecutive-frame
debounce counters, event counts, a bounded event log, and an
integrity_score that starts at 100 and is docked per debounced event).

A single flagged frame is deliberately never enough on its own to record
an event — the same "a transition, not a single frame" reasoning
biometric-auth's blink detector uses for EAR. A brief natural glance away
from the screen shouldn't tank a candidate's integrity score; only a
*sustained* deviation should.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .config import get_settings
from .vision.detector import DetectedFace, FaceDetector
from .vision.gaze import estimate_gaze_offset, estimate_head_pose_deviation

NO_FACE = "no_face"
MULTIPLE_FACES = "multiple_faces"
LOOKING_AWAY = "looking_away"
HEAD_TURNED = "head_turned"

_ALL_EVENT_TYPES = (NO_FACE, MULTIPLE_FACES, LOOKING_AWAY, HEAD_TURNED)


@dataclass
class FrameAnalysis:
    faces_detected: int
    head_pose_deviation_degrees: float | None
    gaze_offset: float | None
    flagged_this_frame: list[str]
    events_recorded: list[str]
    integrity_score: float


def _severity_weights(settings) -> dict[str, float]:  # noqa: ANN001
    return {
        NO_FACE: settings.severity_no_face,
        MULTIPLE_FACES: settings.severity_multiple_faces,
        LOOKING_AWAY: settings.severity_looking_away,
        HEAD_TURNED: settings.severity_head_turned,
    }


def new_session_state() -> dict:
    return {
        "frames_processed": 0,
        "consecutive": dict.fromkeys(_ALL_EVENT_TYPES, 0),
        "event_counts": dict.fromkeys(_ALL_EVENT_TYPES, 0),
        "integrity_score": 100.0,
        "events": [],
    }


def analyze_frame(detector: FaceDetector, image_bgr, state: dict) -> FrameAnalysis:  # noqa: ANN001
    settings = get_settings()
    faces: list[DetectedFace] = detector.detect_all(image_bgr)

    flagged: list[str] = []
    head_pose_deviation: float | None = None
    gaze_offset: float | None = None

    if len(faces) == 0:
        flagged.append(NO_FACE)
    elif len(faces) > 1:
        flagged.append(MULTIPLE_FACES)
    else:
        face = faces[0]
        head_pose_deviation = estimate_head_pose_deviation(face.landmarks, image_bgr.shape[:2])
        gaze_offset = estimate_gaze_offset(face.landmarks)
        if head_pose_deviation > settings.head_turn_threshold_degrees:
            flagged.append(HEAD_TURNED)
        if gaze_offset > settings.gaze_offset_threshold:
            flagged.append(LOOKING_AWAY)

    state["frames_processed"] += 1
    weights = _severity_weights(settings)
    recorded: list[str] = []

    for event_type in _ALL_EVENT_TYPES:
        if event_type in flagged:
            state["consecutive"][event_type] += 1
            if state["consecutive"][event_type] == settings.consecutive_frames_to_flag:
                # Debounce threshold just reached this frame — record the
                # event and reset the streak so a *sustained* condition
                # (e.g. head turned away for 30 straight frames) produces a
                # fresh event every `consecutive_frames_to_flag` frames
                # rather than firing exactly once for the entire session.
                state["consecutive"][event_type] = 0
                state["event_counts"][event_type] += 1
                state["integrity_score"] = max(0.0, state["integrity_score"] - weights[event_type])
                state["events"].append(
                    {"type": event_type, "timestamp": time.time(), "severity": weights[event_type]}
                )
                state["events"] = state["events"][-settings.max_events_retained :]
                recorded.append(event_type)
        else:
            state["consecutive"][event_type] = 0

    return FrameAnalysis(
        faces_detected=len(faces),
        head_pose_deviation_degrees=head_pose_deviation,
        gaze_offset=gaze_offset,
        flagged_this_frame=flagged,
        events_recorded=recorded,
        integrity_score=state["integrity_score"],
    )
