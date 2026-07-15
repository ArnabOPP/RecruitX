"""Unit tests for event classification + severity scoring, isolated from
face detection via a mocked FaceDetector (detection correctness itself is
covered separately in test_detector.py/test_gaze.py)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from app.config import get_settings
from app.severity import (
    HEAD_TURNED,
    LOOKING_AWAY,
    MULTIPLE_FACES,
    NO_FACE,
    analyze_frame,
    new_session_state,
)
from app.vision.detector import DetectedFace


@pytest.fixture(autouse=True)
def _reset_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _frontal_landmarks() -> np.ndarray:
    """A landmark array that estimate_head_pose_deviation/estimate_gaze_offset
    resolve to small ("looking at the screen") values — built from the same
    geometry verified in test_gaze.py's frontal-photo checks, but synthetic
    so this test doesn't depend on loading the real model."""
    from app.vision.detector import POSE_LANDMARK_INDICES

    lm = np.zeros((478, 3))
    lm[POSE_LANDMARK_INDICES["nose_tip"]] = [200, 150, 0]
    lm[POSE_LANDMARK_INDICES["chin"]] = [200, 220, 0]
    lm[POSE_LANDMARK_INDICES["left_eye_corner"]] = [160, 110, 0]
    lm[POSE_LANDMARK_INDICES["right_eye_corner"]] = [240, 110, 0]
    lm[POSE_LANDMARK_INDICES["left_mouth_corner"]] = [175, 190, 0]
    lm[POSE_LANDMARK_INDICES["right_mouth_corner"]] = [225, 190, 0]

    from app.vision.detector import (
        LEFT_EYE_BOTTOM,
        LEFT_EYE_INNER,
        LEFT_EYE_OUTER,
        LEFT_EYE_TOP,
        LEFT_IRIS_CENTER,
        RIGHT_EYE_BOTTOM,
        RIGHT_EYE_INNER,
        RIGHT_EYE_OUTER,
        RIGHT_EYE_TOP,
        RIGHT_IRIS_CENTER,
    )

    for outer, inner, top, bottom, iris, cx in [
        (LEFT_EYE_OUTER, LEFT_EYE_INNER, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_IRIS_CENTER, 160),
        (RIGHT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_IRIS_CENTER, 240),
    ]:
        lm[outer] = [cx - 15, 110, 0]
        lm[inner] = [cx + 15, 110, 0]
        lm[top] = [cx, 103, 0]
        lm[bottom] = [cx, 117, 0]
        lm[iris] = [cx, 110, 0]  # centered gaze
    return lm


def _mock_detector(face_counts: list[int]) -> MagicMock:
    """face_counts[i] is how many faces detect_all should report for the
    i-th call: 0 -> [], 1 -> [one frontal face], 2+ -> [n placeholder faces]."""
    detector = MagicMock()
    state = {"i": 0}

    def fake_detect_all(image):  # noqa: ARG001
        count = face_counts[state["i"]]
        state["i"] += 1
        if count == 0:
            return []
        if count == 1:
            return [DetectedFace(landmarks=_frontal_landmarks())]
        return [DetectedFace(landmarks=_frontal_landmarks()) for _ in range(count)]

    detector.detect_all.side_effect = fake_detect_all
    return detector


def test_no_face_requires_consecutive_frames_before_recording(monkeypatch):
    monkeypatch.setenv("PROCTORING_CONSECUTIVE_FRAMES_TO_FLAG", "3")
    get_settings.cache_clear()

    detector = _mock_detector([0, 0])
    state = new_session_state()
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    result1 = analyze_frame(detector, frame, state)
    assert result1.events_recorded == []
    result2 = analyze_frame(detector, frame, state)
    assert result2.events_recorded == []


def test_no_face_records_event_on_third_consecutive_frame(monkeypatch):
    monkeypatch.setenv("PROCTORING_CONSECUTIVE_FRAMES_TO_FLAG", "3")
    get_settings.cache_clear()

    detector = _mock_detector([0, 0, 0])
    state = new_session_state()
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    analyze_frame(detector, frame, state)
    analyze_frame(detector, frame, state)
    result3 = analyze_frame(detector, frame, state)

    assert result3.events_recorded == [NO_FACE]
    assert state["event_counts"][NO_FACE] == 1
    assert state["integrity_score"] < 100.0


def test_multiple_faces_flagged_immediately_in_this_frame(monkeypatch):
    monkeypatch.setenv("PROCTORING_CONSECUTIVE_FRAMES_TO_FLAG", "1")
    get_settings.cache_clear()

    detector = _mock_detector([2])
    state = new_session_state()
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    result = analyze_frame(detector, frame, state)
    assert result.flagged_this_frame == [MULTIPLE_FACES]
    assert result.events_recorded == [MULTIPLE_FACES]


def test_interrupted_no_face_sequence_resets_consecutive_counter(monkeypatch):
    monkeypatch.setenv("PROCTORING_CONSECUTIVE_FRAMES_TO_FLAG", "3")
    get_settings.cache_clear()

    detector = _mock_detector([0, 0, 1, 0, 0, 0])
    state = new_session_state()
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    recorded = []
    for _ in range(6):
        recorded.extend(analyze_frame(detector, frame, state).events_recorded)

    # A face reappearing after 2 no-face frames must reset the streak — the
    # 3-in-a-row requirement starts over, so only the final 3-frame run
    # (frames 4,5,6) produces a recorded event, not a phantom one at frame 5.
    assert recorded == [NO_FACE]


def test_frontal_face_never_flags_head_turn_or_looking_away():
    detector = _mock_detector([1, 1, 1])
    state = new_session_state()
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    for _ in range(3):
        result = analyze_frame(detector, frame, state)
        assert HEAD_TURNED not in result.flagged_this_frame
        assert LOOKING_AWAY not in result.flagged_this_frame
    assert state["integrity_score"] == 100.0


def test_integrity_score_floors_at_zero(monkeypatch):
    monkeypatch.setenv("PROCTORING_CONSECUTIVE_FRAMES_TO_FLAG", "1")
    monkeypatch.setenv("PROCTORING_SEVERITY_MULTIPLE_FACES", "60")
    get_settings.cache_clear()

    detector = _mock_detector([2, 2, 2])
    state = new_session_state()
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    for _ in range(3):
        analyze_frame(detector, frame, state)

    assert state["integrity_score"] == 0.0


def test_events_list_is_bounded_by_max_events_retained(monkeypatch):
    monkeypatch.setenv("PROCTORING_CONSECUTIVE_FRAMES_TO_FLAG", "1")
    monkeypatch.setenv("PROCTORING_MAX_EVENTS_RETAINED", "5")
    get_settings.cache_clear()

    detector = _mock_detector([0] * 20)
    state = new_session_state()
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    for _ in range(20):
        analyze_frame(detector, frame, state)

    assert len(state["events"]) == 5
