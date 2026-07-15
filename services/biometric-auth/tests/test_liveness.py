"""Tests for liveness detection: head-pose sanity (against a real photo and
against known in-plane rotations — the ground truth that exposed and then
confirmed the fix for a solvePnP axis-convention bug) and the blink
open->closed->open state machine (unit-level, against synthetic landmark
sequences, isolated from face detection which is covered separately in
test_detector.py)."""

from __future__ import annotations

from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from app.vision.detector import (
    LEFT_EYE_EAR_INDICES,
    POSE_LANDMARK_INDICES,
    RIGHT_EYE_EAR_INDICES,
    DetectedFace,
    FaceDetector,
)
from app.vision.liveness import analyze_liveness, estimate_head_pose_deviation


@pytest.fixture(scope="module")
def detector() -> FaceDetector:
    from app.config import get_settings

    settings = get_settings()
    d = FaceDetector(settings.face_landmarker_model_path, settings.min_face_detection_confidence)
    yield d
    d.close()


# --- Head pose -----------------------------------------------------------


def test_head_pose_deviation_is_small_for_frontal_photo(detector: FaceDetector, face_image_bgr: np.ndarray):
    """Regression test for the solvePnP bug: a straightforward frontal
    photo used to produce a physically impossible ~146-degree deviation
    because of a Y-axis convention mismatch between the 3D model points
    and image-space landmarks. It must now be small."""
    face = detector.detect_single(face_image_bgr)
    deviation = estimate_head_pose_deviation(face.landmarks, face_image_bgr.shape[:2])
    assert deviation < 20.0


@pytest.mark.parametrize("rotation_degrees", [15, -15, 30])
def test_head_pose_tracks_known_in_plane_rotation(
    detector: FaceDetector, face_image_bgr: np.ndarray, rotation_degrees: float
):
    """A pure in-plane image rotation is pure roll — the estimated
    deviation should scale with it (loosely; this is a sanity/regression
    check, not a precision calibration test) rather than staying pinned
    near the baseline frontal value or exploding to ~180 degrees."""
    h, w = face_image_bgr.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, rotation_degrees, 1.0)
    rotated = cv2.warpAffine(face_image_bgr, matrix, (w, h))

    face = detector.detect_single(rotated)
    deviation = estimate_head_pose_deviation(face.landmarks, rotated.shape[:2])

    assert abs(deviation - abs(rotation_degrees)) < 10.0


# --- Blink state machine (unit-level, synthetic landmarks) ---------------


def _make_landmarks(eyes_open: bool) -> np.ndarray:
    lm = np.zeros((478, 3))
    vertical_gap = 8.0 if eyes_open else 0.5
    for idx_set, center_x in [(LEFT_EYE_EAR_INDICES, 100), (RIGHT_EYE_EAR_INDICES, 300)]:
        p1, p2, p3, p4, p5, p6 = idx_set
        lm[p1] = [center_x - 20, 100, 0]
        lm[p4] = [center_x + 20, 100, 0]
        lm[p2] = [center_x - 7, 100 - vertical_gap, 0]
        lm[p3] = [center_x + 7, 100 - vertical_gap, 0]
        lm[p6] = [center_x - 7, 100 + vertical_gap, 0]
        lm[p5] = [center_x + 7, 100 + vertical_gap, 0]
    # A non-degenerate, roughly frontal spread for the pose landmarks —
    # solvePnP needs real geometric spread, not six coincident points.
    lm[POSE_LANDMARK_INDICES["nose_tip"]] = [200, 150, 0]
    lm[POSE_LANDMARK_INDICES["chin"]] = [200, 220, 0]
    lm[POSE_LANDMARK_INDICES["left_eye_corner"]] = [160, 110, 0]
    lm[POSE_LANDMARK_INDICES["right_eye_corner"]] = [240, 110, 0]
    lm[POSE_LANDMARK_INDICES["left_mouth_corner"]] = [175, 190, 0]
    lm[POSE_LANDMARK_INDICES["right_mouth_corner"]] = [225, 190, 0]
    return lm


def _mock_detector(eye_state_sequence: list[bool]) -> MagicMock:
    detector = MagicMock()
    state = {"i": 0}

    def fake_detect_single(frame):  # noqa: ARG001
        eyes_open = eye_state_sequence[state["i"]]
        state["i"] += 1
        return DetectedFace(
            landmarks=_make_landmarks(eyes_open),
            bbox=(0, 0, 400, 300),
            aligned_face=np.zeros((224, 224, 3), dtype=np.uint8),
        )

    detector.detect_single.side_effect = fake_detect_single
    return detector


def _dummy_frames(n: int) -> list[np.ndarray]:
    return [np.zeros((300, 400, 3), dtype=np.uint8) for _ in range(n)]


def test_open_closed_open_sequence_counts_one_blink():
    sequence = [True, True, False, True, True]
    detector = _mock_detector(sequence)
    result = analyze_liveness(
        detector, _dummy_frames(len(sequence)),
        ear_threshold=0.21, min_blinks_required=1, max_head_pose_deviation_degrees=45.0,
    )
    assert result.blink_count == 1
    assert result.live is True


def test_two_separate_blinks_counted_correctly():
    sequence = [True, True, False, False, True, True, True, False, True, True]
    detector = _mock_detector(sequence)
    result = analyze_liveness(
        detector, _dummy_frames(len(sequence)),
        ear_threshold=0.21, min_blinks_required=1, max_head_pose_deviation_degrees=45.0,
    )
    assert result.blink_count == 2


def test_never_closed_counts_zero_blinks_and_fails_liveness():
    sequence = [True] * 10
    detector = _mock_detector(sequence)
    result = analyze_liveness(
        detector, _dummy_frames(len(sequence)),
        ear_threshold=0.21, min_blinks_required=1, max_head_pose_deviation_degrees=45.0,
    )
    assert result.blink_count == 0
    assert result.live is False
    assert "blink" in result.reason.lower()


def test_closed_without_reopening_does_not_count_as_blink():
    """A blink is an open->closed->open transition — ending the sequence
    still closed must not count, since that's just as consistent with the
    frame burst simply cutting off mid-blink."""
    sequence = [True, True, False, False, False]
    detector = _mock_detector(sequence)
    result = analyze_liveness(
        detector, _dummy_frames(len(sequence)),
        ear_threshold=0.21, min_blinks_required=1, max_head_pose_deviation_degrees=45.0,
    )
    assert result.blink_count == 0


def test_no_face_in_any_frame_fails_with_zero_frames_analyzed():
    detector = MagicMock()
    from app.vision.detector import NoFaceDetectedError

    detector.detect_single.side_effect = NoFaceDetectedError("no face")
    result = analyze_liveness(
        detector, _dummy_frames(5),
        ear_threshold=0.21, min_blinks_required=1, max_head_pose_deviation_degrees=45.0,
    )
    assert result.live is False
    assert result.frames_analyzed == 0
