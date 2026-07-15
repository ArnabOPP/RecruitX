"""Tests for head-pose and gaze estimation. Head-pose reuses the exact
solvePnP approach fixed and regression-tested in biometric-auth's
liveness.py (see that service's test_liveness.py for the original bug:
Y-axis convention mismatch + wrong solvePnP solver producing a physically
impossible ~146-degree deviation for a frontal face) — the same real-photo
and known-in-plane-rotation checks are repeated here since this is an
independently deployable copy of that logic."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.config import get_settings
from app.vision.detector import FaceDetector
from app.vision.gaze import estimate_gaze_offset, estimate_head_pose_deviation


@pytest.fixture(scope="module")
def detector() -> FaceDetector:
    settings = get_settings()
    d = FaceDetector(settings.face_landmarker_model_path, settings.min_face_detection_confidence)
    yield d
    d.close()


def test_head_pose_deviation_is_small_for_frontal_photo(detector: FaceDetector, face_image_bgr: np.ndarray):
    faces = detector.detect_all(face_image_bgr)
    deviation = estimate_head_pose_deviation(faces[0].landmarks, face_image_bgr.shape[:2])
    assert deviation < 20.0


@pytest.mark.parametrize("rotation_degrees", [15, -15, 30])
def test_head_pose_tracks_known_in_plane_rotation(
    detector: FaceDetector, face_image_bgr: np.ndarray, rotation_degrees: float
):
    h, w = face_image_bgr.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, rotation_degrees, 1.0)
    rotated = cv2.warpAffine(face_image_bgr, matrix, (w, h))

    faces = detector.detect_all(rotated)
    deviation = estimate_head_pose_deviation(faces[0].landmarks, rotated.shape[:2])
    assert abs(deviation - abs(rotation_degrees)) < 10.0


def test_gaze_offset_is_small_for_forward_looking_photo(detector: FaceDetector, face_image_bgr: np.ndarray):
    faces = detector.detect_all(face_image_bgr)
    offset = estimate_gaze_offset(faces[0].landmarks)
    assert 0.0 <= offset < 0.35


def test_gaze_offset_is_bounded_reasonably():
    """Unit-level sanity check on the ratio math itself, independent of
    detection: a perfectly centered iris must score 0 offset."""
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

    lm = np.zeros((478, 3))
    for outer, inner, top, bottom, iris, cx in [
        (LEFT_EYE_OUTER, LEFT_EYE_INNER, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_IRIS_CENTER, 100),
        (RIGHT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_IRIS_CENTER, 300),
    ]:
        lm[outer] = [cx - 20, 100, 0]
        lm[inner] = [cx + 20, 100, 0]
        lm[top] = [cx, 90, 0]
        lm[bottom] = [cx, 110, 0]
        lm[iris] = [cx, 100, 0]  # dead center

    offset = estimate_gaze_offset(lm)
    assert offset == pytest.approx(0.0, abs=1e-6)
