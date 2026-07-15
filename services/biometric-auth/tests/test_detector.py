"""Live tests against the real MediaPipe FaceLandmarker model — detection,
single-vs-multiple-face enforcement, and eye-leveling alignment."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.config import get_settings
from app.vision.detector import (
    FaceDetector,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)


@pytest.fixture(scope="module")
def detector() -> FaceDetector:
    settings = get_settings()
    d = FaceDetector(settings.face_landmarker_model_path, settings.min_face_detection_confidence)
    yield d
    d.close()


def test_detects_exactly_one_face_in_real_photo(detector: FaceDetector, face_image_bgr: np.ndarray):
    face = detector.detect_single(face_image_bgr)
    assert face.landmarks.shape == (478, 3)
    assert face.bbox[2] > 0 and face.bbox[3] > 0


def test_no_face_raises_on_blank_image(detector: FaceDetector):
    blank = np.full((480, 640, 3), 200, dtype=np.uint8)
    with pytest.raises(NoFaceDetectedError):
        detector.detect_single(blank)


def test_multiple_faces_raises_on_two_face_image(detector: FaceDetector, face_image_bgr: np.ndarray):
    h, w = face_image_bgr.shape[:2]
    half = w // 2
    face_small = cv2.resize(face_image_bgr, (half, h))
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    canvas[:, :half] = face_small
    canvas[:, half : half * 2] = face_small

    with pytest.raises(MultipleFacesDetectedError) as exc_info:
        detector.detect_single(canvas)
    assert exc_info.value.count == 2


def test_alignment_produces_correctly_sized_output(detector: FaceDetector, face_image_bgr: np.ndarray):
    face = detector.detect_single(face_image_bgr)
    assert face.aligned_face.shape == (224, 224, 3)


def test_alignment_levels_the_eyes(detector: FaceDetector, face_image_bgr: np.ndarray):
    """Re-detecting on the aligned output, the two outer-eye landmarks
    should be at (almost) the same height — that's the entire point of the
    alignment step."""
    face = detector.detect_single(face_image_bgr)
    realigned = detector.detect_single(face.aligned_face)

    left_eye_y = realigned.landmarks[33][1]
    right_eye_y = realigned.landmarks[263][1]
    assert abs(left_eye_y - right_eye_y) < 5.0
