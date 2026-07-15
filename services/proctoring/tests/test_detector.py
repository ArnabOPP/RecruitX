"""Live tests against the real MediaPipe FaceLandmarker model."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.config import get_settings
from app.vision.detector import FaceDetector


@pytest.fixture(scope="module")
def detector() -> FaceDetector:
    settings = get_settings()
    d = FaceDetector(settings.face_landmarker_model_path, settings.min_face_detection_confidence)
    yield d
    d.close()


def test_detects_exactly_one_face_in_real_photo(detector: FaceDetector, face_image_bgr: np.ndarray):
    faces = detector.detect_all(face_image_bgr)
    assert len(faces) == 1
    assert faces[0].landmarks.shape == (478, 3)


def test_no_face_in_blank_image(detector: FaceDetector):
    blank = np.full((480, 640, 3), 200, dtype=np.uint8)
    assert detector.detect_all(blank) == []


def test_detects_two_faces_in_two_face_image(detector: FaceDetector, face_image_bgr: np.ndarray):
    h, w = face_image_bgr.shape[:2]
    half = w // 2
    face_small = cv2.resize(face_image_bgr, (half, h))
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    canvas[:, :half] = face_small
    canvas[:, half : half * 2] = face_small

    faces = detector.detect_all(canvas)
    assert len(faces) == 2
