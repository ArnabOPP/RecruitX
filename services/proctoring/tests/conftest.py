"""Shared pytest fixtures/config.

Env defaults set here only apply if the environment doesn't already
specify a value (setdefault), so CI can still override.
"""

from __future__ import annotations

import os

os.environ.setdefault("PROCTORING_RATE_LIMIT_ENABLED", "0")
os.environ.setdefault("PROCTORING_REQUIRE_API_KEY", "0")
# See biometric-auth/tests/conftest.py for the empirical reasoning: a
# scaled-down (half-width) face in the synthetic two-faces fixture needs a
# lower confidence floor than a full-frame single face to be reliably
# detected by MediaPipe's FaceLandmarker.
os.environ.setdefault("PROCTORING_MIN_FACE_DETECTION_CONFIDENCE", "0.3")

from pathlib import Path

import cv2
import numpy as np
import pytest

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
os.environ.setdefault("PROCTORING_FACE_LANDMARKER_MODEL_PATH", str(_MODELS_DIR / "face_landmarker.task"))


@pytest.fixture(scope="session")
def face_image_bgr() -> np.ndarray:
    """A real, public-domain face photo (skimage's standard test image) —
    a legitimate stand-in for a camera capture in an environment with no
    webcam access."""
    from skimage import data

    rgb = data.astronaut()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


@pytest.fixture(scope="session")
def face_image_bytes(face_image_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", face_image_bgr)
    assert ok
    return buf.tobytes()


@pytest.fixture(scope="session")
def two_faces_image_bytes(face_image_bgr: np.ndarray) -> bytes:
    """Two real, separated faces side by side in one square frame — kept
    square since MediaPipe's FaceLandmarker was found empirically to miss
    faces entirely on a very wide/non-square canvas (see
    biometric-auth/tests/conftest.py)."""
    h, w = face_image_bgr.shape[:2]
    half = w // 2
    face_small = cv2.resize(face_image_bgr, (half, h))
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    canvas[:, :half] = face_small
    canvas[:, half : half * 2] = face_small
    ok, buf = cv2.imencode(".jpg", canvas)
    assert ok
    return buf.tobytes()


@pytest.fixture(scope="session")
def blank_image_bytes() -> bytes:
    blank = np.full((480, 640, 3), 200, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", blank)
    assert ok
    return buf.tobytes()
