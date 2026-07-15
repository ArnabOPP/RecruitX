"""Shared pytest fixtures/config.

Env defaults set here only apply if the environment doesn't already specify
a value (setdefault), so CI can still override. Real vision models (a
downloaded MediaPipe bundle + insightface's self-downloading buffalo_l
pack) and a real face photo are used throughout this suite rather than
mocks — consistent with the rest of Recruitix: verification against a real
pipeline is the only way to actually trust a computer-vision result.
"""

from __future__ import annotations

import os

os.environ.setdefault("BIOMETRIC_AUTH_RATE_LIMIT_ENABLED", "0")
os.environ.setdefault("BIOMETRIC_AUTH_REQUIRE_API_KEY", "0")
# Verified empirically: MediaPipe's FaceLandmarker needs a lower confidence
# floor to reliably detect both faces in the synthetic two-faces-in-one-
# frame fixture below (each face there is scaled to half width, which
# lowers its detection confidence versus a full-frame single face). 0.3 is
# still a reasonable, non-default-changing floor for real deployments too.
os.environ.setdefault("BIOMETRIC_AUTH_MIN_FACE_DETECTION_CONFIDENCE", "0.3")

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
os.environ.setdefault("BIOMETRIC_AUTH_FACE_LANDMARKER_MODEL_PATH", str(_MODELS_DIR / "face_landmarker.task"))

_tmp_db_dir = tempfile.mkdtemp(prefix="biometric-auth-test-db-")
os.environ.setdefault("BIOMETRIC_AUTH_DB_PATH", str(Path(_tmp_db_dir) / "test_biometric.db"))


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    shutil.rmtree(_tmp_db_dir, ignore_errors=True)


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
    """Two real, separated faces side by side in one square frame — used
    to exercise the "more than one face" rejection path with genuine
    detections rather than a mocked count. Kept square: verified
    empirically that MediaPipe's FaceLandmarker misses faces entirely on
    a very wide/non-square canvas, so faces are scaled to half-width
    rather than the canvas being widened."""
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
