"""Face detection + landmark extraction via MediaPipe FaceLandmarker (468
base + 10 iris landmarks) — the detection half of biometric-auth's
detector.py, without the alignment/crop step proctoring doesn't need (no
embedding is computed here; only landmark positions matter, for gaze and
head-pose).

Uses MediaPipe's Tasks API (`mediapipe.tasks.python.vision`), consistent
with biometric-auth: the installed mediapipe version no longer exposes the
legacy `mp.solutions.face_mesh` at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from ..config import get_settings


@dataclass
class DetectedFace:
    landmarks: np.ndarray  # (478, 3) pixel-space x, y, z


# Same fixed MediaPipe FaceMesh topology indices biometric-auth uses for
# EAR and head-pose, plus the iris center points (only present because
# this model bundle outputs the 478-point iris-refined topology) used here
# for gaze estimation.
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_OUTER = 263
RIGHT_EYE_INNER = 362
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

POSE_LANDMARK_INDICES = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 33,
    "right_eye_corner": 263,
    "left_mouth_corner": 61,
    "right_mouth_corner": 291,
}


class FaceDetector:
    def __init__(self, model_path: str, min_detection_confidence: float) -> None:
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_faces=5,
            min_face_detection_confidence=min_detection_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect_all(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        h, w = image_bgr.shape[:2]
        faces: list[DetectedFace] = []
        for face_landmarks in result.face_landmarks:
            pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in face_landmarks])
            faces.append(DetectedFace(landmarks=pts))
        return faces

    def close(self) -> None:
        self._landmarker.close()


def _build_detector(settings) -> FaceDetector:  # noqa: ANN001
    return FaceDetector(settings.face_landmarker_model_path, settings.min_face_detection_confidence)


@lru_cache(maxsize=1)
def get_face_detector() -> FaceDetector:
    return _build_detector(get_settings())
