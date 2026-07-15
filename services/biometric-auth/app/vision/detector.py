"""Face detection, landmark extraction, and alignment via MediaPipe
FaceLandmarker (468+10 iris landmarks) and OpenCV.

Uses MediaPipe's newer Tasks API (`mediapipe.tasks.python.vision`), not
the legacy `mp.solutions.face_mesh` used in most older tutorials — the
installed mediapipe version (0.10.35) no longer exposes `mp.solutions` at
all, confirmed directly rather than assumed from documentation. The Tasks
API needs a downloaded `.task` model bundle (see README for the official
Google-hosted download URL and Dockerfile for how it's baked into the
image), the same "pre-bake weights at build time" pattern used for
cv-parser's transformer model and answer-grading's embedding model.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from ..config import get_settings


class FaceDetectionError(Exception):
    pass


class NoFaceDetectedError(FaceDetectionError):
    pass


class MultipleFacesDetectedError(FaceDetectionError):
    def __init__(self, count: int) -> None:
        super().__init__(f"Expected exactly one face, found {count}.")
        self.count = count


@dataclass
class DetectedFace:
    landmarks: np.ndarray  # (N, 3) pixel-space x, y, z
    bbox: tuple[int, int, int, int]  # x, y, w, h in pixel coordinates
    aligned_face: np.ndarray  # cropped + eye-leveled BGR image, ready for embedding


# MediaPipe FaceMesh's fixed landmark topology — these indices are stable
# across every detection (the model always outputs landmarks in the same
# order), used both for alignment (outer eye corners as reference points)
# and EAR-based liveness (the full 6-point eye contour per the original
# Soukupova & Cech formulation).
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_OUTER = 263
LEFT_EYE_EAR_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_INDICES = [263, 387, 385, 362, 380, 373]

# A handful of stable landmarks spanning the face depth axis, used for
# head-pose estimation via solvePnP (see liveness.py) — nose tip, chin,
# eye corners, mouth corners, the standard 6-point set for this technique.
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
            xs, ys = pts[:, 0], pts[:, 1]
            x1, x2 = int(max(0, xs.min())), int(min(w, xs.max()))
            y1, y2 = int(max(0, ys.min())), int(min(h, ys.max()))
            bbox = (x1, y1, x2 - x1, y2 - y1)
            aligned = _align_face(image_bgr, pts)
            faces.append(DetectedFace(landmarks=pts, bbox=bbox, aligned_face=aligned))
        return faces

    def detect_single(self, image_bgr: np.ndarray) -> DetectedFace:
        """The common case for enrollment/verification: exactly one clear
        face is required. Zero or multiple faces both raise — a caller
        shouldn't silently pick "the first face" when more than one
        appears, since that's exactly the kind of ambiguity a biometric
        auth system can't afford to guess through."""
        faces = self.detect_all(image_bgr)
        if not faces:
            raise NoFaceDetectedError("No face detected in image.")
        if len(faces) > 1:
            raise MultipleFacesDetectedError(len(faces))
        return faces[0]

    def close(self) -> None:
        self._landmarker.close()


def _align_face(image_bgr: np.ndarray, landmarks: np.ndarray, output_size: int = 224) -> np.ndarray:
    """Rotates/scales/crops so the eyes are level and centered — standard
    face-recognition preprocessing. Embedding models are trained on
    aligned faces and are measurably less accurate on tilted or
    off-center ones, so this isn't cosmetic."""
    left_eye = landmarks[_LEFT_EYE_OUTER][:2]
    right_eye = landmarks[_RIGHT_EYE_OUTER][:2]

    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    eye_distance = np.hypot(dx, dy)
    desired_eye_distance = output_size * 0.4
    scale = desired_eye_distance / max(eye_distance, 1e-6)

    rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale)
    rotation_matrix[0, 2] += output_size * 0.5 - eye_center[0]
    rotation_matrix[1, 2] += output_size * 0.35 - eye_center[1]

    return cv2.warpAffine(image_bgr, rotation_matrix, (output_size, output_size), flags=cv2.INTER_LINEAR)


def _build_detector(settings) -> FaceDetector:  # noqa: ANN001
    return FaceDetector(settings.face_landmarker_model_path, settings.min_face_detection_confidence)


@lru_cache(maxsize=1)
def get_face_detector() -> FaceDetector:
    return _build_detector(get_settings())
