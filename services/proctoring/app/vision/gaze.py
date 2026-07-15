"""Gaze and head-pose estimation, both server-computed from real landmark
positions — the proctoring counterpart to biometric-auth's liveness.py.

Head-pose estimation reuses the exact solvePnP approach biometric-auth's
liveness module uses, including the same two fixes verified there: 3D
model points in a Y-down convention (matching pixel-space image points,
not the Y-up convention many copied tutorials use) and the SOLVEPNP_EPNP
solver (not the default ITERATIVE, which was found to repeatedly converge
to a spurious ~180-degree-rotated solution for this roughly-coplanar
6-point set).
"""

from __future__ import annotations

import cv2
import numpy as np

from .detector import (
    LEFT_EYE_BOTTOM,
    LEFT_EYE_INNER,
    LEFT_EYE_OUTER,
    LEFT_EYE_TOP,
    LEFT_IRIS_CENTER,
    POSE_LANDMARK_INDICES,
    RIGHT_EYE_BOTTOM,
    RIGHT_EYE_INNER,
    RIGHT_EYE_OUTER,
    RIGHT_EYE_TOP,
    RIGHT_IRIS_CENTER,
)

_MODEL_POINTS_3D = np.array(
    [
        [0.0, 0.0, 0.0],  # nose tip
        [0.0, 330.0, -65.0],  # chin
        [-225.0, -170.0, -135.0],  # left eye corner
        [225.0, -170.0, -135.0],  # right eye corner
        [-150.0, 150.0, -125.0],  # left mouth corner
        [150.0, 150.0, -125.0],  # right mouth corner
    ],
    dtype=np.float64,
)


def estimate_head_pose_deviation(landmarks: np.ndarray, image_shape: tuple[int, int]) -> float:
    """Returns the largest of pitch/yaw/roll in degrees. See
    biometric-auth/app/vision/liveness.py for the empirical verification
    (a real frontal photo, and known in-plane rotations) behind this
    specific formulation."""
    h, w = image_shape
    image_points = np.array([landmarks[idx][:2] for idx in POSE_LANDMARK_INDICES.values()], dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64
    )
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, _ = cv2.solvePnP(
        _MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP
    )
    if not success:
        return 0.0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    yaw = np.degrees(np.arctan2(-rotation_matrix[2, 0], sy))
    roll = np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
    pitch = np.degrees(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
    return float(max(abs(pitch), abs(yaw), abs(roll)))


def _eye_gaze_ratios(landmarks: np.ndarray, outer: int, inner: int, top: int, bottom: int, iris: int) -> tuple[float, float]:
    """Returns (horizontal_ratio, vertical_ratio) of the iris center within
    the eye box, each in [0, 1] with 0.5 meaning centered. Independent of
    head pose — this is genuine *eye* gaze direction, not head direction."""
    outer_pt, inner_pt, top_pt, bottom_pt, iris_pt = (
        landmarks[outer][:2], landmarks[inner][:2], landmarks[top][:2], landmarks[bottom][:2], landmarks[iris][:2]
    )
    eye_width = float(np.linalg.norm(inner_pt - outer_pt))
    eye_height = float(np.linalg.norm(bottom_pt - top_pt))
    horizontal = float(np.linalg.norm(iris_pt - outer_pt)) / max(eye_width, 1e-6)
    vertical = float(np.linalg.norm(iris_pt - top_pt)) / max(eye_height, 1e-6)
    return horizontal, vertical


def estimate_gaze_offset(landmarks: np.ndarray) -> float:
    """Returns a single 0..1-ish magnitude of how far the gaze is averted
    from center, averaged across both eyes (max of the horizontal/vertical
    offset per eye, then averaged across eyes) — 0 is a dead-center gaze,
    larger values mean the eyes are looking increasingly off to a side or
    up/down rather than at the screen."""
    left_h, left_v = _eye_gaze_ratios(
        landmarks, LEFT_EYE_OUTER, LEFT_EYE_INNER, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_IRIS_CENTER
    )
    right_h, right_v = _eye_gaze_ratios(
        landmarks, RIGHT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_IRIS_CENTER
    )
    left_offset = max(abs(left_h - 0.5), abs(left_v - 0.5))
    right_offset = max(abs(right_h - 0.5), abs(right_v - 0.5))
    return (left_offset + right_offset) / 2.0 * 2.0  # scaled so a fully-averted eye is ~1.0
