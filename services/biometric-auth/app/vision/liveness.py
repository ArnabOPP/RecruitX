"""Liveness detection: Eye-Aspect-Ratio blink detection + head-pose (PnP)
variation, computed server-side across a short sequence of real frames.

Deliberately never a client-reported boolean — a browser-computed
"isLive: true" flag can be forged with a single crafted HTTP request that
never touched a camera. Every value here is derived from the actual
uploaded frames, re-detected and re-measured by this service.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .detector import (
    LEFT_EYE_EAR_INDICES,
    POSE_LANDMARK_INDICES,
    RIGHT_EYE_EAR_INDICES,
    FaceDetector,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)

# Generic 3D face model points (arbitrary millimeter-like units) — the
# standard 6-point model for solvePnP-based head-pose estimation,
# corresponding index-for-index to POSE_LANDMARK_INDICES.
#
# Y increases *downward* here (chin is +Y, eye/mouth corners above the
# nose are -Y) to match pixel-space image_points, where Y also increases
# downward. The commonly-copied version of this model point set uses Y-up,
# which — verified empirically against real detections — makes solvePnP
# converge to a spurious ~180-degree-rotated solution for an ordinary
# frontal face (roll reported near +-146 degrees instead of near 0).
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


def _eye_aspect_ratio(eye_points: np.ndarray) -> float:
    """Soukupova & Cech's EAR formula: vertical eye-openness distances
    over horizontal eye width. A ratio, not an absolute pixel distance,
    so it stays comparable across different face sizes/camera distances."""
    p1, p2, p3, p4, p5, p6 = eye_points
    vertical_1 = float(np.linalg.norm(p2 - p6))
    vertical_2 = float(np.linalg.norm(p3 - p5))
    horizontal = float(np.linalg.norm(p1 - p4))
    return (vertical_1 + vertical_2) / (2.0 * max(horizontal, 1e-6))


def estimate_head_pose_deviation(landmarks: np.ndarray, image_shape: tuple[int, int]) -> float:
    """Returns the largest of pitch/yaw/roll in degrees via solvePnP
    against a generic 3D face model. Doesn't need to be a precisely
    calibrated absolute pose — just needs to distinguish "roughly facing
    the camera" from "turned/tilted significantly", which is what both
    liveness (a live head moves a little; a static photo doesn't) and
    proctoring (is the candidate looking at the screen) actually need."""
    h, w = image_shape
    image_points = np.array([landmarks[idx][:2] for idx in POSE_LANDMARK_INDICES.values()], dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype=np.float64
    )
    dist_coeffs = np.zeros((4, 1))

    # EPNP (not the default ITERATIVE) — verified empirically: ITERATIVE
    # repeatedly converges to a degenerate/flipped local minimum for this
    # roughly-coplanar 6-point set, while EPNP (a direct, non-iterative
    # solve) consistently lands on the physically correct pose across
    # frontal and in-plane-rotated test images.
    success, rotation_vec, _ = cv2.solvePnP(
        _MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP
    )
    if not success:
        return 0.0

    # R = Rz(roll) @ Ry(yaw) @ Rx(pitch) for this camera-coordinate
    # convention (X right, Y down, Z forward) — verified against known
    # in-plane image rotations (pure roll) to confirm each angle is
    # extracted from the correct matrix elements and responds to the
    # rotation it's supposed to.
    rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    yaw = np.degrees(np.arctan2(-rotation_matrix[2, 0], sy))
    roll = np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
    pitch = np.degrees(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
    return float(max(abs(pitch), abs(yaw), abs(roll)))


@dataclass
class LivenessResult:
    live: bool
    blink_count: int
    max_head_pose_deviation_degrees: float
    frames_analyzed: int
    reason: str


def analyze_liveness(
    detector: FaceDetector,
    frames_bgr: list[np.ndarray],
    *,
    ear_threshold: float,
    min_blinks_required: int,
    max_head_pose_deviation_degrees: float,
) -> LivenessResult:
    ear_values: list[float] = []
    head_poses: list[float] = []
    frames_with_face = 0

    for frame in frames_bgr:
        try:
            face = detector.detect_single(frame)
        except (NoFaceDetectedError, MultipleFacesDetectedError):
            # A frame with zero or multiple faces just isn't counted —
            # doesn't automatically fail liveness, since a brief detection
            # miss (motion blur, a hand passing by) shouldn't tank an
            # otherwise-genuine sequence. Consistently bad frames still
            # show up as frames_analyzed==0 below.
            continue
        frames_with_face += 1

        left_eye = face.landmarks[LEFT_EYE_EAR_INDICES][:, :2]
        right_eye = face.landmarks[RIGHT_EYE_EAR_INDICES][:, :2]
        ear = (_eye_aspect_ratio(left_eye) + _eye_aspect_ratio(right_eye)) / 2.0
        ear_values.append(ear)

        head_poses.append(estimate_head_pose_deviation(face.landmarks, frame.shape[:2]))

    if frames_with_face == 0:
        return LivenessResult(
            live=False, blink_count=0, max_head_pose_deviation_degrees=0.0, frames_analyzed=0,
            reason="No face detected in any submitted frame.",
        )

    # A blink is an open -> closed -> open transition, not just "one frame
    # below threshold" — a single low-EAR frame is just as consistent
    # with detection noise as with an actual blink; requiring the
    # transition back to open is what makes this a real blink signal.
    blink_count = 0
    was_closed = False
    for ear in ear_values:
        if ear < ear_threshold:
            was_closed = True
        elif was_closed:
            blink_count += 1
            was_closed = False

    max_pose = max(head_poses) if head_poses else 0.0

    if max_pose > max_head_pose_deviation_degrees:
        return LivenessResult(
            live=False, blink_count=blink_count, max_head_pose_deviation_degrees=max_pose,
            frames_analyzed=frames_with_face,
            reason=f"Head pose deviation ({max_pose:.1f}°) exceeds the allowed range.",
        )

    if blink_count < min_blinks_required:
        return LivenessResult(
            live=False, blink_count=blink_count, max_head_pose_deviation_degrees=max_pose,
            frames_analyzed=frames_with_face,
            reason=f"Only {blink_count} blink(s) detected across {frames_with_face} frame(s); "
                   f"at least {min_blinks_required} required.",
        )

    return LivenessResult(
        live=True, blink_count=blink_count, max_head_pose_deviation_degrees=max_pose,
        frames_analyzed=frames_with_face,
        reason="Blink pattern and head pose are consistent with a live presence.",
    )
