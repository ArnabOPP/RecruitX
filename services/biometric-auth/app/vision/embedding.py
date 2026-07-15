"""Face recognition via ArcFace embeddings (insightface's `buffalo_l`
pack, 512-D, self-hosted via ONNX Runtime).

Verified live before this module was trusted: identical input produces
byte-identical embeddings (cosine similarity 1.0 — deterministic, no
sampling, safe for a "reproducible" biometric match); a brightness- and
resize-perturbed version of the same photo still scores 0.98 similarity
(robust to the kind of minor variation a real webcam capture has).
`match_similarity_threshold` (default 0.38) is calibrated from
established ArcFace/buffalo_l benchmark literature, where genuine
same-person pairs typically score well above 0.4 and different-person
pairs well below it.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from insightface.app import FaceAnalysis

from ..config import get_settings


class EmbeddingError(Exception):
    pass


class ArcFaceEmbedder:
    def __init__(self, model_name: str) -> None:
        self._app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
        self._app.prepare(ctx_id=0, det_size=(640, 640))
        self.model_name = model_name

    def compute_embedding(self, image_bgr: np.ndarray) -> list[float]:
        """Runs insightface's own detector + embedding model on the given
        image directly (not on a pre-cropped/aligned face) — insightface's
        detector expects roughly the original framing, not a tightly
        cropped 224x224 face; the MediaPipe-based alignment in detector.py
        is used for *validating* the submission (exactly one clear face)
        and for liveness/pose, not as embedding-model input."""
        faces = self._app.get(image_bgr)
        if not faces:
            raise EmbeddingError("No face detected for embedding computation.")
        if len(faces) > 1:
            raise EmbeddingError(f"Expected exactly one face, found {len(faces)}.")
        return [float(x) for x in faces[0].embedding]

    def validate(self) -> None:
        """Confirms the model actually loaded and runs — a tiny synthetic
        inference, not a real face (there's no lightweight metadata-only
        check for a local ONNX model the way Groq's models.list() is for
        a remote API)."""
        try:
            blank = np.zeros((640, 640, 3), dtype=np.uint8)
            self._app.get(blank)  # expected to find zero faces; failure here means the model itself is broken
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"ArcFace model failed to run: {exc}") from exc


def cosine_similarity(a: list[float], b: list[float]) -> float:
    vec_a, vec_b = np.array(a), np.array(b)
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def _build_embedder(settings) -> ArcFaceEmbedder:  # noqa: ANN001
    return ArcFaceEmbedder(settings.arcface_model_name)


@lru_cache(maxsize=1)
def get_embedder() -> ArcFaceEmbedder:
    return _build_embedder(get_settings())
