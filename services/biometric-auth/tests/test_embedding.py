"""Live tests against the real ArcFace (insightface buffalo_l) embedding
model — determinism, robustness to minor perturbation, and similarity
behavior. No mocks: the whole point of a biometric match is that it's
computed by the real model, so a mocked embedder would test nothing."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.vision.embedding import ArcFaceEmbedder, EmbeddingError, cosine_similarity


@pytest.fixture(scope="module")
def embedder() -> ArcFaceEmbedder:
    return ArcFaceEmbedder("buffalo_l")


def test_embedding_is_512_dimensional(embedder: ArcFaceEmbedder, face_image_bgr: np.ndarray):
    emb = embedder.compute_embedding(face_image_bgr)
    assert len(emb) == 512


def test_embedding_is_deterministic(embedder: ArcFaceEmbedder, face_image_bgr: np.ndarray):
    emb_a = embedder.compute_embedding(face_image_bgr)
    emb_b = embedder.compute_embedding(face_image_bgr)
    assert cosine_similarity(emb_a, emb_b) == pytest.approx(1.0, abs=1e-6)


def test_embedding_robust_to_minor_perturbation(embedder: ArcFaceEmbedder, face_image_bgr: np.ndarray):
    """A brightness/resize perturbation of the same photo should still be
    recognized as the same person (high similarity), the way a real webcam
    capture varies frame to frame."""
    brighter = cv2.convertScaleAbs(face_image_bgr, alpha=1.15, beta=10)
    resized = cv2.resize(brighter, (480, 480))

    emb_original = embedder.compute_embedding(face_image_bgr)
    emb_perturbed = embedder.compute_embedding(resized)
    similarity = cosine_similarity(emb_original, emb_perturbed)
    assert similarity > 0.9


def test_no_face_raises_embedding_error(embedder: ArcFaceEmbedder):
    blank = np.full((480, 640, 3), 200, dtype=np.uint8)
    with pytest.raises(EmbeddingError):
        embedder.compute_embedding(blank)


def test_validate_does_not_raise(embedder: ArcFaceEmbedder):
    embedder.validate()  # must not raise


def test_cosine_similarity_of_identical_vectors_is_one():
    v = [1.0, 2.0, 3.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_zero_vector_is_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
