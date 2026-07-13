"""Semantic similarity via a self-hosted sentence-embedding model.

Embeddings are a deterministic function of the model's fixed weights and
the input text — no sampling, no randomness — which is what makes this
safe to use in a "reproducible, auditable" scoring engine, unlike an LLM
chat completion. Same self-hosted-model philosophy as cv-parser's
spaCy/BERT pipeline: the model loads once at process startup, not
per-request, and inference runs on CPU.
"""

from __future__ import annotations

from functools import lru_cache

from ..config import get_settings


class SemanticSimilarityError(Exception):
    """Raised if the embedding model fails to load or run."""


class SemanticSimilarityScorer:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SemanticSimilarityError(f"sentence-transformers is not installed: {exc}") from exc

        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:  # noqa: BLE001
            raise SemanticSimilarityError(f"Could not load embedding model {model_name!r}: {exc}") from exc
        self.model_name = model_name

    def similarity(self, text_a: str, text_b: str) -> float:
        if not text_a.strip() or not text_b.strip():
            return 0.0
        try:
            embeddings = self._model.encode([text_a, text_b], normalize_embeddings=True)
        except Exception as exc:  # noqa: BLE001
            raise SemanticSimilarityError(f"Embedding inference failed: {exc}") from exc

        # Both vectors are L2-normalized, so their dot product is the
        # cosine similarity directly — no separate norm division needed.
        score = float(embeddings[0] @ embeddings[1])
        # Cosine similarity is mathematically in [-1, 1]; clamp to [0, 1]
        # since a "negative relevance" score isn't meaningful for grading.
        return max(0.0, min(1.0, score))

    def validate(self) -> None:
        self.similarity("test", "test")


def _build_scorer(settings) -> SemanticSimilarityScorer:  # noqa: ANN001
    return SemanticSimilarityScorer(settings.semantic_model_name)


@lru_cache(maxsize=1)
def get_semantic_scorer() -> SemanticSimilarityScorer:
    return _build_scorer(get_settings())
