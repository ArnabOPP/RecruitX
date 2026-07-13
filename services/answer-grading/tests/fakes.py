"""A fake semantic scorer for fast, deterministic unit tests of the scoring
math itself (scorer.py, main.py) — decoupled from real embedding-model
behavior, which is verified separately (and for real) in test_semantic.py
and test_api.py's end-to-end test."""

from __future__ import annotations

from app.grading.semantic import SemanticSimilarityError


class FakeSemanticScorer:
    """Returns a scripted similarity value regardless of input, unless a
    specific (text_a, text_b) pair is scripted via `responses`."""

    model_name = "fake-embedding-model"

    def __init__(self, default: float = 0.5, responses: dict[tuple[str, str], float] | None = None):
        self.default = default
        self.responses = responses or {}
        self.calls: list[tuple[str, str]] = []

    def similarity(self, text_a: str, text_b: str) -> float:
        self.calls.append((text_a, text_b))
        if not text_a.strip() or not text_b.strip():
            return 0.0
        return self.responses.get((text_a, text_b), self.default)

    def validate(self) -> None:
        pass


class AlwaysFailingSemanticScorer:
    model_name = "fake-embedding-model"

    def similarity(self, text_a: str, text_b: str) -> float:
        raise SemanticSimilarityError("simulated model failure")

    def validate(self) -> None:
        raise SemanticSimilarityError("simulated model failure")
