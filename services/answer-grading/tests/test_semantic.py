"""Tests against the real, self-hosted embedding model — not mocked. Unlike
interview-qa/speech-io's "live" tests, there's no external credential to
skip on: everything here runs locally, so these always run. Module-scoped
so the (one-time, ~seconds) model load is shared across this file's tests."""

from __future__ import annotations

import pytest

from app.grading.semantic import SemanticSimilarityScorer


@pytest.fixture(scope="module")
def scorer() -> SemanticSimilarityScorer:
    return SemanticSimilarityScorer("all-MiniLM-L6-v2")


def test_identical_text_scores_near_one(scorer):
    score = scorer.similarity("I optimized the database queries.", "I optimized the database queries.")
    assert score > 0.99


def test_paraphrase_scores_meaningfully_higher_than_unrelated(scorer):
    """The entire reason semantic similarity exists in this engine: catch
    paraphrases that share zero literal keywords with the reference."""
    reference = "Optimized PostgreSQL queries using indexes to reduce latency."
    paraphrase = "I sped up the database by adding indexes to slow queries."
    unrelated = "I really enjoy playing football on weekends with my friends."

    paraphrase_score = scorer.similarity(reference, paraphrase)
    unrelated_score = scorer.similarity(reference, unrelated)

    assert paraphrase_score > unrelated_score
    assert paraphrase_score > 0.4
    assert unrelated_score < 0.3


def test_similarity_is_symmetric(scorer):
    a = "Built a REST API with FastAPI."
    b = "Created a FastAPI-based REST API."
    assert abs(scorer.similarity(a, b) - scorer.similarity(b, a)) < 1e-5


def test_similarity_is_reproducible(scorer):
    a, b = "some answer text", "some reference text"
    assert scorer.similarity(a, b) == scorer.similarity(a, b)


def test_empty_text_returns_zero(scorer):
    assert scorer.similarity("", "something") == 0.0
    assert scorer.similarity("something", "") == 0.0
    assert scorer.similarity("   ", "something") == 0.0


def test_similarity_bounded_between_zero_and_one(scorer):
    score = scorer.similarity("anything", "something completely different entirely")
    assert 0.0 <= score <= 1.0


def test_validate_succeeds(scorer):
    scorer.validate()  # must not raise
