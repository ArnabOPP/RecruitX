"""Tests for the deterministic Jaccard/keyword layer — no model involved,
pure set/string math."""

from __future__ import annotations

from app.grading.keywords import jaccard_similarity, keyword_coverage, normalize_tokens


def test_normalize_tokens_lowercases_and_strips_punctuation():
    tokens = normalize_tokens("Hello, World! This is a TEST.")
    assert "hello" in tokens
    assert "world" in tokens
    assert "test" in tokens
    # stopwords/short tokens dropped
    assert "is" not in tokens
    assert "a" not in tokens
    assert "this" not in tokens


def test_normalize_tokens_keeps_tech_tokens_intact():
    tokens = normalize_tokens("I used Node.js and C++ and C#")
    assert "node.js" in tokens
    assert "c++" in tokens
    assert "c#" in tokens


def test_normalize_tokens_can_keep_stopwords():
    tokens = normalize_tokens("this is a test", drop_stopwords=False)
    assert "this" in tokens
    assert "is" in tokens
    assert "a" in tokens


def test_jaccard_similarity_identical_sets():
    a = {"python", "fastapi", "postgresql"}
    assert jaccard_similarity(a, a) == 1.0


def test_jaccard_similarity_disjoint_sets():
    a = {"python", "fastapi"}
    b = {"java", "spring"}
    assert jaccard_similarity(a, b) == 0.0


def test_jaccard_similarity_partial_overlap():
    a = {"python", "fastapi", "postgresql"}
    b = {"python", "django", "mysql"}
    # intersection={python}=1, union={python,fastapi,postgresql,django,mysql}=5
    assert jaccard_similarity(a, b) == 1 / 5


def test_jaccard_similarity_empty_sets_returns_zero():
    assert jaccard_similarity(set(), set()) == 0.0
    assert jaccard_similarity({"a"}, set()) == 0.0
    assert jaccard_similarity(set(), {"a"}) == 0.0


def test_keyword_coverage_finds_matches_and_misses():
    matched, missing = keyword_coverage(
        ["PostgreSQL", "query plan", "sharding"],
        "I optimized the PostgreSQL query plan by adding indexes.",
    )
    assert matched == ["PostgreSQL", "query plan"]
    assert missing == ["sharding"]


def test_keyword_coverage_allows_reordered_phrase_tokens():
    """A candidate saying "the query's execution plan" instead of "query
    plan" verbatim should still count — token-set containment, not exact
    phrase adjacency."""
    matched, missing = keyword_coverage(["query plan"], "I looked at the plan for this query carefully.")
    assert matched == ["query plan"]
    assert missing == []


def test_keyword_coverage_empty_keyword_list():
    matched, missing = keyword_coverage([], "any answer text")
    assert matched == []
    assert missing == []


def test_keyword_coverage_preserves_input_order():
    matched, missing = keyword_coverage(["zebra", "apple", "python"], "I used python and apple products.")
    assert missing == ["zebra"]
    assert matched == ["apple", "python"]
