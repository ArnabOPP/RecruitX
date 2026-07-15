"""Tests for the SQLite-backed embedding store — full CRUD, against a real
temp-file database (no mocking sqlite3)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.db import EmbeddingStore, EnrollmentNotFoundError


@pytest.fixture
def store() -> EmbeddingStore:
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield EmbeddingStore(str(Path(tmp_dir) / "test.db"))


def test_enroll_then_get_round_trips(store: EmbeddingStore):
    embedding = [0.1, 0.2, 0.3]
    store.enroll("candidate-1", embedding, "buffalo_l")
    assert store.get_embedding("candidate-1") == embedding


def test_get_missing_candidate_raises(store: EmbeddingStore):
    with pytest.raises(EnrollmentNotFoundError):
        store.get_embedding("nonexistent")


def test_enroll_upserts_existing_candidate(store: EmbeddingStore):
    store.enroll("candidate-1", [0.1, 0.2], "buffalo_l")
    store.enroll("candidate-1", [0.9, 0.8], "buffalo_l")
    assert store.get_embedding("candidate-1") == [0.9, 0.8]


def test_is_enrolled(store: EmbeddingStore):
    assert store.is_enrolled("candidate-1") is False
    store.enroll("candidate-1", [0.1], "buffalo_l")
    assert store.is_enrolled("candidate-1") is True


def test_delete_removes_enrollment(store: EmbeddingStore):
    store.enroll("candidate-1", [0.1], "buffalo_l")
    store.delete("candidate-1")
    assert store.is_enrolled("candidate-1") is False


def test_delete_nonexistent_candidate_does_not_raise(store: EmbeddingStore):
    store.delete("never-enrolled")  # must not raise


def test_ping_does_not_raise(store: EmbeddingStore):
    store.ping()


def test_two_candidates_are_independent(store: EmbeddingStore):
    store.enroll("candidate-1", [0.1], "buffalo_l")
    store.enroll("candidate-2", [0.2], "buffalo_l")
    assert store.get_embedding("candidate-1") == [0.1]
    assert store.get_embedding("candidate-2") == [0.2]
