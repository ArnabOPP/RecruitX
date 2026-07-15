"""SQLite-backed durable storage for enrolled face embeddings.

The first *durable* store in Recruitix — every other service is either
stateless or holds Redis-TTL'd ephemeral session state. An enrolled
candidate's face embedding is reference data that must survive
indefinitely (until explicitly deleted), not expire with a session, so a
proper on-disk store is the right tool here rather than stretching
Redis's TTL semantics to mean something they don't.

A single SQLite file was chosen over standing up Postgres: this data is
low-throughput (occasional enroll/verify calls, not a hot path) and a
single file needs zero additional infrastructure — consistent with this
project's "reuse what's already proven, don't add infra unless the
workload actually needs it" pattern. Postgres is the natural upgrade path
if this ever needs concurrent-writer scale.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from .config import get_settings


class EnrollmentNotFoundError(Exception):
    pass


class EmbeddingStore:
    """Thread-safe (via a lock — SQLite's own connection is not safe to
    share across threads without one) wrapper around a single SQLite file
    holding candidate_id -> embedding mappings."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS enrollments (
                    candidate_id TEXT PRIMARY KEY,
                    embedding TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    enrolled_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.commit()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self._db_path)
        try:
            yield conn
        finally:
            conn.close()

    def enroll(self, candidate_id: str, embedding: list[float], model_name: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO enrollments (candidate_id, embedding, model_name, updated_at)
                VALUES (?, ?, ?, datetime('now'))
                ON CONFLICT(candidate_id) DO UPDATE SET
                    embedding = excluded.embedding,
                    model_name = excluded.model_name,
                    updated_at = datetime('now')
                """,
                (candidate_id, json.dumps(embedding), model_name),
            )
            conn.commit()

    def get_embedding(self, candidate_id: str) -> list[float]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT embedding FROM enrollments WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
        if row is None:
            raise EnrollmentNotFoundError(candidate_id)
        return json.loads(row[0])

    def is_enrolled(self, candidate_id: str) -> bool:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM enrollments WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
        return row is not None

    def delete(self, candidate_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM enrollments WHERE candidate_id = ?", (candidate_id,))
            conn.commit()

    def ping(self) -> None:
        with self._connect() as conn:
            conn.execute("SELECT 1")


_store: EmbeddingStore | None = None


def get_embedding_store() -> EmbeddingStore:
    global _store
    if _store is None:
        settings = get_settings()
        _store = EmbeddingStore(settings.db_path)
    return _store
