"""An in-memory fake session store, for unit tests that don't need a real
Redis — see test_session_store_live.py for the tests that do."""

from __future__ import annotations

import uuid

from app.session_store import SessionNotFoundError


class FakeSessionStore:
    def __init__(self) -> None:
        self._data: dict[str, dict] = {}

    async def create(self, data: dict) -> str:
        session_id = uuid.uuid4().hex
        data["session_id"] = session_id
        self._data[session_id] = data
        return session_id

    async def save(self, session_id: str, data: dict) -> None:
        self._data[session_id] = data

    async def load(self, session_id: str) -> dict:
        if session_id not in self._data:
            raise SessionNotFoundError(session_id)
        return self._data[session_id]

    async def ping(self) -> None:
        pass
