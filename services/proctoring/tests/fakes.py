"""An in-memory fake session store, for unit/API tests that don't need a
real Redis — see test_session_store_live.py for the tests that do."""

from __future__ import annotations

from app.session_store import SessionNotFoundError
from app.severity import new_session_state


class FakeProctoringSessionStore:
    def __init__(self) -> None:
        self._data: dict[str, dict] = {}

    async def get_or_create(self, session_id: str) -> dict:
        if session_id not in self._data:
            return new_session_state()
        return self._data[session_id]

    async def save(self, session_id: str, state: dict) -> None:
        self._data[session_id] = state

    async def load(self, session_id: str) -> dict:
        if session_id not in self._data:
            raise SessionNotFoundError(session_id)
        return self._data[session_id]

    async def delete(self, session_id: str) -> None:
        self._data.pop(session_id, None)

    async def ping(self) -> None:
        pass
