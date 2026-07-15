"""Redis-backed proctoring session state store — same role and pattern as
the orchestrator's session_store.py. Proctoring state (event counts,
integrity score, event log) is ephemeral and scoped to one interview
sitting, not durable reference data, so Redis (already relied on by every
service for rate limiting) is the right store, with no in-memory fallback:
state must survive across requests and be visible to any replica.
"""

from __future__ import annotations

import json

from redis.asyncio import Redis

from .config import get_settings
from .severity import new_session_state


class SessionNotFoundError(Exception):
    pass


class ProctoringSessionStore:
    def __init__(self, redis_client: Redis) -> None:
        self._redis = redis_client

    def _key(self, session_id: str) -> str:
        return f"proctoring:session:{session_id}"

    async def get_or_create(self, session_id: str) -> dict:
        raw = await self._redis.get(self._key(session_id))
        if raw is None:
            return new_session_state()
        return json.loads(raw)

    async def save(self, session_id: str, state: dict) -> None:
        settings = get_settings()
        await self._redis.set(self._key(session_id), json.dumps(state), ex=settings.session_ttl_seconds)

    async def load(self, session_id: str) -> dict:
        raw = await self._redis.get(self._key(session_id))
        if raw is None:
            raise SessionNotFoundError(session_id)
        return json.loads(raw)

    async def delete(self, session_id: str) -> None:
        await self._redis.delete(self._key(session_id))

    async def ping(self) -> None:
        await self._redis.ping()


_redis_client: Redis | None = None
_session_store: ProctoringSessionStore | None = None


def get_redis_client() -> Redis:
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = Redis.from_url(settings.redis_uri, decode_responses=True)
    return _redis_client


def get_session_store() -> ProctoringSessionStore:
    global _session_store
    if _session_store is None:
        _session_store = ProctoringSessionStore(get_redis_client())
    return _session_store
