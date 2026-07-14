"""Redis-backed session state store.

Unlike the other five services (where Redis is an optional, purely-for-
rate-limiting dependency with an in-memory fallback), Redis here *is* the
persistence layer — session state must survive across requests and be
visible to any orchestrator replica, so there's no in-memory fallback
mode. Every other service in Recruitix already depends on Redis for
shared rate limiting; this reuses that same instance rather than
introducing a second piece of infrastructure.
"""

from __future__ import annotations

import json
import uuid

from redis.asyncio import Redis

from .config import get_settings


class SessionNotFoundError(Exception):
    pass


class SessionStore:
    def __init__(self, redis_client: Redis) -> None:
        self._redis = redis_client

    def _key(self, session_id: str) -> str:
        return f"orchestrator:session:{session_id}"

    async def create(self, data: dict) -> str:
        session_id = uuid.uuid4().hex
        data["session_id"] = session_id
        await self.save(session_id, data)
        return session_id

    async def save(self, session_id: str, data: dict) -> None:
        settings = get_settings()
        await self._redis.set(self._key(session_id), json.dumps(data), ex=settings.session_ttl_seconds)

    async def load(self, session_id: str) -> dict:
        raw = await self._redis.get(self._key(session_id))
        if raw is None:
            raise SessionNotFoundError(session_id)
        return json.loads(raw)

    async def ping(self) -> None:
        await self._redis.ping()


_redis_client: Redis | None = None
_session_store: SessionStore | None = None


def get_redis_client() -> Redis:
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = Redis.from_url(settings.redis_uri, decode_responses=True)
    return _redis_client


def get_session_store() -> SessionStore:
    global _session_store
    if _session_store is None:
        _session_store = SessionStore(get_redis_client())
    return _session_store
