"""Tests against a real Redis-backed SessionStore — not mocked. Unlike the
other services (where Redis is an optional rate-limit backend, tested via
the shared `limits`-library pattern), this service's session store *is*
Redis, so this tests the actual SessionStore class end-to-end against a
real container.

Requires a reachable Docker daemon to spin up a throwaway `redis:7-alpine`
container; skips cleanly if Docker isn't available.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import time

import pytest
from redis.asyncio import Redis

from app.session_store import SessionNotFoundError, SessionStore


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=10, check=False)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def redis_uri():
    if not _docker_available():
        pytest.skip("Docker is not available in this environment")

    port = _free_port()
    container_name = "orchestrator-test-redis-session-store"
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, check=False)
    run = subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", container_name, "-p", f"{port}:6379", "redis:7-alpine"],
        capture_output=True, text=True, check=False,
    )
    if run.returncode != 0:
        pytest.skip(f"Could not start test Redis container: {run.stderr}")

    uri = f"redis://localhost:{port}"
    try:
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                subprocess.run(["docker", "exec", container_name, "redis-cli", "ping"], capture_output=True, timeout=3, check=True)
                break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                time.sleep(0.3)
        else:
            pytest.skip("Test Redis container never became reachable")
        yield uri
    finally:
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, check=False)


@pytest.fixture
async def store(redis_uri):
    client = Redis.from_url(redis_uri, decode_responses=True)
    yield SessionStore(client)
    await client.aclose()


@pytest.mark.asyncio
async def test_ping_succeeds_against_real_redis(store):
    await store.ping()


@pytest.mark.asyncio
async def test_create_and_load_round_trips(store):
    session_id = await store.create({"round": "personal", "history": []})
    loaded = await store.load(session_id)
    assert loaded["round"] == "personal"
    assert loaded["session_id"] == session_id


@pytest.mark.asyncio
async def test_save_updates_existing_session(store):
    session_id = await store.create({"round": "personal"})
    await store.save(session_id, {"session_id": session_id, "round": "hr"})
    loaded = await store.load(session_id)
    assert loaded["round"] == "hr"


@pytest.mark.asyncio
async def test_load_unknown_session_raises(store):
    with pytest.raises(SessionNotFoundError):
        await store.load("does-not-exist")


@pytest.mark.asyncio
async def test_sessions_are_independently_stored(store):
    id_a = await store.create({"round": "personal", "marker": "a"})
    id_b = await store.create({"round": "hr", "marker": "b"})
    loaded_a = await store.load(id_a)
    loaded_b = await store.load(id_b)
    assert loaded_a["marker"] == "a"
    assert loaded_b["marker"] == "b"
