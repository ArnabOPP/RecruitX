"""Tests against a real Redis-backed ProctoringSessionStore — not mocked.
Same pattern as orchestrator/tests/test_session_store_live.py: spins up a
throwaway `redis:7-alpine` container and skips cleanly if Docker isn't
available."""

from __future__ import annotations

import shutil
import socket
import subprocess
import time

import pytest
from redis.asyncio import Redis

from app.session_store import ProctoringSessionStore, SessionNotFoundError
from app.severity import new_session_state


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
    container_name = "proctoring-test-redis-session-store"
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
    yield ProctoringSessionStore(client)
    await client.aclose()


@pytest.mark.asyncio
async def test_ping_succeeds_against_real_redis(store):
    await store.ping()


@pytest.mark.asyncio
async def test_get_or_create_returns_fresh_state_for_unknown_session(store):
    state = await store.get_or_create("brand-new-session")
    assert state == new_session_state()


@pytest.mark.asyncio
async def test_save_then_get_or_create_round_trips(store):
    state = new_session_state()
    state["frames_processed"] = 7
    await store.save("session-a", state)

    loaded = await store.get_or_create("session-a")
    assert loaded["frames_processed"] == 7


@pytest.mark.asyncio
async def test_load_unknown_session_raises(store):
    with pytest.raises(SessionNotFoundError):
        await store.load("does-not-exist")


@pytest.mark.asyncio
async def test_delete_removes_session(store):
    await store.save("session-to-delete", new_session_state())
    await store.delete("session-to-delete")
    with pytest.raises(SessionNotFoundError):
        await store.load("session-to-delete")


@pytest.mark.asyncio
async def test_sessions_are_independently_stored(store):
    state_a = new_session_state()
    state_a["frames_processed"] = 1
    state_b = new_session_state()
    state_b["frames_processed"] = 2

    await store.save("session-x", state_a)
    await store.save("session-y", state_b)

    loaded_a = await store.load("session-x")
    loaded_b = await store.load("session-y")
    assert loaded_a["frames_processed"] == 1
    assert loaded_b["frames_processed"] == 2
