"""Verifies the Redis-backed rate limiter actually shares state across
separate processes — the entire point of CV_PARSER_RATE_LIMIT_STORAGE_URI.

This exercises the same `limits` library storage layer that slowapi uses
internally (Limiter(storage_uri=...) just calls `storage_from_string` under
the hood), by creating two independent storage connections — simulating two
app replicas that have never talked to each other except through Redis —
and confirming a hit recorded through one is visible to the other.

Requires a reachable Docker daemon to spin up a throwaway `redis:7-alpine`
container; skips cleanly if Docker isn't available (e.g. some CI runners),
mirroring the OCR tests' skip-if-Tesseract-unavailable pattern.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import time

import pytest
from limits import RateLimitItemPerMinute, storage
from limits.strategies import FixedWindowRateLimiter


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=10, check=False
        )
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
    container_name = "cv-parser-test-redis-ratelimit"
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, check=False)
    run = subprocess.run(
        [
            "docker", "run", "-d", "--rm",
            "--name", container_name,
            "-p", f"{port}:6379",
            "redis:7-alpine",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if run.returncode != 0:
        pytest.skip(f"Could not start test Redis container: {run.stderr}")

    uri = f"redis://localhost:{port}"
    try:
        # Wait for Redis to accept connections rather than a fixed sleep.
        deadline = time.time() + 15
        last_error = None
        while time.time() < deadline:
            try:
                probe = storage.storage_from_string(uri)
                probe.check()
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(0.3)
        else:
            pytest.skip(f"Test Redis container never became reachable: {last_error}")

        yield uri
    finally:
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, check=False)


def test_redis_storage_shares_hits_across_independent_connections(redis_uri):
    """The core guarantee: two separate `storage_from_string(uri)` calls —
    standing in for two app replicas that each construct their own Limiter
    at startup — must see the same hit count, because they're both backed
    by the same Redis instance rather than each process's own memory.
    """
    limit = RateLimitItemPerMinute(5)

    storage_a = storage.storage_from_string(redis_uri)
    storage_b = storage.storage_from_string(redis_uri)
    limiter_a = FixedWindowRateLimiter(storage_a)
    limiter_b = FixedWindowRateLimiter(storage_b)

    key = "shared-key-for-multi-replica-test"

    # Alternate hits between the two independent connections, exactly like
    # alternating requests between two app replicas behind a load balancer.
    results = []
    for i in range(8):
        limiter = limiter_a if i % 2 == 0 else limiter_b
        results.append(limiter.hit(limit, key))

    allowed = sum(results)
    denied = len(results) - allowed

    assert allowed == 5, f"expected exactly 5 allowed hits (the shared limit), got {allowed}"
    assert denied == 3

    # And an independent third connection agrees on the current count —
    # further proof this is server-side shared state, not something cached
    # in either Python process's memory.
    storage_c = storage.storage_from_string(redis_uri)
    limiter_c = FixedWindowRateLimiter(storage_c)
    assert limiter_c.hit(limit, key) is False


def test_redis_storage_is_isolated_per_key(redis_uri):
    """Sanity check the fixture/limit mechanics themselves: a different
    rate-limit key (e.g. a different client IP) must not be affected by
    another key's hits, confirming test isolation is real and not a fluke
    of shared global state."""
    limit = RateLimitItemPerMinute(5)
    store = storage.storage_from_string(redis_uri)
    limiter = FixedWindowRateLimiter(store)

    for _ in range(5):
        assert limiter.hit(limit, "key-one") is True
    assert limiter.hit(limit, "key-one") is False

    # A fresh key starts with its own budget.
    assert limiter.hit(limit, "key-two") is True
