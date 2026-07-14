"""API-level tests: HTTP status codes, error schema, and endpoint wiring.

The sandbox runner is mocked at the point main.py actually imports it
(app.main.get_sandbox_runner) for fast, deterministic tests. Real sandboxed
execution through the full HTTP stack is proven separately by
test_full_stack_evaluation_with_real_sandbox below, which mocks nothing,
and by test_sandbox_live.py, which tests the runner directly.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from app.sandbox.docker_runner import SandboxRunResult
from tests.fakes import AlwaysFailingSandboxRunner, FakeSandboxRunner


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=10, check=False)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


@pytest.fixture
def client(monkeypatch):
    fake = FakeSandboxRunner(
        results=[SandboxRunResult(stdout="10\n", stderr="", exit_code=0, runtime_ms=5.0, timed_out=False)] * 20
    )
    monkeypatch.setattr("app.main.get_sandbox_runner", lambda: fake)

    from app.main import app

    with TestClient(app) as c:
        yield c


def test_liveness(client):
    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readiness(client):
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_capabilities(client):
    resp = client.get("/api/v1/capabilities")
    assert resp.status_code == 200
    body = resp.json()
    assert "python" in body["supported_languages"]
    assert "javascript" in body["supported_languages"]


def test_metrics_exposed(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200


def test_request_id_header_present(client):
    resp = client.get("/health/live")
    assert "x-request-id" in resp.headers


def test_evaluate_success(client):
    resp = client.post(
        "/api/v1/code/evaluate",
        json={
            "language": "python",
            "source_code": "n = int(input())\nprint(n * 2)",
            "test_cases": [{"input": "5", "expected_output": "10"}],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["correctness"]["passed"] == 1
    assert body["overall_score"] > 0


def test_evaluate_unsupported_language_is_422(client):
    resp = client.post(
        "/api/v1/code/evaluate",
        json={"language": "ruby", "source_code": "puts 1", "test_cases": [{"expected_output": "1"}]},
    )
    assert resp.status_code == 422


def test_evaluate_empty_source_is_422(client):
    resp = client.post(
        "/api/v1/code/evaluate",
        json={"language": "python", "source_code": "", "test_cases": [{"expected_output": "1"}]},
    )
    assert resp.status_code == 422


def test_evaluate_source_too_long_is_422(client):
    from app.main import settings

    resp = client.post(
        "/api/v1/code/evaluate",
        json={
            "language": "python",
            "source_code": "x" * (settings.max_source_code_chars + 1),
            "test_cases": [{"expected_output": "1"}],
        },
    )
    assert resp.status_code == 422


def test_evaluate_too_many_test_cases_is_422(client):
    from app.main import settings

    test_cases = [{"input": str(i), "expected_output": str(i)} for i in range(settings.max_test_cases + 1)]
    resp = client.post(
        "/api/v1/code/evaluate", json={"language": "python", "source_code": "print(1)", "test_cases": test_cases}
    )
    assert resp.status_code == 422


def test_evaluate_no_test_cases_is_422(client):
    resp = client.post(
        "/api/v1/code/evaluate", json={"language": "python", "source_code": "print(1)", "test_cases": []}
    )
    assert resp.status_code == 422


def test_evaluate_returns_502_on_sandbox_failure(monkeypatch):
    monkeypatch.setattr("app.main.get_sandbox_runner", lambda: AlwaysFailingSandboxRunner())
    from app.main import app

    with TestClient(app) as c:
        resp = c.post(
            "/api/v1/code/evaluate",
            json={"language": "python", "source_code": "print(1)", "test_cases": [{"expected_output": "1"}]},
        )
    assert resp.status_code == 502
    assert resp.json()["error"] == "request_error"


def test_error_responses_never_leak_tracebacks(client):
    resp = client.post(
        "/api/v1/code/evaluate", json={"language": "python", "source_code": "", "test_cases": [{"expected_output": "1"}]}
    )
    body = resp.json()
    assert "Traceback" not in body["detail"]
    assert 'File "' not in body["detail"]


def test_production_without_required_auth_logs_security_warning(caplog):
    from app.config import Settings
    from app.main import _warn_if_unprotected_in_production

    settings = Settings(environment="production", require_api_key=False)

    with caplog.at_level("WARNING"):
        _warn_if_unprotected_in_production(settings)

    assert any(
        "environment=production" in r.message and "CODE_EVAL_REQUIRE_API_KEY" in r.message for r in caplog.records
    )


def test_production_with_required_auth_logs_no_warning(caplog):
    from app.config import Settings
    from app.main import _warn_if_unprotected_in_production

    settings = Settings(environment="production", require_api_key=True, api_keys="some-key")

    with caplog.at_level("WARNING"):
        _warn_if_unprotected_in_production(settings)

    assert not any("environment=production" in r.message for r in caplog.records)


@pytest.mark.skipif(not _docker_available(), reason="Docker is not available in this environment")
def test_full_stack_evaluation_with_real_sandbox():
    """No mocking anywhere: a real O(n^2) submission, run through the real
    HTTP endpoint against the real sandboxed Docker execution path, must
    be correctly graded as correct-but-inefficient against an O(n) target.
    This is the equivalent of interview-qa/speech-io's live-provider tests
    — except here what's "live" is the sandbox, not an external API."""
    from app.main import app

    def make_case(n: int) -> dict:
        import random

        rng = random.Random(n)
        arr = [rng.randint(0, 1000) for _ in range(n)]
        return {"input": f"{n}\n{' '.join(map(str, arr))}", "expected_output": str(sum(arr)), "size_n": n}

    bubble_sort_then_sum = (
        "n = int(input())\n"
        "arr = list(map(int, input().split()))\n"
        "for i in range(len(arr)):\n"
        "    for j in range(len(arr) - 1):\n"
        "        if arr[j] > arr[j + 1]:\n"
        "            arr[j], arr[j + 1] = arr[j + 1], arr[j]\n"
        "print(sum(arr))\n"
    )

    with TestClient(app) as c:
        resp = c.post(
            "/api/v1/code/evaluate",
            json={
                "language": "python",
                "source_code": bubble_sort_then_sum,
                "test_cases": [make_case(n) for n in [200, 800, 1600, 3200]],
                "expected_complexity": "O(n)",
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["correctness"]["pass_rate"] == 1.0
    assert body["efficiency"]["estimated_complexity"] == "O(n^2)"
    # Correct but inefficient relative to the O(n) target -> penalized
    # below a perfect score, but not catastrophically (correctness still
    # dominates the weighting).
    assert 0.5 < body["overall_score"] < 1.0
