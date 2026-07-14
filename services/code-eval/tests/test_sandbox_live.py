"""Tests against the real Docker sandbox — not mocked. This is the most
important test file in this service: it proves the isolation guarantees
documented in sandbox/docker_runner.py actually hold, the same properties
verified manually before this service was built (memory bombs get
OOM-killed, network is unreachable, filesystem writes outside /tmp are
blocked, infinite loops are actually terminated) — codified here so a
future change can't silently weaken them without a test failing.

Requires a reachable Docker daemon with the python:3.11-slim and
node:20-slim images available; skips cleanly if Docker isn't reachable,
matching the pattern used for the other Docker-gated tests in this
project (test_rate_limit_redis.py in every other service).
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from app.sandbox.docker_runner import DockerSandboxRunner


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=10, check=False)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


pytestmark = pytest.mark.skipif(not _docker_available(), reason="Docker is not available in this environment")


@pytest.fixture(scope="module")
def runner() -> DockerSandboxRunner:
    return DockerSandboxRunner()


def test_validate_succeeds_when_docker_and_images_are_present(runner):
    runner.validate()  # must not raise


def test_correct_python_program_produces_expected_output(runner):
    result = runner.run("python", "n = int(input())\nprint(sum(range(n)))", "5000")
    assert result.exit_code == 0
    assert result.stdout.strip() == "12497500"
    assert result.timed_out is False


def test_correct_javascript_program_produces_expected_output(runner):
    result = runner.run(
        "javascript",
        "const readline = require('readline').createInterface({ input: process.stdin });"
        "readline.on('line', (line) => { console.log(parseInt(line) * 2); readline.close(); });",
        "21",
    )
    assert result.exit_code == 0
    assert result.stdout.strip() == "42"


def test_program_with_runtime_error_reports_nonzero_exit(runner):
    result = runner.run("python", "raise ValueError('boom')", "")
    assert result.exit_code != 0
    assert "ValueError" in result.stderr


def test_infinite_loop_is_actually_terminated_by_timeout(runner):
    """Proves the wall-clock timeout genuinely kills a hanging
    submission (not just that the client gives up waiting) — the earlier
    manual verification of this exact property, now codified."""
    from app.config import get_settings

    get_settings.cache_clear()
    result = runner.run("python", "while True:\n    pass", "")
    assert result.timed_out is True
    assert result.exit_code is None


def test_no_orphaned_containers_after_timeout(runner):
    """A timed-out submission's container must actually be stopped, not
    left running and burning CPU/memory indefinitely."""
    runner.run("python", "while True:\n    pass", "")
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=code-eval-sandbox-", "--format", "{{.Names}}"],
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.stdout.strip() == ""


def test_memory_bomb_is_contained_not_crashing_the_runner(runner):
    """A submission that tries to exhaust memory must be killed (OOM,
    exit code 137) by the container's own --memory limit, not allowed to
    affect the host or this process."""
    bomb = "x = []\nwhile True:\n    x.append(bytearray(10**7))"
    result = runner.run("python", bomb, "")
    assert result.exit_code == 137


def test_network_access_is_unreachable(runner):
    """--network none must actually block outbound connections, not just
    be present in the command line."""
    probe = (
        "import socket\n"
        "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "s.settimeout(3)\n"
        "try:\n"
        "    s.connect(('8.8.8.8', 53))\n"
        "    print('NETWORK ACCESSIBLE')\n"
        "except Exception as e:\n"
        "    print(f'blocked: {e}')\n"
    )
    result = runner.run("python", probe, "")
    assert "NETWORK ACCESSIBLE" not in result.stdout
    assert "blocked" in result.stdout


def test_filesystem_writes_outside_tmp_are_blocked(runner):
    """--read-only must actually prevent writes to the root filesystem,
    not just be present in the command line."""
    probe = (
        "try:\n"
        "    with open('/etc/malicious', 'w') as f:\n"
        "        f.write('pwned')\n"
        "    print('WROTE TO ROOTFS')\n"
        "except Exception as e:\n"
        "    print(f'blocked: {e}')\n"
    )
    result = runner.run("python", probe, "")
    assert "WROTE TO ROOTFS" not in result.stdout
    assert "blocked" in result.stdout


def test_tmp_is_writable_for_legitimate_use(runner):
    """The isolation must not be so strict that ordinary submissions
    (e.g. ones using a temp file) can't function at all."""
    probe = "with open('/tmp/scratch.txt', 'w') as f:\n    f.write('ok')\nprint(open('/tmp/scratch.txt').read())"
    result = runner.run("python", probe, "")
    assert result.exit_code == 0
    assert result.stdout.strip() == "ok"


def test_runs_as_non_root_user(runner):
    result = runner.run("python", "import os\nprint(os.getuid())", "")
    assert result.stdout.strip() == "1000"


def test_unsupported_language_raises_before_touching_docker(runner):
    from app.sandbox.docker_runner import UnsupportedLanguageError

    with pytest.raises(UnsupportedLanguageError):
        runner.run("ruby", "puts 1", "")
