"""Sandboxed code execution via plain, non-privileged `docker run`.

The BRD calls for a "sandboxed test-case runner." Engines like Piston
achieve strong isolation by nesting nsjail inside their container, but
that requires the *outer* container to run with `--privileged` (or an
equally broad capability set) so it can construct nsjail's namespaces —
which would weaken Docker's own isolation on the container running this
service, on the same Docker daemon as everything else on this host. That
tradeoff wasn't worth it here.

Instead, this runs each submission in its own plain, ephemeral Docker
container using only *standard* Docker isolation, verified empirically
(not just assumed) to actually hold:
  - `--network none`            no network access — verified: connect() -> "Network is unreachable"
  - `--memory` / `--memory-swap` a memory bomb gets OOM-killed (exit 137), not the host
  - `--cpus` / `--pids-limit`   bounds CPU share and fork-bomb potential
  - `--cap-drop=ALL`             no Linux capabilities beyond the bare minimum
  - `--security-opt no-new-privileges`  can't escalate via setuid binaries
  - `--read-only` + a small `--tmpfs /tmp`  writes outside /tmp are blocked — verified: open("/etc/x") -> Read-only file system
  - `--user 1000:1000`           never runs as root inside the container

This is a real, weaker-than-nsjail isolation boundary — a container escape
vulnerability in the Docker runtime itself isn't defended against, the way
it might be by a second layer of sandboxing. It does not require any
privilege that weakens this service's own container security, which was
the deciding tradeoff.

Source code reaches the sandbox via `docker exec` writing into a
detached placeholder container's tmpfs, not a bind mount (`-v`). A bind
mount's host-side path is resolved by the *Docker daemon*, not by this
process — fine when this service runs directly on the same host as the
daemon, but wrong once this service itself runs inside a container
talking to a mounted-in daemon socket (Docker-outside-of-Docker, the
documented production deployment model, and also how GitHub Actions' own
containerized runners work): the temp file lives in *this* container's
filesystem, which the daemon can't see by that path at all. This was a
real bug, not a hypothetical one — it surfaced as every sandboxed run
failing with "can't find '__main__' module" the first time this ran
inside `act`'s own containerized CI runner, despite passing every local
(bare-metal) test.

`docker cp` was the first fix attempted — it streams file bytes through
the Docker API rather than resolving a host path, which should sidestep
the problem. It turned out Docker's CLI refuses `docker cp` into *any*
`--read-only` container outright, as a blanket check unrelated to whether
the specific destination (our tmpfs `/tmp`) is actually writable at the
mount level. `docker exec` isn't subject to that same check — a live
process inside the container can write to `/tmp` same as any program
running there normally — so the actual flow is: start a detached
placeholder container (`sh -c "sleep <timeout>"`), `docker exec` a
`cat > /tmp/code.py` with the source piped via stdin, then `docker exec`
the real interpreter command with the test case's stdin, then remove the
container. Confirmed working both on bare metal and inside a
containerized CI runner before this was trusted.
"""

from __future__ import annotations

import subprocess
import time
import uuid
from dataclasses import dataclass

from ..config import get_settings

_LANGUAGE_RUN_COMMAND: dict[str, list[str]] = {
    "python": ["python3", "/tmp/code.py"],
    "javascript": ["node", "/tmp/code.js"],
}
_LANGUAGE_FILENAME: dict[str, str] = {
    "python": "code.py",
    "javascript": "code.js",
}

SUPPORTED_LANGUAGES = frozenset(_LANGUAGE_RUN_COMMAND)


class SandboxError(Exception):
    """Raised for any sandbox-level failure (Docker unreachable, image
    missing, etc.) — distinct from a candidate submission simply failing a
    test, which is a normal, expected result, not an error."""


class UnsupportedLanguageError(SandboxError):
    pass


@dataclass
class SandboxRunResult:
    stdout: str
    stderr: str
    exit_code: int | None
    runtime_ms: float
    timed_out: bool


class DockerSandboxRunner:
    def __init__(self) -> None:
        pass

    def _image_for(self, language: str, settings) -> str:  # noqa: ANN001
        if language == "python":
            return settings.python_image
        if language == "javascript":
            return settings.javascript_image
        raise UnsupportedLanguageError(f"Unsupported language: {language!r}")

    def run(self, language: str, source_code: str, stdin_input: str) -> SandboxRunResult:
        settings = get_settings()
        if language not in _LANGUAGE_RUN_COMMAND:
            raise UnsupportedLanguageError(f"Unsupported language: {language!r}")

        image = self._image_for(language, settings)
        filename = _LANGUAGE_FILENAME[language]
        # Distinctive prefix, not just "code-eval-" — that substring also
        # turns up inside CI runners' own container names (e.g. act names
        # its job container "act-code-eval-CI-...-<hash>"), which made an
        # early version of the orphan-container test below false-positive
        # by matching the CI runner's own container, not a real leak.
        container_name = f"code-eval-sandbox-{uuid.uuid4().hex[:12]}"
        # A generous but bounded sleep — this is just a placeholder process
        # keeping the container alive between the two `docker exec` calls
        # below; the sandbox_timeout_seconds enforcement happens on the
        # exec that actually runs the submission, not on this placeholder.
        placeholder_seconds = max(60, int(settings.sandbox_timeout_seconds) * 3)

        run_cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "--network", "none",
            "--memory", f"{settings.sandbox_memory_mb}m",
            "--memory-swap", f"{settings.sandbox_memory_mb}m",
            "--cpus", str(settings.sandbox_cpus),
            "--pids-limit", str(settings.sandbox_pids_limit),
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--read-only",
            "--tmpfs", "/tmp:rw,exec,size=32m",
            "--user", "1000:1000",
            image,
            "sh", "-c", f"sleep {placeholder_seconds}",
        ]

        start = time.monotonic()
        run_result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=15, check=False)
        if run_result.returncode != 0:
            raise SandboxError(f"Failed to start sandbox container: {run_result.stderr.strip()}")

        try:
            # `docker exec` (unlike `docker cp`, which refuses outright on
            # any --read-only container regardless of the destination
            # being on a writable tmpfs) can write into /tmp the same as
            # any process running inside the container normally would.
            write_result = subprocess.run(
                ["docker", "exec", "-i", container_name, "sh", "-c", f"cat > /tmp/{filename}"],
                input=source_code, capture_output=True, text=True, timeout=15, check=False,
            )
            if write_result.returncode != 0:
                raise SandboxError(f"Failed to write source into sandbox: {write_result.stderr.strip()}")

            try:
                proc = subprocess.run(
                    ["docker", "exec", "-i", container_name, *_LANGUAGE_RUN_COMMAND[language]],
                    input=stdin_input,
                    capture_output=True,
                    text=True,
                    timeout=settings.sandbox_timeout_seconds,
                )
                elapsed_ms = (time.monotonic() - start) * 1000
                return SandboxRunResult(
                    stdout=proc.stdout, stderr=proc.stderr, exit_code=proc.returncode,
                    runtime_ms=elapsed_ms, timed_out=False,
                )
            except subprocess.TimeoutExpired as exc:
                elapsed_ms = (time.monotonic() - start) * 1000
                # Killing the `docker exec` CLI process (what the timeout
                # does) doesn't stop the exec'd process or the container on
                # the daemon side — the `docker rm -f` in `finally` below
                # is what actually terminates a hanging submission.
                stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
                stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", "replace")
                return SandboxRunResult(
                    stdout=stdout, stderr=stderr, exit_code=None,
                    runtime_ms=elapsed_ms, timed_out=True,
                )
        finally:
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, timeout=10, check=False)

    def validate(self) -> None:
        """Confirms Docker is reachable and the language images this
        service depends on are actually present — cheap metadata calls,
        no container execution."""
        settings = get_settings()
        try:
            info = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxError(f"Docker is not reachable: {exc}") from exc
        if info.returncode != 0:
            raise SandboxError(f"Docker is not reachable: {info.stderr.strip()}")

        for image in (settings.python_image, settings.javascript_image):
            inspect = subprocess.run(
                ["docker", "image", "inspect", image], capture_output=True, text=True, timeout=10, check=False
            )
            if inspect.returncode != 0:
                raise SandboxError(f"Required sandbox image not present: {image!r}. Pull it before starting this service.")


_runner: DockerSandboxRunner | None = None


def get_sandbox_runner() -> DockerSandboxRunner:
    global _runner
    if _runner is None:
        _runner = DockerSandboxRunner()
    return _runner
