"""A fake sandbox runner for fast, deterministic unit tests that don't
spin up real Docker containers — see test_sandbox_live.py for the tests
that exercise the real sandbox, including its isolation guarantees."""

from __future__ import annotations

from app.sandbox.docker_runner import SandboxError, SandboxRunResult


class FakeSandboxRunner:
    """Returns a scripted sequence of results, one per call — lets a test
    simulate "first call succeeds, second times out" etc."""

    def __init__(self, results: list[SandboxRunResult] | None = None):
        self._results = list(results or [])
        self.calls: list[tuple[str, str, str]] = []

    def run(self, language: str, source_code: str, stdin_input: str) -> SandboxRunResult:
        self.calls.append((language, source_code, stdin_input))
        if not self._results:
            raise SandboxError("FakeSandboxRunner has no scripted results left.")
        return self._results.pop(0)

    def validate(self) -> None:
        pass


class AlwaysFailingSandboxRunner:
    def run(self, language: str, source_code: str, stdin_input: str) -> SandboxRunResult:
        raise SandboxError("simulated sandbox failure")

    def validate(self) -> None:
        raise SandboxError("simulated sandbox failure")
