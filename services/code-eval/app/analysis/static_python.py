"""Static analysis for Python submissions: cyclomatic complexity (radon)
and style/lint issues (ruff). Both are pure static analysis over source
text — neither executes the candidate's code — so this runs directly in
this process, no sandbox needed, unlike test-case execution.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from radon.complexity import cc_visit
from radon.metrics import mi_visit


class StaticAnalysisResult:
    def __init__(
        self,
        max_cyclomatic_complexity: float | None,
        average_cyclomatic_complexity: float | None,
        maintainability_index: float | None,
        lint_issues: list[str],
        syntax_valid: bool,
    ) -> None:
        self.max_cyclomatic_complexity = max_cyclomatic_complexity
        self.average_cyclomatic_complexity = average_cyclomatic_complexity
        self.maintainability_index = maintainability_index
        self.lint_issues = lint_issues
        self.syntax_valid = syntax_valid


def analyze_python(source_code: str) -> StaticAnalysisResult:
    try:
        blocks = cc_visit(source_code)
        complexities = [b.complexity for b in blocks]
        max_complexity = max(complexities) if complexities else 1.0
        avg_complexity = sum(complexities) / len(complexities) if complexities else 1.0
        maintainability = mi_visit(source_code, multi=True)
        syntax_valid = True
    except SyntaxError:
        # A submission that doesn't even parse still gets a result object
        # (with syntax_valid=False) rather than raising — the caller
        # combines this with correctness (which will already be 0, since
        # unparseable code can't pass any test either) rather than treating
        # a syntax error as a service-level failure.
        max_complexity = None
        avg_complexity = None
        maintainability = None
        syntax_valid = False

    lint_issues = _run_ruff(source_code) if syntax_valid else []

    return StaticAnalysisResult(
        max_cyclomatic_complexity=max_complexity,
        average_cyclomatic_complexity=avg_complexity,
        maintainability_index=maintainability,
        lint_issues=lint_issues,
        syntax_valid=syntax_valid,
    )


def _run_ruff(source_code: str) -> list[str]:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "submission.py"
        path.write_text(source_code, encoding="utf-8")
        try:
            result = subprocess.run(
                ["ruff", "check", "--no-cache", "--output-format=json", str(path)],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return []

        if not result.stdout.strip():
            return []
        try:
            issues = json.loads(result.stdout)
        except json.JSONDecodeError:
            return []
        return [
            f"{issue['code']}: {issue['message']} (line {issue['location']['row']})"
            for issue in issues
            if issue.get("code")
        ]
