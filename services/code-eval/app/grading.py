"""Combines sandboxed test-case execution, static analysis, and empirical
efficiency estimation into a single deterministic, auditable score.

Same "measure, don't guess" philosophy as answer-grading: correctness
comes from actually running the code against test cases in the sandbox,
efficiency comes from actually measuring runtime growth (not inferring
Big-O from source structure), and any component this service has no real
basis to judge (efficiency without a target complexity, static analysis
for a language it doesn't statically analyze) is excluded from
overall_score entirely rather than filled in with a guess.
"""

from __future__ import annotations

from .analysis.efficiency import COMPLEXITY_ORDER, estimate_complexity
from .analysis.static_python import analyze_python
from .config import get_settings
from .sandbox.docker_runner import DockerSandboxRunner
from .schemas import (
    CorrectnessSummary,
    EfficiencySummary,
    EvaluateRequest,
    EvaluateResponse,
    StaticAnalysisSummary,
    TestCaseResult,
)

_NOOP_SOURCE = {
    "python": "pass",
    "javascript": "",
}


def _normalize_output(s: str) -> str:
    return s.strip()


def _measure_baseline_overhead_ms(runner: DockerSandboxRunner, language: str) -> float:
    """Runs a no-op program through the same sandbox pathway to measure
    fixed per-run overhead (container startup + interpreter startup).

    Found via live testing, not assumed: for small inputs this overhead
    (several hundred ms) completely swamps the actual algorithmic work,
    which would make every submission's timing curve look flat regardless
    of its real complexity. Subtracting a measured baseline — rather than
    an assumed constant, since actual overhead varies by host load and
    language runtime — is what makes the complexity *growth* signal
    visible above the fixed cost of spinning up each sandbox.
    """
    noop = _NOOP_SOURCE.get(language, "")
    result = runner.run(language, noop, "")
    return result.runtime_ms


def run_test_cases(
    runner: DockerSandboxRunner, language: str, source_code: str, test_cases: list
) -> list[TestCaseResult]:
    results: list[TestCaseResult] = []
    for i, tc in enumerate(test_cases):
        run_result = runner.run(language, source_code, tc.input)
        actual = _normalize_output(run_result.stdout)
        expected = _normalize_output(tc.expected_output)
        passed = (not run_result.timed_out) and run_result.exit_code == 0 and actual == expected
        results.append(
            TestCaseResult(
                index=i,
                passed=passed,
                expected_output=tc.expected_output,
                actual_output=run_result.stdout,
                stderr=run_result.stderr,
                runtime_ms=round(run_result.runtime_ms, 4),
                timed_out=run_result.timed_out,
                size_n=tc.size_n,
            )
        )
    return results


def _static_quality_score(static: StaticAnalysisSummary) -> float | None:
    if not static.syntax_valid:
        return 0.0
    score = 1.0
    if static.cyclomatic_complexity is not None:
        # radon's own bands: 1-5 simple, 6-10 moderate, 11-20 complex, 21+ very complex
        if static.cyclomatic_complexity > 20:
            score -= 0.5
        elif static.cyclomatic_complexity > 10:
            score -= 0.25
        elif static.cyclomatic_complexity > 5:
            score -= 0.1
    score -= min(0.3, 0.05 * len(static.lint_issues))
    return max(0.0, score)


def _efficiency_score(estimated_complexity: str | None, expected_complexity: str | None) -> float | None:
    if estimated_complexity is None or expected_complexity is None:
        return None
    if estimated_complexity not in COMPLEXITY_ORDER or expected_complexity not in COMPLEXITY_ORDER:
        return None
    measured_idx = COMPLEXITY_ORDER.index(estimated_complexity)
    target_idx = COMPLEXITY_ORDER.index(expected_complexity)
    if measured_idx <= target_idx:
        return 1.0
    penalty_per_class = 0.3
    return max(0.0, 1.0 - penalty_per_class * (measured_idx - target_idx))


def _combine_score(
    correctness: CorrectnessSummary,
    static_summary: StaticAnalysisSummary,
    efficiency_score: float | None,
    settings,  # noqa: ANN001
) -> float:
    """Weight-normalized average over whichever components actually have a
    real signal — a component with nothing to judge against (efficiency
    with no target, or no static analyzer for this language) is dropped
    rather than silently scored as if it were neutral or perfect."""
    components: list[tuple[float, float]] = [(correctness.pass_rate, settings.correctness_weight)]

    static_score = _static_quality_score(static_summary)
    if static_score is not None:
        components.append((static_score, settings.static_quality_weight))

    if efficiency_score is not None:
        components.append((efficiency_score, settings.efficiency_weight))

    total_weight = sum(w for _, w in components)
    if total_weight == 0:
        return 0.0
    return sum(s * w for s, w in components) / total_weight


def _build_explanation(
    correctness: CorrectnessSummary,
    static_summary: StaticAnalysisSummary,
    efficiency_summary: EfficiencySummary,
    expected_complexity: str | None,
    overall_score: float,
) -> str:
    parts = [f"Passed {correctness.passed}/{correctness.total} test cases ({correctness.pass_rate:.0%})."]

    if not static_summary.syntax_valid:
        parts.append("Source code has a syntax error.")
    elif static_summary.lint_issues:
        parts.append(f"{len(static_summary.lint_issues)} style/lint issue(s) found.")

    if efficiency_summary.estimated_complexity:
        line = f"Estimated time complexity: {efficiency_summary.estimated_complexity} (confidence: {efficiency_summary.confidence})."
        if expected_complexity:
            line += f" Target was {expected_complexity}."
        parts.append(line)
    elif efficiency_summary.confidence == "insufficient_data":
        parts.append("Not enough size-tagged test cases to estimate complexity.")

    parts.append(f"Overall score: {overall_score:.2f}.")
    return " ".join(parts)


def grade_submission(request: EvaluateRequest, runner: DockerSandboxRunner) -> EvaluateResponse:
    settings = get_settings()

    test_results = run_test_cases(runner, request.language, request.source_code, request.test_cases)

    passed_count = sum(1 for r in test_results if r.passed)
    total = len(test_results)
    correctness = CorrectnessSummary(
        passed=passed_count, total=total, pass_rate=(passed_count / total if total else 0.0)
    )

    if request.language == "python":
        static = analyze_python(request.source_code)
        static_summary = StaticAnalysisSummary(
            syntax_valid=static.syntax_valid,
            cyclomatic_complexity=static.max_cyclomatic_complexity,
            maintainability_index=static.maintainability_index,
            lint_issues=static.lint_issues,
        )
    else:
        # No static analyzer registered for this language yet (see README
        # known limitations) — reported as syntax_valid=True (we have no
        # evidence otherwise; a real syntax error would show up as every
        # test case failing/erroring instead) with no complexity/lint data,
        # and excluded from overall_score via _combine_score.
        static_summary = StaticAnalysisSummary(syntax_valid=True)

    # Efficiency only draws on *passing* tests with a tagged size — a
    # failing test's runtime isn't a meaningful efficiency signal (it may
    # have errored out near-instantly, which looks "fast" but isn't
    # informative about the algorithm's actual growth rate).
    sized_passing = [(r.size_n, r.runtime_ms) for r in test_results if r.passed and r.size_n is not None]
    size_runtime_pairs: list[tuple[int, float]]
    if len(sized_passing) >= 2:
        # Only pay for the extra baseline run when there's actually enough
        # size-tagged data for a complexity estimate to be attempted.
        baseline_ms = _measure_baseline_overhead_ms(runner, request.language)
        size_runtime_pairs = [(n, max(0.01, runtime_ms - baseline_ms)) for n, runtime_ms in sized_passing]
    else:
        size_runtime_pairs = []
    efficiency = estimate_complexity(size_runtime_pairs)
    efficiency_summary = EfficiencySummary(
        estimated_complexity=efficiency.estimated_complexity,
        fit_quality=efficiency.fit_quality,
        confidence=efficiency.confidence,
        runtime_by_size=efficiency.runtime_by_size,
    )

    efficiency_score = (
        _efficiency_score(efficiency.estimated_complexity, request.expected_complexity)
        if efficiency.confidence not in ("insufficient_data", "low")
        else None
    )

    overall_score = _combine_score(correctness, static_summary, efficiency_score, settings)
    explanation = _build_explanation(
        correctness, static_summary, efficiency_summary, request.expected_complexity, overall_score
    )

    return EvaluateResponse(
        language=request.language,
        correctness=correctness,
        test_results=test_results,
        static_analysis=static_summary,
        efficiency=efficiency_summary,
        overall_score=round(overall_score, 4),
        explanation=explanation,
    )
