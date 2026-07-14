"""Tests for the grading combination logic — correctness, static quality,
and efficiency weighting — using a fake sandbox runner for speed and
determinism. Real sandboxed execution is verified in test_sandbox_live.py
and test_api.py's full-stack test."""

from __future__ import annotations

from app.grading import _combine_score, _efficiency_score, _static_quality_score
from app.sandbox.docker_runner import SandboxRunResult
from app.schemas import CorrectnessSummary, EvaluateRequest, StaticAnalysisSummary, TestCase
from tests.fakes import FakeSandboxRunner


def _settings():
    from app.config import get_settings

    return get_settings()


def test_grade_submission_all_tests_pass():
    from app.grading import grade_submission

    runner = FakeSandboxRunner(
        results=[
            SandboxRunResult(stdout="10\n", stderr="", exit_code=0, runtime_ms=5.0, timed_out=False),
            SandboxRunResult(stdout="6\n", stderr="", exit_code=0, runtime_ms=4.0, timed_out=False),
        ]
    )
    request = EvaluateRequest(
        language="python",
        source_code="n = int(input())\nprint(n * 2)",
        test_cases=[TestCase(input="5", expected_output="10"), TestCase(input="3", expected_output="6")],
    )
    result = grade_submission(request, runner)
    assert result.correctness.passed == 2
    assert result.correctness.total == 2
    assert result.correctness.pass_rate == 1.0
    assert result.test_results[0].passed is True


def test_grade_submission_output_whitespace_is_normalized():
    """A trailing newline (which every `print()` adds) must not itself
    count as a failure — only meaningfully different output should."""
    from app.grading import grade_submission

    runner = FakeSandboxRunner(
        results=[SandboxRunResult(stdout="  10  \n", stderr="", exit_code=0, runtime_ms=5.0, timed_out=False)]
    )
    request = EvaluateRequest(
        language="python", source_code="print(10)", test_cases=[TestCase(input="", expected_output="10")]
    )
    result = grade_submission(request, runner)
    assert result.test_results[0].passed is True


def test_grade_submission_wrong_output_fails():
    from app.grading import grade_submission

    runner = FakeSandboxRunner(
        results=[SandboxRunResult(stdout="99\n", stderr="", exit_code=0, runtime_ms=5.0, timed_out=False)]
    )
    request = EvaluateRequest(
        language="python", source_code="print(99)", test_cases=[TestCase(input="", expected_output="10")]
    )
    result = grade_submission(request, runner)
    assert result.test_results[0].passed is False
    assert result.correctness.pass_rate == 0.0


def test_grade_submission_nonzero_exit_code_fails_even_with_matching_stdout():
    from app.grading import grade_submission

    runner = FakeSandboxRunner(
        results=[SandboxRunResult(stdout="10\n", stderr="Traceback...", exit_code=1, runtime_ms=5.0, timed_out=False)]
    )
    request = EvaluateRequest(
        language="python", source_code="print(10); raise ValueError()", test_cases=[TestCase(input="", expected_output="10")]
    )
    result = grade_submission(request, runner)
    assert result.test_results[0].passed is False


def test_grade_submission_timeout_fails_the_test():
    from app.grading import grade_submission

    runner = FakeSandboxRunner(
        results=[SandboxRunResult(stdout="", stderr="", exit_code=None, runtime_ms=10000.0, timed_out=True)]
    )
    request = EvaluateRequest(
        language="python", source_code="while True: pass", test_cases=[TestCase(input="", expected_output="10")]
    )
    result = grade_submission(request, runner)
    assert result.test_results[0].passed is False
    assert result.test_results[0].timed_out is True


def test_static_quality_score_penalizes_high_complexity():
    simple = StaticAnalysisSummary(syntax_valid=True, cyclomatic_complexity=2, lint_issues=[])
    complex_ = StaticAnalysisSummary(syntax_valid=True, cyclomatic_complexity=25, lint_issues=[])
    assert _static_quality_score(simple) > _static_quality_score(complex_)


def test_static_quality_score_zero_for_syntax_error():
    result = StaticAnalysisSummary(syntax_valid=False)
    assert _static_quality_score(result) == 0.0


def test_static_quality_score_penalizes_lint_issues():
    clean = StaticAnalysisSummary(syntax_valid=True, cyclomatic_complexity=1, lint_issues=[])
    messy = StaticAnalysisSummary(syntax_valid=True, cyclomatic_complexity=1, lint_issues=["F401: x", "F841: y"])
    assert _static_quality_score(clean) > _static_quality_score(messy)


def test_efficiency_score_full_credit_when_meeting_target():
    assert _efficiency_score("O(n)", "O(n)") == 1.0


def test_efficiency_score_full_credit_when_beating_target():
    assert _efficiency_score("O(n)", "O(n^2)") == 1.0


def test_efficiency_score_penalized_when_worse_than_target():
    score = _efficiency_score("O(n^2)", "O(n)")
    assert score is not None
    assert score < 1.0


def test_efficiency_score_none_without_target():
    assert _efficiency_score("O(n)", None) is None


def test_efficiency_score_none_without_estimate():
    assert _efficiency_score(None, "O(n)") is None


def test_combine_score_excludes_efficiency_when_none():
    """When there's no basis to judge efficiency (no target, or
    insufficient data), it must be dropped from the weighted average
    entirely — not silently treated as a 0 or a 1."""
    correctness = CorrectnessSummary(passed=1, total=1, pass_rate=1.0)
    static = StaticAnalysisSummary(syntax_valid=True, cyclomatic_complexity=1, lint_issues=[])
    score_without_efficiency = _combine_score(correctness, static, None, _settings())
    score_with_perfect_efficiency = _combine_score(correctness, static, 1.0, _settings())
    # Both should be high (correctness=1.0, static=~1.0) but not
    # necessarily identical, since the weight redistribution differs.
    assert score_without_efficiency > 0.9
    assert score_with_perfect_efficiency > 0.9


def test_combine_score_reflects_correctness_dominating():
    perfect_correctness = CorrectnessSummary(passed=10, total=10, pass_rate=1.0)
    zero_correctness = CorrectnessSummary(passed=0, total=10, pass_rate=0.0)
    static = StaticAnalysisSummary(syntax_valid=True, cyclomatic_complexity=1, lint_issues=[])
    assert _combine_score(perfect_correctness, static, 1.0, _settings()) > _combine_score(
        zero_correctness, static, 1.0, _settings()
    )
