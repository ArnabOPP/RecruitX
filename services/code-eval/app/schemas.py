"""Data contracts for the code-eval API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TestCase(BaseModel):
    # Not a pytest test class — pytest's collector matches on the `Test`
    # name prefix regardless of what the class actually is.
    __test__ = False

    input: str = ""
    expected_output: str
    # Tags this test case's input size for empirical complexity
    # estimation — e.g. the length of a list the candidate's function
    # processes. Optional: correctness grading works without it, but
    # efficiency estimation needs at least 3 test cases with distinct
    # size_n values to fit a growth curve.
    size_n: int | None = None


class EvaluateRequest(BaseModel):
    language: str
    source_code: str
    test_cases: list[TestCase] = Field(min_length=1)
    # What complexity class a correct solution to this problem should
    # achieve, e.g. "O(n log n)" — set by whoever authored the coding
    # question. Without it, efficiency has nothing to grade *against*
    # (a measured O(n^2) isn't inherently bad without knowing O(n) was
    # achievable), so the efficiency component is simply excluded from
    # overall_score rather than scored against a made-up baseline.
    expected_complexity: str | None = None


class TestCaseResult(BaseModel):
    index: int
    passed: bool
    expected_output: str
    actual_output: str
    stderr: str
    runtime_ms: float
    timed_out: bool
    size_n: int | None = None


class CorrectnessSummary(BaseModel):
    passed: int
    total: int
    pass_rate: float


class StaticAnalysisSummary(BaseModel):
    syntax_valid: bool
    cyclomatic_complexity: float | None = None
    maintainability_index: float | None = None
    lint_issues: list[str] = Field(default_factory=list)


class EfficiencySummary(BaseModel):
    estimated_complexity: str | None = None
    fit_quality: float | None = None
    confidence: str
    runtime_by_size: list[dict] = Field(default_factory=list)


class EvaluateResponse(BaseModel):
    language: str
    correctness: CorrectnessSummary
    test_results: list[TestCaseResult]
    static_analysis: StaticAnalysisSummary
    efficiency: EfficiencySummary
    overall_score: float
    explanation: str
