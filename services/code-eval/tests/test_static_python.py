"""Tests for Python static analysis (radon complexity + ruff lint) — pure
static analysis over source text, no code execution, no sandbox needed."""

from __future__ import annotations

from app.analysis.static_python import analyze_python


def test_simple_function_has_low_complexity():
    source = "def add(a, b):\n    return a + b\n"
    result = analyze_python(source)
    assert result.syntax_valid is True
    assert result.max_cyclomatic_complexity == 1


def test_nested_branches_increase_complexity():
    source = """
def classify(n):
    if n < 0:
        return "negative"
    elif n == 0:
        return "zero"
    else:
        if n % 2 == 0:
            return "positive even"
        else:
            return "positive odd"
"""
    result = analyze_python(source)
    assert result.syntax_valid is True
    assert result.max_cyclomatic_complexity is not None
    assert result.max_cyclomatic_complexity > 1


def test_syntax_error_reported_not_raised():
    result = analyze_python("def foo(:\n    pass")
    assert result.syntax_valid is False
    assert result.max_cyclomatic_complexity is None
    assert result.lint_issues == []


def test_unused_import_and_variable_flagged():
    source = "import os\ndef foo():\n    x = 1\n    return\n"
    result = analyze_python(source)
    codes = [issue.split(":")[0] for issue in result.lint_issues]
    assert "F401" in codes  # unused import
    assert "F841" in codes  # unused variable


def test_clean_code_has_no_lint_issues():
    source = "def add(a, b):\n    return a + b\n"
    result = analyze_python(source)
    assert result.lint_issues == []


def test_maintainability_index_is_reported_for_valid_code():
    source = "def add(a, b):\n    return a + b\n"
    result = analyze_python(source)
    assert result.maintainability_index is not None
    assert 0 <= result.maintainability_index <= 100
