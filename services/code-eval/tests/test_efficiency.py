"""Tests for the empirical complexity estimator — pure math, no sandbox
execution, deterministic and fast."""

from __future__ import annotations

import math
import random

from app.analysis.efficiency import estimate_complexity


def test_linear_data_classified_as_on():
    rng = random.Random(1)
    pairs = [(n, n * 0.01 + rng.uniform(-0.05, 0.05)) for n in [100, 500, 1000, 5000, 10000]]
    result = estimate_complexity(pairs)
    assert result.estimated_complexity == "O(n)"
    assert result.confidence == "high"


def test_quadratic_data_classified_as_on_squared():
    rng = random.Random(2)
    pairs = [(n, (n**2) * 0.00001 + rng.uniform(-0.1, 0.1)) for n in [100, 500, 1000, 5000, 10000]]
    result = estimate_complexity(pairs)
    assert result.estimated_complexity == "O(n^2)"
    assert result.confidence == "high"


def test_constant_data_classified_as_o1():
    """The regression case: O(1) is mathematically degenerate under
    textbook R² (its own best fit collapses to R²'s "predict the mean"
    baseline, forcing a score of 0 regardless of fit quality) — this must
    stay correctly classified as high-confidence O(1), not silently break
    if the goodness-of-fit metric ever reverts to plain R²."""
    rng = random.Random(3)
    pairs = [(n, 5.0 + rng.uniform(-0.2, 0.2)) for n in [100, 500, 1000, 5000, 10000]]
    result = estimate_complexity(pairs)
    assert result.estimated_complexity == "O(1)"
    assert result.confidence == "high"
    assert result.fit_quality is not None
    assert result.fit_quality > 0.9


def test_n_log_n_data_classified_correctly_with_wide_enough_range():
    """Distinguishing O(n log n) from O(n) empirically requires a wide
    enough size range — over a narrow range the two curves are nearly
    collinear and genuinely hard to tell apart (an inherent property of
    the technique, not a bug)."""
    rng = random.Random(4)
    pairs = [(n, n * math.log2(n) * 0.001 + rng.uniform(-0.1, 0.1)) for n in [10, 1_000, 100_000, 1_000_000, 10_000_000]]
    result = estimate_complexity(pairs)
    assert result.estimated_complexity == "O(n log n)"
    assert result.confidence == "high"


def test_non_monotonic_data_gets_low_confidence():
    # Alternating extremes rather than a random draw: with only 5 points
    # and 7 candidate models, unstructured random noise can occasionally
    # look like a passable fit to *some* model by chance (a real, known
    # limitation of curve-fitting on few points) — an oscillating pattern
    # is reliably a bad fit for every monotonic growth model, so this
    # exercises the "should be low confidence" path deterministically.
    pairs = [(100, 100.0), (500, 0.0), (1000, 100.0), (5000, 0.0), (10000, 100.0)]
    result = estimate_complexity(pairs)
    assert result.confidence == "low"


def test_insufficient_data_below_minimum_distinct_sizes():
    result = estimate_complexity([(100, 1.0), (200, 2.0)])
    assert result.estimated_complexity is None
    assert result.confidence == "insufficient_data"


def test_repeated_sizes_are_averaged_not_over_weighted():
    # Five measurements at n=100 (noisy) but only one each at n=500, n=1000
    # — must not let n=100's five samples dominate the fit purely by count.
    rng = random.Random(6)
    pairs = [(100, 1.0 + rng.uniform(-0.1, 0.1)) for _ in range(5)]
    pairs += [(500, 5.0), (1000, 10.0)]
    result = estimate_complexity(pairs)
    assert result.estimated_complexity == "O(n)"
    sizes_reported = [r["size_n"] for r in result.runtime_by_size]
    assert sizes_reported == [100, 500, 1000]


def test_negative_or_zero_sizes_are_ignored():
    pairs = [(0, 1.0), (-5, 2.0), (100, 1.0), (500, 5.0), (1000, 10.0)]
    result = estimate_complexity(pairs)
    assert all(r["size_n"] > 0 for r in result.runtime_by_size)


def test_estimate_is_deterministic():
    pairs = [(100, 1.2), (500, 6.1), (1000, 11.9), (5000, 60.5)]
    a = estimate_complexity(pairs)
    b = estimate_complexity(pairs)
    assert a == b
