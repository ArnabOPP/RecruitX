"""Empirical time-complexity estimation from measured runtimes.

True Big-O inference from static code alone is an unsolved general
problem (it's undecidable in the worst case — equivalent to the halting
problem for arbitrary programs). Rather than guess from source structure,
this measures what the code actually does: given (input_size, runtime_ms)
pairs from running the *same* submission against test cases of increasing
size, it fits each candidate growth model via least squares and reports
whichever fits best — the same "measure, don't guess" principle as
answer-grading's semantic scoring. Deterministic and auditable: the raw
timing data and R² fit quality are returned alongside the verdict, not
just a label.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

# Ordered from simplest to most complex — used as a tiebreak (Occam's
# razor) when two models fit the data almost equally well, since with only
# a handful of data points, a more complex model can spuriously fit noise
# just as well as the true, simpler underlying complexity class.
_COMPLEXITY_MODELS: list[tuple[str, Callable[[int], float]]] = [
    ("O(1)", lambda n: 1.0),
    ("O(log n)", lambda n: math.log2(n) if n > 1 else 1.0),
    ("O(n)", lambda n: float(n)),
    ("O(n log n)", lambda n: n * math.log2(n) if n > 1 else float(n)),
    ("O(n^2)", lambda n: float(n) ** 2),
    ("O(n^3)", lambda n: float(n) ** 3),
    ("O(2^n)", lambda n: 2.0**n if n < 64 else float("inf")),
]

_MIN_DISTINCT_SIZES = 3

# Ordinal ranking (simplest/fastest first) — grading.py uses this to
# compare a measured complexity against a target one without duplicating
# the growth-model list.
COMPLEXITY_ORDER = [name for name, _ in _COMPLEXITY_MODELS]


@dataclass
class EfficiencyEstimate:
    estimated_complexity: str | None
    fit_quality: float | None
    confidence: str  # "high" | "medium" | "low" | "insufficient_data"
    runtime_by_size: list[dict]


def _goodness_of_fit(actual: list[float], predicted: list[float]) -> float:
    """1.0 = perfect fit, 0.0 = as bad as it gets (or worse, clamped).

    Deliberately *not* textbook R² (1 - ss_res/ss_tot): R² is degenerate
    for the O(1) growth model specifically. Its own least-squares fit
    collapses to "predict the mean" — but that's exactly R²'s own
    baseline, so R² forces a score of 0 even when the constant model is a
    perfect fit for genuinely constant data. Normalized RMSE has no such
    baseline-collision and works uniformly across every candidate model,
    including the constant one.
    """
    n = len(actual)
    mean_abs = sum(abs(a) for a in actual) / n or 1.0
    rmse = math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted, strict=True)) / n)
    return max(0.0, 1 - rmse / mean_abs)


def estimate_complexity(size_runtime_pairs: list[tuple[int, float]]) -> EfficiencyEstimate:
    """size_runtime_pairs: list of (input_size_n, runtime_ms), one per
    successfully-run test case that was tagged with a size. Only points
    with a genuinely distinct size contribute to the fit — repeated sizes
    are averaged first so one size isn't over-weighted."""
    by_size: dict[int, list[float]] = {}
    for n, runtime_ms in size_runtime_pairs:
        if n <= 0:
            continue
        by_size.setdefault(n, []).append(runtime_ms)

    runtime_by_size = [
        {"size_n": n, "runtime_ms": round(sum(times) / len(times), 4)} for n, times in sorted(by_size.items())
    ]

    if len(by_size) < _MIN_DISTINCT_SIZES:
        return EfficiencyEstimate(
            estimated_complexity=None,
            fit_quality=None,
            confidence="insufficient_data",
            runtime_by_size=runtime_by_size,
        )

    xs_n = sorted(by_size)
    ys = [sum(by_size[n]) / len(by_size[n]) for n in xs_n]

    best_name: str | None = None
    best_fit = -math.inf
    for name, model in _COMPLEXITY_MODELS:
        model_xs = [model(n) for n in xs_n]
        if any(math.isinf(x) or math.isnan(x) for x in model_xs):
            continue
        sum_xx = sum(x * x for x in model_xs)
        if sum_xx == 0:
            continue
        sum_xy = sum(x * y for x, y in zip(model_xs, ys, strict=True))
        scale = sum_xy / sum_xx
        predicted = [scale * x for x in model_xs]
        fit = _goodness_of_fit(ys, predicted)
        # A strictly-better fit wins outright; a near-tie (within 0.02)
        # keeps the *earlier* (simpler) model already chosen, implementing
        # the Occam's-razor tiebreak via the model list's own ordering.
        if fit > best_fit + 0.02:
            best_fit = fit
            best_name = name

    if best_name is None:
        return EfficiencyEstimate(
            estimated_complexity=None, fit_quality=None, confidence="insufficient_data", runtime_by_size=runtime_by_size
        )

    if best_fit >= 0.9:
        confidence = "high"
    elif best_fit >= 0.7:
        confidence = "medium"
    else:
        confidence = "low"

    return EfficiencyEstimate(
        estimated_complexity=best_name,
        fit_quality=round(best_fit, 4),
        confidence=confidence,
        runtime_by_size=runtime_by_size,
    )
