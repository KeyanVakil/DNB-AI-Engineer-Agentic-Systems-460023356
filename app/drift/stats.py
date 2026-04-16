"""Statistical drift detection: KS test and PSI."""

from __future__ import annotations

import math


def ks_test(baseline: list[float], current: list[float]) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test.

    Returns (statistic, p_value). Uses scipy when available, falls back to
    a simple implementation for environments without scipy.
    """
    if not baseline or not current:
        return 0.0, 1.0

    try:
        from scipy import stats  # type: ignore[import]

        result = stats.ks_2samp(baseline, current)
        return float(result.statistic), float(result.pvalue)
    except ImportError:
        return _ks_simple(baseline, current)


def _ks_simple(a: list[float], b: list[float]) -> tuple[float, float]:
    all_values = sorted(set(a + b))
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return 0.0, 1.0
    a_sorted = sorted(a)
    b_sorted = sorted(b)
    max_diff = 0.0
    for v in all_values:
        cdf_a = sum(1 for x in a_sorted if x <= v) / n_a
        cdf_b = sum(1 for x in b_sorted if x <= v) / n_b
        max_diff = max(max_diff, abs(cdf_a - cdf_b))
    # Approximate p-value using the Kolmogorov distribution
    n = (n_a * n_b) / (n_a + n_b)
    z = max_diff * math.sqrt(n)
    p_value = _kolmogorov_p(z)
    return max_diff, p_value


def _kolmogorov_p(z: float) -> float:
    if z <= 0:
        return 1.0
    p = 0.0
    for k in range(1, 100):
        p += ((-1) ** (k - 1)) * math.exp(-2 * k * k * z * z)
    return max(0.0, min(1.0, 2 * p))


def psi(baseline_bins: list[float], current_bins: list[float]) -> float:
    """Population Stability Index.

    Both lists are probability vectors (sum ≈ 1 each). Uses the standard
    symmetric PSI formula.
    """
    if len(baseline_bins) != len(current_bins):
        raise ValueError("baseline_bins and current_bins must have the same length")

    _EPS = 1e-10
    result = 0.0
    for b, c in zip(baseline_bins, current_bins):
        b = max(b, _EPS)
        c = max(c, _EPS)
        result += (c - b) * math.log(c / b)
    return result
