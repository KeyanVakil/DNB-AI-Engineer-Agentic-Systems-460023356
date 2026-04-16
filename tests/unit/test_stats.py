"""Unit tests for drift detection statistics (no database required)."""

from __future__ import annotations

import math

import pytest


def test_ks_test_identical_distributions():
    from app.drift.stats import ks_test

    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    stat, p = ks_test(data, data)
    assert stat == pytest.approx(0.0, abs=0.01)
    assert p > 0.05


def test_ks_test_completely_different_distributions():
    from app.drift.stats import ks_test

    baseline = [1.0] * 50
    current = [10.0] * 50
    stat, p = ks_test(baseline, current)
    assert stat == pytest.approx(1.0, abs=0.01)
    assert p < 0.01


def test_ks_test_empty_lists_return_no_diff():
    from app.drift.stats import ks_test

    stat, p = ks_test([], [])
    assert stat == 0.0
    assert p == 1.0


def test_psi_identical_distributions_is_zero():
    from app.drift.stats import psi

    bins = [0.25, 0.25, 0.25, 0.25]
    assert psi(bins, bins) == pytest.approx(0.0, abs=1e-9)


def test_psi_detects_large_shift():
    from app.drift.stats import psi

    baseline = [0.25, 0.25, 0.25, 0.25]
    shifted = [0.7, 0.1, 0.1, 0.1]
    assert psi(baseline, shifted) > 0.2


def test_psi_raises_on_mismatched_lengths():
    from app.drift.stats import psi

    with pytest.raises(ValueError):
        psi([0.5, 0.5], [0.3, 0.3, 0.4])


def test_psi_uses_eps_floor_to_avoid_log_zero():
    from app.drift.stats import psi

    baseline = [1.0, 0.0, 0.0, 0.0]
    current = [0.0, 1.0, 0.0, 0.0]
    result = psi(baseline, current)
    assert math.isfinite(result)
