"""
Tests for src.scale_up_model.

Focus on: correct shapes, conservation of baseline, scaling by relative_cadr_ratio,
cumulative growth, stats ordering, and median_scaleup_curves_for_ratios consistency.
"""

import numpy as np
import pytest

from src.scale_up_model import (
    growth_model,
    calculate_stats,
    median_scaleup_curves_for_ratios,
)


def _make_deterministic_inputs(n_simulations=10, seed=42):
    """Shared deterministic inputs for growth_model."""
    rng = np.random.default_rng(seed)
    return {
        "baseline_cadr": rng.uniform(1e8, 5e8, size=n_simulations),
        "global_market_usable": rng.uniform(1e9, 1e10, size=n_simulations),
        "cost_per_unit": rng.uniform(100, 500, size=n_simulations),
        "cadr_per_unit": rng.uniform(100, 400, size=n_simulations),
    }


class TestGrowthModel:
    """growth_model output shape, baseline, scaling, and growth logic."""

    def test_output_shape(self):
        """Output must be (months + 1, n_simulations)."""
        n_sims = 7
        months = 6
        kwargs = _make_deterministic_inputs(n_simulations=n_sims)
        out = growth_model(**kwargs, months=months)
        assert out.shape == (months + 1, n_sims)

    def test_month_zero_is_scaled_baseline(self):
        """Month 0 must equal baseline_cadr * relative_cadr_ratio (no other terms)."""
        kwargs = _make_deterministic_inputs(n_simulations=5)
        ratio = 0.66
        out = growth_model(**kwargs, months=3, relative_cadr_ratio=ratio)
        expected_month0 = kwargs["baseline_cadr"] * ratio
        np.testing.assert_array_almost_equal(out[0], expected_month0)

    def test_relative_cadr_ratio_scales_entire_trajectory(self):
        """Doubling relative_cadr_ratio must double every time step (same inputs)."""
        kwargs = _make_deterministic_inputs(n_simulations=20)
        out_1 = growth_model(**kwargs, months=4, relative_cadr_ratio=1.0)
        out_2 = growth_model(**kwargs, months=4, relative_cadr_ratio=2.0)
        np.testing.assert_array_almost_equal(out_2, 2.0 * out_1)

    def test_no_growth_implies_flat_trajectory(self):
        """With no new production and no repurposing, CADR stays at month-0 level."""
        kwargs = _make_deterministic_inputs(n_simulations=5)
        # Zero additional production and zero repurposing
        out = growth_model(
            **kwargs,
            months=4,
            additional_annual_production=0.0,
            repurposed_ramp_months=0,
            repurposed_annual_production=0.0,
        )
        for t in range(1, out.shape[0]):
            np.testing.assert_array_almost_equal(out[t], out[0])

    def test_cadr_non_decreasing_over_time(self):
        """Cumulative CADR must be non-decreasing month to month."""
        kwargs = _make_deterministic_inputs(n_simulations=15)
        out = growth_model(**kwargs, months=6)
        for t in range(1, out.shape[0]):
            assert np.all(out[t] >= out[t - 1] - 1e-6), f"Month {t} decreased"

    def test_utilization_ramp_affects_growth(self):
        """Higher utilization_end should give higher final CADR (more production)."""
        kwargs = _make_deterministic_inputs(n_simulations=20)
        low = growth_model(
            **kwargs, months=6, utilization_start=0.5, utilization_end=0.5
        )
        high = growth_model(
            **kwargs, months=6, utilization_start=0.5, utilization_end=1.0
        )
        assert np.all(high[-1] >= low[-1])
        assert np.any(high[-1] > low[-1])

    def test_repurposing_adds_to_cadr(self):
        """With repurposing, later months should exceed no-repurposing case."""
        kwargs = _make_deterministic_inputs(n_simulations=20)
        no_rep = growth_model(
            **kwargs,
            months=6,
            repurposed_ramp_months=0,
            repurposed_annual_production=0.0,
        )
        with_rep = growth_model(
            **kwargs,
            months=6,
            repurposed_ramp_months=2,
            repurposed_annual_production=0.5 / 12,
        )
        assert np.all(with_rep[-1] >= no_rep[-1])
        assert np.any(with_rep[-1] > no_rep[-1])


class TestCalculateStats:
    """calculate_stats output shape and percentile ordering."""

    def test_output_shapes(self):
        """Returns 5 arrays, each of shape (n_months,)."""
        n_sims = 100
        n_months = 7
        monthly = np.random.rand(n_months, n_sims)
        med, p5, p25, p75, p95 = calculate_stats(monthly)
        assert med.shape == (n_months,)
        assert p5.shape == p25.shape == p75.shape == p95.shape == (n_months,)

    def test_percentile_ordering(self):
        """For each time step: p5 <= p25 <= median <= p75 <= p95."""
        n_sims = 200
        n_months = 5
        monthly = np.random.rand(n_months, n_sims)
        med, p5, p25, p75, p95 = calculate_stats(monthly)
        assert np.all(p5 <= p25)
        assert np.all(p25 <= med)
        assert np.all(med <= p75)
        assert np.all(p75 <= p95)

    def test_median_matches_numpy(self):
        """Median series must match np.median(monthly_cadr, axis=1)."""
        monthly = np.random.rand(4, 50)
        med, *_ = calculate_stats(monthly)
        np.testing.assert_array_almost_equal(med, np.median(monthly, axis=1))


class TestMedianScaleupCurvesForRatios:
    """median_scaleup_curves_for_ratios length, shape, and scaling."""

    def test_returns_one_series_per_ratio(self):
        """Length of result must equal length of ratios."""
        kwargs = _make_deterministic_inputs(n_simulations=10)
        ratios = [1.0, 0.66, 1.22]
        result = median_scaleup_curves_for_ratios(**kwargs, ratios=ratios, months=4)
        assert len(result) == len(ratios)

    def test_each_series_shape(self):
        """Each median series has shape (months + 1,)."""
        kwargs = _make_deterministic_inputs(n_simulations=10)
        result = median_scaleup_curves_for_ratios(**kwargs, ratios=[1.0, 0.5], months=6)
        for arr in result:
            assert arr.shape == (7,)

    def test_ratio_scaling_consistency(self):
        """Median for ratio 2.0 should be 2x median for ratio 1.0 (same inputs)."""
        kwargs = _make_deterministic_inputs(n_simulations=30)
        result = median_scaleup_curves_for_ratios(**kwargs, ratios=[1.0, 2.0], months=4)
        np.testing.assert_array_almost_equal(result[1], 2.0 * result[0])
