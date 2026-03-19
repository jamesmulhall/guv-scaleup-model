"""
Tests for src.mc_distributions.

Focus on: output shapes, valid/invalid confidence levels, bounds (normal/lognormal),
and reproducibility with fixed seeds.
"""

import numpy as np
import pytest

from src.mc_distributions import sample_normal, sample_lognormal, sample_gpd


class TestSampleNormal:
    """sample_normal shape, confidence validation, and basic distribution properties."""

    def test_output_shape(self):
        """Returned array must have shape (n,)."""
        out = sample_normal(10.0, 20.0, n=100)
        assert out.shape == (100,)

    def test_invalid_confidence_raises(self):
        """Confidence other than 90, 95, 99 must raise ValueError."""
        with pytest.raises(ValueError, match="Confidence level must be 90, 95, or 99"):
            sample_normal(10.0, 20.0, n=5, confidence=80)
        with pytest.raises(ValueError, match="Confidence level must be 90, 95, or 99"):
            sample_normal(10.0, 20.0, n=5, confidence=100)

    def test_accepts_90_95_99(self):
        """Valid confidence levels do not raise."""
        for conf in (90, 95, 99):
            out = sample_normal(10.0, 20.0, n=10, confidence=conf)
            assert out.shape == (10,)

    def test_absolute_true_samples_non_negative(self):
        """With absolute=True, all samples must be >= 0."""
        out = sample_normal(0.0, 20.0, n=500, absolute=True)
        assert np.all(out >= 0)

    def test_absolute_false_allows_negatives(self):
        """With absolute=False (default), samples near zero can go negative."""
        np.random.seed(0)
        out = sample_normal(0.0, 20.0, n=500, absolute=False)
        assert np.any(out < 0)

    def test_reproducibility_with_seed(self):
        """With same global seed, two calls give same samples."""
        np.random.seed(123)
        a = sample_normal(10.0, 20.0, n=20)
        np.random.seed(123)
        b = sample_normal(10.0, 20.0, n=20)
        np.testing.assert_array_equal(a, b)

    def test_mean_near_midpoint_for_symmetric_bounds(self):
        """For symmetric low/high, sample mean should be close to (low+high)/2."""
        low, high = 50.0, 150.0
        out = sample_normal(low, high, n=5000)
        assert np.abs(np.mean(out) - (low + high) / 2) < 5.0


class TestSampleLognormal:
    """sample_lognormal shape, low > 0, confidence, and positivity."""

    def test_output_shape(self):
        """Returned array must have shape (n,)."""
        out = sample_lognormal(1.0, 10.0, n=100)
        assert out.shape == (100,)

    def test_low_must_be_positive(self):
        """low <= 0 must raise (AssertionError from assert)."""
        with pytest.raises(AssertionError, match="Low must be greater than 0"):
            sample_lognormal(0.0, 10.0, n=5)
        with pytest.raises(AssertionError, match="Low must be greater than 0"):
            sample_lognormal(-1.0, 10.0, n=5)

    def test_all_samples_positive(self):
        """Lognormal samples must be strictly positive."""
        out = sample_lognormal(0.1, 100.0, n=500)
        assert np.all(out > 0)

    def test_invalid_confidence_raises(self):
        """Confidence other than 90, 95, 99 must raise ValueError."""
        with pytest.raises(ValueError, match="Confidence level must be 90, 95, or 99"):
            sample_lognormal(1.0, 10.0, n=5, confidence=50)

    def test_accepts_90_95_99(self):
        """Valid confidence levels do not raise."""
        for conf in (90, 95, 99):
            out = sample_lognormal(1.0, 10.0, n=10, confidence=conf)
            assert out.shape == (10,) and np.all(out > 0)


class TestSampleGpd:
    """sample_gpd shape and basic behavior."""

    def test_output_shape(self):
        """Returned array must have shape (n,)."""
        out = sample_gpd(0.1, 1.0, n=100)
        assert out.shape == (100,)

    def test_default_n(self):
        """Default n=1000 gives 1000 samples."""
        out = sample_gpd(0.1, 1.0)
        assert out.shape == (1000,)

    def test_loc_param_shift(self):
        """Larger loc_param should shift samples upward (on average)."""
        np.random.seed(42)
        low_loc = sample_gpd(0.1, 1.0, loc_param=1.0, n=500)
        np.random.seed(42)
        high_loc = sample_gpd(0.1, 1.0, loc_param=10.0, n=500)
        assert np.mean(high_loc) > np.mean(low_loc)
