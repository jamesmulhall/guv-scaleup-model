"""
Tests for src.plotter.

Focus on: return types, correct use of data (shapes), optional cadr_requirement,
and save without display. Uses non-interactive backend and optional style patch
to avoid display/network in CI.
"""

import numpy as np
import pytest

# Use Agg backend so no display is required
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.plotter import (
    plot_uv_fluoro_ramp,
    plot_median_uv_scaleup_by_cadr_ratio,
    plot_forecast_pct_at_month,
)


@pytest.fixture(autouse=True)
def _no_style_network(monkeypatch):
    """Avoid loading style from URL in tests (no network dependency)."""
    monkeypatch.setattr(plt.style, "use", lambda x: None)


def _make_uv_fluoro_data(n_months=7, n_sims=20):
    """Minimal valid arrays for plot_uv_fluoro_ramp."""
    rng = np.random.default_rng(42)
    uv = np.cumsum(rng.uniform(1e8, 1e9, (n_months, n_sims)), axis=0)
    fluoro = np.cumsum(rng.uniform(1e7, 1e8, (n_months, n_sims)), axis=0)
    far_uvc = np.cumsum(rng.uniform(1e7, 2e8, (n_months, n_sims)), axis=0)
    cadr_req = rng.uniform(1e10, 5e10, n_sims)
    return uv, fluoro, far_uvc, cadr_req


class TestPlotUvFluoroRamp:
    """plot_uv_fluoro_ramp return value and save."""

    def test_returns_fig_ax(self):
        """Must return (fig, ax) tuple."""
        uv, fluoro, far_uvc, cadr_req = _make_uv_fluoro_data()
        fig, ax = plot_uv_fluoro_ramp(
            uv, fluoro, cadr_req, far_uvc_monthly_cadr=far_uvc, show=False, save_path=None
        )
        assert fig is not None
        assert ax is not None
        assert hasattr(ax, "plot")
        plt.close(fig)

    def test_save_creates_file(self, tmp_path):
        """With save_path set, file must be created."""
        uv, fluoro, far_uvc, cadr_req = _make_uv_fluoro_data()
        path = tmp_path / "ramp.png"
        plot_uv_fluoro_ramp(
            uv,
            fluoro,
            cadr_req,
            far_uvc_monthly_cadr=far_uvc,
            show=False,
            save_path=str(path),
        )
        assert path.exists()
        assert path.stat().st_size > 0

    def test_uses_correct_month_count(self):
        """Plot x-axis should reflect number of time steps (months)."""
        n_months = 5
        uv, fluoro, far_uvc, cadr_req = _make_uv_fluoro_data(
            n_months=n_months, n_sims=10
        )
        fig, ax = plot_uv_fluoro_ramp(
            uv, fluoro, cadr_req, far_uvc_monthly_cadr=far_uvc, show=False, save_path=None
        )
        # One line per series; first line's x data length should match
        lines = [
            line for line in ax.get_children() if hasattr(line, "get_xdata")
        ]  # noqa: F841
        # get_xdata() on Line2D; at least one line should have len n_months
        assert len(lines[0].get_xdata()) == n_months
        plt.close(fig)

    def test_far_uvc_adds_series(self):
        """With far_uvc_monthly_cadr set, an extra median line is drawn."""
        uv, fluoro, far_uvc, cadr_req = _make_uv_fluoro_data(n_months=4, n_sims=8)
        fig, ax = plot_uv_fluoro_ramp(uv, fluoro, cadr_req, show=False)
        n_without = len(ax.lines)
        plt.close(fig)
        fig, ax = plot_uv_fluoro_ramp(
            uv,
            fluoro,
            cadr_req,
            far_uvc_monthly_cadr=far_uvc,
            show=False,
        )
        assert len(ax.lines) == n_without + 1
        plt.close(fig)

    def test_left_axis_is_percent_of_median_requirement(self):
        """Median requirement line is at 100; series are scaled to that percent."""
        uv, fluoro, far_uvc, cadr_req = _make_uv_fluoro_data(n_months=4, n_sims=15)
        fig, ax = plot_uv_fluoro_ramp(
            uv,
            fluoro,
            cadr_req,
            far_uvc_monthly_cadr=far_uvc,
            show=False,
        )
        assert "% of Median Global CADR Requirement" in ax.get_ylabel()
        dashed = [
            line.get_ydata()[0]
            for line in ax.lines
            if line.get_linestyle() == "--"
        ]
        assert any(abs(y - 100.0) < 1e-9 for y in dashed)
        median_req = np.median(cadr_req)
        expected_month0 = 100.0 * np.median(uv[0]) / median_req
        np.testing.assert_allclose(ax.lines[0].get_ydata()[0], expected_month0)
        plt.close(fig)

    def test_absolute_axis_optional_and_synced(self):
        """Right-hand Global CADR axis is off by default and matches percent when on."""
        uv, fluoro, far_uvc, cadr_req = _make_uv_fluoro_data(n_months=4, n_sims=10)
        fig, ax = plot_uv_fluoro_ramp(
            uv, fluoro, cadr_req, far_uvc_monthly_cadr=far_uvc, show=False
        )
        assert len(fig.axes) == 1
        plt.close(fig)

        fig, ax = plot_uv_fluoro_ramp(
            uv,
            fluoro,
            cadr_req,
            far_uvc_monthly_cadr=far_uvc,
            show=False,
            show_absolute_cadr_axis=True,
            ylim=(0, 200),
        )
        assert len(fig.axes) == 2
        ax_abs = fig.axes[1]
        assert ax_abs.get_ylabel() == "Global CADR"
        median_req = np.median(cadr_req)
        np.testing.assert_allclose(ax_abs.get_ylim()[1], 200.0 / 100.0 * median_req)
        plt.close(fig)


class TestPlotMedianUvScaleupByCadrRatio:
    """plot_median_uv_scaleup_by_cadr_ratio return value, curves, optional requirement."""

    def test_returns_fig_ax(self):
        """Must return (fig, ax) tuple."""
        months = np.arange(7)
        curves = [(np.linspace(1e9, 2e9, 7), "Pathogen A (1.0)")]
        fig, ax = plot_median_uv_scaleup_by_cadr_ratio(months, curves, show=False)
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plots_all_curves(self):
        """Number of lines from curve data should match len(curves)."""
        months = np.arange(5)
        curves = [
            (np.linspace(1e9, 2e9, 5), "A (1.0)"),
            (np.linspace(0.8e9, 1.8e9, 5), "B (0.66)"),
        ]
        fig, ax = plot_median_uv_scaleup_by_cadr_ratio(months, curves, show=False)
        # One line per (median_series, label)
        assert len(ax.lines) == 2
        plt.close(fig)

    def test_without_cadr_requirement_no_hline(self):
        """With cadr_requirement=None, no horizontal line (axhline) for requirement."""
        months = np.arange(4)
        curves = [(np.array([1.0, 2.0, 3.0, 4.0]), "Only curve")]
        fig, ax = plot_median_uv_scaleup_by_cadr_ratio(
            months, curves, cadr_requirement=None, show=False
        )
        # ax.get_lines() gives Line2D; axhline adds a Line2D. So we have 1 line.
        # axhspan adds PolyCollection. Count lines only.
        n_lines = len(ax.lines)
        assert n_lines == 1
        plt.close(fig)

    def test_with_cadr_requirement_adds_hline(self):
        """With cadr_requirement provided, requirement line is drawn (extra line)."""
        months = np.arange(4)
        curves = [(np.array([1.0, 2.0, 3.0, 4.0]), "Curve")]
        cadr_req = np.array([2.5, 2.6, 2.4])
        fig, ax = plot_median_uv_scaleup_by_cadr_ratio(
            months, curves, cadr_requirement=cadr_req, show=False
        )
        # 1 curve line + 1 axhline
        assert len(ax.lines) >= 2
        plt.close(fig)

    def test_save_creates_file(self, tmp_path):
        """With save_path set, file must be created."""
        months = np.arange(6)
        curves = [(np.linspace(1e9, 3e9, 6), "Test (1.0)")]
        path = tmp_path / "pathogen_plot.png"
        plot_median_uv_scaleup_by_cadr_ratio(
            months, curves, show=False, save_path=str(path)
        )
        assert path.exists()
        assert path.stat().st_size > 0

    def test_curve_length_must_match_months_axis(self):
        """Curve length mismatch with months_axis would misplot; we assert consistent use."""
        months = np.arange(5)
        curves = [(np.linspace(0, 1, 5), "OK")]  # same length
        fig, ax = plot_median_uv_scaleup_by_cadr_ratio(months, curves, show=False)
        line = ax.lines[0]
        assert len(line.get_xdata()) == 5
        assert len(line.get_ydata()) == 5
        plt.close(fig)


class TestPlotForecastPctAtMonth:
    """plot_forecast_pct_at_month return value and series count."""

    def _forecast(self, n_years=6, n_sims=12, seed=1):
        rng = np.random.default_rng(seed)
        years = np.arange(2025, 2025 + n_years)
        return {
            "years": years,
            "month": 3,
            "pct_total": np.cumsum(rng.uniform(5, 15, (n_years, n_sims)), axis=0),
            "pct_uv": rng.uniform(1, 5, (n_years, n_sims)),
            "pct_far_uvc": rng.uniform(1, 4, (n_years, n_sims)),
            "pct_fluoro": rng.uniform(10, 40, (n_years, n_sims)),
        }

    def test_returns_fig_ax(self):
        series = [(self._forecast(), "Policy phase-out")]
        fig, ax = plot_forecast_pct_at_month(series, show=False)
        assert fig is not None
        assert "Year" in ax.get_xlabel()
        plt.close(fig)

    def test_plots_one_total_line_per_scenario(self):
        series = [
            (self._forecast(seed=1), "A"),
            (self._forecast(seed=2), "B"),
            (self._forecast(seed=3), "C"),
        ]
        fig, ax = plot_forecast_pct_at_month(series, show_ci=False, show=False)
        # 3 scenario medians + 1 requirement hline
        assert len(ax.lines) == 4
        plt.close(fig)

    def test_save_creates_file(self, tmp_path):
        path = tmp_path / "forecast.png"
        plot_forecast_pct_at_month(
            [(self._forecast(), "Scenario")],
            show=False,
            save_path=str(path),
        )
        assert path.exists()
        assert path.stat().st_size > 0

