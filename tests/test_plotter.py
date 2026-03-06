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

from src.plotter import plot_uv_fluoro_ramp, plot_median_uv_scaleup_by_cadr_ratio


@pytest.fixture(autouse=True)
def _no_style_network(monkeypatch):
    """Avoid loading style from URL in tests (no network dependency)."""
    monkeypatch.setattr(plt.style, "use", lambda x: None)


def _make_uv_fluoro_data(n_months=7, n_sims=20):
    """Minimal valid arrays for plot_uv_fluoro_ramp."""
    rng = np.random.default_rng(42)
    uv = np.cumsum(rng.uniform(1e8, 1e9, (n_months, n_sims)), axis=0)
    fluoro = np.cumsum(rng.uniform(1e7, 1e8, (n_months, n_sims)), axis=0)
    cadr_req = rng.uniform(1e10, 5e10, n_sims)
    return uv, fluoro, cadr_req


class TestPlotUvFluoroRamp:
    """plot_uv_fluoro_ramp return value and save."""

    def test_returns_fig_ax(self):
        """Must return (fig, ax) tuple."""
        uv, fluoro, cadr_req = _make_uv_fluoro_data()
        fig, ax = plot_uv_fluoro_ramp(uv, fluoro, cadr_req, show=False, save_path=None)
        assert fig is not None
        assert ax is not None
        assert hasattr(ax, "plot")
        plt.close(fig)

    def test_save_creates_file(self, tmp_path):
        """With save_path set, file must be created."""
        uv, fluoro, cadr_req = _make_uv_fluoro_data()
        path = tmp_path / "ramp.png"
        plot_uv_fluoro_ramp(uv, fluoro, cadr_req, show=False, save_path=str(path))
        assert path.exists()
        assert path.stat().st_size > 0

    def test_uses_correct_month_count(self):
        """Plot x-axis should reflect number of time steps (months)."""
        n_months = 5
        uv, fluoro, cadr_req = _make_uv_fluoro_data(n_months=n_months, n_sims=10)
        fig, ax = plot_uv_fluoro_ramp(uv, fluoro, cadr_req, show=False, save_path=None)
        # One line per series; first line's x data length should match
        lines = [l for l in ax.get_children() if hasattr(l, "get_xdata")]
        # get_xdata() on Line2D; at least one line should have len n_months
        line = ax.lines[0]
        assert len(line.get_xdata()) == n_months
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
