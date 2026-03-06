"""
Plotting utilities for the GUV scale-up model.
"""

import numpy as np
import matplotlib.pyplot as plt

from .scale_up_model import calculate_stats


def plot_uv_fluoro_ramp(
    uv_monthly_cadr,
    fluoro_monthly_cadr,
    cadr_requirement,
    confidence_interval=90,
    title=(
        "Ramp-up of Germicidal UV Lamps vs Vital Worker Requirements\n"
        "(Pandemic 5X Transmissibility of COVID-19)"
    ),
    ylim=(0, 2.5e11),
    save_path=None,
    show=True,
):
    """
    Generate a ramp-up plot comparing UV lamp supply, repurposed fluorescent
    lamp supply, and CADR requirement over time. Stats (median and percentile
    bounds) are computed from the raw simulation arrays via calculate_stats.

    Arguments:
        uv_monthly_cadr (array-like): Shape (months, n_simulations). UV lamp CADR per month.
        fluoro_monthly_cadr (array-like): Shape (months, n_simulations). Repurposed fluorescent
            CADR per month.
        cadr_requirement (array-like): Shape (n_simulations,). CADR requirement (constant).
        confidence_interval (int or float): Shaded interval width in percent. E.g. 90 → 5th/95th,
            50 → 25th/75th. Default 90.
        title (str): Plot title. Defaults to pandemic scenario title.
        ylim (tuple): (ymin, ymax) for y-axis. Defaults to (0, 2.5e11).
        save_path (str or None): If set, save figure to this path (e.g. 'results/ramp.png').
        show (bool): If True, call plt.show(). Default True.

    Returns:
        tuple: (fig, ax) for further customization or saving by the caller.
    """
    p_lower = (100 - confidence_interval) / 2
    p_upper = 100 - p_lower
    ci_label = f"{int(confidence_interval)}% CI"

    uv_median, _, _, _, _ = calculate_stats(uv_monthly_cadr)
    uv_p_lo = np.percentile(uv_monthly_cadr, p_lower, axis=1)
    uv_p_hi = np.percentile(uv_monthly_cadr, p_upper, axis=1)

    fluoro_median, _, _, _, _ = calculate_stats(fluoro_monthly_cadr)
    fluoro_p_lo = np.percentile(fluoro_monthly_cadr, p_lower, axis=1)
    fluoro_p_hi = np.percentile(fluoro_monthly_cadr, p_upper, axis=1)

    months = uv_monthly_cadr.shape[0]
    median_cadr_req = np.median(cadr_requirement)
    cadr_req_lo = np.percentile(cadr_requirement, p_lower)
    cadr_req_hi = np.percentile(cadr_requirement, p_upper)

    plt.style.use(
        "https://raw.githubusercontent.com/allfed/ALLFED-matplotlib-style-sheet/main/"
        "ALLFED.mplstyle"
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    months_array = np.arange(months)

    # UV lamp supply
    ax.plot(
        months_array,
        uv_median,
        color="#3A913F",
        linewidth=3,
        label="UV Lamps (Median)",
        marker="o",
        markersize=6,
    )
    ax.fill_between(
        months_array,
        uv_p_lo,
        uv_p_hi,
        alpha=0.3,
        color="#3A913F",
        label=f"UV Lamps ({ci_label})",
    )

    # Repurposed fluorescent lamp supply
    ax.plot(
        months_array,
        fluoro_median,
        color="#F0B323",
        linewidth=3,
        label="Repurposed fluorescent lamps (Median)",
        marker="s",
        markersize=6,
    )
    ax.fill_between(
        months_array,
        fluoro_p_lo,
        fluoro_p_hi,
        alpha=0.3,
        color="#F0B323",
        label=f"Repurposed fluorescent lamps ({ci_label})",
    )

    # CADR requirement (constant over time)
    ax.axhline(
        y=median_cadr_req,
        color="#6c7075",
        linestyle="--",
        linewidth=3,
        label="CADR Requirement (Median)",
    )
    ax.axhspan(
        cadr_req_lo,
        cadr_req_hi,
        alpha=0.2,
        color="#6c7075",
        label=f"CADR Requirement ({ci_label})",
    )

    ax.set_xlabel("Month", fontsize=12, fontweight="bold")
    ax.set_ylabel("Global CADR", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_ylim(ylim)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_median_uv_scaleup_by_cadr_ratio(
    months_axis,
    curves,
    title=None,
    ylim=None,
    save_path=None,
    show=True,
    cadr_requirement=None,
    cadr_requirement_ci=90,
):
    """
    Plot median UV scale-up curves for different relative CADR ratios (e.g. by pathogen).

    Arguments:
        months_axis (array-like): Month indices, length matches each median_series.
        curves (list of tuple): Each element is (median_series, label). median_series
            is 1D, length len(months_axis); label is used in the legend.
        title (str or None): Plot title. If None, a default title is used.
        ylim (tuple or None): (ymin, ymax) for y-axis. If None, axes auto-scale.
        save_path (str or None): If set, save figure to this path.
        show (bool): If True, call plt.show(). Default True.
        cadr_requirement (array-like or None): If provided, shape (n_simulations,). Draw
            median CADR requirement as a horizontal line with optional CI band. If None,
            do not show requirement. Default None.
        cadr_requirement_ci (int or float): Confidence interval in percent for the
            requirement band when cadr_requirement is set (e.g. 90 → 5th/95th). Default 90.

    Returns:
        tuple: (fig, ax) for further customization or saving by the caller.
    """
    if title is None:
        title = "Median UV scale-up by relative CADR ratio (pathogen)"

    plt.style.use(
        "https://raw.githubusercontent.com/allfed/ALLFED-matplotlib-style-sheet/main/"
        "ALLFED.mplstyle"
    )

    colors = ["#3A913F", "#2563EB", "#DC2626"]
    markers = ["o", "s", "^"]

    fig, ax = plt.subplots(figsize=(14, 8))

    for i, (median_series, label) in enumerate(curves):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        ax.plot(
            months_axis,
            median_series,
            color=color,
            linewidth=3,
            label=label,
            marker=marker,
            markersize=6,
        )

    if cadr_requirement is not None:
        p_lower = (100 - cadr_requirement_ci) / 2
        p_upper = 100 - p_lower
        ci_label = f"{int(cadr_requirement_ci)}% CI"
        median_cadr_req = np.median(cadr_requirement)
        cadr_req_lo = np.percentile(cadr_requirement, p_lower)
        cadr_req_hi = np.percentile(cadr_requirement, p_upper)
        ax.axhline(
            y=median_cadr_req,
            color="#6c7075",
            linestyle="--",
            linewidth=3,
            label="CADR Requirement (Median)",
        )
        ax.axhspan(
            cadr_req_lo,
            cadr_req_hi,
            alpha=0.2,
            color="#6c7075",
            label=f"CADR Requirement ({ci_label})",
        )

    ax.set_xlabel("Month", fontsize=12, fontweight="bold")
    ax.set_ylabel("Global CADR", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper left", fontsize=10)
    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax
