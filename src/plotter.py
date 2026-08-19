"""
Plotting utilities for the GUV scale-up model.
"""

import numpy as np
import matplotlib.pyplot as plt

from .scale_up_model import calculate_stats


def _plot_cadr_series(
    ax,
    months_array,
    monthly_cadr,
    color,
    median_label,
    ci_label,
    marker,
    p_lower,
    p_upper,
    y_scale=1.0,
):
    """Plot median CADR trajectory with a percentile band."""
    median, *_ = calculate_stats(monthly_cadr)
    p_lo = np.percentile(monthly_cadr, p_lower, axis=1)
    p_hi = np.percentile(monthly_cadr, p_upper, axis=1)
    ax.plot(
        months_array,
        median * y_scale,
        color=color,
        linewidth=3,
        label=median_label,
        marker=marker,
        markersize=6,
    )
    ax.fill_between(
        months_array,
        p_lo * y_scale,
        p_hi * y_scale,
        alpha=0.3,
        color=color,
        label=ci_label,
    )


def plot_uv_fluoro_ramp(
    uv_monthly_cadr,
    fluoro_monthly_cadr,
    cadr_requirement,
    far_uvc_monthly_cadr=None,
    confidence_interval=90,
    title=(
        "Ramp-up of Germicidal UV Lamps vs Vital Worker Requirements\n"
        "(Pandemic 5X Transmissibility of COVID-19)"
    ),
    ylim=None,
    save_path=None,
    show=True,
    show_absolute_cadr_axis=False,
):
    """
    Generate a ramp-up plot comparing conventional UV lamp supply, far-UVC
    supply, repurposed fluorescent lamp supply, and CADR requirement over time.
    The left y-axis is percent of the median CADR requirement. Stats (median
    and percentile bounds) are computed from the raw simulation arrays via
    calculate_stats.

    Arguments:
        uv_monthly_cadr (array-like): Shape (months, n_simulations). Conventional UV CADR per month.
        fluoro_monthly_cadr (array-like): Shape (months, n_simulations). Repurposed fluorescent
            CADR per month.
        cadr_requirement (array-like): Shape (n_simulations,). CADR requirement (constant).
        far_uvc_monthly_cadr (array-like or None): Shape (months, n_simulations). Far-UVC CADR
            per month. If None, far-UVC is omitted. Default None.
        confidence_interval (int or float): Shaded interval width in percent. E.g. 90 → 5th/95th,
            50 → 25th/75th. Default 90.
        title (str): Plot title. Defaults to pandemic scenario title.
        ylim (tuple or None): (ymin, ymax) for the left y-axis in percent of median
            requirement. If None, the axis starts at 0 and autoscales. Default None.
        save_path (str or None): If set, save figure to this path (e.g. 'results/ramp.png').
        show (bool): If True, call plt.show(). Default True.
        show_absolute_cadr_axis (bool): If True, add a right y-axis in absolute
            Global CADR units. Default False.

    Returns:
        tuple: (fig, ax) for further customization or saving by the caller.
    """
    p_lower = (100 - confidence_interval) / 2
    p_upper = 100 - p_lower
    ci_label = f"{int(confidence_interval)}% CI"

    months = uv_monthly_cadr.shape[0]
    median_cadr_req = np.median(cadr_requirement)
    if median_cadr_req == 0:
        raise ValueError("Median CADR requirement is zero; cannot scale to percent.")
    y_scale = 100.0 / median_cadr_req
    cadr_req_lo = np.percentile(cadr_requirement, p_lower) * y_scale
    cadr_req_hi = np.percentile(cadr_requirement, p_upper) * y_scale

    plt.style.use(
        "https://raw.githubusercontent.com/allfed/ALLFED-matplotlib-style-sheet/main/"
        "ALLFED.mplstyle"
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    months_array = np.arange(months)

    _plot_cadr_series(
        ax,
        months_array,
        uv_monthly_cadr,
        color="#3A913F",
        median_label="UR-UV Lamps (Median)",
        ci_label=f"UR-UV Lamps ({ci_label})",
        marker="o",
        p_lower=p_lower,
        p_upper=p_upper,
        y_scale=y_scale,
    )

    if far_uvc_monthly_cadr is not None:
        _plot_cadr_series(
            ax,
            months_array,
            far_uvc_monthly_cadr,
            color="#2563EB",
            median_label="Far-UVC (Median)",
            ci_label=f"Far-UVC ({ci_label})",
            marker="^",
            p_lower=p_lower,
            p_upper=p_upper,
            y_scale=y_scale,
        )

    _plot_cadr_series(
        ax,
        months_array,
        fluoro_monthly_cadr,
        color="#F0B323",
        median_label="Repurposed fluorescent lamps (Median)",
        ci_label=f"Repurposed fluorescent lamps ({ci_label})",
        marker="s",
        p_lower=p_lower,
        p_upper=p_upper,
        y_scale=y_scale,
    )

    # CADR requirement (constant over time); median is 100% by construction
    ax.axhline(
        y=100.0,
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
    ax.set_ylabel("% of Median Global CADR Requirement", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper left", fontsize=10)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(bottom=0)

    if show_absolute_cadr_axis:
        ax_abs = ax.twinx()
        y_lo, y_hi = ax.get_ylim()
        ax_abs.set_ylim(y_lo / y_scale, y_hi / y_scale)
        ax_abs.set_ylabel("Global CADR", fontsize=12, fontweight="bold")

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


def plot_forecast_pct_at_month(
    series,
    confidence_interval=50,
    show_ci=True,
    title=None,
    ylim=None,
    save_path=None,
    show=True,
):
    """
    Plot total percent of CADR requirement at a fixed ramp month versus year.

    Each series is one fluorescent-market scenario; only the total (UR-UV +
    far-UVC + fluorescents) is drawn.

    Arguments:
        series (list of tuple): Each element is (forecast_dict, label).
            forecast_dict is the output of forecast_pct_of_requirement_at_month.
        confidence_interval (int or float): Shaded interval for each total.
            Default 50.
        show_ci (bool): If True, draw percentile bands. Default True.
        title (str or None): Plot title. If None, a default using the snapshot month.
        ylim (tuple or None): (ymin, ymax) in percent. If None, start at 0.
        save_path (str or None): If set, save figure to this path.
        show (bool): If True, call plt.show(). Default True.

    Returns:
        tuple: (fig, ax)
    """
    if not series:
        raise ValueError("series must contain at least one (forecast, label) pair")

    month = series[0][0].get("month", 3)
    p_lower = (100 - confidence_interval) / 2
    p_upper = 100 - p_lower
    ci_label = f"{int(confidence_interval)}% CI"

    if title is None:
        title = (
            f"CADR at {month} months as % of requirement\n"
            "by fluorescent phase-out scenario"
        )

    plt.style.use(
        "https://raw.githubusercontent.com/allfed/ALLFED-matplotlib-style-sheet/main/"
        "ALLFED.mplstyle"
    )

    colors = ["#DC2626", "#2563EB", "#3A913F"]
    markers = ["o", "s", "^"]

    fig, ax = plt.subplots(figsize=(14, 8))

    for i, (forecast, label) in enumerate(series):
        years = np.asarray(forecast["years"])
        total = forecast["pct_total"]
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        median, *_ = calculate_stats(total)
        ax.plot(
            years,
            median,
            color=color,
            linewidth=3,
            marker=marker,
            markersize=6,
            label=f"{label} (Median)",
        )
        if show_ci:
            ax.fill_between(
                years,
                np.percentile(total, p_lower, axis=1),
                np.percentile(total, p_upper, axis=1),
                alpha=0.18,
                color=color,
                label=f"{label} ({ci_label})",
            )

    ax.axhline(
        y=100.0,
        color="#6c7075",
        linestyle="--",
        linewidth=2,
        label="CADR requirement (100%)",
    )

    ax.set_xlabel("Year", fontsize=12, fontweight="bold")
    ax.set_ylabel(
        f"% of Global CADR requirement at {month} months",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper left", fontsize=10)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(bottom=0)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax
