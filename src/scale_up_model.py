"""
Scale-up model for CADR (Clean Air Delivery Rate) production over time.

Models growth of germicidal UV and repurposed fluorescent lamp supply with
utilization ramp-up and optional repurposing phase.
"""

import numpy as np


def growth_model(
    baseline_cadr,
    global_market_usable,
    cost_per_unit,
    cadr_per_unit,
    months=6,
    utilization_start=0.7,
    utilization_end=1.0,
    utilization_ramp_months=3,
    additional_annual_production=1 / 12,
    repurposed_ramp_months=1,
    repurposed_annual_production=2 / 12,
):
    """
    Model growth of CADR production over time with factory utilization ramp-up.

    Arguments:
        baseline_cadr (array-like): Initial CADR values, shape (n_simulations,).
        global_market_usable (array-like): Annual usable market size, shape (n_simulations,).
        cost_per_unit (array-like): Cost per unit, shape (n_simulations,).
        cadr_per_unit (array-like): CADR per unit, shape (n_simulations,).
        months (int): Number of months to model. Default 6.
        utilization_start (float): Starting utilization (0-1). Default 0.7.
        utilization_end (float): Ending utilization (0-1). Default 1.0.
        utilization_ramp_months (int): Months over which utilization ramps up. Default 3.
        additional_annual_production (float): Additional production per month as fraction of annual. Default 1/12.
        repurposed_ramp_months (int): Months over which repurposed production ramps up. Default 1.
        repurposed_annual_production (float): Repurposed production per month as fraction of annual. Default 2/12.

    Returns:
        numpy.ndarray: Cumulative CADR per month, shape (months, n_simulations).
    """
    n_simulations = len(baseline_cadr)
    monthly_cadr = np.zeros((months, n_simulations))

    # Month 0 (baseline)
    monthly_cadr[0] = baseline_cadr

    # Calculate growth for each month
    for month in range(1, months):
        # Calculate utilization factor
        if month <= utilization_ramp_months:
            utilization_factor = (
                utilization_start
                + (utilization_end - utilization_start)
                * month
                / utilization_ramp_months
            )
        else:
            utilization_factor = utilization_end

        # Calculate new production for this month
        new_monthly_production = (
            global_market_usable * additional_annual_production * utilization_factor
        )
        new_units = new_monthly_production / cost_per_unit

        # Calculate repurposing
        if month <= repurposed_ramp_months:
            repurposed_factor = (
                repurposed_annual_production * month / repurposed_ramp_months
            )
            repurposed_value = (
                global_market_usable * repurposed_factor * utilization_factor
            )
            repurposed_units = repurposed_value / cost_per_unit
        else:
            repurposed_units = 0

        new_monthly_cadr = (new_units + repurposed_units) * cadr_per_unit

        # Cumulative CADR (previous months + new month)
        monthly_cadr[month] = monthly_cadr[month - 1] + new_monthly_cadr

    return monthly_cadr


def calculate_stats(monthly_cadr):
    """
    Compute median and percentile series across simulations for each month.

    Arguments:
        monthly_cadr (array-like): CADR per month per simulation, shape (months, n_simulations).

    Returns:
        tuple: (median_cadr, p5_cadr, p25_cadr, p75_cadr, p95_cadr), each shape (months,).
    """
    median_cadr = np.median(monthly_cadr, axis=1)
    p5_cadr = np.percentile(monthly_cadr, 5, axis=1)
    p25_cadr = np.percentile(monthly_cadr, 25, axis=1)
    p75_cadr = np.percentile(monthly_cadr, 75, axis=1)
    p95_cadr = np.percentile(monthly_cadr, 95, axis=1)
    return median_cadr, p5_cadr, p25_cadr, p75_cadr, p95_cadr
