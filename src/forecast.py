"""
Multi-year forecast of 3-month CADR coverage as markets and demand evolve.
"""

import numpy as np

from .scale_up_model import growth_model


def yearly_index(years, base_year, cagr):
    """
    Compound growth index relative to a base year.

    Arguments:
        years (array-like): Calendar years, shape (n_years,).
        base_year (float): Year at which the index equals 1.
        cagr (float): Compound annual growth rate (e.g. 0.007 for 0.7%/yr).

    Returns:
        numpy.ndarray: Index values, shape (n_years,).
    """
    years = np.asarray(years, dtype=float)
    return (1.0 + cagr) ** (years - base_year)


def linear_remaining_share(years, start_year, end_year):
    """
    Linear phase-out share: 1 at start_year, 0 at end_year, clipped to [0, 1].

    Arguments:
        years (array-like): Calendar years, shape (n_years,).
        start_year (float): Last year at full remaining share (1.0).
        end_year (float): First year at zero remaining share.

    Returns:
        numpy.ndarray: Remaining share, shape (n_years,).
    """
    if end_year <= start_year:
        raise ValueError("end_year must be greater than start_year")
    years = np.asarray(years, dtype=float)
    share = 1.0 - (years - start_year) / (end_year - start_year)
    return np.clip(share, 0.0, 1.0)


def exponential_remaining_share(years, base_year, annual_replacement_rate):
    """
    Remaining share if a constant fraction of what is left is replaced each year.

    remaining(year) = (1 - annual_replacement_rate) ** (year - base_year)

    Arguments:
        years (array-like): Calendar years, shape (n_years,).
        base_year (float): Year at which the share equals 1.
        annual_replacement_rate (float): Fraction of remaining fluorescent
            market replaced in one year (e.g. 0.20 for 20%/yr). Must be < 1.

    Returns:
        numpy.ndarray: Remaining share, shape (n_years,).
    """
    if annual_replacement_rate >= 1.0:
        raise ValueError("annual_replacement_rate must be less than 1")
    years = np.asarray(years, dtype=float)
    return (1.0 - annual_replacement_rate) ** (years - base_year)


def biphasic_remaining_share(
    years,
    start_year,
    fast_end_year,
    remaining_at_fast_end=0.10,
    tail_half_life_years=8.0,
):
    """
    Two-exponential remaining share: a fast term plus a slow tail.

    remaining(t) = (1 - R) * exp(-k_fast * t) + R * exp(-k_slow * t)

    where t is years since ``start_year``, R is ``remaining_at_fast_end``,
    k_slow is set from ``tail_half_life_years``, and k_fast is chosen so
    remaining(fast_end_year) equals R. That hits a 90% phase-out by the
    compliance date without a sharp corner, then the residual decays with
    the slow half-life (exemptions, specialty uses, leftover stock).

    Before ``start_year`` the share is 1.

    Arguments:
        years (array-like): Calendar years, shape (n_years,).
        start_year (float): Year at which the share equals 1.
        fast_end_year (float): Year at which remaining share equals
            remaining_at_fast_end.
        remaining_at_fast_end (float): Share still remaining at fast_end_year
            (e.g. 0.10 means 90% phased out). Default 0.10.
        tail_half_life_years (float): Half-life of the slow exponential.
            Default 8.

    Returns:
        numpy.ndarray: Remaining share, shape (n_years,).
    """
    if fast_end_year <= start_year:
        raise ValueError("fast_end_year must be greater than start_year")
    if not 0.0 < remaining_at_fast_end < 1.0:
        raise ValueError("remaining_at_fast_end must be in (0, 1)")
    if tail_half_life_years <= 0:
        raise ValueError("tail_half_life_years must be positive")

    years = np.asarray(years, dtype=float)
    t = years - start_year
    t_fast = float(fast_end_year - start_year)
    residual = remaining_at_fast_end
    k_slow = np.log(2.0) / tail_half_life_years
    slow_at_fast_end = np.exp(-k_slow * t_fast)
    fast_frac_at_end = residual * (1.0 - slow_at_fast_end) / (1.0 - residual)
    if not 0.0 < fast_frac_at_end < 1.0:
        raise ValueError(
            "Cannot fit a fast exponential for these phase-out parameters."
        )
    k_fast = -np.log(fast_frac_at_end) / t_fast

    share = (1.0 - residual) * np.exp(-k_fast * t) + residual * np.exp(-k_slow * t)
    share = np.where(t <= 0, 1.0, share)
    return np.clip(share, 0.0, 1.0)


def demand_index_for_driver(years, base_year, driver, population_cagr, labour_force_cagr):
    """
    Demand index from either population or labour-force growth.

    Arguments:
        years (array-like): Calendar years.
        base_year (float): Year at which the index equals 1.
        driver (str): ``"population"`` or ``"labour_force"``.
        population_cagr (float): Population compound annual growth rate.
        labour_force_cagr (float): Labour-force compound annual growth rate.

    Returns:
        numpy.ndarray: Demand index, shape (n_years,).
    """
    if driver == "population":
        return yearly_index(years, base_year, population_cagr)
    if driver == "labour_force":
        return yearly_index(years, base_year, labour_force_cagr)
    raise ValueError("driver must be 'population' or 'labour_force'")


def inventory_baseline_cadr(market_usable, inventory_pct, cost_per_unit, cadr_per_unit):
    """CADR from inventory stock proportional to the annual usable market."""
    return (market_usable * inventory_pct / cost_per_unit) * cadr_per_unit


def _cadr_at_month(
    market_usable,
    cost_per_unit,
    cadr_per_unit,
    inventory_pct,
    month,
    growth_kwargs,
):
    """Run growth_model and return CADR at ``month`` (shape n_simulations,)."""
    if inventory_pct is None:
        baseline = np.zeros_like(market_usable, dtype=float)
    else:
        baseline = inventory_baseline_cadr(
            market_usable, inventory_pct, cost_per_unit, cadr_per_unit
        )
    monthly = growth_model(
        baseline_cadr=baseline,
        global_market_usable=market_usable,
        cost_per_unit=cost_per_unit,
        cadr_per_unit=cadr_per_unit,
        months=month,
        **growth_kwargs,
    )
    return monthly[month]


def forecast_pct_of_requirement_at_month(
    years,
    cadr_requirement_base,
    demand_index,
    uv_market_usable_base,
    uv_market_index,
    cost_per_unit_uv,
    cadr_per_unit_uv,
    inventory_pct_uv,
    uv_growth_kwargs,
    far_uvc_market_usable_base,
    far_uvc_market_index,
    cost_per_unit_far_uvc,
    cadr_per_unit_far_uvc,
    inventory_pct_far_uvc,
    far_uvc_growth_kwargs,
    fluoro_market_usable_base,
    fluoro_market_index,
    cost_per_unit_fluoro,
    cadr_per_unit_fluoro,
    fluoro_growth_kwargs,
    inventory_pct_fluoro=None,
    month=3,
):
    """
    For each year, scale markets and CADR demand, run a ``month``-month
    ramp, and return percent of that year's CADR requirement.

    All ``*_index`` arrays are relative to the same base year as
    ``cadr_requirement_base`` and the ``*_base`` markets (typically 1.0
    in the first forecast year). Fluorescent phase-out is expressed as a
    remaining-share index on the fluorescent market.

    Arguments:
        years (array-like): Calendar years, shape (n_years,).
        cadr_requirement_base (array-like): CADR requirement in the base year,
            shape (n_simulations,).
        demand_index (array-like): Population or labour-force index, shape (n_years,).
        uv_market_usable_base, far_uvc_market_usable_base, fluoro_market_usable_base:
            Usable annual markets in the base year, shape (n_simulations,).
        uv_market_index, far_uvc_market_index, fluoro_market_index:
            Market scale factors by year, shape (n_years,).
        cost_per_unit_* / cadr_per_unit_* / inventory_pct_*: Simulation draws
            used to rebuild inventory baseline CADR as markets scale.
        *_growth_kwargs (dict): Extra arguments to growth_model (utilization, etc.).
            Do not include months, baseline, market, cost, or cadr_per_unit.
        inventory_pct_fluoro (array-like or None): If None, fluorescent baseline
            CADR is zero. Default None.
        month (int): Snapshot month for coverage (0 is baseline). Default 3.

    Returns:
        dict: years (n_years,), pct_total / pct_uv / pct_far_uvc / pct_fluoro
            each shape (n_years, n_simulations), and cadr_requirement
            shape (n_years, n_simulations).
    """
    years = np.asarray(years)
    demand_index = np.asarray(demand_index, dtype=float)
    uv_market_index = np.asarray(uv_market_index, dtype=float)
    far_uvc_market_index = np.asarray(far_uvc_market_index, dtype=float)
    fluoro_market_index = np.asarray(fluoro_market_index, dtype=float)

    n_years = len(years)
    for name, arr in (
        ("demand_index", demand_index),
        ("uv_market_index", uv_market_index),
        ("far_uvc_market_index", far_uvc_market_index),
        ("fluoro_market_index", fluoro_market_index),
    ):
        if arr.shape != (n_years,):
            raise ValueError(f"{name} must have shape (n_years,), got {arr.shape}")

    n_sims = len(cadr_requirement_base)
    pct_uv = np.zeros((n_years, n_sims))
    pct_far = np.zeros((n_years, n_sims))
    pct_fluoro = np.zeros((n_years, n_sims))
    pct_total = np.zeros((n_years, n_sims))
    cadr_requirement = np.zeros((n_years, n_sims))

    for i in range(n_years):
        req = cadr_requirement_base * demand_index[i]
        cadr_requirement[i] = req

        uv_m = uv_market_usable_base * uv_market_index[i]
        far_m = far_uvc_market_usable_base * far_uvc_market_index[i]
        fl_m = fluoro_market_usable_base * fluoro_market_index[i]

        uv_c = _cadr_at_month(
            uv_m,
            cost_per_unit_uv,
            cadr_per_unit_uv,
            inventory_pct_uv,
            month,
            uv_growth_kwargs,
        )
        far_c = _cadr_at_month(
            far_m,
            cost_per_unit_far_uvc,
            cadr_per_unit_far_uvc,
            inventory_pct_far_uvc,
            month,
            far_uvc_growth_kwargs,
        )
        fl_c = _cadr_at_month(
            fl_m,
            cost_per_unit_fluoro,
            cadr_per_unit_fluoro,
            inventory_pct_fluoro,
            month,
            fluoro_growth_kwargs,
        )

        pct_uv[i] = 100.0 * uv_c / req
        pct_far[i] = 100.0 * far_c / req
        pct_fluoro[i] = 100.0 * fl_c / req
        pct_total[i] = 100.0 * (uv_c + far_c + fl_c) / req

    return {
        "years": years,
        "month": month,
        "pct_total": pct_total,
        "pct_uv": pct_uv,
        "pct_far_uvc": pct_far,
        "pct_fluoro": pct_fluoro,
        "cadr_requirement": cadr_requirement,
    }


def forecast_pct_for_fluoro_indices(
    fluoro_indices,
    **kwargs,
):
    """
    Run forecast_pct_of_requirement_at_month once per fluorescent market index.

    Arguments:
        fluoro_indices (sequence of (array-like, str)): Each item is
            (fluoro_market_index, legend_label).
        **kwargs: Passed through to forecast_pct_of_requirement_at_month
            except fluoro_market_index.

    Returns:
        list of tuple: (forecast_dict, label) for each index.
    """
    series = []
    for fluoro_market_index, label in fluoro_indices:
        forecast = forecast_pct_of_requirement_at_month(
            fluoro_market_index=fluoro_market_index,
            **kwargs,
        )
        series.append((forecast, label))
    return series
