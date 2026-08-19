"""
Tests for src.forecast.
"""

import numpy as np
import pytest

from src.forecast import (
    demand_index_for_driver,
    exponential_remaining_share,
    forecast_pct_for_fluoro_indices,
    forecast_pct_of_requirement_at_month,
    inventory_baseline_cadr,
    linear_remaining_share,
    biphasic_remaining_share,
    yearly_index,
)
from src.scale_up_model import growth_model


def test_yearly_index_is_one_at_base_year():
    years = np.array([2025, 2026, 2027])
    idx = yearly_index(years, base_year=2025, cagr=0.1)
    np.testing.assert_allclose(idx[0], 1.0)
    np.testing.assert_allclose(idx[1], 1.1)
    np.testing.assert_allclose(idx[2], 1.21)


def test_linear_remaining_share_clips():
    years = np.array([2024, 2025, 2030, 2035, 2040])
    share = linear_remaining_share(years, start_year=2025, end_year=2035)
    np.testing.assert_allclose(share, [1.0, 1.0, 0.5, 0.0, 0.0])


def test_exponential_remaining_share():
    years = np.array([2025, 2026, 2027])
    share = exponential_remaining_share(
        years, base_year=2025, annual_replacement_rate=0.2
    )
    np.testing.assert_allclose(share, [1.0, 0.8, 0.64])


def test_biphasic_remaining_share_hits_residual_then_tails():
    years = np.array([2024, 2025, 2026, 2027, 2035])
    share = biphasic_remaining_share(
        years,
        start_year=2025,
        fast_end_year=2027,
        remaining_at_fast_end=0.10,
        tail_half_life_years=8.0,
    )
    np.testing.assert_allclose(share[0], 1.0)
    np.testing.assert_allclose(share[1], 1.0)
    np.testing.assert_allclose(share[3], 0.10)
    assert 0.0 < share[4] < share[3]
    assert share[2] > 0.10
    assert share[2] < 1.0
    # Smooth: 2026 is not the linear midpoint (0.55)
    assert share[2] != pytest.approx(0.55)


def test_demand_driver_switches_cagr():
    years = np.array([2025, 2026])
    pop = demand_index_for_driver(years, 2025, "population", 0.1, 0.0)
    lf = demand_index_for_driver(years, 2025, "labour_force", 0.0, 0.05)
    np.testing.assert_allclose(pop[1], 1.1)
    np.testing.assert_allclose(lf[1], 1.05)
    with pytest.raises(ValueError, match="driver"):
        demand_index_for_driver(years, 2025, "gdp", 0.1, 0.1)


def _constant_source(n_sims=8, seed=0):
    rng = np.random.default_rng(seed)
    market = rng.uniform(1e9, 2e9, n_sims)
    cost = rng.uniform(100, 200, n_sims)
    cadr = rng.uniform(100, 200, n_sims)
    inv = np.full(n_sims, 1 / 12)
    req = rng.uniform(1e10, 2e10, n_sims)
    return market, cost, cadr, inv, req


def test_forecast_matches_single_growth_model_when_indices_are_one():
    """With all indices 1, year-0 coverage equals a standalone 3-month run."""
    n_sims = 6
    market, cost, cadr, inv, req = _constant_source(n_sims)
    growth_kw = {
        "utilization_start": 1.0,
        "utilization_end": 1.0,
        "utilization_ramp_months": 3,
        "monthly_production_pct_of_annual": 1 / 12,
        "repurposed_ramp_months": 0,
        "repurposed_pct_of_annual": 0.0,
    }
    years = np.array([2025])
    ones = np.array([1.0])
    out = forecast_pct_of_requirement_at_month(
        years=years,
        cadr_requirement_base=req,
        demand_index=ones,
        uv_market_usable_base=market,
        uv_market_index=ones,
        cost_per_unit_uv=cost,
        cadr_per_unit_uv=cadr,
        inventory_pct_uv=inv,
        uv_growth_kwargs=growth_kw,
        far_uvc_market_usable_base=market * 0.2,
        far_uvc_market_index=ones,
        cost_per_unit_far_uvc=cost,
        cadr_per_unit_far_uvc=cadr,
        inventory_pct_far_uvc=inv,
        far_uvc_growth_kwargs=growth_kw,
        fluoro_market_usable_base=market,
        fluoro_market_index=ones,
        cost_per_unit_fluoro=cost,
        cadr_per_unit_fluoro=cadr,
        fluoro_growth_kwargs={
            **growth_kw,
            "utilization_start": 0.0,
            "utilization_end": 0.7,
        },
        month=3,
    )
    uv_traj = growth_model(
        baseline_cadr=inventory_baseline_cadr(market, inv, cost, cadr),
        global_market_usable=market,
        cost_per_unit=cost,
        cadr_per_unit=cadr,
        months=3,
        **growth_kw,
    )
    np.testing.assert_allclose(out["pct_uv"][0], 100.0 * uv_traj[3] / req)


def test_doubling_demand_halves_percent_if_markets_fixed():
    market, cost, cadr, inv, req = _constant_source(5)
    growth_kw = {
        "utilization_start": 1.0,
        "utilization_end": 1.0,
        "utilization_ramp_months": 1,
        "monthly_production_pct_of_annual": 1 / 12,
        "repurposed_ramp_months": 0,
        "repurposed_pct_of_annual": 0.0,
    }
    years = np.array([2025, 2026])
    ones = np.array([1.0, 1.0])
    demand = np.array([1.0, 2.0])
    out = forecast_pct_of_requirement_at_month(
        years=years,
        cadr_requirement_base=req,
        demand_index=demand,
        uv_market_usable_base=market,
        uv_market_index=ones,
        cost_per_unit_uv=cost,
        cadr_per_unit_uv=cadr,
        inventory_pct_uv=inv,
        uv_growth_kwargs=growth_kw,
        far_uvc_market_usable_base=np.zeros_like(market),
        far_uvc_market_index=ones,
        cost_per_unit_far_uvc=cost,
        cadr_per_unit_far_uvc=cadr,
        inventory_pct_far_uvc=inv,
        far_uvc_growth_kwargs=growth_kw,
        fluoro_market_usable_base=np.zeros_like(market),
        fluoro_market_index=ones,
        cost_per_unit_fluoro=cost,
        cadr_per_unit_fluoro=cadr,
        fluoro_growth_kwargs=growth_kw,
        month=3,
    )
    np.testing.assert_allclose(out["pct_total"][1], 0.5 * out["pct_total"][0])


def test_zero_fluoro_share_drops_fluoro_component():
    market, cost, cadr, inv, req = _constant_source(4)
    growth_kw = {
        "utilization_start": 0.5,
        "utilization_end": 0.5,
        "utilization_ramp_months": 1,
        "monthly_production_pct_of_annual": 1 / 12,
        "repurposed_ramp_months": 0,
        "repurposed_pct_of_annual": 0.0,
    }
    years = np.array([2025, 2035])
    ones = np.array([1.0, 1.0])
    fluoro_index = np.array([1.0, 0.0])
    out = forecast_pct_of_requirement_at_month(
        years=years,
        cadr_requirement_base=req,
        demand_index=ones,
        uv_market_usable_base=np.zeros_like(market),
        uv_market_index=ones,
        cost_per_unit_uv=cost,
        cadr_per_unit_uv=cadr,
        inventory_pct_uv=inv,
        uv_growth_kwargs=growth_kw,
        far_uvc_market_usable_base=np.zeros_like(market),
        far_uvc_market_index=ones,
        cost_per_unit_far_uvc=cost,
        cadr_per_unit_far_uvc=cadr,
        inventory_pct_far_uvc=inv,
        far_uvc_growth_kwargs=growth_kw,
        fluoro_market_usable_base=market,
        fluoro_market_index=fluoro_index,
        cost_per_unit_fluoro=cost,
        cadr_per_unit_fluoro=cadr,
        fluoro_growth_kwargs=growth_kw,
        month=3,
    )
    assert np.all(out["pct_fluoro"][0] > 0)
    np.testing.assert_allclose(out["pct_fluoro"][1], 0.0)
    np.testing.assert_allclose(out["pct_total"][1], 0.0)


def test_forecast_pct_for_fluoro_indices_runs_each_scenario():
    market, cost, cadr, inv, req = _constant_source(4)
    growth_kw = {
        "utilization_start": 1.0,
        "utilization_end": 1.0,
        "utilization_ramp_months": 1,
        "monthly_production_pct_of_annual": 1 / 12,
        "repurposed_ramp_months": 0,
        "repurposed_pct_of_annual": 0.0,
    }
    years = np.array([2025, 2027])
    ones = np.array([1.0, 1.0])
    series = forecast_pct_for_fluoro_indices(
        [
            (np.array([1.0, 0.0]), "policy"),
            (np.array([1.0, 0.5]), "slow"),
        ],
        years=years,
        cadr_requirement_base=req,
        demand_index=ones,
        uv_market_usable_base=np.zeros_like(market),
        uv_market_index=ones,
        cost_per_unit_uv=cost,
        cadr_per_unit_uv=cadr,
        inventory_pct_uv=inv,
        uv_growth_kwargs=growth_kw,
        far_uvc_market_usable_base=np.zeros_like(market),
        far_uvc_market_index=ones,
        cost_per_unit_far_uvc=cost,
        cadr_per_unit_far_uvc=cadr,
        inventory_pct_far_uvc=inv,
        far_uvc_growth_kwargs=growth_kw,
        fluoro_market_usable_base=market,
        cost_per_unit_fluoro=cost,
        cadr_per_unit_fluoro=cadr,
        fluoro_growth_kwargs=growth_kw,
        month=3,
    )
    assert [label for _, label in series] == ["policy", "slow"]
    np.testing.assert_allclose(series[0][0]["pct_fluoro"][1], 0.0)
    assert np.all(series[1][0]["pct_fluoro"][1] > 0)
    assert np.all(series[1][0]["pct_total"][1] > series[0][0]["pct_total"][1])
