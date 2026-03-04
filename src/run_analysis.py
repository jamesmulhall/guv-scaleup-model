"""
Command-line entry point for running the GUV scale-up analysis.

The analysis mirrors the example in the `scripts/estimate_scaleup.ipynb`
notebook, but is fully configurable via a YAML file and saves outputs to
disk (CSV + PNG).

Default configuration file path (relative to the repository root):
    `analysis_config.yml`
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

from .mc_distributions import sample_lognormal, sample_normal
from .plotter import plot_uv_fluoro_ramp
from .scale_up_model import growth_model


def _get_project_root() -> Path:
    """Return the repository root assumed to be the parent of `src`."""
    return Path(__file__).resolve().parents[1]


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load YAML configuration for the analysis.

    Arguments:
        config_path (pathlib.Path): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration.
    """
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_analysis_from_config(config: Dict[str, Any]) -> None:
    """
    Run the GUV scale-up analysis using a configuration dictionary.

    This function:
        1. Draws Monte Carlo samples for UV and fluorescent markets.
        2. Computes baseline CADR for UV and repurposed fluorescents.
        3. Samples CADR requirement for vital workers.
        4. Runs the growth model for UV and repurposed fluorescents.
        5. Writes a CSV summary and a PNG plot to the results folder.

    Arguments:
        config (dict): Configuration with keys:
            - n_sims (int)
            - months (int)
            - confidence_interval (int or float)
            - uv (dict): UV market and growth parameters.
            - fluoro (dict): Fluorescent market and growth parameters.
            - requirement (dict): CADR requirement parameters.
            - outputs (dict): Output paths for CSV and PNG.
    """
    n_sims = int(config.get("n_sims", 1000))
    months = int(config.get("months", 6))
    confidence_interval = float(config.get("confidence_interval", 90))

    uv_cfg = config["uv"]
    fluoro_cfg = config["fluoro"]
    req_cfg = config["requirement"]
    outputs_cfg = config["outputs"]

    # 1. Parameter sampling
    # UV market and supply chain
    global_uv_market = sample_normal(
        uv_cfg["global_market_low"],
        uv_cfg["global_market_high"],
        n_sims,
    )
    percent_usable_uv = sample_normal(
        uv_cfg["percent_usable_low"],
        uv_cfg["percent_usable_high"],
        n_sims,
    )
    percent_of_annual_uv = sample_normal(
        uv_cfg["percent_of_annual_low"],
        uv_cfg["percent_of_annual_high"],
        n_sims,
    )
    cost_per_unit_uv = sample_lognormal(
        uv_cfg["cost_per_unit_low"],
        uv_cfg["cost_per_unit_high"],
        n_sims,
    )
    cadr_per_unit = sample_lognormal(
        uv_cfg["cadr_per_unit_low"],
        uv_cfg["cadr_per_unit_high"],
        n_sims,
    )

    # Repurposed fluorescent market
    global_fluoro_market = sample_normal(
        fluoro_cfg["global_market_low"],
        fluoro_cfg["global_market_high"],
        n_sims,
    )
    percent_usable_fluoro = sample_normal(
        fluoro_cfg["percent_usable_low"],
        fluoro_cfg["percent_usable_high"],
        n_sims,
    )
    cost_per_unit_fluoro = sample_lognormal(
        fluoro_cfg["cost_per_unit_low"],
        fluoro_cfg["cost_per_unit_high"],
        n_sims,
    )

    # 2. Baseline supply
    global_uv_market_usable = global_uv_market * percent_usable_uv
    annual_uv_market_usable = global_uv_market_usable * percent_of_annual_uv
    uv_lamps = annual_uv_market_usable / cost_per_unit_uv
    uv_baseline_cadr = uv_lamps * cadr_per_unit

    # No current annual repurposed fluoro production → baseline 0
    fluoro_baseline_cadr = np.zeros(n_sims)
    global_fluoro_market_usable = global_fluoro_market * percent_usable_fluoro

    # 3. CADR requirement
    ashrae_req = sample_normal(
        req_cfg["ashrae_low"],
        req_cfg["ashrae_high"],
        n_sims,
    )
    vital_workers = sample_normal(
        req_cfg["vital_workers_low"],
        req_cfg["vital_workers_high"],
        n_sims,
    )
    total_cadr_req = ashrae_req * vital_workers

    # 4. Growth models
    uv_growth = uv_cfg["growth"]
    fluoro_growth = fluoro_cfg["growth"]

    uv_monthly_cadr = growth_model(
        baseline_cadr=uv_baseline_cadr,
        global_market_usable=annual_uv_market_usable,
        cost_per_unit=cost_per_unit_uv,
        cadr_per_unit=cadr_per_unit,
        months=months,
        utilization_start=uv_growth["utilization_start"],
        utilization_end=uv_growth["utilization_end"],
        utilization_ramp_months=uv_growth["utilization_ramp_months"],
        additional_annual_production=uv_growth["additional_annual_production"],
        repurposed_ramp_months=uv_growth["repurposed_ramp_months"],
        repurposed_annual_production=uv_growth["repurposed_annual_production"],
    )

    fluoro_monthly_cadr = growth_model(
        baseline_cadr=fluoro_baseline_cadr,
        global_market_usable=global_fluoro_market_usable,
        cost_per_unit=cost_per_unit_fluoro,
        cadr_per_unit=cadr_per_unit,
        months=months,
        utilization_start=fluoro_growth["utilization_start"],
        utilization_end=fluoro_growth["utilization_end"],
        utilization_ramp_months=fluoro_growth["utilization_ramp_months"],
        additional_annual_production=fluoro_growth["additional_annual_production"],
        repurposed_ramp_months=fluoro_growth["repurposed_ramp_months"],
        repurposed_annual_production=fluoro_growth["repurposed_annual_production"],
    )

    # 5. Save results: CSV summary and PNG plot
    project_root = _get_project_root()
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / outputs_cfg.get("csv_path", "analysis_summary.csv")
    png_path = results_dir / outputs_cfg.get("png_path", "uv_fluoro_ramp.png")

    p_lower = (100.0 - confidence_interval) / 2.0
    p_upper = 100.0 - p_lower

    months_array = np.arange(months)

    uv_median = np.median(uv_monthly_cadr, axis=1)
    uv_p_lo = np.percentile(uv_monthly_cadr, p_lower, axis=1)
    uv_p_hi = np.percentile(uv_monthly_cadr, p_upper, axis=1)

    fluoro_median = np.median(fluoro_monthly_cadr, axis=1)
    fluoro_p_lo = np.percentile(fluoro_monthly_cadr, p_lower, axis=1)
    fluoro_p_hi = np.percentile(fluoro_monthly_cadr, p_upper, axis=1)

    median_cadr_req = np.median(total_cadr_req)
    cadr_req_lo = np.percentile(total_cadr_req, p_lower)
    cadr_req_hi = np.percentile(total_cadr_req, p_upper)

    df = pd.DataFrame(
        {
            "month": months_array,
            "uv_median": uv_median,
            "uv_p_lower": uv_p_lo,
            "uv_p_upper": uv_p_hi,
            "fluoro_median": fluoro_median,
            "fluoro_p_lower": fluoro_p_lo,
            "fluoro_p_upper": fluoro_p_hi,
            "median_cadr_requirement": median_cadr_req,
            "cadr_requirement_p_lower": cadr_req_lo,
            "cadr_requirement_p_upper": cadr_req_hi,
        }
    )
    df.to_csv(csv_path, index=False)

    # Use the existing plotting utility; it handles style and DPI.
    plot_uv_fluoro_ramp(
        uv_monthly_cadr=uv_monthly_cadr,
        fluoro_monthly_cadr=fluoro_monthly_cadr,
        cadr_requirement=total_cadr_req,
        confidence_interval=confidence_interval,
        save_path=str(png_path),
        show=False,
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments with `config` attribute.
    """
    parser = argparse.ArgumentParser(
        description="Run the GUV scale-up analysis from a YAML configuration file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="analysis_config.yml",
        help=(
            "Path to the YAML configuration file, relative to the repository root. "
            "Default: analysis_config.yml"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for running the analysis from the command line."""
    args = parse_args()
    project_root = _get_project_root()
    config_path = (project_root / args.config).resolve()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            "Create it or pass a different path with --config."
        )

    config = load_config(config_path)
    run_analysis_from_config(config)


if __name__ == "__main__":
    main()
