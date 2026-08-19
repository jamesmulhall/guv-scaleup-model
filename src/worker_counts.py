"""
Load global indoor CADR requirements from the country-level worker table.
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKER_CSV = REPO_ROOT / "data" / "EssentialWorkersByCountry.csv"

INDOOR_VITAL_CADR_COL = "Indoor Vital CADR Requirement (L/s)"
INDOOR_ESSENTIAL_CADR_COL = "Indoor Essential CADR Requirement (L/s)"


def load_global_cadr_requirements(csv_path=None):
    """
    Sum country-level indoor CADR requirements to global totals.

    Indoor vital CADR is the lower bound and indoor essential CADR is the
    upper bound. These columns already include occupancy-weighted ASHRAE 241
    equivalent clean airflow scaled for a severe pandemic.

    Arguments:
        csv_path (str or Path or None): Path to EssentialWorkersByCountry.csv.
            If None, use data/EssentialWorkersByCountry.csv in this repo.

    Returns:
        dict: Keys indoor_vital and indoor_essential (floats, L/s).
    """
    path = Path(csv_path) if csv_path is not None else DEFAULT_WORKER_CSV
    df = pd.read_csv(path)
    required = (INDOOR_VITAL_CADR_COL, INDOOR_ESSENTIAL_CADR_COL)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Worker table missing columns: {missing}")

    return {
        "indoor_vital": float(df[INDOOR_VITAL_CADR_COL].sum(skipna=True)),
        "indoor_essential": float(df[INDOOR_ESSENTIAL_CADR_COL].sum(skipna=True)),
    }
