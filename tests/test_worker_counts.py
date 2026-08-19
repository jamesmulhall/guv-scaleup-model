"""
Tests for src.worker_counts.
"""

import pytest

from src.worker_counts import DEFAULT_WORKER_CSV, load_global_cadr_requirements


def test_load_from_fixture_csv(tmp_path):
    """Sums indoor vital and essential CADR columns across rows."""
    csv_path = tmp_path / "workers.csv"
    csv_path.write_text(
        "Indoor Vital CADR Requirement (L/s),"
        "Indoor Essential CADR Requirement (L/s)\n"
        "10,40\n"
        "20,60\n"
    )
    bounds = load_global_cadr_requirements(csv_path)
    assert bounds["indoor_vital"] == 30.0
    assert bounds["indoor_essential"] == 100.0


def test_missing_column_raises(tmp_path):
    """Missing required columns must raise ValueError."""
    csv_path = tmp_path / "workers.csv"
    csv_path.write_text("Indoor Vital CADR Requirement (L/s)\n10\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_global_cadr_requirements(csv_path)


def test_default_csv_has_vital_cadr_below_essential():
    """Repo table: global indoor vital CADR is below indoor essential CADR."""
    assert DEFAULT_WORKER_CSV.exists()
    bounds = load_global_cadr_requirements()
    assert bounds["indoor_vital"] > 0
    assert bounds["indoor_essential"] > bounds["indoor_vital"]
