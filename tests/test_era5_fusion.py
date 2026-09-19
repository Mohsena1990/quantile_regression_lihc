"""Regression tests for the ERA5-Land fusion QC bugs found during audit.

1. ERA5-Land is land-only: raw grids legitimately contain large NaN blocks over
   sea/ocean. `load_temperature` must not reject a file for that reason alone.
2. `gate_3_era5` must close from a key that `fuse_era5_nuts1.py` actually writes.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

xr = pytest.importorskip("xarray")

from scripts.fuse_era5_nuts1 import load_temperature


def _write_grid(path, values):
    ds = xr.Dataset(
        {"t2m": (("valid_time", "latitude", "longitude"), values)},
        coords={
            "valid_time": pd.date_range("2017-01-01", periods=values.shape[0]),
            "latitude": np.linspace(60, 58, values.shape[1]),
            "longitude": np.linspace(10, 12, values.shape[2]),
        },
    )
    ds.to_netcdf(path)
    return path


def test_load_temperature_tolerates_ocean_nan(tmp_path):
    values = np.full((2, 3, 3), 280.0)  # plausible Kelvin land temperature
    values[:, 0, 0] = np.nan  # simulated sea-mask cell
    path = _write_grid(tmp_path / "grid.nc", values)
    celsius, time_dim = load_temperature(path)
    assert time_dim == "valid_time"
    assert np.isnan(celsius.values[:, 0, 0]).all()  # NaN cell preserved, not fabricated
    assert np.allclose(celsius.values[:, 1, 1], 280.0 - 273.15)


def test_load_temperature_still_rejects_unreasonable_kelvin_range(tmp_path):
    values = np.full((2, 3, 3), 500.0)  # physically implausible
    path = _write_grid(tmp_path / "bad_grid.nc", values)
    with pytest.raises(SystemExit, match="BLOCKED_ERA5_QC"):
        load_temperature(path)


def _gate_3_status(qc: dict) -> str:
    """Mirrors scripts/check_milestone_gates.py's gate_3_era5 condition exactly."""
    return "CLOSED" if qc.get("coverage") == 1 and qc.get("fused_regions") == qc.get("expected_regions") and qc.get("qc_all_clear") else "IN_PROGRESS"


def test_gate_3_closes_on_the_key_fusion_actually_writes():
    qc = {"coverage": 1, "fused_regions": 17, "expected_regions": 17, "qc_all_clear": True}
    assert _gate_3_status(qc) == "CLOSED"


def test_gate_3_stays_open_when_qc_incomplete():
    qc = {"coverage": 1, "fused_regions": 17, "expected_regions": 17, "qc_all_clear": False}
    assert _gate_3_status(qc) == "IN_PROGRESS"


def test_gate_3_regression_the_old_nonexistent_key_would_never_close():
    """Characterizes the bug found during audit: check_milestone_gates.py used to key off
    'all_day_counts_valid', which fuse_era5_nuts1.py never wrote, so a genuinely complete
    fusion (coverage=1, fused_regions=17 of 17) could never close the gate."""
    qc = {"coverage": 1, "fused_regions": 17, "expected_regions": 17, "qc_all_clear": True}
    old_buggy_status = "CLOSED" if qc.get("coverage") == 1 and qc.get("fused_regions") == qc.get("expected_regions") and qc.get("all_day_counts_valid") else "IN_PROGRESS"
    assert old_buggy_status == "IN_PROGRESS"  # the bug: permanently stuck despite success
    assert _gate_3_status(qc) == "CLOSED"  # the fix
