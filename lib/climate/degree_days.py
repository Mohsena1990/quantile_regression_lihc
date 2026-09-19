"""Configurable daily degree-day summaries; temperatures are Celsius."""
from __future__ import annotations
import numpy as np

def climate_summary(daily_mean, daily_min, *, hdd_base: float=18.0, cdd_base: float=22.0) -> dict[str,float|int]:
    mean=np.asarray(daily_mean,dtype=float); minimum=np.asarray(daily_min,dtype=float)
    if mean.shape!=minimum.shape or mean.ndim!=1 or not np.isfinite(mean).all() or not np.isfinite(minimum).all():
        raise ValueError("daily mean/min arrays must be equally sized, finite, and one-dimensional")
    # Defensive guards (added 2026-08-11 audit): this module assumes Celsius input and never
    # itself converts from Kelvin. A caller that forgot the Kelvin->Celsius step would silently
    # produce enormous bogus HDD/CDD sums with no error otherwise -- catch that here instead of
    # relying solely on the one current caller (scripts/fuse_era5_nuts1.py) to have done it right.
    if (mean>100).any() or (minimum>100).any():
        raise ValueError("temperatures look like Kelvin, not Celsius -- convert before calling climate_summary")
    if (mean<minimum-1e-6).any():
        raise ValueError("daily mean temperature must be >= daily minimum temperature")
    return {"hdd":float(np.maximum(0,hdd_base-mean).sum()),"cdd":float(np.maximum(0,mean-cdd_base).sum()),"temperature_mean":float(mean.mean()),"temperature_sd":float(mean.std(ddof=1)) if len(mean)>1 else 0.0,"frost_days":int((minimum<0).sum()),"days":len(mean)}
