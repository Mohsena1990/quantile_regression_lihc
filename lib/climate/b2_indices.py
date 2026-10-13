"""Climate indices of feature stage B2, one value per household climate region (NUTS3; Poland NUTS2), from Open-Meteo
daily reanalysis series (scripts/acquire_openmeteo_b2.py; temperatures in Celsius).

Survey reference: ENABLE interviews ran Nov 2017 - Feb 2018, so the "last heating season" is winter 2016/17
(HEATING_SEASON) and the "last 12 months" of electricity bills is roughly 2017 (WARM_SEASON for cooling and heat).

  hdd_hs1617       heating degree days, base 18 C, heating season Oct 2016 - Apr 2017
  cdd_s2017        cooling degree days, base 22 C, May - Sep 2017
  hdd_normal       mean heating-season HDD (base 18 C), 1991/92 - 2019/20   -- typical heating need
  cdd_normal       mean warm-season CDD (base 22 C), 1991 - 2020            -- typical cooling need
                   (both from ERA5-Land MONTHLY means: days x max(0, base - monthly mean); scripts/acquire_era5_normals.py)
  tmin_hs          mean daily minimum temperature, heating season
  apparent_tmin_hs mean daily minimum apparent ("feels-like": wind chill, humidity) temperature, heating season
  frost_days_hs    days with daily minimum < 0 C, heating season
  cold_spell_max   longest run of consecutive days with daily MEAN < 0 C, heating season
  heatwave_days    days inside runs of >= 3 consecutive days with daily MAXIMUM >= 30 C, warm season
  tropical_nights  days with daily minimum > 20 C, warm season
  rh_hs            mean relative humidity (%), heating season   -- damp and mould
  precip_hs        precipitation (mm), heating season
  snowfall_hs      snowfall (cm), heating season
  wind_hs          mean 10 m wind speed (km/h), heating season  -- infiltration heat loss
  gust_max_hs      mean of daily maximum gusts (km/h), heating season
  solar_hs         mean daily shortwave radiation (MJ/m2), heating season -- passive solar gains
  sunshine_hs      mean daily sunshine (hours), heating season
  cloud_hs         mean cloud cover (%), heating season
Base temperatures match lib/climate/degree_days.py (HDD 18 C, CDD 22 C).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

HEATING_SEASON = ("2016-10-01", "2017-04-30")
WARM_SEASON = ("2017-05-01", "2017-09-30")
HDD_BASE, CDD_BASE = 18.0, 22.0
HEATWAVE_TMAX, HEATWAVE_MIN_RUN = 30.0, 3
TROPICAL_TMIN = 20.0
B2_CLIMATE = ["hdd_hs1617", "cdd_s2017", "hdd_normal", "cdd_normal", "tmin_hs", "apparent_tmin_hs", "frost_days_hs",
              "cold_spell_max", "heatwave_days", "tropical_nights", "rh_hs", "precip_hs", "snowfall_hs", "wind_hs",
              "gust_max_hs", "solar_hs", "sunshine_hs", "cloud_hs"]
B2_NAMES = {"hdd_hs1617": "HDD winter 2016/17", "cdd_s2017": "CDD summer 2017", "hdd_normal": "HDD normal 1991-2020",
            "cdd_normal": "CDD normal 1991-2020", "tmin_hs": "min temperature (winter)", "apparent_tmin_hs": "feels-like min (winter)",
            "frost_days_hs": "frost days (winter)", "cold_spell_max": "longest cold spell", "heatwave_days": "heatwave days 2017",
            "tropical_nights": "tropical nights 2017", "rh_hs": "relative humidity (winter)", "precip_hs": "precipitation (winter)",
            "snowfall_hs": "snowfall (winter)", "wind_hs": "wind speed (winter)", "gust_max_hs": "wind gusts (winter)",
            "solar_hs": "solar radiation (winter)", "sunshine_hs": "sunshine hours (winter)", "cloud_hs": "cloud cover (winter)"}


def longest_run(flags) -> int:
    """Length of the longest run of True values."""
    best = cur = 0
    for f in np.asarray(flags, bool):
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return int(best)


def days_in_runs(flags, min_run: int) -> int:
    """Number of True days that belong to a run of at least `min_run` consecutive True days."""
    total = cur = 0
    for f in list(np.asarray(flags, bool)) + [False]:
        if f:
            cur += 1
        else:
            total += cur if cur >= min_run else 0
            cur = 0
    return int(total)


def _window(frame: pd.DataFrame, window: tuple[str, str]) -> pd.DataFrame:
    out = frame.loc[window[0]:window[1]]
    if len(out) != len(pd.date_range(*window, freq="D")) or out.isna().any().any():
        raise ValueError(f"incomplete daily series for window {window}: {len(out)} days")
    return out


def survey_indices(daily: pd.DataFrame) -> dict[str, float]:
    """`daily`: one region's Open-Meteo daily frame indexed by date (columns as in acquire_openmeteo_b2.SURVEY_DAILY)."""
    hs, ws = _window(daily, HEATING_SEASON), _window(daily, WARM_SEASON)
    if (hs.temperature_2m_mean < hs.temperature_2m_min - 1e-6).any() or (ws.temperature_2m_max < ws.temperature_2m_mean - 1e-6).any():
        raise ValueError("daily statistics out of order (mean < minimum or maximum < mean)")
    return {"hdd_hs1617": float(np.maximum(0, HDD_BASE - hs.temperature_2m_mean).sum()),
            "cdd_s2017": float(np.maximum(0, ws.temperature_2m_mean - CDD_BASE).sum()),
            "tmin_hs": float(hs.temperature_2m_min.mean()),
            "apparent_tmin_hs": float(hs.apparent_temperature_min.mean()),
            "frost_days_hs": int((hs.temperature_2m_min < 0).sum()),
            "cold_spell_max": longest_run(hs.temperature_2m_mean < 0),
            "heatwave_days": days_in_runs(ws.temperature_2m_max >= HEATWAVE_TMAX, HEATWAVE_MIN_RUN),
            "tropical_nights": int((ws.temperature_2m_min > TROPICAL_TMIN).sum()),
            "rh_hs": float(hs.relative_humidity_2m_mean.mean()),
            "precip_hs": float(hs.precipitation_sum.sum()),
            "snowfall_hs": float(hs.snowfall_sum.sum()),
            "wind_hs": float(hs.wind_speed_10m_mean.mean()),
            "gust_max_hs": float(hs.wind_gusts_10m_max.mean()),
            "solar_hs": float(hs.shortwave_radiation_sum.mean()),
            "sunshine_hs": float(hs.sunshine_duration.mean() / 3600),
            "cloud_hs": float(hs.cloud_cover_mean.mean())}


def normal_indices(monthly: pd.Series) -> dict[str, float]:
    """Mean heating-season HDD (Oct-Apr, 1991/92 - 2019/20) and warm-season CDD (May-Sep, 1991 - 2020) from a monthly mean
    temperature series indexed by month start: days_in_month x max(0, base - monthly mean). Understates degree days in
    shoulder months (days that cross the base) but ranks regions consistently."""
    m = monthly.sort_index()
    if m.isna().any() or len(m.loc["1991-01-01":"2020-12-01"]) != 360:
        raise ValueError("incomplete 1991-2020 monthly series")
    days = m.index.days_in_month
    hdd_m = pd.Series(days * np.maximum(0, HDD_BASE - m.to_numpy()), index=m.index)
    cdd_m = pd.Series(days * np.maximum(0, m.to_numpy() - CDD_BASE), index=m.index)
    hdd = [hdd_m.loc[f"{y}-10-01":f"{y + 1}-04-01"].sum() for y in range(1991, 2020)]
    cdd = [cdd_m.loc[f"{y}-05-01":f"{y}-09-01"].sum() for y in range(1991, 2021)]
    return {"hdd_normal": float(np.mean(hdd)), "cdd_normal": float(np.mean(cdd))}
