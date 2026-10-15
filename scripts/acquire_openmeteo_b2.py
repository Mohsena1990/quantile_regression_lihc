"""Acquire the stage-B2 climate inputs from the Open-Meteo historical weather API (archive-api.open-meteo.com).

Open-Meteo serves the ECMWF reanalyses (ERA5, ERA5-Land, and CERRA over Europe; default "best match" model) as daily
aggregates for any coordinate, without the CDS queue. One point per household climate region of the version-5 cohort (four countries + Spain; NUTS3, Poland NUTS2): the
region polygon's representative point (always inside the polygon; Open-Meteo picks a land grid cell of similar elevation
and downscales to the 90 m terrain height). Requests (UTC days):
  survey  2016-10-01 .. 2017-12-31, 14 daily variables (temperature mean/max/min, apparent temperature mean/min, relative
          humidity, dew point, precipitation, snowfall, wind speed, gusts, shortwave radiation, sunshine, cloud cover)
(The 1991-2020 normals come from ERA5-Land monthly means, scripts/acquire_era5_normals.py: 30 years of daily data per
region would cost ~780 Open-Meteo calls each under its fair-use weighting.)
Free-tier limits (600 / minute, 5,000 / hour, 10,000 / day; long or many-variable requests count as several calls):
requests are paced, a limit message triggers a wait, and every response is cached (artifacts/climate/openmeteo/http_cache),
so a rerun resumes where it stopped. Writes artifacts/climate/openmeteo/survey_daily.parquet and points.csv.

Citation (README "Data"): Zippenfenig (2023) Open-Meteo.com Weather API, doi:10.5281/zenodo.7970649, and the reanalyses it
serves (Hersbach et al. 2023 ERA5; Munoz Sabater 2019 ERA5-Land; Schimanke et al. 2021 CERRA).
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.enable import EXTENDED_COUNTRY_CODES, HOUSEHOLD_SHEET, retained_cohort
from lib.geography.climate_regions import climate_region, region_geometries

OUT = ROOT / "artifacts/climate/openmeteo"
URL = "https://archive-api.open-meteo.com/v1/archive"
SURVEY_DAILY = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min", "apparent_temperature_mean",
                "apparent_temperature_min", "relative_humidity_2m_mean", "dew_point_2m_mean", "precipitation_sum",
                "snowfall_sum", "wind_speed_10m_mean", "wind_gusts_10m_max", "shortwave_radiation_sum", "sunshine_duration",
                "cloud_cover_mean"]
REQUESTS = {"survey": ("2016-10-01", "2017-12-31", SURVEY_DAILY)}


def region_points() -> pd.DataFrame:
    hh = retained_cohort(pd.read_excel(ROOT / "data/raw/ENABLE.EU_dataset_survey of households.xlsx", sheet_name=HOUSEHOLD_SHEET,
                                       usecols=["UID", "Country", "NUTS1", "NUTS2", "NUTS3"]), EXTENDED_COUNTRY_CODES)
    hh["climate_region"] = climate_region(hh)
    geo = region_geometries(hh.climate_region.unique())
    pts = geo.geometry.representative_point()
    return pd.DataFrame({"climate_region": geo.NUTS_ID, "latitude": pts.y.round(4), "longitude": pts.x.round(4)}).sort_values("climate_region")


def fetch(client, lat: float, lon: float, start: str, end: str, daily: list[str]) -> pd.DataFrame:
    for attempt in range(40):
        try:
            r = client.weather_api(URL, params={"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                                                "daily": daily, "timezone": "GMT"})[0]
            break
        except Exception as e:                                                  # rate limits: wait for the window to pass
            msg = str(e)
            if "Daily API request limit" in msg:
                raise SystemExit("OPEN_METEO_DAILY_LIMIT: rerun tomorrow; cached responses are kept")
            wait = 3600 if "Hourly" in msg else 65
            print(f"limit/err ({msg[:80]}); waiting {wait}s", flush=True)
            time.sleep(wait)
    else:
        raise SystemExit("OPEN_METEO: giving up after repeated errors")
    d = r.Daily()
    out = pd.DataFrame({"date": pd.date_range(pd.to_datetime(d.Time(), unit="s"), pd.to_datetime(d.TimeEnd(), unit="s"),
                                              freq=pd.Timedelta(seconds=d.Interval()), inclusive="left")})
    for i, v in enumerate(daily):
        out[v] = d.Variables(i).ValuesAsNumpy()
    out["grid_latitude"], out["grid_longitude"], out["grid_elevation"] = r.Latitude(), r.Longitude(), r.Elevation()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--which", choices=["survey"], default="survey")
    p.add_argument("--pause", type=float, default=2.0, help="seconds between uncached requests")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    session = retry(requests_cache.CachedSession(str(OUT / "http_cache"), expire_after=-1), retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=session)
    pts = region_points()
    pts.to_csv(OUT / "points.csv", index=False)
    for which in [a.which]:
        start, end, daily = REQUESTS[which]
        frames = []
        for i, r in enumerate(pts.itertuples()):
            f = fetch(client, r.latitude, r.longitude, start, end, daily).assign(climate_region=r.climate_region)
            frames.append(f)
            if not getattr(session, "_last_from_cache", False):
                time.sleep(a.pause)
            print(f"{which} {i + 1}/{len(pts)} {r.climate_region}", flush=True)
        out = pd.concat(frames, ignore_index=True)
        bad = out[daily].isna().mean()
        if (bad > 0).any():
            print(f"WARNING missing values in {which}: {bad[bad > 0].round(4).to_dict()}")
        out.to_parquet(OUT / f"{which}_daily.parquet", index=False)
        print(f"{which}: {len(out)} region-days -> {OUT.relative_to(ROOT)}/{which}_daily.parquet", flush=True)


if __name__ == "__main__":
    main()
