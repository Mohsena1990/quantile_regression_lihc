"""1991-2020 climate normals for stage B2: one Copernicus CDS request, ERA5-Land monthly mean 2m temperature 1991-2020
over the four countries, sampled at each climate region's representative point (the point used for the Open-Meteo
series, artifacts/climate/openmeteo/points.csv; nearest valid land grid cell).

Why not Open-Meteo: 30 years of daily data per region counts as ~780 API calls under its fair-use weighting, i.e. more
than a week of the free daily quota for 121 regions. Degree days from MONTHLY means (days x max(0, base - monthly mean))
understate HDD/CDD in shoulder months, where some days cross the base temperature; they rank regions the same way and are
used only as the "typical climate" covariate (`hdd_normal`, `cdd_normal`).

Outputs: artifacts/climate/era5_normals/t2m_monthly_1991_2020.nc (+ manifest), artifacts/climate/era5_normals/points_monthly.parquet.
Credentials: ~/.cdsapirc. Skips the download when the NetCDF exists.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/climate/era5_normals"
AREA = [55.5, 6.0, 35.0, 30.0]
AREA_SPAIN = [44.5, -18.5, 27.5, 4.5]          # mainland Spain, Balearic and Canary Islands (version-5 cohort)
REQUEST = {"product_type": ["monthly_averaged_reanalysis"], "variable": ["2m_temperature"],
           "year": [str(y) for y in range(1991, 2021)], "month": [f"{m:02d}" for m in range(1, 13)], "time": ["00:00"],
           "data_format": "netcdf", "download_format": "unarchived", "area": AREA}


def main() -> None:
    import cdsapi
    OUT.mkdir(parents=True, exist_ok=True)
    files = {"t2m_monthly_1991_2020.nc": AREA, "t2m_monthly_1991_2020_spain.nc": AREA_SPAIN}
    manifest = {}
    for name, area in files.items():
        nc = OUT / name
        if not nc.exists():
            try:
                cdsapi.Client(quiet=True).retrieve("reanalysis-era5-land-monthly-means", {**REQUEST, "area": area}, str(nc))
            except Exception as e:
                raise SystemExit(f"BLOCKED_EXTERNAL_CDS: {e}")
        manifest[name] = {"dataset": "reanalysis-era5-land-monthly-means", "doi": "10.24381/cds.68d2bb30",
                          "request": {**REQUEST, "area": area}, "sha256": hashlib.sha256(nc.read_bytes()).hexdigest()}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    grids = []
    for name in files:
        ds = xr.open_dataset(OUT / name)
        t = ds["t2m"].rename({"valid_time": "time"}) if "valid_time" in ds["t2m"].dims else ds["t2m"]
        iy, ix = np.where(np.isfinite(t.isel(time=0).to_numpy()))
        grids.append((t, t.latitude.to_numpy(), t.longitude.to_numpy(), iy, ix))
    pts = pd.read_csv(ROOT / "artifacts/climate/openmeteo/points.csv")
    rows = []
    for r in pts.itertuples():
        best = None
        for t, lat, lon, iy, ix in grids:                       # nearest valid land cell over both grids
            dist = (lat[iy] - r.latitude) ** 2 + ((lon[ix] - r.longitude) * np.cos(np.radians(r.latitude))) ** 2
            k = int(np.argmin(dist))
            if best is None or dist[k] < best[0]:
                best = (dist[k], t, lat[iy[k]], lon[ix[k]], iy[k], ix[k])
        _, t, la, lo, y_, x_ = best
        series = t.isel(latitude=y_, longitude=x_).to_numpy() - 273.15
        rows.append(pd.DataFrame({"climate_region": r.climate_region, "month": pd.DatetimeIndex(t.time.to_numpy()).to_period("M").to_timestamp(),
                                  "t2m": series, "grid_latitude": la, "grid_longitude": lo}))
    out = pd.concat(rows, ignore_index=True)
    if out.t2m.isna().any() or len(out) != len(pts) * 360:
        raise SystemExit("BLOCKED_NORMALS_QC: incomplete monthly series")
    out.to_parquet(OUT / "points_monthly.parquet", index=False)
    print(f"normals: {len(pts)} regions x 360 months -> {OUT.relative_to(ROOT)}/points_monthly.parquet")


if __name__ == "__main__":
    sys.exit(main())
