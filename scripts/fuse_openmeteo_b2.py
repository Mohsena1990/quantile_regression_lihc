"""Compute the stage-B2 climate indices per household climate region from the Open-Meteo downloads.

Reads artifacts/climate/openmeteo/survey_daily.parquet (scripts/acquire_openmeteo_b2.py) and the 1991-2020 monthly normals
artifacts/climate/era5_normals/points_monthly.parquet (scripts/acquire_era5_normals.py); indices are defined in
lib/climate/b2_indices.py. QC gate (fails the script): every cohort climate region present, complete daily windows,
mean >= minimum and maximum >= mean, plausible ranges.

Outputs: artifacts/climate/b2/region_climate.parquet (+ .qc.json), outputs/tables/table_b2_climate_regions.csv,
outputs/figures/b2_climate_maps.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.climate.b2_indices import B2_CLIMATE, B2_NAMES, normal_indices, survey_indices
from lib.geography.climate_regions import region_geometries
from lib.reporting.figures import save_figure

OM, B2 = ROOT / "artifacts/climate/openmeteo", ROOT / "artifacts/climate/b2"
RANGES = {"hdd_hs1617": (300, 6000), "cdd_s2017": (0, 800), "hdd_normal": (300, 6000), "cdd_normal": (0, 800),
          "tmin_hs": (-20, 15), "apparent_tmin_hs": (-30, 15), "frost_days_hs": (0, 212), "cold_spell_max": (0, 212),
          "heatwave_days": (0, 153), "tropical_nights": (0, 153), "rh_hs": (40, 100), "precip_hs": (0, 3000),
          "snowfall_hs": (0, 1000), "wind_hs": (1, 60), "gust_max_hs": (5, 120), "solar_hs": (1, 20), "sunshine_hs": (0, 12),
          "cloud_hs": (0, 100)}


def main() -> None:
    points = pd.read_csv(OM / "points.csv")
    survey = pd.read_parquet(OM / "survey_daily.parquet").set_index("date")
    normal = pd.read_parquet(ROOT / "artifacts/climate/era5_normals/points_monthly.parquet").set_index("month")
    rows = []
    for code in points.climate_region:
        s, n = survey[survey.climate_region == code], normal[normal.climate_region == code]
        rows.append({"climate_region": code, **survey_indices(s), **normal_indices(n.t2m),
                     "grid_elevation": float(s.grid_elevation.iloc[0])})
    res = pd.DataFrame(rows).merge(points, on="climate_region", validate="one_to_one")
    res["level"] = np.where(res.climate_region.str.len() == 5, "NUTS3", "NUTS2")
    res["country"] = res.climate_region.str[:2].map({"BG": "Bulgaria", "HU": "Hungary", "IT": "Italy", "PL": "Poland", "ES": "Spain"})
    bad = {f: res.loc[~res[f].between(*RANGES[f]), "climate_region"].tolist() for f in B2_CLIMATE}
    bad = {f: v for f, v in bad.items() if v}
    qc = {"source": "Open-Meteo historical weather API (best-match ECMWF reanalysis) + ERA5-Land monthly normals, one representative point per region",
          "regions": len(res), "finite": bool(np.isfinite(res[B2_CLIMATE].to_numpy(float)).all()), "out_of_range": bad,
          "survey_sha256": hashlib.sha256((OM / "survey_daily.parquet").read_bytes()).hexdigest(),
          "normal_sha256": hashlib.sha256((ROOT / "artifacts/climate/era5_normals/points_monthly.parquet").read_bytes()).hexdigest()}
    qc["qc_all_clear"] = qc["regions"] == len(points) and qc["finite"] and not bad
    B2.mkdir(parents=True, exist_ok=True)
    (B2 / "region_climate.qc.json").write_text(json.dumps(qc, indent=2) + "\n")
    if not qc["qc_all_clear"]:
        raise SystemExit(f"BLOCKED_B2_QC: {json.dumps(qc)}")
    res.to_parquet(B2 / "region_climate.parquet", index=False)
    res.to_csv(ROOT / "outputs/tables/table_b2_climate_regions.csv", index=False)
    print(res.groupby("country")[B2_CLIMATE].agg(["min", "max"]).round(1).T.to_string())

    geo = region_geometries(res.climate_region).merge(res, left_on="NUTS_ID", right_on="climate_region")
    panels = ["hdd_hs1617", "hdd_normal", "cold_spell_max", "heatwave_days", "rh_hs", "wind_hs", "solar_hs", "snowfall_hs"]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5))
    for ax, col in zip(axes.ravel(), panels):
        geo.plot(column=col, ax=ax, cmap="viridis", legend=True, edgecolor="white", linewidth=0.2, legend_kwds={"shrink": 0.6})
        ax.set_axis_off()
        ax.set_title(B2_NAMES[col], fontsize=9, loc="left")
    fig.suptitle(f"B2 climate indices (Open-Meteo reanalysis) by household climate region, n = {len(res)}", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    save_figure(fig, ROOT / "outputs/figures/b2_climate_maps")
    plt.close(fig)
    print(json.dumps(qc, indent=2))


if __name__ == "__main__":
    main()
