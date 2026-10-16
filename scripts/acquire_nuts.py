"""Eurostat GISCO NUTS 2016 region polygons (1:1 million, WGS84) for levels 1-3, and a check that every study household's climate
region exists in them (lib/geography/climate_regions.py). Downloads only the files that are missing.

Outputs: artifacts/reference/NUTS_RG_01M_2016_4326_LEVL_{1,2,3}.geojson
"""
from __future__ import annotations
import sys
from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.enable import HOUSEHOLD_SHEET, retained_cohort
from lib.geography.climate_regions import climate_region, region_geometries

URL = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_01M_2016_4326_LEVL_{level}.geojson"


def main() -> None:
    out = ROOT / "artifacts/reference"
    out.mkdir(parents=True, exist_ok=True)
    for level in (1, 2, 3):
        f = out / f"NUTS_RG_01M_2016_4326_LEVL_{level}.geojson"
        if not f.exists():
            urlretrieve(URL.format(level=level), f)
    hh = retained_cohort(pd.read_excel(ROOT / "data/raw/ENABLE.EU_dataset_survey of households.xlsx", sheet_name=HOUSEHOLD_SHEET,
                                       usecols=["UID", "Country", "NUTS1", "NUTS2", "NUTS3"]))
    regions = climate_region(hh).dropna().unique()
    region_geometries(regions)                       # raises if any study region is missing from GISCO NUTS 2016
    print(f"NUTS 2016 polygons present; all {len(regions)} study climate regions found")


if __name__ == "__main__":
    main()
