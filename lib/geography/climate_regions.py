"""Finest climate region each household can be placed in (feature stage B2).

The ENABLE workbook carries NUTS3 for Bulgaria, Hungary and Italy and only NUTS2 for Poland (NUTS3 is blank for all
1,000 Polish households), so the climate region is NUTS3 where available and NUTS2 otherwise: 28 + 20 + 56 NUTS3 and
17 Polish NUTS2 regions, against 17 NUTS1 regions in total before.

Hungary's survey codes are NUTS 2013 for the capital region: HU101 (Budapest) and HU102 (Pest) became HU110 and HU120
in NUTS 2016 (HU10 was split into HU11 and HU12) with unchanged borders. Every other code is NUTS 2016.
"""
from __future__ import annotations
from pathlib import Path
import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NUTS2013_TO_2016 = {"HU101": "HU110", "HU102": "HU120"}
NUTS2_ONLY = {"Poland"}


def climate_region(frame: pd.DataFrame) -> pd.Series:
    """NUTS 2016 code of each household's climate region (needs columns country, NUTS2, NUTS3)."""
    n3 = frame["NUTS3"].astype("string").replace({"nan": pd.NA}).replace(NUTS2013_TO_2016)
    n2 = frame["NUTS2"].astype("string").replace({"nan": pd.NA})
    out = n3.where(~frame["country"].isin(NUTS2_ONLY), n2)
    return out.where(out.notna(), n2)


def region_geometries(codes) -> gpd.GeoDataFrame:
    """GISCO NUTS 2016 (1:1M) polygons for the given level-2/3 codes; raises if any code is unknown."""
    parts = [gpd.read_file(ROOT / f"artifacts/reference/NUTS_RG_01M_2016_4326_LEVL_{lvl}.geojson")[["NUTS_ID", "LEVL_CODE", "geometry"]]
             for lvl in (2, 3)]
    geo = pd.concat(parts, ignore_index=True)
    geo = geo[geo.NUTS_ID.isin(set(codes))].reset_index(drop=True)
    missing = sorted(set(codes) - set(geo.NUTS_ID))
    if missing:
        raise ValueError(f"climate regions missing from GISCO NUTS 2016: {missing}")
    return gpd.GeoDataFrame(geo, geometry="geometry", crs=4326)
