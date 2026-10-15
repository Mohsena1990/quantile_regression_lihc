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


# ---- Version 6: regions of the extra training countries (pooled models only) ----------------------------------------------
# Germany: the survey's NUTS1 is the official state number (1 Schleswig-Holstein .. 16 Thuringia; the sample shares match state
# populations, e.g. 5 North Rhine-Westphalia 21%, 9 Bavaria 15%, 4 Bremen 0.7%).
DE_STATE_TO_NUTS1 = {1: "DEF", 2: "DE6", 3: "DE9", 4: "DE5", 5: "DEA", 6: "DE7", 7: "DEB", 8: "DE1", 9: "DE2", 10: "DEC", 11: "DE3",
                     12: "DE4", 13: "DE8", 14: "DED", 15: "DEE", 16: "DEG"}
# Norway: NUTS3 2013 codes; Sor- and Nord-Trondelag (NO061, NO062) merged into Trondelag NO060 in NUTS 2016.
NO_2013_TO_2016 = {"NO061": "NO060", "NO062": "NO060"}
RS_NAME_TO_NUTS2 = {"Belgrade": "RS11", "Vojvodina": "RS12", "Sumadija and Western Serbia": "RS21", "East and South Serbia": "RS22"}
# Ukraine (no NUTS): the survey's NUTS1 2-26 are the oblasts in Ukrainian alphabetical order with Kyiv city after Kyiv oblast
# (Crimea, 1, not surveyed; sample shares match oblast populations within 1-3 households). One point per oblast: its administrative
# centre (Donetsk and Luhansk oblasts: the 2018 administrative centres Kramatorsk and Sievierodonetsk; Kyiv oblast: Bila Tserkva).
UA_POINTS = {"UA02": (49.2331, 28.4682), "UA03": (50.7472, 25.3254), "UA04": (48.4647, 35.0462), "UA05": (48.7230, 37.5560),
             "UA06": (50.2547, 28.6587), "UA07": (48.6208, 22.2879), "UA08": (47.8388, 35.1396), "UA09": (48.9226, 24.7111),
             "UA10": (49.7988, 30.1153), "UA11": (50.4501, 30.5234), "UA12": (48.5079, 32.2623), "UA13": (48.9480, 38.4870),
             "UA14": (49.8397, 24.0297), "UA15": (46.9750, 31.9946), "UA16": (46.4825, 30.7233), "UA17": (49.5883, 34.5514),
             "UA18": (50.6199, 26.2516), "UA19": (50.9077, 34.7981), "UA20": (49.5535, 25.5948), "UA21": (49.9935, 36.2304),
             "UA22": (46.6354, 32.6169), "UA23": (49.4229, 26.9871), "UA24": (49.4444, 32.0598), "UA25": (48.2921, 25.9358),
             "UA26": (51.4982, 31.2893)}


def extra_region(frame: pd.DataFrame) -> pd.Series:
    """Climate region of the extra-pool households (Germany, Norway, Serbia, Ukraine); France and the UK have none (missing)."""
    n1 = frame["NUTS1"].astype(str).str.strip()
    out = pd.Series(pd.NA, index=frame.index, dtype="string")
    c = frame["country"]
    out[c == "Germany"] = pd.to_numeric(n1[c == "Germany"], errors="coerce").map(DE_STATE_TO_NUTS1)
    out[c == "Norway"] = n1[c == "Norway"].replace(NO_2013_TO_2016)
    out[c == "Serbia"] = frame.loc[c == "Serbia", "NUTS2"].astype(str).str.strip().map(RS_NAME_TO_NUTS2)
    out[c == "Ukraine"] = pd.to_numeric(n1[c == "Ukraine"], errors="coerce").map(lambda v: f"UA{int(v):02d}" if pd.notna(v) else pd.NA)
    return out


def region_points(codes) -> pd.DataFrame:
    """One point per region: the representative point of its GISCO NUTS 2016 polygon (levels 1-3), or the Ukrainian oblast centre."""
    codes = sorted(set(codes))
    ua = [c for c in codes if c.startswith("UA")]
    parts = [gpd.read_file(ROOT / f"artifacts/reference/NUTS_RG_01M_2016_4326_LEVL_{lvl}.geojson")[["NUTS_ID", "geometry"]] for lvl in (1, 2, 3)]
    geo = pd.concat(parts, ignore_index=True)
    geo = geo[geo.NUTS_ID.isin(set(codes) - set(ua))]
    missing = sorted(set(codes) - set(ua) - set(geo.NUTS_ID))
    if missing:
        raise ValueError(f"regions missing from GISCO NUTS 2016: {missing}")
    pts = gpd.GeoSeries(geo.geometry.values, crs=4326).representative_point()
    out = pd.DataFrame({"climate_region": geo.NUTS_ID.values, "latitude": pts.y.round(4).values, "longitude": pts.x.round(4).values})
    out = pd.concat([out, pd.DataFrame([{"climate_region": c, "latitude": UA_POINTS[c][0], "longitude": UA_POINTS[c][1]} for c in ua])])
    return out.sort_values("climate_region").reset_index(drop=True)
