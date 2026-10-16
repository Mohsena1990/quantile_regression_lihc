"""Household frames of the study.

`load_development_frame()`  the five study countries' development households (train + validation of the frozen 60/20/20 split,
                            config/household_split_v1.csv). The 20% test partition is never loaded: it was evaluated
                            once (2026-09-20, recorded in config/final_selection.json) and stays locked.
`load_pool_frame()`         the same + every household of the six other ENABLE countries (`pool` = "focus" / "extra"); the extra
                            households are only ever TRAINING households of pooled models (they are not in the frozen split).

Each row: survey features (sentinel codes as missing), annual energy cost in EUR (`cost`), income bracket, survey weight, climate
region (NUTS3; Poland NUTS2; extra countries see lib/geography/climate_regions.py) with its climate measures, regional context,
household indices and climate x home features (lib/features/indices.py), partly asked survey modules and the region (NUTS2) code.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from data.enable import ALL_COUNTRY_CODES, HOUSEHOLD_SHEET, POOL_EXTRA_NAMES, STUDY_COUNTRY_CODES, STUDY_COUNTRY_NAMES, retained_cohort
from lib.features.engineered import RAW_EXTRA, engineer_extra
from lib.features.household_features import BASE_RAW_COLUMNS, RAW_COLUMNS as FEATURE_RAW_COLUMNS, REFERENCE_YEAR, engineer
from lib.features.partial_modules import RAW_PARTIAL, partial_modules
from lib.features.survey_items import ITEMS, clean_items
from lib.geography.climate_regions import climate_region

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "config/household_split_v1.csv"
COMP = [f"S1{g}c{i}" for g in ("A", "B") for i in (1, 2, 3)]
DEV_PARTITIONS = ("train", "validation")
COUNTRIES = list(STUDY_COUNTRY_NAMES)
WELFARE_SCREEN = ["S7", "S8"]
COMFORT_RAW = ["C1A", "C1B", "C2"]          # usual indoor temperature in winter / summer, air conditioning (comfort outcomes)
HEATING_STRATEGY = {1: "one_setting", 2: "manual_adjustment", 3: "programmed_thermostat", 4: "no_control"}   # H9


def load_development_frame() -> pd.DataFrame:
    """The five study countries' train + validation households."""
    return _load_frame(pool=False)


def load_pool_frame() -> pd.DataFrame:
    """The five study countries' development households + every household of the six other ENABLE countries."""
    return _load_frame(pool=True)


def _load_frame(pool: bool) -> pd.DataFrame:
    cost_file = "energy_cost_all.parquet" if pool else "energy_cost_study.parquet"
    base = pd.read_parquet(ROOT / "artifacts/data" / cost_file)[["UID", "income_bracket", "w", "observed_energy_cost_eur_reference"]]
    cols = list(dict.fromkeys(["UID", "Country", "NUTS1", "NUTS2", "NUTS3"] + BASE_RAW_COLUMNS + FEATURE_RAW_COLUMNS + COMP
                              + WELFARE_SCREEN + RAW_EXTRA + ITEMS + RAW_PARTIAL + COMFORT_RAW))
    hh = retained_cohort(pd.read_excel(ROOT / "data/raw/ENABLE.EU_dataset_survey of households.xlsx", sheet_name=HOUSEHOLD_SHEET,
                                       usecols=cols), ALL_COUNTRY_CODES if pool else STUDY_COUNTRY_CODES, require_nuts1=not pool)
    hh["household_size"] = hh[COMP].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=len(COMP))
    hh = engineer(hh)
    # Floor-area code 7 is "don't know" (not a band above "> 200 m2"); a household size of 0 is impossible: both become missing.
    hh["H3"] = pd.to_numeric(hh["H3"], errors="coerce").where(lambda s: s.between(1, 6))
    hh["household_size"] = hh["household_size"].where(hh["household_size"] >= 1)
    hh = engineer_extra(hh)
    hh[ITEMS] = clean_items(hh)
    hh["S7"] = pd.to_numeric(hh["S7"], errors="coerce").replace({9: np.nan}).map({1: 1.0, 2: 0.0})
    hh["S8"] = pd.to_numeric(hh["S8"], errors="coerce").where(lambda s: s.between(1, 4))
    # Heating strategy = H9 (how the household controls its main heating), four strategies + "don't know / no answer" as its own
    # category; birth year of the respondent from S4.
    hh["heating_strategy"] = hh["heating_control"].map(HEATING_STRATEGY).fillna("dont_know_no_answer")
    hh["birth_year"] = REFERENCE_YEAR - hh["respondent_age"]
    d = hh.merge(base, on="UID", how="left", validate="one_to_one")
    d["climate_region"] = climate_region(d)
    if pool:
        from lib.geography.climate_regions import extra_region
        extra = d.country.isin(POOL_EXTRA_NAMES)
        d.loc[extra, "climate_region"] = extra_region(d[extra])
    b2 = ROOT / "artifacts/climate/b2/region_climate.parquet"
    if b2.exists():
        from lib.climate.b2_indices import B2_CLIMATE
        rc = pd.read_parquet(b2)
        b2x = ROOT / "artifacts/climate/b2/region_climate_extra.parquet"
        if pool and b2x.exists():
            rc = pd.concat([rc, pd.read_parquet(b2x)], ignore_index=True)
        d = d.merge(rc[["climate_region"] + B2_CLIMATE], on="climate_region", how="left", validate="many_to_one")
    split = pd.read_csv(SPLIT, dtype={"UID": str})
    part = dict(zip(split.UID, split.partition))
    d["partition"] = d.UID.astype(str).map(part)
    if pool:
        d.loc[d.country.isin(POOL_EXTRA_NAMES), "partition"] = "extra"
    d = d[d.partition.isin(DEV_PARTITIONS + (("extra",) if pool else ()))].copy()
    d["pool"] = np.where(d.partition == "extra", "extra", "focus")
    d["income_bracket"] = pd.to_numeric(d.income_bracket, errors="coerce").where(lambda s: s.between(1, 10))
    d["cost"] = pd.to_numeric(d.observed_energy_cost_eur_reference, errors="coerce")
    d = partial_modules(d)
    d["region"] = d["NUTS2"].astype(str).str.strip()             # NUTS2 code as one categorical variable
    if pool:                                                     # extra countries: Germany NUTS1, Norway NUTS2, Serbia NUTS2, Ukraine oblast
        x = d.pool == "extra"
        cr = d.loc[x, "climate_region"].astype("string")
        d.loc[x, "region"] = cr.where(~cr.str.startswith("NO", na=False), cr.str[:4])
    context = ROOT / "artifacts/reference/regional_context.csv"
    if context.exists():
        from lib.features.indices import REGIONAL_CONTEXT, climate_features, household_indices
        d = d.merge(pd.read_csv(context)[["climate_region"] + REGIONAL_CONTEXT], on="climate_region", how="left", validate="many_to_one")
        d = household_indices(d)
        if b2.exists():
            d = climate_features(d)
    return d.reset_index(drop=True)
