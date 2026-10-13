"""Household frames for the identification study.

`load_development_frame()` returns ONLY train + validation households; it has no parameter for the test
partition. The test partition can be read only through `lib/identification/final_test.py`, which refuses unless
`config/final_selection.json` records an explicit human authorisation and no earlier test evaluation (one look, ever).

Information tiers (what a policymaker must collect to reach a given accuracy):
  T1  structural   A0 (dwelling/household/fuel core from config/pipeline.json) + climate
  T2  + socio-economic   education, employment, vehicles, respondent gender, area type, appliance
                         ages, heating control, household composition counts
  T3  + welfare screen   S7 (received public energy-bill aid) and S8 (subjective income
                         adequacy, 1 comfortable .. 4 very difficult). These are survey answers a
                         targeting scheme can ask; they are NOT inputs to the income-bracket label
                         (S9MONTH). Income itself is a separate switch (`include_income`).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from data.enable import HOUSEHOLD_SHEET, RETAINED_COUNTRY_NAMES, retained_cohort
from lib.features.household_features import (BASE_RAW_COLUMNS, RAW_COLUMNS as FEATURE_RAW_COLUMNS, REFERENCE_YEAR, engineer,
                                             stack_features)
from lib.features.engineered import RAW_EXTRA, engineer_extra
from lib.features.survey_items import ITEMS, clean_items
from lib.geography.climate_regions import climate_region
from lib.labels.income_harmonization import harmonized_income_percentile

ROOT = Path(__file__).resolve().parents[2]
COMP = [f"S1{g}c{i}" for g in ("A", "B") for i in (1, 2, 3)]
DEV_PARTITIONS = ("train", "validation")
COUNTRIES = list(RETAINED_COUNTRY_NAMES)
NOMINAL = {"H1", "employment_status", "primary_heating_fuel", "gender"}  # one-hot in linear models
SOCIO_ECONOMIC = ["education_level", "employment_status", "vehicle_count", "gender", "area_type",
                  "tv_age", "dishwasher_age", "heating_control", "elderly_count", "children_count"]
WELFARE_SCREEN = ["S7", "S8"]
NEED_FEATURES = ["H3", "household_size", "H1", "H2", "hdd", "cdd"]  # physical heating/cooling need only


HEATING_STRATEGY = {1: "one_setting", 2: "manual_adjustment", 3: "programmed_thermostat", 4: "no_control"}   # H9
INCOME_PREDICTOR = "income_bracket"   # Van Hove's design: income bracket (1-10) as an ordinary predictor


def tier_features(cfg: dict | None = None, *, include_income: bool = False) -> dict[str, list[str]]:
    """Information tiers. `include_income=True` adds the income bracket to every tier -- Van Hove's
    replication state (income is easy for a policymaker to obtain from a benefits/tax register); the default
    excludes it (the proxy-targeting state used where income cannot be observed)."""
    cfg = cfg or json.loads((ROOT / "config/pipeline.json").read_text())
    t1 = stack_features(cfg["features"], cfg["climate_features"])
    t2 = stack_features(t1, SOCIO_ECONOMIC)
    t3 = stack_features(t2, WELFARE_SCREEN)
    tiers = {"T1": t1, "T2": t2, "T3": t3}
    if include_income:
        tiers = {k: stack_features(v, [INCOME_PREDICTOR]) for k, v in tiers.items()}
    return tiers


def load_development_frame() -> pd.DataFrame:
    """One row per train/validation household with features, cost, income bracket, weights.
    Sentinel codes are already NaN (H1/H2 via engineer(); S7/S8 here)."""
    return _load_frame(DEV_PARTITIONS)


def _load_frame(partitions: tuple[str, ...]) -> pd.DataFrame:
    """Shared loader. Callers outside this package must use load_development_frame(); the only other caller is the
    guarded final-test module."""
    cfg = json.loads((ROOT / "config/pipeline.json").read_text())
    base = pd.read_parquet(ROOT / "artifacts/data/harmonised_observed_cost.parquet")[
        ["UID", "income_bracket", "w", "observed_energy_cost_eur_reference"]]
    cols = list(dict.fromkeys(["UID", "Country", "NUTS1", "NUTS2", "NUTS3"] + BASE_RAW_COLUMNS + FEATURE_RAW_COLUMNS + COMP + WELFARE_SCREEN
                              + RAW_EXTRA + ITEMS))
    hh = retained_cohort(pd.read_excel(ROOT / "data/raw/ENABLE.EU_dataset_survey of households.xlsx",
                                       sheet_name=HOUSEHOLD_SHEET, usecols=cols))
    hh["household_size"] = hh[COMP].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=len(COMP))
    hh = engineer(hh)
    # Impossible / "don't know" values (fixed 2026-10-13): floor-area code 7 is "don't know" and was read as the band above
    # "> 200 m2"; a household size of 0 (every composition count 0) is impossible. 41 of 1,996 eligible development
    # households (15 and 26). Both become missing (median-imputed by the models).
    hh["H3"] = pd.to_numeric(hh["H3"], errors="coerce").where(lambda s: s.between(1, 6))
    hh["household_size"] = hh["household_size"].where(hh["household_size"] >= 1)
    hh = engineer_extra(hh)
    hh[ITEMS] = clean_items(hh)                     # raw survey items, missing codes -> NaN (B1 "all variables")
    hh["S7"] = pd.to_numeric(hh["S7"], errors="coerce").replace({9: np.nan}).map({1: 1.0, 2: 0.0})
    hh["S8"] = pd.to_numeric(hh["S8"], errors="coerce").where(lambda s: s.between(1, 4))
    # Van Hove et al. (2022), Table 1 / thesis Sec. 5.3: "heating strategy" is H9 (how the household controls its main
    # heating equipment), categorical with 5 values -- the four strategies plus "don't know / no answer" kept as its own
    # category; "birth year respondent" is S4 itself (79 distinct values).
    hh["heating_strategy"] = hh["heating_control"].map(HEATING_STRATEGY).fillna("dont_know_no_answer")
    hh["birth_year"] = REFERENCE_YEAR - hh["respondent_age"]
    d = hh.merge(base, on="UID", how="left", validate="one_to_one")
    climate = pd.read_parquet(ROOT / "artifacts/climate/nuts1_annual_2017.parquet")[["NUTS1"] + cfg["climate_features"]]
    d = d.merge(climate, on="NUTS1", how="left", validate="many_to_one")
    # Stage-B2 climate (NUTS3 / Polish NUTS2; scripts/fuse_openmeteo_b2.py). Optional so the earlier stages run without it.
    d["climate_region"] = climate_region(d)
    b2 = ROOT / "artifacts/climate/b2/region_climate.parquet"
    if b2.exists():
        from lib.climate.b2_indices import B2_CLIMATE
        d = d.merge(pd.read_parquet(b2)[["climate_region"] + B2_CLIMATE], on="climate_region", how="left", validate="many_to_one")
    split = json.loads((ROOT / cfg["split"]).read_text())["assignments"]
    part = {str(a["household_id"]): a["partition"] for a in split}
    d["partition"] = d.UID.astype(str).map(part)
    d = d[d.partition.isin(partitions)].copy()
    d["income_bracket"] = pd.to_numeric(d.income_bracket, errors="coerce").where(lambda s: s.between(1, 10))
    d["cost"] = pd.to_numeric(d.observed_energy_cost_eur_reference, errors="coerce")
    d["income_percentile"] = harmonized_income_percentile(d.income_bracket, d.country)
    return d.reset_index(drop=True)
