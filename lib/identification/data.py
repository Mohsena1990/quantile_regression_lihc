"""Development-partition household frame for the identification protocol.

Returns ONLY train + validation households. The test partition is locked
(`artifacts/manifests/final_selection_frozen.json`, `test_unlock_authorized: false`); nothing in
this package can load it -- there is deliberately no parameter for it.

Information tiers (what a policymaker must collect to reach a given accuracy):
  T1  structural   A0 (dwelling/household/fuel core from config/pipeline.json) + climate
  T2  + socio-economic   education, employment, vehicles, respondent gender, area type, appliance
                         ages, heating control, household composition counts
  T3  + welfare screen   S7 (received public energy-bill aid) and S8 (subjective income
                         adequacy, 1 comfortable .. 4 very difficult). These are survey answers a
                         targeting scheme can ask; they are NOT inputs to the income-bracket label
                         (S9MONTH), unlike income itself, which stays excluded everywhere.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from data.enable import HOUSEHOLD_SHEET, RETAINED_COUNTRY_NAMES, retained_cohort
from lib.features.household_features import BASE_RAW_COLUMNS, RAW_COLUMNS as FEATURE_RAW_COLUMNS, engineer, stack_features
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
    cfg = json.loads((ROOT / "config/pipeline.json").read_text())
    base = pd.read_parquet(ROOT / "artifacts/data/harmonised_observed_cost.parquet")[
        ["UID", "income_bracket", "w", "observed_energy_cost_eur_reference"]]
    cols = list(dict.fromkeys(["UID", "Country", "NUTS1"] + BASE_RAW_COLUMNS + FEATURE_RAW_COLUMNS + COMP + WELFARE_SCREEN))
    hh = retained_cohort(pd.read_excel(ROOT / "data/raw/ENABLE.EU_dataset_survey of households.xlsx",
                                       sheet_name=HOUSEHOLD_SHEET, usecols=cols))
    hh["household_size"] = hh[COMP].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=len(COMP))
    hh = engineer(hh)
    hh["S7"] = pd.to_numeric(hh["S7"], errors="coerce").replace({9: np.nan}).map({1: 1.0, 2: 0.0})
    hh["S8"] = pd.to_numeric(hh["S8"], errors="coerce").where(lambda s: s.between(1, 4))
    d = hh.merge(base, on="UID", how="left", validate="one_to_one")
    climate = pd.read_parquet(ROOT / "artifacts/climate/nuts1_annual_2017.parquet")[["NUTS1"] + cfg["climate_features"]]
    d = d.merge(climate, on="NUTS1", how="left", validate="many_to_one")
    split = json.loads((ROOT / cfg["split"]).read_text())["assignments"]
    part = {str(a["household_id"]): a["partition"] for a in split}
    d["partition"] = d.UID.astype(str).map(part)
    d = d[d.partition.isin(DEV_PARTITIONS)].copy()
    d["income_bracket"] = pd.to_numeric(d.income_bracket, errors="coerce").where(lambda s: s.between(1, 10))
    d["cost"] = pd.to_numeric(d.observed_energy_cost_eur_reference, errors="coerce")
    d["income_percentile"] = harmonized_income_percentile(d.income_bracket, d.country)
    return d.reset_index(drop=True)
