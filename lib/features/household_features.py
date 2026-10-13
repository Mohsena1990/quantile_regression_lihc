"""Household-conditioning feature engineering for M1/M2/M3 / screening.

Audit finding: household composition detail, heating-source composition, and the
PV/solar/heat-pump technology block exist in the ENABLE workbook but were only ever
fed into the (since removed) BREDEM required-cost engine, never into the M1/M2/M3/screening feature set -- weakening
the "threshold conditional on household need" claim these families make. This module
derives a defensible subset of that raw survey information into model-ready columns.

Every raw column referenced here was confirmed present in the ENABLE.EU_dataset_HH
sheet and cross-checked against the workbook's own variable/value-label dictionary
sheets before being added (see the variable labels: S1Ac1-3/S1Bc1-3 age-sex bands,
H6A1-11 heating-source shares, H10A1/H10A3/H10A4 PV/solar-water/heat-pump technology
flags, H9 heating control, SettlementSize).

2026-08-17 follow-up: mutual-information audit against `observed_energy_cost_eur_reference`
and the four-class label (see AUDIT_AND_REPAIR_REPORT.md-era SHAP audit, corroborated here)
found has_pv/has_solar_water_heating/has_heat_pump_tech/has_heat_pump_heating carry ~zero
signal in this cohort (rare technology adoption -> near-constant columns) -- dropped from
config/pipeline.json's `features` whitelist (kept here in ENGINEERED_COLUMNS for anyone who
wants them, just no longer selected by default). Added 7 previously-unused raw columns with
verified (5-fold CV, two regression families) predictive value: H4A vehicle ownership,
H11B/H11E/H11H appliance age (dishwasher/AC/TV), S2 education, S3 employment status,
S6 settlement type. None overlap with the label's own inputs (income_bracket/S9MONTH/S9YEAR),
so no leakage risk -- confirmed against tests/test_governance.py's FORBIDDEN_SCREENING_COLUMNS.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

BASE_RAW_COLUMNS = ["H1", "H2", "H3", "H5A1", "H5A2", "H5A3", "H5A4"]  # dwelling type/age/floor area/insulation
AGE_SEX_COLUMNS = [f"S1{g}c{i}" for g in ("A", "B") for i in (1, 2, 3)]  # Women/Men x under-18/18-65/65+
HEATING_SHARE_COLUMNS = [f"H6A{i}" for i in range(1, 12)]
TECHNOLOGY_COLUMNS = ["H10A1", "H10A3", "H10A4"]  # PV, solar water heating, geothermal/air-source heat pump
CONTROL_COLUMN = "H9"
SETTLEMENT_COLUMN = "SettlementSize"
# Vehicle ownership (wealth proxy), appliance age (dishwasher/AC/TV -- consumption/efficiency
# proxy), education, employment status, settlement type (S6 -- distinct from the raw
# SettlementSize sampling-stratum column above).
VEHICLE_COLUMN = "H4A"
APPLIANCE_AGE_COLUMNS = {"H11B": "dishwasher_age", "H11E": "ac_age", "H11H": "tv_age"}
EDUCATION_COLUMN = "S2"
EMPLOYMENT_COLUMN = "S3"
AREA_TYPE_COLUMN = "S6"
# Respondent gender (S5, 1=Male/2=Female) -- reporting/policy-map segmentation only,
# deliberately NOT added to config/pipeline.json's predictor set: this project's
# standard predictor set (A0) is built by scripts/build_a0_predictor_selection.py's
# recursive search, not hand-added; gender is exposed here so
# scripts/build_policy_maps_by_feature.py can show how risk concentration differs
# by respondent gender without making it a classifier input.
GENDER_COLUMN = "S5"
# Van Hove et al.'s "birth year of respondent" predictor -- one of their 7 base
# predictors (see docs/methodology_sources.md / the 2026-08-26 Van Hove comparison),
# previously missing from this project's feature set entirely. Converted to an age
# (REFERENCE_YEAR - birth year) rather than kept as a raw year: a monotonic affine
# transform, so it carries identical information to every model in this pipeline
# (tree splits and linear coefficients alike), just more directly interpretable in
# SHAP/coefficient output than a bare four-digit year.
BIRTH_YEAR_COLUMN = "S4"
_BIRTH_YEAR_MISSING_CODE = 9999
REFERENCE_YEAR = 2018  # matches this project's other 2018-anchored reference data (Eurostat prices/income deciles)

RAW_COLUMNS = (AGE_SEX_COLUMNS + HEATING_SHARE_COLUMNS + TECHNOLOGY_COLUMNS
               + [CONTROL_COLUMN, SETTLEMENT_COLUMN, VEHICLE_COLUMN, EDUCATION_COLUMN,
                  EMPLOYMENT_COLUMN, AREA_TYPE_COLUMN, BIRTH_YEAR_COLUMN, GENDER_COLUMN]
               + list(APPLIANCE_AGE_COLUMNS))

# Solid/biomass fuels whose combined heating share flags a household as solid-fuel heated.
_SOLID_FUEL_COLUMNS = ["H6A4", "H6A5", "H6A6", "H6A8", "H6A9"]  # wood, coal/coke, pellets, waste, biomass
_HEAT_PUMP_HEATING_COLUMN = "H6A10"
_HEATING_FUEL_LABELS = {
    "H6A1": "electricity", "H6A2": "district_heating", "H6A3": "gas_lpg", "H6A4": "wood",
    "H6A5": "coal_coke", "H6A6": "pellets", "H6A7": "fuel_oil", "H6A8": "waste",
    "H6A9": "biomass", "H6A10": "heat_pump", "H6A11": "other",
}
_HEATING_CONTROL_MISSING_CODE = 9  # "Don't know / No answer" -- distinct from the generic 98/99 sentinel family
# "Don't know" sentinels for the 2026-08-17 additions, per the workbook's own value-label sheet.
_VEHICLE_MISSING_CODE = 99
_APPLIANCE_AGE_MISSING_CODE = 99
_EDUCATION_MISSING_CODE = 9
_EMPLOYMENT_MISSING_CODE = 99
_AREA_TYPE_MISSING_CODE = 6
# Dwelling type / age "don't know" sentinels (workbook value-label sheet). H2 == 99 covers 5.8%-14.7% of
# households per country and was previously passed to every model as the number 99 (an extreme
# "dwelling age" outlier for M1/M2); H1 == 9 is 5 households. Found 2026-09-19; data/enable.py's
# MISSING_CODES was defined but never applied to these columns.
_DWELLING_SENTINELS = {"H1": 9, "H2": 99}

ENGINEERED_COLUMNS = ["children_count", "working_age_count", "elderly_count", "dependency_ratio",
                      "primary_heating_fuel", "has_solid_fuel_heating", "has_heat_pump_heating",
                      "has_pv", "has_solar_water_heating", "has_heat_pump_tech",
                      "heating_control", "settlement_size", "vehicle_count", "dishwasher_age",
                      "ac_age", "tv_age", "education_level", "employment_status", "area_type",
                      "respondent_age", "gender"]
_GENDER_LABELS = {1: "Male", 2: "Female"}


def stack_features(*groups: list[str]) -> list[str]:
    """Concatenate feature-name lists (e.g. A0 + climate_features, or + CONTEXT_COLUMNS)
    while de-duplicating, order preserved. Needed because A0 (config/pipeline.json's
    `features`, built by scripts/build_a0_predictor_selection.py's recursive search)
    can itself already contain a climate or country-context column if the search found
    it useful -- a plain `a0 + climate_features` would then pass a duplicate column
    name into the model design matrix, which breaks `pd.api.types.is_numeric_dtype`
    (a duplicate label makes `frame[name]` return a DataFrame, not a Series)."""
    return list(dict.fromkeys(x for group in groups for x in group))


def engineer(hh: pd.DataFrame) -> pd.DataFrame:
    """Return `hh` with ENGINEERED_COLUMNS added. `hh` must already contain RAW_COLUMNS."""
    out = hh.copy()
    for column, sentinel in _DWELLING_SENTINELS.items():
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").replace(sentinel, np.nan)
    numeric = out[RAW_COLUMNS].apply(pd.to_numeric, errors="coerce")

    out["children_count"] = numeric[["S1Ac1", "S1Bc1"]].sum(axis=1, min_count=1)
    out["working_age_count"] = numeric[["S1Ac2", "S1Bc2"]].sum(axis=1, min_count=1)
    out["elderly_count"] = numeric[["S1Ac3", "S1Bc3"]].sum(axis=1, min_count=1)
    dependants = out["children_count"] + out["elderly_count"]
    out["dependency_ratio"] = np.where(out["working_age_count"] > 0, dependants / out["working_age_count"], np.nan)

    shares = numeric[HEATING_SHARE_COLUMNS]
    has_any_share = shares.notna().any(axis=1)
    dominant = pd.Series(np.nan, index=shares.index, dtype="object")
    dominant.loc[has_any_share] = shares.loc[has_any_share].idxmax(axis=1)
    out["primary_heating_fuel"] = np.where(has_any_share, dominant.map(_HEATING_FUEL_LABELS), np.nan)
    out["has_solid_fuel_heating"] = np.where(has_any_share, (shares[_SOLID_FUEL_COLUMNS].fillna(0).sum(axis=1) > 0).astype(int), np.nan)
    out["has_heat_pump_heating"] = np.where(has_any_share, (shares[_HEAT_PUMP_HEATING_COLUMN].fillna(0) > 0).astype(int), np.nan)

    out["has_pv"] = numeric["H10A1"]
    out["has_solar_water_heating"] = numeric["H10A3"]
    out["has_heat_pump_tech"] = numeric["H10A4"]

    control = numeric[CONTROL_COLUMN].replace(_HEATING_CONTROL_MISSING_CODE, np.nan)
    out["heating_control"] = control
    out["settlement_size"] = numeric[SETTLEMENT_COLUMN]

    out["vehicle_count"] = numeric[VEHICLE_COLUMN].replace(_VEHICLE_MISSING_CODE, np.nan)
    for raw_col, name in APPLIANCE_AGE_COLUMNS.items():
        out[name] = numeric[raw_col].replace(_APPLIANCE_AGE_MISSING_CODE, np.nan)
    out["education_level"] = numeric[EDUCATION_COLUMN].replace(_EDUCATION_MISSING_CODE, np.nan)
    out["employment_status"] = numeric[EMPLOYMENT_COLUMN].replace(_EMPLOYMENT_MISSING_CODE, np.nan)
    out["area_type"] = numeric[AREA_TYPE_COLUMN].replace(_AREA_TYPE_MISSING_CODE, np.nan)
    birth_year = numeric[BIRTH_YEAR_COLUMN].replace(_BIRTH_YEAR_MISSING_CODE, np.nan)
    out["respondent_age"] = REFERENCE_YEAR - birth_year
    out["gender"] = numeric[GENDER_COLUMN].map(_GENDER_LABELS)
    return out
