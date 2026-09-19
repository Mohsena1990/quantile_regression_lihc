"""Named policy factors for cross-country interpretation.

Each factor is a binary exposure with a documented recode, so "the effect of X" means the same
thing in every country. Missing/sentinel values stay NaN (never coded as 'unexposed').

  solid_fuel     primary heating fuel is wood / coal / pellets (biomass, waste): the fuel-poverty and
                 hidden-energy-poverty lever (self-collected fuel is often absent from recorded cost)
  old_dwelling   built before 1970 (H2 code <= 3) vs 1970 or later -- the retrofit lever
  elderly        respondent aged 65+ -- the vulnerability lever
  large_hh       four or more household members -- the need lever
  cold_region    NUTS1 heating-degree-days above the country's household median -- the climate
                 moderator (within-country: Bulgaria has only 2 NUTS1 regions, Hungary 3)
Controls always included in effect models: floor area (H3, standardised) and HDD (standardised),
so a factor's effect is net of dwelling size and climate. Household size is NOT a control: it is the
`large_hh` factor itself.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

SOLID_FUELS = {"wood", "coal_coke", "pellets", "biomass", "waste"}
FACTORS = ["solid_fuel", "old_dwelling", "elderly", "large_hh"]
FACTOR_LABELS = {"solid_fuel": "Solid-fuel heating", "old_dwelling": "Dwelling built pre-1970",
                 "elderly": "Respondent 65+", "large_hh": "Household of 4+", "cold_region": "Colder region (HDD)"}
PAIRS = [("solid_fuel", "old_dwelling"), ("elderly", "large_hh")]
TRIPLE = ("solid_fuel", "old_dwelling", "cold_region")

# leave-one-group-out blocks for the accuracy-contribution analysis
FEATURE_GROUPS = {
    "Heating fuel": ["primary_heating_fuel"],
    "Dwelling type/age": ["H1", "H2"],
    "Floor area & size": ["H3", "household_size", "children_count", "elderly_count"],
    "Respondent": ["respondent_age", "gender"],
    "Appliances": ["ac_age", "tv_age", "dishwasher_age"],
    "Climate": ["hdd", "cdd", "temperature_mean", "temperature_sd", "frost_days"],
    "Socio-economic": ["education_level", "employment_status", "vehicle_count", "area_type", "heating_control"],
    "Welfare screen (S7/S8)": ["S7", "S8"],
}


def add_factors(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    fuel = d["primary_heating_fuel"]
    d["solid_fuel"] = np.where(fuel.isna(), np.nan, fuel.isin(SOLID_FUELS).astype(float))
    d["old_dwelling"] = np.where(d["H2"].isna(), np.nan, (d["H2"] <= 3).astype(float))
    d["elderly"] = np.where(d["respondent_age"].isna(), np.nan, (d["respondent_age"] >= 65).astype(float))
    d["large_hh"] = np.where(d["household_size"].isna(), np.nan, (d["household_size"] >= 4).astype(float))
    median_hdd = d.groupby("country")["hdd"].transform("median")
    d["cold_region"] = np.where(d["hdd"].isna(), np.nan, (d["hdd"] > median_hdd).astype(float))
    for c in ("H3", "household_size", "hdd"):
        d[f"{c}_z"] = d.groupby("country")[c].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0))
    return d


# Feature -> conceptual domain (colours/markers in the discovery figures; economic / social / political
# interpretation in the paper)
FEATURE_DOMAIN = {
    "H1": "Dwelling & efficiency", "H2": "Dwelling & efficiency", "H3": "Dwelling & efficiency",
    "primary_heating_fuel": "Heating & appliances", "heating_control": "Heating & appliances", "ac_age": "Heating & appliances",
    "tv_age": "Heating & appliances", "dishwasher_age": "Heating & appliances",
    "household_size": "Household", "children_count": "Household", "elderly_count": "Household",
    "respondent_age": "Household", "gender": "Household",
    "education_level": "Socio-economic", "employment_status": "Socio-economic", "vehicle_count": "Socio-economic",
    "area_type": "Socio-economic",
    "hdd": "Climate", "cdd": "Climate", "temperature_mean": "Climate", "temperature_sd": "Climate", "frost_days": "Climate",
    "S7": "Welfare screen", "S8": "Welfare screen", "income_bracket": "Income",
}
