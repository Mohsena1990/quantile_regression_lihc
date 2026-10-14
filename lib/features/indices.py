"""Household indices (B1, Strategy 2), regional context (B1, Dalla Longa et al. 2021) and climate x dwelling features (B2).

Indices -- each the mean of standardised items (z-scores over the development households; no labels involved), so a higher
value always means "more of" the named property:
  dwelling_efficiency_index    + insulation types, + newer construction (H2), + apartment rather than house (H1),
                               - "no insulation"
  wealth_index                 + cars, + appliances owned, + air conditioning, + renewables, + floor area, + dishwasher
                               owned, + smart meter            (stand-in for house value, which ENABLE does not ask)
  inefficient_equipment_index  + appliances older than 10 years, + portable electric heater, + incandescent bulbs,
                               + electric water heater
  vulnerability_index          + elderly living alone, + single person, + share of elderly, + share of children,
                               + respondent not employed (unemployed, retired, inactive)
  heating_system_type          dominant heating group: district / gas / solid fuel / electric / other (categorical)
  income_eur_equiv             income per adult equivalent, EUR (bracket -> Eurostat national decile; income_proxy_frame)

Regional context (artifacts/reference/regional_context.csv): pop_density, owner_share, gdp_per_capita, unemployment_rate,
poverty_rate, pre1981_share.

Climate x dwelling -- weather acting on a particular home, so they vary household by household:
  heating_need_kwh       floor area x heat-loss coefficient (age, type) x HDD of winter 2016/17 x 24/1000
                         (lib/identification/required_heat.py)
  heating_need_pp        heating need per person
  need_burden            heating need per EUR of household income (kWh / EUR)  -- income-derived
  hdd_x_old              HDD x built before 1970
  hdd_x_uninsulated      HDD x no additional insulation
  hdd_x_solid            HDD x share of heating from solid fuel
  hdd_x_electric         HDD x share of heating from electricity
  cdd_x_ac               CDD of summer 2017 x owns air conditioning
  rh_x_old               winter humidity x built before 1970 (damp risk)
  wind_x_detached        winter wind x detached house (exposure)
  density_x_urban        log population density x urbanness of the respondent's area (S6: 5 big city .. 1 farm)
Regional anomalies: winter_anomaly = HDD 2016/17 / 1991-2020 normal; summer_anomaly = CDD 2017 - normal CDD (a difference:
many regions have a cooling normal near 0, where a ratio explodes).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from lib.climate.b2_indices import B2_CLIMATE
from lib.identification.required_heat import required_heat_kwh
from lib.labels.income_harmonization import equivalence_scale, income_proxy_frame

REGIONAL_CONTEXT = ["pop_density", "owner_share", "gdp_per_capita", "unemployment_rate", "poverty_rate", "pre1981_share"]
INDICES = ["dwelling_efficiency_index", "wealth_index", "inefficient_equipment_index", "vulnerability_index",
           "heating_system_type", "income_eur_equiv"]
CLIMATE_REGIONAL = list(B2_CLIMATE) + ["winter_anomaly", "summer_anomaly"]
CLIMATE_HOUSEHOLD = ["heating_need_kwh", "heating_need_pp", "need_burden", "hdd_x_old", "hdd_x_uninsulated", "hdd_x_solid",
                     "hdd_x_electric", "cdd_x_ac", "rh_x_old", "wind_x_detached", "density_x_urban"]
CLIMATE_CANDIDATES = CLIMATE_REGIONAL + CLIMATE_HOUSEHOLD
INCOME_DERIVED = {"income_bracket", "income_eur_equiv", "need_burden"}
NAMES = {"dwelling_efficiency_index": "dwelling efficiency index", "wealth_index": "wealth index (house-value stand-in)",
         "inefficient_equipment_index": "inefficient equipment index", "vulnerability_index": "household vulnerability index",
         "heating_system_type": "heating system type", "income_eur_equiv": "income per adult equivalent (EUR)",
         "pop_density": "population density", "owner_share": "home-ownership share (region)", "gdp_per_capita": "GDP per head (region)",
         "unemployment_rate": "unemployment rate (region)", "poverty_rate": "poverty rate (region)",
         "pre1981_share": "share of dwellings built before 1981 (region)", "winter_anomaly": "winter 2016/17 vs normal",
         "summer_anomaly": "summer 2017 minus normal (CDD)", "heating_need_kwh": "heating need (kWh)", "heating_need_pp": "heating need per person",
         "need_burden": "heating need per EUR income", "hdd_x_old": "cold winter x old home", "hdd_x_uninsulated": "cold winter x no insulation",
         "hdd_x_solid": "cold winter x solid-fuel heating", "hdd_x_electric": "cold winter x electric heating",
         "cdd_x_ac": "hot summer x air conditioning", "rh_x_old": "humidity x old home", "wind_x_detached": "wind x detached house",
         "density_x_urban": "density x urban area"}


def _z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    sd = s.std()
    return (s - s.mean()) / sd if sd and np.isfinite(sd) else s * 0


def _index(parts: list[pd.Series]) -> pd.Series:
    return pd.concat([_z(p) for p in parts], axis=1).mean(axis=1, skipna=True)


def household_indices(d: pd.DataFrame) -> pd.DataFrame:
    """Adds INDICES (needs the loader's engineered columns, raw items H1/H2/H11B/S3 and income_bracket)."""
    out = d.copy()
    out["dwelling_efficiency_index"] = _index([d.insulation_count, d.H2, d.H1, -d.no_insulation])
    dishwasher_owned = pd.to_numeric(d.H11B, errors="coerce").isin([1, 2, 4]).astype(float).where(d.H11B.notna())
    out["wealth_index"] = _index([d.car_count, d.appliance_count, d.has_air_conditioning, d.renewable_count, d.H3,
                                  dishwasher_owned, d.smart_meter])
    out["inefficient_equipment_index"] = _index([d.old_appliance_count, d.has_portable_heater, d.incandescent_level,
                                                 d.has_electric_water_heater])
    s3 = d.S3.astype(str).str.replace("code_", "", regex=False)
    not_employed = s3.isin(["3", "4", "6"]).astype(float).where(d.S3.notna())
    out["vulnerability_index"] = _index([d.elderly_alone, d.single_person, d.elderly_share, d.children_share, not_employed])
    shares = pd.DataFrame({"district": d.district_heating_share, "gas": d.gas_heating_share, "solid fuel": d.solid_fuel_share,
                           "electric": d.electric_heating_share}).fillna(0)
    top = shares.idxmax(axis=1).where(shares.max(axis=1) > 0, "other")
    out["heating_system_type"] = top.where(d.district_heating_share.notna(), np.nan)
    scale = equivalence_scale(d.working_age_count + d.elderly_count, d.children_count)
    prox = income_proxy_frame(d.income_bracket, d.country, scale)
    out["income_eur_total"], out["income_eur_equiv"] = prox.proxy_total_eur, prox.proxy_equiv_eur
    return out


def climate_features(d: pd.DataFrame) -> pd.DataFrame:
    """Adds the regional anomalies and CLIMATE_HOUSEHOLD from the regional climate columns (recomputed for the placebo)."""
    out = d.copy()
    out["winter_anomaly"] = d.hdd_hs1617 / d.hdd_normal
    out["summer_anomaly"] = d.cdd_s2017 - d.cdd_normal
    old = (pd.to_numeric(d.H2, errors="coerce") <= 3).astype(float).where(d.H2.notna())
    need = required_heat_kwh(d, "hdd_hs1617")
    out["heating_need_kwh"] = need
    out["heating_need_pp"] = need / d.household_size
    out["need_burden"] = need / d.income_eur_total
    out["hdd_x_old"] = d.hdd_hs1617 * old
    out["hdd_x_uninsulated"] = d.hdd_hs1617 * d.no_insulation
    out["hdd_x_solid"] = d.hdd_hs1617 * d.solid_fuel_share / 100
    out["hdd_x_electric"] = d.hdd_hs1617 * d.electric_heating_share / 100
    out["cdd_x_ac"] = d.cdd_s2017 * d.has_air_conditioning
    out["rh_x_old"] = d.rh_hs * old
    out["wind_x_detached"] = d.wind_hs * (pd.to_numeric(d.H1, errors="coerce") == 1).astype(float).where(d.H1.notna())
    out["density_x_urban"] = np.log(d.pop_density) * (6 - pd.to_numeric(d.S6, errors="coerce"))
    return out
