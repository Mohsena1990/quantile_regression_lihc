"""Modelled heating need and the hidden under-heating flag (sensitivity check on the expenditure-based cost axis).

An expenditure-based label cannot see households that keep their home cold to save money: low spending puts them below
any cost threshold. This module estimates each household's heating NEED with a steady-state degree-day model and flags
households that spend far less than comparable households for the same need.

  required_heat_kwh = heated floor area x HLC per m2 (dwelling age, dwelling type) x HDD x 24 / 1000
    heated floor area  midpoint of the ENABLE H3 band (open top band > 200 m2 set to 250 m2)
    HLC per m2         stylised heat-loss coefficient (W/K per m2 of floor) by construction period, in the range of the
                       TABULA/EPISCOPE residential archetypes (older and detached homes lose more heat per m2), scaled
                       by an exposed-envelope factor for the dwelling type
    HDD                heating degree days (base 18 C) of the survey's heating season, winter 2016/17, at the
                       household's NUTS3 (Poland NUTS2) region (Open-Meteo reanalysis; lib/climate/b2_indices.py)
  spend_per_need     = observed annual energy cost (EUR) / required_heat_kwh
  hidden under-heating = spend_per_need below HALF the median of households in the same country with the same main
                       heating fuel (country median if the fuel group has fewer than MIN_GROUP households) -- an
                       EU Energy Poverty Observatory "M/2"-style rule, made relative to need.

Because the flag compares each household with the median of its own country x fuel group, only the RELATIVE values of
the archetype coefficients matter (a common scale factor cancels), and fuel-price differences between groups cancel too.
Cost includes electricity for appliances, which adds noise roughly proportional to household size; this is a
sensitivity check, not a replacement label.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

FLOOR_AREA_M2 = {1: 35.0, 2: 54.0, 3: 78.0, 4: 105.5, 5: 160.0, 6: 250.0}        # H3 bands
HLC_PER_M2 = {1: 3.0, 2: 2.8, 3: 2.7, 4: 2.5, 5: 2.2, 6: 1.9, 7: 1.5, 8: 1.1}    # H2: <1950 ... 2010-16, W/K per m2
TYPE_FACTOR = {1: 1.0, 2: 0.85, 3: 0.75, 4: 0.65}                                  # H1: detached, attached, 2-5 flats, 6+ flats
UNDER_SPEND_RATIO = 0.5
MIN_GROUP = 20


def required_heat_kwh(frame: pd.DataFrame, hdd_col: str = "hdd_hs1617") -> pd.Series:
    area = frame["H3"].map(FLOOR_AREA_M2)
    hlc = frame["H2"].map(HLC_PER_M2) * frame["H1"].map(TYPE_FACTOR)
    return area * hlc * frame[hdd_col] * 24 / 1000


def hidden_underheating(frame: pd.DataFrame, hdd_col: str = "hdd_hs1617") -> pd.DataFrame:
    """Adds required_heat_kwh, spend_per_need, comparison_group, hidden_underheating (NaN where need or cost is unknown)."""
    d = frame.copy()
    d["required_heat_kwh"] = required_heat_kwh(d, hdd_col)
    ok = d["required_heat_kwh"].gt(0) & d["cost"].notna()
    d["spend_per_need"] = np.where(ok, d["cost"] / d["required_heat_kwh"], np.nan)
    fuel = d["primary_heating_fuel"].fillna("unknown").astype(str)
    size = d[ok].groupby([d["country"], fuel]).size()
    big = {k for k, v in size.items() if v >= MIN_GROUP}
    d["comparison_group"] = [f"{c}|{f}" if (c, f) in big else f"{c}|all" for c, f in zip(d["country"], fuel)]
    med_group = d[ok].groupby("comparison_group")["spend_per_need"].median()
    med_country = d[ok].groupby("country")["spend_per_need"].median()
    ref = d["comparison_group"].map(med_group).fillna(d["country"].map(med_country))
    d["hidden_underheating"] = np.where(ok, (d["spend_per_need"] < UNDER_SPEND_RATIO * ref).astype(float), np.nan)
    return d
