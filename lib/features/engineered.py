"""Engineered household features (B1 candidates) from survey items answered in all four countries.

Coverage screen (2026-10-13, development households): only the H (household / dwelling) and S (socio-demographic) sections
are >= 80% complete in Bulgaria, Hungary, Italy and Poland. Excluded as country-specific: P (PV prosumers, Italy only),
C (comfort / refurbishment barriers, Hungary only), H15 (environmental attitudes, not asked in Bulgaria), SettlementSize
(blank in Poland), H13B (gas smart meter, 6-12% in Bulgaria/Poland). Never used: the cost and income items that build
the label (H7*, H8*, S9*). Survey codes: H4 vehicles 1 none / 2 one / 3 two / 4 three+; H11 appliance age 1 <= 3 years /
2 4-10 / 4 > 10 / 5 not owned; H12 bulb share 1 all / 2 most / 3 half / 4 some / 5 none; H13 1 yes / 2 no;
H5A, H10A 0/1 mentioned; H6A heating-source shares in percent; 99 = don't know (missing) throughout.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

CAR_COLUMNS = ["H4A", "H4B", "H4C", "H4D", "H4E"]                       # petrol, diesel, LPG, electric, hybrid
MOTOR_COLUMNS = CAR_COLUMNS + ["H4F", "H4G", "H4H"]                       # + motorcycles, vans
APPLIANCE_COLUMNS = [f"H11{c}" for c in "ABCDEFGH"]
SHARE_COLUMNS = [f"H6A{i}" for i in range(1, 12)]
SOLID = ["H6A4", "H6A5", "H6A6", "H6A8", "H6A9"]                           # wood, coal, pellets, waste, biomass
RAW_EXTRA = (MOTOR_COLUMNS + APPLIANCE_COLUMNS + SHARE_COLUMNS + ["H5A1", "H5A2", "H5A3", "H5A4", "H10A1", "H10A2", "H10A3",
             "H10A4", "H12A", "H12B", "H13A", "H13C", "H3"])
FLOOR_MID = {1: 35.0, 2: 54.0, 3: 78.0, 4: 105.5, 5: 160.0, 6: 250.0}
BULB_LEVEL = {1: 4, 2: 3, 3: 2, 4: 1, 5: 0}                               # share of bulbs of that kind, 4 = all .. 0 = none

ENGINEERED = {
    "car_count": "cars of any fuel (0-3+ each type, summed)",
    "motor_vehicle_count": "cars + motorcycles + vans",
    "insulation_count": "additional insulation types: roof, cavity wall, external wall (0-3)",
    "no_insulation": "home has no additional insulation",
    "district_heating_share": "share of heating from district heating (%)",
    "electric_heating_share": "share of heating from electricity (%)",
    "gas_heating_share": "share of heating from gas (%)",
    "solid_fuel_share": "share of heating from wood, coal, pellets, waste, biomass (%)",
    "n_heating_sources": "number of heating sources used",
    "renewable_count": "PV, biomass generation, solar water heating, heat pump (0-4)",
    "appliance_count": "appliances owned out of 8",
    "old_appliance_count": "appliances older than 10 years",
    "has_air_conditioning": "owns air conditioning",
    "has_portable_heater": "owns portable electric heater(s)",
    "has_electric_water_heater": "owns a standalone electric water heater",
    "efficient_bulb_level": "share of energy-efficient bulbs (0 none .. 4 all)",
    "incandescent_level": "share of old incandescent bulbs (0 none .. 4 all)",
    "smart_meter": "electricity or heating smart meter",
    "floor_area_per_person": "floor-area band midpoint / household size (m2)",
    "single_person": "one-person household",
    "elderly_alone": "all members above 65",
    "elderly_share": "share of members above 65",
    "children_share": "share of members up to 18",
}


def _num(frame: pd.DataFrame, cols) -> pd.DataFrame:
    return frame[cols].apply(pd.to_numeric, errors="coerce").replace(99, np.nan)


def engineer_extra(hh: pd.DataFrame) -> pd.DataFrame:
    """`hh` must hold RAW_EXTRA plus household_size, elderly_count, children_count. Returns hh with ENGINEERED columns."""
    out = hh.copy()
    veh = _num(hh, MOTOR_COLUMNS) - 1                                      # 1 none -> 0 ... 4 three+ -> 3
    out["car_count"] = veh[CAR_COLUMNS].sum(axis=1, min_count=1)
    out["motor_vehicle_count"] = veh.sum(axis=1, min_count=1)
    ins = _num(hh, ["H5A1", "H5A2", "H5A3", "H5A4"])
    out["insulation_count"] = ins[["H5A1", "H5A2", "H5A3"]].sum(axis=1)
    out["no_insulation"] = ins["H5A4"]
    sh = _num(hh, SHARE_COLUMNS)
    known = sh.notna().any(axis=1)
    s0 = sh.fillna(0)
    for name, cols in (("district_heating_share", ["H6A2"]), ("electric_heating_share", ["H6A1"]),
                       ("gas_heating_share", ["H6A3"]), ("solid_fuel_share", SOLID)):
        out[name] = s0[cols].sum(axis=1).where(known)
    out["n_heating_sources"] = (s0 > 0).sum(axis=1).where(known)
    out["renewable_count"] = _num(hh, ["H10A1", "H10A2", "H10A3", "H10A4"]).sum(axis=1)
    app = _num(hh, APPLIANCE_COLUMNS)
    answered = app.notna()
    out["appliance_count"] = (app.isin([1, 2, 4])).sum(axis=1).where(answered.any(axis=1))
    out["old_appliance_count"] = (app == 4).sum(axis=1).where(answered.any(axis=1))
    for name, col in (("has_air_conditioning", "H11E"), ("has_portable_heater", "H11F"), ("has_electric_water_heater", "H11G")):
        out[name] = app[col].isin([1, 2, 4]).astype(float).where(app[col].notna())
    bulbs = _num(hh, ["H12A", "H12B"])
    out["efficient_bulb_level"] = bulbs["H12B"].map(BULB_LEVEL)
    out["incandescent_level"] = bulbs["H12A"].map(BULB_LEVEL)
    sm = _num(hh, ["H13A", "H13C"])
    out["smart_meter"] = (sm == 1).any(axis=1).astype(float).where(sm.notna().any(axis=1))
    size = pd.to_numeric(hh["household_size"], errors="coerce").where(lambda s: s >= 1)
    out["floor_area_per_person"] = pd.to_numeric(hh["H3"], errors="coerce").map(FLOOR_MID) / size
    out["single_person"] = (size == 1).astype(float).where(size.notna())
    elderly = pd.to_numeric(hh["elderly_count"], errors="coerce")
    out["elderly_alone"] = (elderly == size).astype(float).where(size.notna())
    out["elderly_share"] = elderly / size
    out["children_share"] = pd.to_numeric(hh["children_count"], errors="coerce") / size
    return out
