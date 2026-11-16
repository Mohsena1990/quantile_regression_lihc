"""v12 design: four cost thresholds (R0-R3) x six non-overlapping information blocks x three models, five countries.

Quadrants (Hills' income-cost plane): y = LI + 2*HC -> 0 HILC, 1 LILC, 2 HIHC, 3 LIHC (energy poor).
  R0  80th percentile of absolute energy spending in the country (van Hove et al. 2022)
  R1  80th percentile of the energy burden (spending / income) in the country (Dalla Longa et al. 2021)
  R2  need line: 60th conditional quantile of spending given dwelling + household (quantile regression; Hills-style required cost)
  R3  R2 + heating and cooling degree days
Low income: income bracket <= 3 in every line.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lib.identification.lines import make_labels

QUAD = {0: "HILC", 1: "LILC", 2: "HIHC", 3: "LIHC"}
COUNTRIES = ["Bulgaria", "Hungary", "Italy", "Poland", "Spain"]
RSPEC = {"R0": ("p80", 0.80), "R1": ("burden", 0.80), "R2": ("needqr_noclim", 0.60), "R3": ("needqr", 0.60)}
RNAME = {"R0": "R0 fixed spending line (P80)", "R1": "R1 energy-burden line", "R2": "R2 need line", "R3": "R3 need line + climate"}
BLOCKS = {
    "I": ["income_bracket"],
    "H": ["household_size", "children_share", "elderly_share", "single_person", "elderly_alone", "respondent_age",
          "employment_status", "education_level"],
    "D": ["H3", "H1", "H2", "insulation_count", "no_insulation"],
    "S": ["heating_strategy", "primary_heating_fuel", "solid_fuel_share", "electric_heating_share", "gas_heating_share",
          "district_heating_share", "n_heating_sources", "has_heat_pump_heating", "old_appliance_count", "has_portable_heater",
          "has_electric_water_heater", "has_air_conditioning", "incandescent_level"],
    "R": ["region", "area_type"],
    "C": ["hdd_hs1617", "cdd_s2017", "heatwave_days", "tropical_nights", "cold_spell_max", "hdd_normal"],
}
BLOCK_NAME = {"I": "Income", "H": "Household", "D": "Dwelling fabric", "S": "Heating system & fuel", "R": "Region", "C": "Climate"}
BASE = ["country"]
NOMINAL = {"country", "region", "H1", "employment_status", "heating_strategy", "primary_heating_fuel"}
CLIMATE_VARS = BLOCKS["C"]
ORDER = ["I", "H", "D", "S", "R", "C"]


def feats(blocks) -> list[str]:
    return BASE + [f for b in ORDER if b in blocks for f in BLOCKS[b]]


def model_sets() -> dict[str, tuple[str, ...]]:
    s = {}
    for k in range(1, 7):
        s["+".join(ORDER[:k])] = tuple(ORDER[:k])
    for b in ORDER:
        if b != "C":
            s["ALL-" + b] = tuple(x for x in ORDER if x != b)
    s["ALL_placebo"] = tuple(ORDER)
    return s


def labels(dev: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for r, (fam, tau) in RSPEC.items():
        e = make_labels(dev, fam, 3, tau)
        e = e[e.country.isin(COUNTRIES)].reset_index(drop=True)
        out[r] = e
    return out


def placebo(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Each climate region gets the climate of another climate region of the same country (derangement)."""
    rng = np.random.default_rng(seed)
    f = frame.copy()
    for c, g in f.groupby("country"):
        regs = np.array(sorted(g.climate_region.dropna().unique()))
        if len(regs) < 2:
            continue
        while True:
            perm = rng.permutation(len(regs))
            if not np.any(perm == np.arange(len(regs))):
                break
        clim = g.groupby("climate_region")[CLIMATE_VARS].first()
        mapping = dict(zip(regs, regs[perm]))
        src = g.climate_region.map(mapping)
        f.loc[g.index, CLIMATE_VARS] = clim.reindex(src.to_numpy()).to_numpy()
    return f
