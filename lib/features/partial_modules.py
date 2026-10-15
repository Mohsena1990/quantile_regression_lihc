"""Survey modules asked in only some countries (version 5, B1 group "partial_modules").

These questions are missing for whole countries, so earlier versions left them out. In the pooled model a household from a country
where the question was not asked simply has a missing value; CatBoost and XGBoost route missing values natively, and the
FT-Transformer fills them with the training median (a constant, so it carries no signal by itself beyond "country").

  env_attitude_index    H15 environmental attitudes (asked in Hungary, Italy, Poland, Spain; not Bulgaria). Mean of the
                        pro-environment answers on the 1-4 agreement scale: D (willing to compromise my lifestyle) and G
                        (protecting the environment stimulates growth) as asked, A, B, C and F reversed (5 - answer). E ("policies
                        should not cost me extra") is left out: it mixes attitude with money worry, which is the outcome's
                        territory. "Do not know" (99) -> missing; the index needs at least 3 answered items.
  energy_saving_index   G module energy-saving behaviour (asked in Bulgaria, Hungary, Poland; not Italy or Spain). Number of the
                        four energy-efficient choices mentioned in G1 (fuel-efficient car, car alternatives, efficient appliance,
                        green supplier) + number of "Yes" in G3 (uses renewable energy, electric/hybrid car, cuts CO2, improves
                        energy efficiency, cleaner vehicle). G3 "Do not apply" counts as no.
  knows_kwh_price       energy literacy, E1 "how much does 1 kWh cost?" (asked in Bulgaria only): 1 = gave a guess, 0 = "do not know".
  bill_too_complicated  energy literacy, C5C "my energy bill is too complicated, I cannot interpret it" (asked in Hungary and
                        Spain): 1 not at all .. 5 very much; "not applicable" (6) and "do not know" (99) -> missing.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

H15 = ["H15A", "H15B", "H15C", "H15D", "H15E", "H15F", "H15G"]
G1 = ["G1A1", "G1A2", "G1A3", "G1A4"]
G3 = ["G3A", "G3B", "G3C", "G3D", "G3E"]
RAW_PARTIAL = H15 + G1 + ["G1A5"] + G3 + ["E1", "C5C"]
PARTIAL = ["env_attitude_index", "energy_saving_index", "knows_kwh_price", "bill_too_complicated"]
PARTIAL_NAMES = {"env_attitude_index": "environmental attitude index (H15)", "energy_saving_index": "energy-saving behaviour index (G1, G3)",
                 "knows_kwh_price": "knows the price of 1 kWh (E1)", "bill_too_complicated": "energy bill too complicated (C5C)"}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def partial_modules(d: pd.DataFrame) -> pd.DataFrame:
    """Adds PARTIAL from the raw columns RAW_PARTIAL (missing for households of countries where a module was not asked)."""
    out = d.copy()
    h = pd.DataFrame({c: _num(d[c]).where(lambda s: s.between(1, 4)) for c in H15})
    pro = pd.concat([h.H15D, h.H15G] + [5 - h[c] for c in ("H15A", "H15B", "H15C", "H15F")], axis=1)
    out["env_attitude_index"] = pro.mean(axis=1).where(pro.notna().sum(axis=1) >= 3)
    g1 = pd.DataFrame({c: _num(d[c]) for c in G1 + ["G1A5"]})
    g3 = pd.DataFrame({c: _num(d[c]).where(lambda s: s.between(1, 3)) for c in G3})
    asked = g1.notna().any(axis=1) | g3.notna().any(axis=1)
    out["energy_saving_index"] = ((g1[G1] == 1).sum(axis=1) + (g3 == 1).sum(axis=1)).astype(float).where(asked)
    e1 = _num(d["E1"])
    out["knows_kwh_price"] = (e1 != 99).astype(float).where(e1.notna())
    out["bill_too_complicated"] = _num(d["C5C"]).where(lambda s: s.between(1, 5))
    return out
