"""The extra training countries in pooled models: their four-group labels and the fixed feature sets used when comparing
training sets (scripts/build_pooled_extra.py, build_bill_climate.py, build_climate_power.py, build_loco_validation.py).

Every line is computed within each extra country with the same rule and thresholds as for the study countries. The need lines
need climate (France and the UK have no regions, so they do not enter the line WITH climate); the burden line needs income in EUR
(Ukraine has no Eurostat income deciles). A country enters a line only with at least 50 eligible households and all four groups.
"""
from __future__ import annotations
import pandas as pd
from .lines import eligible, make_labels
from .stages import B0_FEATURES

B0 = B0_FEATURES + ["country"]
B1 = B0 + ["region", "wealth_index", "vulnerability_index", "income_eur_equiv"]      # the B1 variables chosen most often
N_REPEATS = 3


def extra_labels(extra: pd.DataFrame, spec: dict) -> pd.DataFrame:
    e = extra[extra.hdd_hs1617.notna()] if spec["family"] == "needqr" else extra
    out = []
    for _, g in e.groupby("country"):
        if len(eligible(g, spec["family"])) < 50:
            continue
        lab = make_labels(g, spec["family"], spec["cutoff"], spec["tau"])
        if lab.y.nunique() == 4:
            out.append(lab)
    return pd.concat(out, ignore_index=True)
