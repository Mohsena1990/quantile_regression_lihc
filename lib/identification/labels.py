"""Four-class labels with a partition-stable income axis and two cost-axis definitions.

Income axis (`income_rule`, see lib/labels/income_harmonization.py):
  bracket3 (HEADLINE)  Van Hove's fixed rule: income bracket <= 3
  percentile           frozen survey-weighted empirical bracket percentile <= 0.30   (sensitivity)
  equiv60 / residual60 DYNAMIC household-size-adjusted / residual-after-energy-cost rules (sensitivity)
Cost axis (`high_cost`):
  raw_q80        Van Hove / M0: cost above the country's 80th percentile of raw energy cost.
                 Mostly measures dwelling *need* (big homes cost more), so it is learnable from
                 need variables but says little about over-spending.
  need_adjusted  cost above the country's conditional q-quantile of log-cost given physical NEED
                 only (floor area, household size, dwelling type/age, HDD/CDD) -- "spends more than
                 comparable homes". Deliberately does NOT condition on fuel, respondent age,
                 appliances or socio-economics, so those remain genuinely informative for a
                 classifier (conditioning the threshold on everything the classifier sees makes the
                 label independent of its inputs by construction: pooled AUC 0.51 on the previous
                 R1-R3 labels).
Thresholds for need_adjusted are out-of-fold (each household's threshold comes from a per-country
quantile regression that never saw that household).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lib.labels.income_harmonization import (HEADLINE_INCOME_RULE, INCOME_RULES, equivalence_scale,
                                             income_proxy_frame, low_income_flag)
from .data import NEED_FEATURES

COST_KINDS = ("raw_q80", "need_adjusted")
DEFAULT_Q = 0.63          # the M2 champion's OOF-calibrated pooled quantile (table_family_champions_final)
LOW_INCOME_BOUNDARY = 0.30
CLASS_NAMES = {0: "No Risk", 1: "Income Risk", 2: "Expenditure Risk", 3: "Double Risk"}


def _need_design(frame: pd.DataFrame) -> ColumnTransformer:
    num = [f for f in NEED_FEATURES if f != "H1"]
    return ColumnTransformer([
        ("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("c", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore", drop="first")), ["H1"])])


def _need_threshold_oof(g: pd.DataFrame, q: float, *, folds: int, seed: int) -> np.ndarray:
    """Out-of-fold conditional q-quantile (in EUR) of cost given need, for one country."""
    x = g[NEED_FEATURES].copy()
    x["H1"] = x["H1"].astype("object")
    logc = np.log(g["cost"].where(g["cost"] > 0))
    thr = np.full(len(g), np.nan)
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(g):
        fit = logc.iloc[tr].notna().to_numpy()
        model = make_pipeline(_need_design(g), QuantileRegressor(quantile=q, alpha=0.001, solver="highs"))
        model.fit(x.iloc[tr][fit], logc.iloc[tr][fit])
        thr[te] = np.exp(model.predict(x.iloc[te]))
    return thr


def add_labels(dev: pd.DataFrame, *, cost_kind: str, income_rule: str = HEADLINE_INCOME_RULE, q: float = DEFAULT_Q,
               boundary: float = LOW_INCOME_BOUNDARY, folds: int = 5, seed: int = 7) -> pd.DataFrame:
    """Return the label-eligible households with columns li, hc, y (0 No / 1 Income / 2 Expenditure /
    3 Double), threshold, income_rule. Eligible = valid cost and a valid income bracket (plus, for the
    dynamic rules, a known household composition)."""
    if cost_kind not in COST_KINDS:
        raise ValueError(f"cost_kind must be one of {COST_KINDS}")
    if income_rule not in INCOME_RULES:
        raise ValueError(f"income_rule must be one of {INCOME_RULES}")
    d = dev.copy()
    if income_rule in ("equiv60", "residual60"):
        scale = equivalence_scale(d["working_age_count"] + d["elderly_count"], d["children_count"])
        prox = income_proxy_frame(d["income_bracket"], d["country"], scale, cost=d["cost"])
        d["li_raw"] = prox[f"low_income_{income_rule}"]
        ok_income = d["li_raw"].notna()
    else:
        ok_income = d["income_percentile"].notna()          # valid bracket 1-10 and mapped
        if income_rule == "bracket3":
            d["li_raw"] = low_income_flag(d["income_bracket"], d["country"], rule="bracket3")
        else:
            d["li_raw"] = low_income_flag(d["income_bracket"], d["country"], rule="percentile", percentile_boundary=boundary)
    e = d[d["cost"].notna() & ok_income].copy().reset_index(drop=True)
    e["li"] = e["li_raw"].astype(int)
    e["threshold"] = np.nan
    for country, idx in e.groupby("country").groups.items():
        g = e.loc[idx]
        if cost_kind == "raw_q80":
            e.loc[idx, "threshold"] = g["cost"].quantile(0.8)
        else:
            e.loc[idx, "threshold"] = _need_threshold_oof(g, q, folds=folds, seed=seed)
    e["hc"] = (e["cost"] > e["threshold"]).astype(int)
    e["y"] = e["li"] + 2 * e["hc"]
    e["cost_kind"] = cost_kind
    e["income_rule"] = income_rule
    return e


def support_table(labelled: pd.DataFrame) -> pd.DataFrame:
    """Households per country x four-class cell, plus the two axes' positive counts."""
    t = pd.crosstab(labelled["country"], labelled["y"]).rename(columns=CLASS_NAMES)
    t["n"] = t.sum(axis=1)
    t["low_income_n"] = labelled.groupby("country")["li"].sum()
    t["high_cost_n"] = labelled.groupby("country")["hc"].sum()
    return t.reset_index()


def viability(support: pd.DataFrame, *, min_per_class: int = 30, min_axis: int = 30) -> pd.DataFrame:
    """Which claims a country's sample can carry.
      four_class_ok  every one of the 4 classes has >= min_per_class households
      axis_ok        both binary axes have >= min_axis positives AND >= min_axis negatives
    A country failing four_class_ok is evaluated on the two binary axes / pooled model only --
    a 4-class macro-F1 over a cell with a handful of households is noise, not evidence."""
    s = support.copy()
    classes = list(CLASS_NAMES.values())
    s["min_class_n"] = s[classes].min(axis=1)
    s["four_class_ok"] = s["min_class_n"] >= min_per_class
    s["axis_ok"] = ((s["low_income_n"] >= min_axis) & (s["n"] - s["low_income_n"] >= min_axis)
                    & (s["high_cost_n"] >= min_axis) & (s["n"] - s["high_cost_n"] >= min_axis))
    return s
