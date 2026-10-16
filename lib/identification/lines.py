"""Who counts as energy poor: four groups from an income cut-off and an expenditure line, each line a threshold in EUR per household.

Four groups: y = low income + 2 x high expenditure (0 No Risk, 1 Income Risk, 2 Expenditure Risk, 3 Double Risk).
Low income = income bracket (1-10) <= cutoff. High expenditure = the household's annual energy cost > its threshold `thr`:
  p80 / pct      thr = the country's tau-percentile of energy cost (the same for every household of a country).
                 P80 = cutoff 3, tau 0.80: the study's BASELINE line (the definition of Van Hove et al. 2022).
  needqr         thr = linear quantile regression at level tau of cost on physical need -- floor area, household size, dwelling
                 type and age, insulation, heating and cooling degree days (out of fold, within each country)
  needqr_noclim  the same without the degree days -- the fair test for climate predictors
  burden         thr = the country's tau-percentile of the energy burden (cost / income in EUR) x the household's income
                 (the burden definition of Dalla Longa et al. 2021)
The thresholds (cutoff, tau) of the optimised lines come from scripts/build_threshold_moo.py (artifacts/identification/
moo_thresholds.json). `fold_threshold` refits a line on training households only, for the bill models and imputation.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
NEED_CLIM = ["H3", "household_size", "H1", "H2", "insulation_count", "hdd_hs1617", "cdd_s2017"]
NEED_NOCLIM = ["H3", "household_size", "H1", "H2", "insulation_count"]
MOO_PATH = ROOT / "artifacts/identification/moo_thresholds.json"
STUDY_LINES = ("p80", "needqr", "needqr_noclim", "burden")
CLASS_NAMES = {0: "No Risk", 1: "Income Risk", 2: "Expenditure Risk", 3: "Double Risk"}
QR_ALPHA = 0.001


def _feats(family: str) -> list[str]:
    return NEED_CLIM if family == "needqr" else NEED_NOCLIM


def quantile_line(train: pd.DataFrame, target: pd.DataFrame, feats: list[str], tau: float) -> np.ndarray:
    """Linear quantile regression of cost (median-imputed, standardised numbers; one-hot dwelling type), L1 penalty QR_ALPHA."""
    def prep(g):
        x = g[feats].copy()
        x["H1"] = x["H1"].map(lambda v: f"type_{int(v)}" if pd.notna(v) else np.nan)
        return x
    xa, xb = prep(train), prep(target)
    y = pd.to_numeric(train["cost"], errors="coerce")
    ok = y.notna() & (y > 0)
    cat = [f for f in feats if not pd.api.types.is_numeric_dtype(xa[f])]
    num = [f for f in feats if f not in cat]
    design = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
                                ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                                      OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")), cat)])
    model = make_pipeline(design, QuantileRegressor(quantile=float(tau), alpha=QR_ALPHA, solver="highs"))
    model.fit(xa.loc[ok], y.loc[ok])
    return np.maximum(np.asarray(model.predict(xb), float), 0.0)


def eligible(dev: pd.DataFrame, family: str) -> pd.DataFrame:
    ok = dev.cost.notna() & dev.income_bracket.between(1, 10)
    if family == "burden":
        ok &= dev.income_eur_total.notna() & (dev.income_eur_total > 0)
    return dev[ok].copy()


def expenditure_threshold(e: pd.DataFrame, family: str, tau: float, *, folds: int = 5, seed: int = 7) -> pd.Series:
    """Per-household threshold (EUR) for the eligible households `e`; high expenditure = cost > threshold."""
    thr = pd.Series(np.nan, index=e.index)
    for _, g in e.groupby("country"):
        if family in ("pct", "p80"):
            thr[g.index] = g.cost.quantile(tau)
        elif family == "burden":
            thr[g.index] = (g.cost / g.income_eur_total).quantile(tau) * g.income_eur_total
        else:
            g = g.reset_index()
            t = np.full(len(g), np.nan)
            for tr, te in KFold(folds, shuffle=True, random_state=seed).split(g):
                t[te] = quantile_line(g.iloc[tr], g.iloc[te], _feats(family), tau)
            thr[g["index"].to_numpy()] = t
    return thr


def fold_threshold(train: pd.DataFrame, target: pd.DataFrame, family: str, tau: float) -> np.ndarray:
    """Threshold for the `target` households, every quantity estimated on the (labelled) `train` households only."""
    out = np.full(len(target), np.nan)
    country = target["country"].to_numpy()
    for c, g in train.groupby("country"):
        m = country == c
        if not m.any():
            continue
        t = target[m]
        if family in ("pct", "p80"):
            out[m] = g.cost.quantile(tau)
        elif family == "burden":
            out[m] = (g.cost / g.income_eur_total).quantile(tau) * t.income_eur_total.to_numpy()
        else:
            out[m] = quantile_line(g, t, _feats(family), tau)
    return out


def make_labels(dev: pd.DataFrame, family: str, cutoff: int, tau: float, thr: pd.Series | None = None) -> pd.DataFrame:
    """Eligible households with thr, hc, li, y."""
    e = eligible(dev, family)
    e["thr"] = expenditure_threshold(e, family, tau) if thr is None else thr.reindex(e.index)
    e["hc"] = (e.cost > e.thr).astype(int)
    e["li"] = (e.income_bracket <= cutoff).astype(int)
    e["y"] = e.li + 2 * e.hc
    return e.reset_index(drop=True)


def unlabelled(dev: pd.DataFrame, family: str, cutoff: int) -> pd.DataFrame:
    """Households with a valid income answer but no energy cost (extra, imputed TRAINING households; never scored)."""
    ok = dev.cost.isna() & dev.income_bracket.between(1, 10)
    if family == "burden":
        ok &= dev.income_eur_total.notna() & (dev.income_eur_total > 0)
    u = dev[ok].copy()
    u["li"] = (u.income_bracket <= cutoff).astype(int)
    return u.reset_index(drop=True)


def cost_only(dev: pd.DataFrame) -> pd.DataFrame:
    """Households with an energy cost but no usable income answer: they inform a model of the bill, not the label."""
    return dev[dev.cost.notna() & ~dev.income_bracket.between(1, 10)].reset_index(drop=True)


def chosen_thresholds() -> dict:
    if not MOO_PATH.exists():
        raise SystemExit("threshold optimisation missing: run scripts/build_threshold_moo.py first")
    return json.loads(MOO_PATH.read_text())


def line_specs() -> dict[str, dict]:
    """The four study lines: the P80 baseline + the optimised need, climate-free need and burden lines."""
    ch = chosen_thresholds()
    return {"p80": {"family": "p80", "cutoff": 3, "tau": 0.80},
            "needqr": {"family": "needqr", "cutoff": ch["needqr"]["cutoff"], "tau": ch["needqr"]["tau"]},
            "needqr_noclim": {"family": "needqr_noclim", "cutoff": ch["needqr"]["cutoff"], "tau": ch["needqr"]["tau"]},
            "burden": {"family": "burden", "cutoff": ch["burden"]["cutoff"], "tau": ch["burden"]["tau"]}}


def study_labels(dev: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, dict]]:
    """line -> (labelled frame, spec {family, cutoff, tau})."""
    return {k: (make_labels(dev, s["family"], s["cutoff"], s["tau"]), s) for k, s in line_specs().items()}
