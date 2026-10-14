"""Version-4 labels: four expenditure-line families, each with an adjustable income cut-off c and expenditure level tau.

  pct            Van Hove type: energy expenditure above the country's tau-percentile
  needqr         Quantile Regression on physical need (floor area, household size, dwelling type and age, insulation, heating
                 and cooling degree days at NUTS3) at level tau, out of fold, per country
  needqr_noclim  the same without heating / cooling degree days -- the fair test of climate predictors (the label does not
                 already contain climate)
  burden         energy burden (Dalla Longa et al. 2021): energy expenditure / household income (EUR, from the income bracket via
                 Eurostat deciles) above the country's tau-percentile
Low income = income bracket <= c (Van Hove: c = 3). Four classes: y = low income + 2 x high expenditure.
Van Hove's benchmark is pct with c = 3, tau = 0.80. The chosen (c, tau) per family come from scripts/build_threshold_moo.py
(artifacts/identification/moo_thresholds.json).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("pct", "needqr", "needqr_noclim", "burden")
NEED_CLIM = ["H3", "household_size", "H1", "H2", "insulation_count", "hdd_hs1617", "cdd_s2017"]
NEED_NOCLIM = ["H3", "household_size", "H1", "H2", "insulation_count"]
MOO_PATH = ROOT / "artifacts/identification/moo_thresholds.json"
LABEL_NAMES = {"vh": "Van Hove (bracket 3, 80th percentile)", "pct": "Country percentile (optimised)",
               "needqr": "Quantile Regression on need (optimised)", "needqr_noclim": "Quantile Regression on need, no climate",
               "burden": "Energy burden (optimised)"}


def _qr_line(train: pd.DataFrame, test: pd.DataFrame, feats: list[str], tau: float) -> np.ndarray:
    from measurement.families.model_based.estimators import fit_family
    def prep(g):
        x = g[feats + ["cost", "country"]].copy()
        x["H1"] = x["H1"].map(lambda v: f"type_{int(v)}" if pd.notna(v) else np.nan)
        return x
    model, _, _ = fit_family("M1", prep(train), prep(train), features=feats, target="cost", quantile=float(tau))
    return np.maximum(model.predict(prep(test)[feats]), 0.0)


def expenditure_flag(e: pd.DataFrame, family: str, tau: float, *, folds: int = 5, seed: int = 7) -> pd.Series:
    """High-expenditure flag (0/1) for the eligible households `e` (one or more countries)."""
    hc = pd.Series(np.nan, index=e.index)
    for _, g in e.groupby("country"):
        if family == "pct":
            hc[g.index] = (g.cost > g.cost.quantile(tau)).astype(float)
        elif family == "burden":
            share = g.cost / g.income_eur_total
            hc[g.index] = (share > share.quantile(tau)).astype(float)
        else:
            feats = NEED_CLIM if family == "needqr" else NEED_NOCLIM
            g = g.reset_index()
            thr = np.full(len(g), np.nan)
            for tr, te in KFold(folds, shuffle=True, random_state=seed).split(g):
                thr[te] = _qr_line(g.iloc[tr], g.iloc[te], feats, tau)
            hc[g["index"].to_numpy()] = (g.cost.to_numpy() > thr).astype(float)
    return hc


def eligible(dev: pd.DataFrame, family: str) -> pd.DataFrame:
    ok = dev.cost.notna() & dev.income_bracket.between(1, 10)
    if family == "burden":
        ok &= dev.income_eur_total.notna() & (dev.income_eur_total > 0)
    return dev[ok].copy()


def make_labels(dev: pd.DataFrame, family: str, cutoff: int, tau: float, hc: pd.Series | None = None) -> pd.DataFrame:
    """Eligible households with li, hc, y. Pass a precomputed `hc` (same index as eligible(dev, family)) to reuse it."""
    e = eligible(dev, family)
    e["hc"] = (expenditure_flag(e, family, tau) if hc is None else hc.reindex(e.index)).astype(int)
    e["li"] = (e.income_bracket <= cutoff).astype(int)
    e["y"] = e.li + 2 * e.hc
    return e.reset_index(drop=True)


def chosen_thresholds() -> dict:
    if not MOO_PATH.exists():
        raise SystemExit("threshold optimisation missing: run scripts/build_threshold_moo.py first")
    return json.loads(MOO_PATH.read_text())


def study_labels(dev: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The five version-4 label sets: Van Hove benchmark + the four families at their optimised thresholds (the climate-free
    need line uses the need line's thresholds)."""
    ch = chosen_thresholds()
    out = {"vh": make_labels(dev, "pct", 3, 0.80)}
    for fam in ("pct", "needqr", "burden"):
        if fam == "pct" and (ch[fam]["cutoff"], ch[fam]["tau"]) == (3, 0.80):
            continue                                # the optimised country-percentile line IS Van Hove's
        out[fam] = make_labels(dev, fam, ch[fam]["cutoff"], ch[fam]["tau"])
    out["needqr_noclim"] = make_labels(dev, "needqr_noclim", ch["needqr"]["cutoff"], ch["needqr"]["tau"])
    return out
