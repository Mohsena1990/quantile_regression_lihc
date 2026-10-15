"""Version-5 labels: the version-4 line families, written as a per-household spending threshold in EUR.

Every family can be read as "high expenditure = the household's energy cost is above ITS threshold thr":
  pct            thr = the country's tau-percentile of energy cost (same for every household of a country)
  burden         thr = the country's tau-percentile of the energy burden (cost / income) x the household's income
  needqr         thr = Quantile Regression line at level tau on physical need (out of fold, per country; lines_v4._qr_line)
  needqr_noclim  the same without heating / cooling degree days
This gives exactly the version-4 flags (cost > thr) and adds what version 5 needs:
  * the threshold itself, so the "predict the bill, then classify" model can compare a predicted bill with it (idea 1), and
    so households close to their line can be down-weighted (idea 4);
  * `fold_threshold`, the same lines refitted on the training households of a fold only and applied to other households
    (test households, and households whose heating cost is missing -- idea 2).
Thresholds for the version-5 cohort (four countries + Spain) come from scripts/build_threshold_moo.py --v5
(artifacts/identification/moo_thresholds_v5.json).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from .lines_v4 import NEED_CLIM, NEED_NOCLIM, _qr_line, eligible

ROOT = Path(__file__).resolve().parents[2]
MOO_PATH_V5 = ROOT / "artifacts/identification/moo_thresholds_v5.json"


def _feats(family: str) -> list[str]:
    return NEED_CLIM if family == "needqr" else NEED_NOCLIM


def expenditure_threshold(e: pd.DataFrame, family: str, tau: float, *, folds: int = 5, seed: int = 7) -> pd.Series:
    """Per-household threshold (EUR) for the eligible households `e`; high expenditure = cost > threshold."""
    thr = pd.Series(np.nan, index=e.index)
    for _, g in e.groupby("country"):
        if family == "pct":
            thr[g.index] = g.cost.quantile(tau)
        elif family == "burden":
            thr[g.index] = (g.cost / g.income_eur_total).quantile(tau) * g.income_eur_total
        else:
            g = g.reset_index()
            t = np.full(len(g), np.nan)
            for tr, te in KFold(folds, shuffle=True, random_state=seed).split(g):
                t[te] = _qr_line(g.iloc[tr], g.iloc[te], _feats(family), tau)
            thr[g["index"].to_numpy()] = t
    return thr


def fold_threshold(train: pd.DataFrame, target: pd.DataFrame, family: str, tau: float) -> np.ndarray:
    """Threshold for the `target` households, with every quantity estimated on the (labelled) `train` households only."""
    out = np.full(len(target), np.nan)
    country = target["country"].to_numpy()
    for c, g in train.groupby("country"):
        m = country == c
        if not m.any():
            continue
        t = target[m]
        if family == "pct":
            out[m] = g.cost.quantile(tau)
        elif family == "burden":
            out[m] = (g.cost / g.income_eur_total).quantile(tau) * t.income_eur_total.to_numpy()
        else:
            out[m] = _qr_line(g, t, _feats(family), tau)
    return out


def make_labels(dev: pd.DataFrame, family: str, cutoff: int, tau: float, thr: pd.Series | None = None) -> pd.DataFrame:
    """Eligible households with thr, hc, li, y (y = low income + 2 x high expenditure)."""
    e = eligible(dev, family)
    e["thr"] = expenditure_threshold(e, family, tau) if thr is None else thr.reindex(e.index)
    e["hc"] = (e.cost > e.thr).astype(int)
    e["li"] = (e.income_bracket <= cutoff).astype(int)
    e["y"] = e.li + 2 * e.hc
    return e.reset_index(drop=True)


def unlabelled(dev: pd.DataFrame, family: str, cutoff: int) -> pd.DataFrame:
    """Households with a valid income answer but no energy cost: their income axis is known, their spending axis is not.
    Used only as extra (imputed) TRAINING households, never scored."""
    ok = dev.cost.isna() & dev.income_bracket.between(1, 10)
    if family == "burden":
        ok &= dev.income_eur_total.notna() & (dev.income_eur_total > 0)
    u = dev[ok].copy()
    u["li"] = (u.income_bracket <= cutoff).astype(int)
    return u.reset_index(drop=True)


def cost_only(dev: pd.DataFrame) -> pd.DataFrame:
    """Households with an energy cost but no usable income answer: they cannot be labelled, but they still inform a model of
    the energy bill ("keep households with partial answers for the parts they can inform")."""
    return dev[dev.cost.notna() & ~dev.income_bracket.between(1, 10)].reset_index(drop=True)


def chosen_thresholds_v5() -> dict:
    if not MOO_PATH_V5.exists():
        raise SystemExit("version-5 threshold optimisation missing: run scripts/build_threshold_moo.py --v5 first")
    return json.loads(MOO_PATH_V5.read_text())


def study_labels_v5(dev: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, dict]]:
    """label -> (labelled frame, spec {family, cutoff, tau}). Van Hove's benchmark + the optimised lines, as in version 4."""
    ch = chosen_thresholds_v5()
    specs = {"vh": ("pct", 3, 0.80)}
    for fam in ("pct", "needqr", "burden"):
        if fam == "pct" and (ch[fam]["cutoff"], ch[fam]["tau"]) == (3, 0.80):
            continue
        specs[fam] = (fam, ch[fam]["cutoff"], ch[fam]["tau"])
    specs["needqr_noclim"] = ("needqr_noclim", ch["needqr"]["cutoff"], ch["needqr"]["tau"])
    return {k: (make_labels(dev, f, c, t), {"family": f, "cutoff": c, "tau": t}) for k, (f, c, t) in specs.items()}
