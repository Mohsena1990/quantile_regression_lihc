"""The expenditure line ("high energy cost" threshold) -- three methods, plus Van Hove's benchmark.

  Van Hove                 one line per country: the 80th percentile of observed annual energy expenditure
  Quantile Regression      a line per household: the conditional 80th percentile of expenditure given physical need
                           (linear quantile regression, Koenker & Bassett 1978)
  Hansen Threshold         one line per country at the expenditure TIPPING POINT found by Hansen's Threshold Regression
                           (lib/identification/hansen.py): the spending level above which low-income households' reported
                           hardship jumps
  Hansen + Quantile Reg.   the integrated baseline: Hansen's tipping point, measured as a position among homes with the
                           same physical need, becomes the quantile level tau of the Quantile Regression line

Physical need (NEED_V2): floor area, household size, dwelling type, dwelling age, number of insulation types, heating degree
days of the survey's heating season 2016/17 and cooling degree days of summer 2017 at the household's NUTS3 region.

Tipping points. Among low-income households (income bracket <= 3) of a country, hardship = 1 if the household finds it
"difficult" or "very difficult" to live on present income (S8 >= 3). Hansen's Threshold Regression of hardship on the household's
spending position q (intercept and slope change at the tipping point; income bracket and household size as common controls):
  * q = percentile of observed expenditure within the country            -> gamma_u, the Hansen Threshold line
  * q = conditional rank: share of the household's need-specific quantile lines (tau = 0.05 ... 0.95, out of fold) that its
        expenditure exceeds                                               -> gamma_c, the tau of Hansen + Quantile Regression
A country's own tipping point is used when Hansen's bootstrap test finds one (p < 0.10) AND hardship is higher above it than
below (a tipping point INTO hardship); otherwise the four countries' pooled tipping point (country fixed effects) is used. The
pooled point is used even when it is not significant -- the table and README then say so. Lines are fitted out of fold, so no household's line is estimated on its own cost.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
NEED_V2 = ["H3", "household_size", "H1", "H2", "insulation_count", "hdd_hs1617", "cdd_s2017"]
QR_TAU = 0.80
TAU_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)
P_COUNTRY = 0.10
TIPPING_PATH = ROOT / "artifacts/identification/hansen_tipping_points.json"


def _need_frame(g: pd.DataFrame) -> pd.DataFrame:
    x = g[NEED_V2 + ["cost", "country"]].copy()
    x["H1"] = x["H1"].map(lambda v: f"type_{int(v)}" if pd.notna(v) else np.nan)
    return x


def fit_qr_line(train: pd.DataFrame, tau: float):
    """Linear quantile regression of annual expenditure (EUR) on physical need at level tau. Returns predict(frame)."""
    from measurement.families.model_based.estimators import fit_family
    tr = _need_frame(train)
    model, _, _ = fit_family("M1", tr, tr, features=NEED_V2, target="cost", quantile=float(tau))
    return lambda frame: np.maximum(model.predict(_need_frame(frame)[NEED_V2]), 0.0)


def qr_line_oof(g: pd.DataFrame, tau: float, *, folds: int = 5, seed: int = 7) -> np.ndarray:
    out = np.full(len(g), np.nan)
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(g):
        out[te] = fit_qr_line(g.iloc[tr], tau)(g.iloc[te])
    return out


def conditional_rank_oof(g: pd.DataFrame, *, folds: int = 5, seed: int = 7) -> np.ndarray:
    """Share of the household's out-of-fold need-specific quantile lines (TAU_GRID) that its expenditure exceeds (0..1)."""
    g = g.reset_index(drop=True)
    above = np.zeros(len(g))
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(g):
        cost = g.cost.to_numpy()[te]
        for tau in TAU_GRID:
            above[te] += cost > fit_qr_line(g.iloc[tr], tau)(g.iloc[te])
    return above / len(TAU_GRID)


def load_tipping_points() -> dict:
    if not TIPPING_PATH.exists():
        raise SystemExit("Hansen tipping points missing: run scripts/build_expenditure_lines.py first")
    return json.loads(TIPPING_PATH.read_text())


def chosen_gamma(tp: dict, country: str, kind: str) -> float:
    """kind 'unconditional' or 'conditional'; the country's own tipping point if significant, else the pooled one."""
    c = tp[kind]["countries"][country]
    own = c["p_value"] < P_COUNTRY and c["hardship_above"] > c["hardship_below"]
    return float(c["gamma"] if own else tp[kind]["pooled"]["gamma"])
