"""Idea 4 -- treat borderline and likely mislabelled households as uncertain (training weights only; scoring is unchanged).

Many households sit just above or just below their spending line, and self-reported bills are rounded and noisy, so their label
could easily have been the other one. Two down-weights, multiplied:

  borderline   w = 0.5 + 0.5 x min(1, |log(cost / thr)| / 0.2)
               a household exactly on its line counts half; one 20% or more away from it counts fully.
  confident learning (Northcutt, Jiang & Chuang 2021, "Confident learning: estimating uncertainty in dataset labels", JAIR 70)
               out-of-fold class probabilities on the training households (CatBoost, 3 folds); per-class confidence threshold
               t_j = mean probability of class j among households labelled j. A household labelled k is a likely label error when
               p_k < t_k while the class with the other spending side (same income side, k XOR 2) has p >= its threshold -- the
               income axis is known exactly, so only the spending side can be wrong. With classifiers this weak (balanced accuracy
               about 0.55) that rule alone flags about 38% of households -- a sign of a weak model, not of 38% wrong labels -- so a
               household counts as a likely error only if, in addition, the model prefers the other spending side by at least
               CL_MARGIN (about 3% of households). Likely errors get weight 0.3. Both shares are recorded.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from .models import fit as fit_model

BORDER_DELTA, BORDER_FLOOR, CL_WEIGHT, CL_MARGIN = 0.2, 0.5, 0.3, 0.15


def borderline_weight(cost: np.ndarray, thr: np.ndarray) -> np.ndarray:
    d = np.abs(np.log(np.clip(np.asarray(cost, float), 1.0, None) / np.clip(np.asarray(thr, float), 1.0, None)))
    return BORDER_FLOOR + (1 - BORDER_FLOOR) * np.minimum(1.0, d / BORDER_DELTA)


def _oof(train: pd.DataFrame, feats: list[str], seed: int) -> tuple[np.ndarray, np.ndarray]:
    train = train.reset_index(drop=True)
    y = train["y"].astype(int).to_numpy()
    p = np.zeros((len(y), 4))
    for i, (tr, te) in enumerate(StratifiedKFold(3, shuffle=True, random_state=seed).split(train, y)):
        p[te] = fit_model("catboost", train.iloc[tr], feats, seed + i).predict_proba(train.iloc[te])
    return y, p


def _flags(y: np.ndarray, p: np.ndarray, margin: float) -> np.ndarray:
    t = np.array([p[y == j, j].mean() if (y == j).any() else 1.0 for j in range(4)])
    alt, idx = y ^ 2, np.arange(len(y))
    return (p[idx, y] < t[y]) & (p[idx, alt] >= t[alt]) & (p[idx, alt] - p[idx, y] >= margin)


def confident_learning_flags(train: pd.DataFrame, feats: list[str], seed: int, margin: float = CL_MARGIN) -> np.ndarray:
    return _flags(*_oof(train, feats, seed), margin)


def robust_weight(train: pd.DataFrame, feats: list[str], seed: int) -> tuple[np.ndarray, dict]:
    """Training weights and a summary (share flagged by the conservative rule and by the plain rule, mean weights)."""
    wb = borderline_weight(train.cost.to_numpy(), train.thr.to_numpy())
    y, p = _oof(train, feats, seed)
    flag = _flags(y, p, CL_MARGIN)
    w = wb * np.where(flag, CL_WEIGHT, 1.0)
    return w, {"cl_flagged_share": float(flag.mean()), "cl_plain_rule_share": float(_flags(y, p, -1.0).mean()),
               "border_mean_weight": float(wb.mean()), "mean_weight": float(w.mean())}
