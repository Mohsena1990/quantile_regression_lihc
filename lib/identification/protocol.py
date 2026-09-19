"""Repeated-CV identification protocol with honest uncertainty.

Why this replaces "one validation split, pick the best config": per-country validation sets are
63-164 households; a 4-class macro-F1 on 63 households with a class of 1-7 members has a standard
error larger than the gaps used to pick winners. Here every number comes with a household-level
bootstrap CI over repeat-averaged out-of-fold predictions, next to a chance baseline (predictions
drawn from the training class prior), so "F1 = 0.40" reads as "chance is 0.25, so the lift is 0.15".

Models (all cost-sensitive / class-balanced, small, because n ~ 300-650 per country):
  prior   chance floor: class drawn at random from the training prior
  lr4     multinomial logistic regression, one-hot nominals, standardised numerics
  axis2   two binary logistic regressions (income axis, cost axis); 4-class probability is their
          product. Matches how the label is built and pools strength across the 4 cells
  cb4     shallow CatBoost multiclass
Scopes:
  local   one country's own households
  pooled  all countries, country as a feature, scored per country (borrows strength)
  loco    leave-one-country-out: train on the other four, score the target (pure transfer)
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .data import NOMINAL

MODELS = ("prior", "lr4", "axis2", "cb4")
SEED = 20260811
LABELS = [0, 1, 2, 3]


def _design(frame: pd.DataFrame, feats: list[str]) -> ColumnTransformer:
    cat = [f for f in feats if f in NOMINAL or f == "country" or not pd.api.types.is_numeric_dtype(frame[f])]
    num = [f for f in feats if f not in cat]
    parts = []
    if num:
        parts.append(("n", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()), num))
    if cat:
        parts.append(("c", make_pipeline(SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                                         OneHotEncoder(handle_unknown="ignore")), cat))
    return ColumnTransformer(parts)


def _prep(frame: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    x = frame[feats].copy()
    for f in feats:
        if f in NOMINAL or f == "country":
            x[f] = x[f].astype("object")
    return x


def _lr(frame, feats, C=0.1):
    return make_pipeline(_design(frame, feats), LogisticRegression(C=C, max_iter=4000, class_weight="balanced"))


def _proba4(model, x, classes) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.zeros((len(x), 4))
    out[:, np.asarray(classes, int)] = raw
    return out


def fit_predict_proba(kind: str, train: pd.DataFrame, test: pd.DataFrame, feats: list[str], *, seed: int = SEED) -> np.ndarray:
    """(len(test), 4) class probabilities for the four-class label `y`."""
    ytr = train["y"].astype(int)
    if kind == "prior":
        # Chance floor: each household is assigned a class drawn from the training class prior
        # (a constant-argmax "majority class" floor would be lower still, ~0.14 macro-F1, and
        # would flatter every model).
        prior = ytr.value_counts(normalize=True).reindex(LABELS).fillna(0).to_numpy()
        draw = np.random.default_rng(seed + len(test)).choice(4, size=len(test), p=prior / prior.sum())
        return np.eye(4)[draw]
    xtr, xte = _prep(train, feats), _prep(test, feats)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "lr4":
            m = _lr(xtr, feats).fit(xtr, ytr)
            return _proba4(m, xte, m.classes_)
        if kind == "axis2":
            li_m = _lr(xtr, feats).fit(xtr, train["li"].astype(int)) if train["li"].nunique() > 1 else None
            hc_m = _lr(xtr, feats).fit(xtr, train["hc"].astype(int)) if train["hc"].nunique() > 1 else None
            p_li = li_m.predict_proba(xte)[:, 1] if li_m else np.full(len(test), float(train["li"].mean()))
            p_hc = hc_m.predict_proba(xte)[:, 1] if hc_m else np.full(len(test), float(train["hc"].mean()))
            return np.column_stack([(1 - p_li) * (1 - p_hc), p_li * (1 - p_hc), (1 - p_li) * p_hc, p_li * p_hc])
        if kind == "cb4":
            cat = [f for f in feats if f in NOMINAL or f == "country" or not pd.api.types.is_numeric_dtype(xtr[f])]
            a, b = xtr.copy(), xte.copy()
            for f in feats:
                if f in cat:
                    a[f], b[f] = a[f].fillna("NA").astype(str), b[f].fillna("NA").astype(str)
                else:
                    med = a[f].median()
                    a[f], b[f] = a[f].fillna(med), b[f].fillna(med)
            m = CatBoostClassifier(iterations=100, depth=4, learning_rate=0.05, l2_leaf_reg=10, loss_function="MultiClass",
                                   auto_class_weights="Balanced", verbose=False, random_seed=seed, thread_count=4)
            m.fit(a, ytr, cat_features=cat)
            return _proba4(m, b, m.classes_)
    raise ValueError(f"unknown model {kind}")


def _strata(frame: pd.DataFrame) -> pd.Series:
    """Fold-stratification key that does not depend on threshold-derived labels."""
    med = frame.groupby("country")["cost"].transform("median")
    return frame["country"].astype(str) + "_" + frame["li"].astype(str) + "_" + (frame["cost"] > med).astype(int).astype(str)


def oof_proba(frame: pd.DataFrame, feats: list[str], kind: str, *, n_splits: int = 5, n_repeats: int = 3, seed: int = 1) -> np.ndarray:
    """(n_repeats, len(frame), 4) out-of-fold probabilities. `frame` is one country ('local') or
    all countries ('pooled': `feats` should then include 'country')."""
    frame = frame.reset_index(drop=True)
    out = np.zeros((n_repeats, len(frame), 4))
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for i, (tr, te) in enumerate(cv.split(frame, _strata(frame))):
        r = i // n_splits
        out[r, te] = fit_predict_proba(kind, frame.iloc[tr], frame.iloc[te], feats)
    return out


def loco_proba(frame: pd.DataFrame, target: str, feats: list[str], kind: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Train on every other country, score `target`. Returns (probabilities, target rows)."""
    frame = frame.reset_index(drop=True)
    train, test = frame[frame.country != target], frame[frame.country == target]
    return fit_predict_proba(kind, train, test, feats), test


def summarise(y: np.ndarray, proba: np.ndarray, *, n_boot: int = 200, seed: int = 3) -> dict:
    """Full metric suite (accuracy, precision, recall, F1, specificity, ROC-AUC, PR-AUC, MCC, kappa, log-loss,
    Brier, per class and per axis) with household-bootstrap 95% CIs, averaged over the CV repeats.
    See lib/identification/metrics.py."""
    from .metrics import summarise_all
    return summarise_all(y, proba, n_boot=n_boot, seed=seed)


def binary_axis_auc_cv(frame: pd.DataFrame, feats: list[str], axis: str, *, n_splits: int = 5, n_repeats: int = 3,
                       seed: int = 1, per_fold: bool = False):
    """Repeated-CV ROC-AUC of a logistic model for one binary axis ('li' or 'hc'). Identical folds for
    identical (frame, seed), so two feature sets can be compared fold-by-fold (`per_fold=True`)."""
    frame = frame.reset_index(drop=True)
    y = frame[axis].astype(int)
    if y.sum() < 10 or (1 - y).sum() < 10:
        return np.array([]) if per_fold else np.nan
    x = _prep(frame, feats)
    aucs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, te in RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed).split(frame, y):
            m = _lr(x, feats).fit(x.iloc[tr], y.iloc[tr])
            aucs.append(roc_auc_score(y.iloc[te], m.predict_proba(x.iloc[te])[:, 1]))
    return np.asarray(aucs) if per_fold else float(np.mean(aucs))
