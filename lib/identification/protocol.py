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
from sklearn.model_selection import StratifiedKFold
from .data import NOMINAL
from .imbalance import (ALPHAS, N_BAGS, RESAMPLING, STRATEGIES, WEIGHT_STRATEGIES, alpha_of, catboost_weights, fit_class_scales,
                        lr_class_weight, resample_positions, weight_vector)

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


def _lr(frame, feats, C=0.1, class_weight="balanced"):
    return make_pipeline(_design(frame, feats), LogisticRegression(C=C, max_iter=4000, class_weight=class_weight))


def _proba4(model, x, classes) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.zeros((len(x), 4))
    out[:, np.asarray(classes, int)] = raw
    return out


def _inner_oof(kind, train, feats, seed, imbalance):
    ytr = train["y"].astype(int).to_numpy()
    inner = np.zeros((len(train), 4))
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=seed).split(train, ytr):
        inner[te] = fit_predict_proba(kind, train.iloc[tr], train.iloc[te], feats, seed=seed, imbalance=imbalance)
    return inner


def _alpha_tuned(kind, train, test, feats, seed):
    """Weight exponent chosen by inner 3-fold macro-F1 inside the training fold, then one fit at the chosen exponent."""
    ytr = train["y"].astype(int).to_numpy()
    if np.bincount(ytr, minlength=4).min() < 3:
        return fit_predict_proba(kind, train, test, feats, seed=seed, imbalance="balanced")
    from sklearn.metrics import f1_score
    scores = [f1_score(ytr, _inner_oof(kind, train, feats, seed, f"alpha={a}").argmax(1), average="macro", labels=LABELS, zero_division=0)
              for a in ALPHAS]
    return fit_predict_proba(kind, train, test, feats, seed=seed, imbalance=f"alpha={ALPHAS[int(np.argmax(scores))]}")


def _bagging(kind, train, test, feats, seed):
    """EasyEnsemble-style: average N_BAGS models, each fit on a fresh undersampled bag with balanced weights."""
    ytr = train["y"].astype(int).to_numpy()
    out = np.zeros((len(test), 4))
    for b in range(N_BAGS):
        bag = train.iloc[resample_positions(ytr, "rus", seed + 7919 * b)]
        out += fit_predict_proba(kind, bag, test, feats, seed=seed + b, imbalance="balanced")
    return out / N_BAGS


def _posthoc(kind, train, test, feats, seed, base_strategy="none"):
    """Decision-threshold moving: model (unweighted, or balanced for `balanced_posthoc`) + class scales learned on inner
    out-of-fold predictions of the training fold."""
    ytr = train["y"].astype(int).to_numpy()
    base = fit_predict_proba(kind, train, test, feats, seed=seed, imbalance=base_strategy)
    if np.bincount(ytr, minlength=4).min() < 3:
        return base
    scaled = base * fit_class_scales(_inner_oof(kind, train, feats, seed, base_strategy), ytr)
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_predict_proba(kind: str, train: pd.DataFrame, test: pd.DataFrame, feats: list[str], *, seed: int = SEED,
                      imbalance: str = "balanced") -> np.ndarray:
    """(len(test), 4) class probabilities for the four-class label `y`. `imbalance` selects the class-imbalance strategy
    (lib/identification/imbalance.py); the default reproduces the pipeline's original behaviour exactly."""
    if imbalance not in STRATEGIES and alpha_of(imbalance) is None:
        raise ValueError(f"unknown imbalance strategy {imbalance}")
    ytr = train["y"].astype(int)
    if kind == "prior":
        # Chance floor: each household is assigned a class drawn from the training class prior
        # (a constant-argmax "majority class" floor would be lower still, ~0.14 macro-F1, and
        # would flatter every model).
        prior = ytr.value_counts(normalize=True).reindex(LABELS).fillna(0).to_numpy()
        draw = np.random.default_rng(seed + len(test)).choice(4, size=len(test), p=prior / prior.sum())
        return np.eye(4)[draw]
    if imbalance in ("posthoc", "balanced_posthoc"):
        return _posthoc(kind, train, test, feats, seed, base_strategy="none" if imbalance == "posthoc" else "balanced")
    if imbalance == "alpha_tuned":
        return _alpha_tuned(kind, train, test, feats, seed)
    if imbalance == "bagging":
        return _bagging(kind, train, test, feats, seed)
    xtr, xte = _prep(train, feats), _prep(test, feats)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "axis2":
            if imbalance not in WEIGHT_STRATEGIES:
                raise ValueError(f"axis2 supports weight strategies only, not {imbalance}")
            def axis_model(col):
                y = train[col].astype(int)
                return _lr(xtr, feats, class_weight=lr_class_weight(imbalance, y)).fit(xtr, y) if y.nunique() > 1 else None
            li_m, hc_m = axis_model("li"), axis_model("hc")
            p_li = li_m.predict_proba(xte)[:, 1] if li_m else np.full(len(test), float(train["li"].mean()))
            p_hc = hc_m.predict_proba(xte)[:, 1] if hc_m else np.full(len(test), float(train["hc"].mean()))
            return np.column_stack([(1 - p_li) * (1 - p_hc), p_li * (1 - p_hc), (1 - p_li) * p_hc, p_li * p_hc])
        if imbalance in RESAMPLING:
            pos = resample_positions(ytr.to_numpy(), imbalance, seed)
            xtr, ytr = xtr.iloc[pos], ytr.iloc[pos]
        if kind == "lr4":
            if imbalance == "smote":
                from imblearn.over_sampling import SMOTE
                design = _design(xtr, feats)
                xa, xb = design.fit_transform(xtr), design.transform(xte)
                xa, xb = (m.toarray() if hasattr(m, "toarray") else m for m in (xa, xb))
                k = min(5, int(ytr.value_counts().min()) - 1)
                if k >= 1:
                    xa, ya = SMOTE(k_neighbors=k, random_state=seed).fit_resample(xa, ytr)
                else:
                    ya = ytr
                clf = LogisticRegression(C=0.1, max_iter=4000).fit(xa, ya)
                return _proba4(clf, xb, clf.classes_)
            m = _lr(xtr, feats, class_weight=lr_class_weight(imbalance, ytr)).fit(xtr, ytr)
            return _proba4(m, xte, m.classes_)
        if kind == "cb4":
            if imbalance == "smote":
                raise ValueError("smote is applied to the logistic model only (CatBoost categorical features make synthetic rows ill-defined)")
            cat = [f for f in feats if f in NOMINAL or f == "country" or not pd.api.types.is_numeric_dtype(xtr[f])]
            a, b = xtr.copy(), xte.copy()
            for f in feats:
                if f in cat:
                    a[f], b[f] = a[f].fillna("NA").astype(str), b[f].fillna("NA").astype(str)
                else:
                    med = a[f].median()
                    a[f], b[f] = a[f].fillna(med), b[f].fillna(med)
            extra = ({"class_weights": weight_vector(ytr, alpha_of(imbalance))} if alpha_of(imbalance) is not None
                     else {"auto_class_weights": catboost_weights(imbalance)})
            m = CatBoostClassifier(iterations=100, depth=4, learning_rate=0.05, l2_leaf_reg=10, loss_function="MultiClass",
                                   verbose=False, random_seed=seed, thread_count=4, **extra)
            m.fit(a, ytr, cat_features=cat)
            return _proba4(m, b, m.classes_)
    raise ValueError(f"unknown model {kind}")


def _strata(frame: pd.DataFrame) -> pd.Series:
    """Fold-stratification key that does not depend on threshold-derived labels."""
    med = frame.groupby("country")["cost"].transform("median")
    return frame["country"].astype(str) + "_" + frame["li"].astype(str) + "_" + (frame["cost"] > med).astype(int).astype(str)


def oof_proba(frame: pd.DataFrame, feats: list[str], kind: str, *, n_splits: int = 5, n_repeats: int = 3, seed: int = 1,
              imbalance: str = "balanced") -> np.ndarray:
    """(n_repeats, len(frame), 4) out-of-fold probabilities. `frame` is one country ('local') or
    all countries ('pooled': `feats` should then include 'country')."""
    frame = frame.reset_index(drop=True)
    out = np.zeros((n_repeats, len(frame), 4))
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for i, (tr, te) in enumerate(cv.split(frame, _strata(frame))):
        r = i // n_splits
        out[r, te] = fit_predict_proba(kind, frame.iloc[tr], frame.iloc[te], feats, imbalance=imbalance)
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
