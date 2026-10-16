"""Nested feature selection: "Attention Mechanisms as Embedded Selectors + SHAP", the one-standard-error rule, a paired gate, the
region-swap placebo and the paired bootstrap for differences between two models.

Selection runs inside every outer training fold only, so the households that are scored never influence the choice:
  1. rank: the training rows are split 80 / 20 three times (RANK_SPLITS). Each time three models are fitted on the 80% with the base
     set + all candidates: an FT-Transformer, whose [CLS] attention to each feature token is an EMBEDDED importance, and CatBoost
     and XGBoost, whose exact TreeSHAP values give each feature's contribution; all three are measured on the 20%. RELIABILITY
     WEIGHTING: each measure's weight = how consistently it ranks the candidates across the three splits (mean pairwise Spearman
     correlation, floored at 0). Score = weighted mean rank.
  2. curve: inner repeated cross-validation (2 x 3 folds) of CatBoost on base + the k best-ranked candidates, scored by balanced
     accuracy.
  3. decide: ONE-STANDARD-ERROR RULE (Breiman et al. 1984; Hastie, Tibshirani & Friedman 2009, Sec. 7.10) -- the smallest k whose
     mean inner balanced accuracy is within one standard error of the best k.
  4. gate: the chosen k is kept only if it beats k = 0 in the same inner folds by more than one standard error of the paired
     difference.
Placebo: climate (or any region-level value) lets a model recognise regions, so a region-swap placebo -- each region's values moved
to another region of the same country -- is the control; only a gain over the placebo can be credited to what the values measure.
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from .models import fit as fit_model

INNER_FOLDS, INNER_REPEATS = 3, 2
RANK_SPLITS = 3
MEASURES = ("attention", "shap_catboost", "shap_xgboost")


def _ranks(s: pd.Series) -> pd.Series:
    return s.rank(ascending=False, method="average")


def rank_candidates(train: pd.DataFrame, base: list[str], candidates: list[str], seed: int) -> pd.DataFrame:
    """Reliability-weighted attention + SHAP ranking over RANK_SPLITS stratified 80/20 splits of `train`. Columns: the three
    measures (mean importance), their weights (stability), score (weighted mean rank, lower = better); best first."""
    feats = base + candidates
    per = {m: [] for m in MEASURES}
    for s_ in range(RANK_SPLITS):
        tr, va = train_test_split(train, test_size=0.2, stratify=train["y"], random_state=seed + 97 * s_)
        per["attention"].append(fit_model("ft_transformer", tr, feats, seed + s_, val=va).importance(va).loc[candidates])
        per["shap_catboost"].append(fit_model("catboost", tr, feats, seed + s_).importance(va).loc[candidates])
        per["shap_xgboost"].append(fit_model("xgboost", tr, feats, seed + s_).importance(va).loc[candidates])
    t = pd.DataFrame({m: pd.concat(v, axis=1).mean(axis=1) for m, v in per.items()})
    weights = {}
    for m, v in per.items():
        rhos = [spearmanr(a_, b_)[0] for a_, b_ in itertools.combinations(v, 2)]
        weights[m] = max(0.0, float(np.nanmean(rhos))) if rhos else 1.0
    total = sum(weights.values()) or 1.0
    t["score"] = sum(weights[m] * _ranks(t[m]) for m in MEASURES) / total
    for m in MEASURES:
        t[f"weight_{m}"] = weights[m] / total
    return t.sort_values("score")


def _inner_folds(train: pd.DataFrame, seed: int):
    y = train["y"].astype(int).to_numpy()
    counts = np.bincount(y, minlength=4)
    n_splits = max(2, min(INNER_FOLDS, int(counts[counts > 0].min())))
    return list(RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=INNER_REPEATS, random_state=seed).split(train, y))


def one_se_choice(curve: dict[int, tuple[float, float]]) -> int:
    """Smallest k whose mean score is within one standard error of the best k's mean (curve: k -> (mean, se))."""
    best_k = max(curve, key=lambda k: curve[k][0])
    bar = curve[best_k][0] - curve[best_k][1]
    return min(k for k, (m, _) in curve.items() if m >= bar)


def paired_gate(k: int, per_fold: dict[int, np.ndarray]) -> int:
    """Keep k only if its inner-fold scores beat k = 0's in the same folds by more than one standard error of the paired
    difference; otherwise return 0."""
    if k == 0:
        return 0
    diff = per_fold[k] - per_fold[0]
    return k if diff.mean() > diff.std(ddof=1) / np.sqrt(len(diff)) else 0


def _curve(train: pd.DataFrame, sets: dict, seed: int) -> tuple[dict, dict]:
    """Inner repeated-CV balanced accuracy of CatBoost for each candidate feature set. Returns (curve, per-fold scores)."""
    y = train["y"].astype(int).to_numpy()
    folds = _inner_folds(train, seed)
    curve, per = {}, {}
    for key, feats in sets.items():
        s = [balanced_accuracy_score(y[te], fit_model("catboost", train.iloc[tr], feats, seed + i).predict_proba(train.iloc[te]).argmax(1))
             for i, (tr, te) in enumerate(folds)]
        per[key] = np.asarray(s)
        curve[key] = (float(np.mean(s)), float(np.std(s, ddof=1) / np.sqrt(len(s))))
    return curve, per


def placebo_climate(frame: pd.DataFrame, cols: list[str], seed: int) -> pd.DataFrame:
    """`frame` with the climate columns of each region replaced by those of another region of the same country
    (random permutation of regions with no fixed point where the country has >= 2 regions)."""
    out = frame.copy()
    rng = np.random.default_rng(seed)
    for _, g in frame.groupby("country"):
        regions = g["climate_region"].drop_duplicates().to_numpy()
        if len(regions) < 2:
            continue
        while True:
            perm = rng.permutation(len(regions))
            if not (perm == np.arange(len(regions))).any():
                break
        table = g.drop_duplicates("climate_region").set_index("climate_region")[cols]
        donor = dict(zip(regions, regions[perm]))
        out.loc[g.index, cols] = table.loc[g["climate_region"].map(donor)].to_numpy()
    return out


def paired_delta(y: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray, metrics=("f1_macro", "balanced_accuracy", "f1_weighted"), *,
                 n_boot: int = 500, seed: int = 11) -> dict:
    """Difference b - a of repeat-averaged metrics with a PAIRED household bootstrap 95% CI (same resampled households
    for both), plus the share of resamples in which b > a."""
    from .metrics import average_over_repeats
    y = np.asarray(y, int)
    pa, pb = average_over_repeats(y, proba_a), average_over_repeats(y, proba_b)
    rng = np.random.default_rng(seed)
    draws = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        a, b = average_over_repeats(y[idx], proba_a[:, idx]), average_over_repeats(y[idx], proba_b[:, idx])
        for m in metrics:
            draws[m].append(b[m] - a[m])
    out = {}
    for m in metrics:
        d = np.asarray(draws[m])
        out.update({f"delta_{m}": pb[m] - pa[m], f"delta_{m}_lo": float(np.percentile(d, 2.5)),
                    f"delta_{m}_hi": float(np.percentile(d, 97.5)), f"p_improve_{m}": float((d > 0).mean())})
    return out
