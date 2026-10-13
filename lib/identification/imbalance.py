"""Class-imbalance strategies for the four-class identification task.

Classes are unbalanced (Poland: No Risk 9%, Expenditure Risk 8%, Income Risk 53%, Double Risk 29%), and the current default
(`balanced` class weights) flattens the predicted class shares, which trades accuracy for minority recall. The strategies
compared in scripts/build_imbalance_strategy_comparison.py:

  none      no correction (raw class frequencies)
  balanced  class weights n / (K * n_k)                    -- the pipeline's default until now
  sqrt      square root of the balanced weights            -- a softened correction
  ros       random oversampling of the training fold to the largest class
  rus       random undersampling of the classes above the median class count down to it
  smote     SMOTE oversampling to the largest class in the encoded feature space (logistic model only)
  balanced_posthoc  balanced weights, THEN the same post-hoc class rescaling (does calibrating predicted shares help on top of weights?)
  alpha_tuned       weights (n / (K n_k))**alpha with alpha in {0, .25, .5, .75, 1} chosen by inner 3-fold macro-F1 in each training fold
  bagging           EasyEnsemble-style: 10 undersampled bags (classes capped at the median count), balanced weights, probabilities averaged
  posthoc   train unweighted, then rescale class probabilities so argmax maximises macro-F1 on inner out-of-fold predictions
            (decision-threshold moving; probabilities are rescaled, so log-loss/calibration change)

Every strategy is applied INSIDE the training fold only; test folds are never resampled or used to fit a scale.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score

STRATEGIES = ("none", "balanced", "sqrt", "ros", "rus", "smote", "posthoc", "balanced_posthoc", "alpha_tuned", "bagging")
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
N_BAGS = 10
WEIGHT_STRATEGIES = ("none", "balanced", "sqrt")
RESAMPLING = ("ros", "rus")
SCALE_GRID = np.exp(np.linspace(-2.5, 2.5, 26))


def alpha_of(strategy: str) -> float | None:
    """Parametric weight strategies are named 'alpha=<x>' internally (used by alpha_tuned)."""
    return float(strategy.split("=")[1]) if strategy.startswith("alpha=") else None


def weight_vector(y, alpha: float, k: int = 4) -> list[float]:
    """(n / (K n_k)) ** alpha for classes 0..k-1 (1.0 for a class absent from y)."""
    counts = np.bincount(np.asarray(y, int), minlength=k)
    return [float((len(y) / (k * n)) ** alpha) if n > 0 else 1.0 for n in counts]


def lr_class_weight(strategy: str, y) -> str | dict | None:
    """sklearn `class_weight` for the logistic model."""
    a = alpha_of(strategy)
    if a is not None:
        return {i: w for i, w in enumerate(weight_vector(y, a))}
    if strategy == "balanced":
        return "balanced"
    if strategy == "sqrt":
        classes, counts = np.unique(np.asarray(y), return_counts=True)
        return {int(c): float(np.sqrt(len(y) / (len(classes) * n))) for c, n in zip(classes, counts)}
    return None


def catboost_weights(strategy: str) -> str | None:
    return {"balanced": "Balanced", "sqrt": "SqrtBalanced"}.get(strategy)


def resample_positions(y, strategy: str, seed: int) -> np.ndarray:
    """Row positions of the resampled training fold (`ros`: oversample to the largest class; `rus`: cap classes above the
    median class count at the median)."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    keep = []
    if strategy == "ros":
        target = counts.max()
        for c, n in zip(classes, counts):
            pos = np.flatnonzero(y == c)
            keep.append(pos)
            if n < target:
                keep.append(rng.choice(pos, target - n, replace=True))
    elif strategy == "rus":
        target = int(np.median(counts))
        for c, n in zip(classes, counts):
            pos = np.flatnonzero(y == c)
            keep.append(pos if n <= target else rng.choice(pos, target, replace=False))
    else:
        raise ValueError(f"not a resampling strategy: {strategy}")
    return np.sort(np.concatenate(keep))


def fit_class_scales(proba: np.ndarray, y, sweeps: int = 3) -> np.ndarray:
    """Per-class multiplicative scales (class 0 fixed at 1) maximising macro-F1 of argmax(proba * scale) by coordinate ascent
    on a log grid. Ties are broken toward scale 1 (no change), so a model that is already well balanced is left alone."""
    y = np.asarray(y, int)
    scale = np.ones(proba.shape[1])

    def score(s):
        return f1_score(y, (proba * s).argmax(1), average="macro", labels=list(range(proba.shape[1])), zero_division=0)

    best = score(scale)
    for _ in range(sweeps):
        for k in range(1, proba.shape[1]):
            for g in sorted(SCALE_GRID, key=lambda v: abs(np.log(v))):
                trial = scale.copy()
                trial[k] = g
                s = score(trial)
                if s > best + 1e-12:
                    best, scale = s, trial
    return scale
