"""Validation-safe hyperparameter search.

Every candidate is scored on an internal holdout carved out of whatever
`train`-role data the caller passes in -- never on the caller's actual
validation/test frame. This is what makes it safe to call identically in
`development` and `final_test` stage: at `development` the caller's train-role
frame is pure train, at `final_test` it is train+validation (already unlocked
for fitting), but in neither case does tuning ever see the caller's held-out
evaluation frame. `screening/models.py`'s original `select_hierarchical_lambda`
established this pattern for one hyperparameter; this module generalises it.
"""
from __future__ import annotations
from typing import Callable, Sequence, TypeVar
import numpy as np

T = TypeVar("T")


def holdout_split(n: int, *, val_fraction: float = 0.2, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic (fit_idx, holdout_idx) partition of range(n)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(n * (1 - val_fraction))
    return idx[:cut], idx[cut:]


def kfold_indices(n: int, k: int, *, seed: int) -> list[np.ndarray]:
    """k deterministic near-equal folds over range(n)."""
    rng = np.random.default_rng(seed)
    return np.array_split(rng.permutation(n), k)


def select_by_score(candidates: Sequence[T], score_fn: Callable[[T], float], *,
                     maximize: bool = True) -> tuple[T, float, list[tuple[T, float]]]:
    """Evaluate every candidate with `score_fn` and return (best, best_score, all_scores).

    Falls back to the first candidate if every score is non-finite (e.g. a fold with a
    single class), so callers never receive a NaN "best" silently.
    """
    scored = [(c, float(score_fn(c))) for c in candidates]
    finite = [(c, s) for c, s in scored if np.isfinite(s)]
    if not finite:
        return candidates[0], float("nan"), scored
    best = max(finite, key=lambda cs: cs[1]) if maximize else min(finite, key=lambda cs: cs[1])
    return best[0], best[1], scored


def select_by_kfold(candidates: Sequence[T], fit_score_fn: Callable[[T, np.ndarray, np.ndarray], float],
                     n: int, *, k: int = 3, seed: int, maximize: bool = False) -> tuple[T, float, list[tuple[T, float]]]:
    """Select the candidate minimising (or maximising) the mean k-fold score.

    `fit_score_fn(candidate, fit_idx, hold_idx) -> float` fits on `fit_idx` and scores
    on `hold_idx`, both index arrays into the caller's *train*-role frame only.
    """
    folds = kfold_indices(n, k, seed=seed)
    all_idx = np.arange(n)

    def mean_score(candidate: T) -> float:
        vals = []
        for i in range(k):
            hold_idx = folds[i]
            fit_idx = np.setdiff1d(all_idx, hold_idx, assume_unique=False)
            if len(hold_idx) == 0 or len(fit_idx) == 0:
                continue
            vals.append(fit_score_fn(candidate, fit_idx, hold_idx))
        return float(np.mean(vals)) if vals else float("nan")

    return select_by_score(candidates, mean_score, maximize=maximize)
