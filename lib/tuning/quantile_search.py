"""Continuous per-(family x feature-config) calibration search.

Replaces the discrete CORE_QUANTILES grid (measurement/registry.py, removed) with a
scalar optimizer that finds the single quantile q in `bounds` minimising the 5-fold
out-of-fold absolute calibration error |observed_high_cost_rate - (1-q)| on the
"Five-country overall" pool -- exactly the metric scripts/select_family_champions.py
already computed post-hoc over discrete points, now minimised directly.

bounds=(0.50, 0.80) brackets the same two named anchors this project is grounded in:
the official-LIHC handbook's country-median (~Q50) and Van Hove et al. (2022)'s
country Q80 (M0) -- both fixed by their own external definitions, outside this search.

Hyperparameter tuning (lib/tuning/regression_search.py) is skipped during the search
itself (tune_during_search=False) -- tuning at every candidate q would multiply cost
5-10x for no accuracy benefit -- and run once at the winning q* (tune_final=True).

The calibration objective is a staircase-like function of q (observed_high_cost_rate
moves in discrete jumps as households cross a shifting threshold) layered on a model
whose fit itself changes with q, so it is not provably unimodal. A single bounded
local search is not immune to a bad local plateau, so `find_optimal_quantile` seeds
itself from a coarse global scan (`n_coarse` evenly spaced points) and only then
refines locally around the best coarse point -- the returned quantile is never worse
than the coarse scan's best point, by construction.
"""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from measurement.families.model_based import fit_measurement_strategy

DEFAULT_BOUNDS = (0.50, 0.80)
DEFAULT_XATOL = 0.005
DEFAULT_N_COARSE = 7
N_FOLDS = 5


def oof_fold(uid: str, *, salt: str = "oof-v1") -> int:
    """Deterministic fold assignment, sha256(salt|UID) % 5 -- single definition shared
    by the optimizer and scripts/build_oof_predictions.py (previously duplicated)."""
    return int(hashlib.sha256(f"{salt}|{uid}".encode()).hexdigest()[:8], 16) % N_FOLDS


def _oof_predict(family: str, training: pd.DataFrame, q: float, *, features: list[str], target: str,
                  country_col: str, tune: bool) -> pd.DataFrame:
    parts = []
    for k in range(N_FOLDS):
        fit = training[training.oof_fold.ne(k)]
        hold = training[training.oof_fold.eq(k)]
        _, pred = fit_measurement_strategy(family, fit, hold, features=features, target=target,
                                            quantile=q, country_col=country_col, tune=tune)
        pred = pred.copy()
        pred["oof_fold"] = k
        parts.append(pred)
    return pd.concat(parts).sort_index()


def _calibration_error(oof: pd.DataFrame, q: float) -> float:
    eligible = oof.measurement_eligible
    observed = oof.loc[eligible, "high_cost"].mean()
    return abs(float(observed) - (1 - q)) if pd.notna(observed) else float("nan")


def find_optimal_quantile(family: str, training: pd.DataFrame, *, features: list[str], target: str,
                           country_col: str = "country", bounds: tuple[float, float] = DEFAULT_BOUNDS,
                           xatol: float = DEFAULT_XATOL, n_coarse: int = DEFAULT_N_COARSE,
                           tune_during_search: bool = False, tune_final: bool = True) -> dict:
    """`training` must carry an 'oof_fold' column (see `oof_fold`). Returns
    {'quantile', 'absolute_calibration_error', 'oof_predictions' (train rows only,
    oof_fold-tagged, from the final tuned refit at q*), 'n_evaluations', 'evaluations'}."""
    if "oof_fold" not in training.columns:
        raise ValueError("training frame must carry an 'oof_fold' column before searching")
    trace: list[tuple[float, float]] = []

    def objective(q: float) -> float:
        oof = _oof_predict(family, training, q, features=features, target=target,
                            country_col=country_col, tune=tune_during_search)
        err = _calibration_error(oof, q)
        trace.append((float(q), err))
        return err if np.isfinite(err) else 1.0  # worst-case finite score, never wins over a real point

    # Coarse global scan first (see module docstring): seeds the local refinement from
    # the best of several widely-spaced points instead of trusting a single bounded
    # search to find the global minimum on a non-smooth objective.
    coarse_qs = np.linspace(bounds[0], bounds[1], n_coarse)
    coarse_errs = [objective(float(q)) for q in coarse_qs]
    best_i = int(np.argmin(coarse_errs))
    q_best, err_best = float(coarse_qs[best_i]), coarse_errs[best_i]

    step = (bounds[1] - bounds[0]) / (n_coarse - 1)
    local_bounds = (max(bounds[0], q_best - step), min(bounds[1], q_best + step))
    if local_bounds[1] - local_bounds[0] > xatol:
        result = minimize_scalar(objective, bounds=local_bounds, method="bounded", options={"xatol": xatol})
        if result.fun < err_best:
            q_best, err_best = float(result.x), float(result.fun)

    # Re-run once more at q_best with tuning -- the search phase above deliberately used
    # un-tuned params (tune_during_search), so this is the fit actually written to disk.
    final_oof = _oof_predict(family, training, q_best, features=features, target=target,
                              country_col=country_col, tune=tune_final)
    return {"quantile": q_best, "absolute_calibration_error": _calibration_error(final_oof, q_best),
            "oof_predictions": final_oof, "n_evaluations": len(trace) + 1, "evaluations": trace}
