"""Hansen's Threshold Regression (Hansen 1996, Econometrica 64:413; Hansen 2000, Econometrica 68:575).

Model, with threshold variable q and unknown threshold gamma:
    y = Xs' b1 + Xc' c + e   if q <= gamma
    y = Xs' b2 + Xc' c + e   if q >  gamma
Xs = regressors whose effect may change at the tipping point (here a constant and q itself); Xc = controls with one common
effect. gamma is estimated ENDOGENOUSLY -- chosen by the data, not imposed -- as the value that minimises the residual sum of
squares over a grid of candidate values (the trimmed central range of q). The ENABLE survey is one cross-section, so this is
the cross-section form (Hansen 2000); the panel form (Hansen 1999) needs repeated observations of the same households.

Inference:
  * Is there a tipping point at all? F = n (SSR_linear - SSR_threshold) / SSR_threshold; because gamma is not identified under
    "no threshold", its p-value comes from Hansen's (1996) fixed-regressor bootstrap (heteroskedasticity-robust: y* = e_i x N(0,1)).
  * Where is it? 95% confidence set {gamma : LR(gamma) <= 7.35}, LR(gamma) = n (SSR(gamma) - SSR(gamma_hat)) / SSR(gamma_hat);
    7.35 = -2 log(1 - sqrt(0.95)) (Hansen 2000, Table 1).
"""
from __future__ import annotations
import numpy as np

LR_CRITICAL_95 = -2 * np.log(1 - np.sqrt(0.95))


def _ssr(Z: np.ndarray, y: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    r = y - Z @ beta
    return float(r @ r)


def _design(Xs, Xc, q, gamma):
    low = (q <= gamma)[:, None]
    return np.hstack([Xs * low, Xs * ~low, Xc])


def threshold_grid(q: np.ndarray, *, trim: float = 0.15, n_grid: int = 200) -> np.ndarray:
    lo, hi = np.quantile(q, [trim, 1 - trim])
    cand = np.unique(q[(q >= lo) & (q <= hi)])
    if len(cand) > n_grid:
        cand = np.unique(np.quantile(cand, np.linspace(0, 1, n_grid)))
    return cand


def _sup_f(y, Xs, Xc, q, grid):
    n = len(y)
    ssr0 = _ssr(np.hstack([Xs, Xc]), y)
    ssr = np.array([_ssr(_design(Xs, Xc, q, g), y) for g in grid])
    k = int(np.argmin(ssr))
    return n * (ssr0 - ssr[k]) / ssr[k], k, ssr, ssr0


def hansen_threshold(y, q, Xs, Xc=None, *, trim: float = 0.15, n_grid: int = 200, n_boot: int = 500, seed: int = 0) -> dict:
    """Estimate gamma, test for a threshold and give its 95% confidence set. Xs and Xc must not contain a constant twice
    (Xs carries the constant). Returns gamma, F, bootstrap p-value, CI bounds, the SSR / LR curve and both regimes' slopes."""
    y = np.asarray(y, float); q = np.asarray(q, float)
    Xs = np.asarray(Xs, float); Xc = np.zeros((len(y), 0)) if Xc is None else np.asarray(Xc, float)
    grid = threshold_grid(q, trim=trim, n_grid=n_grid)
    F, k, ssr, ssr0 = _sup_f(y, Xs, Xc, q, grid)
    gamma = float(grid[k])
    Z = _design(Xs, Xc, q, gamma)
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    resid = y - Z @ beta
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):                                    # fixed-regressor bootstrap under H0: no threshold
        ystar = resid * rng.standard_normal(len(y))
        boot[b] = _sup_f(ystar, Xs, Xc, q, grid)[0]
    lr = len(y) * (ssr - ssr[k]) / ssr[k]
    inside = grid[lr <= LR_CRITICAL_95]
    p = Xs.shape[1]
    return {"gamma": gamma, "F": float(F), "p_value": float((boot >= F).mean()), "ci_low": float(inside.min()),
            "ci_high": float(inside.max()), "n": int(len(y)), "grid": grid, "lr": lr,
            "coef_below": beta[:p].tolist(), "coef_above": beta[p:2 * p].tolist()}
