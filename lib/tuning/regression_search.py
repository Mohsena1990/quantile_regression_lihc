"""Internal k-fold hyperparameter search for the M2/M5/M7 cost-threshold families.

Every candidate is scored by mean_pinball_loss at the family's target quantile,
folded strictly inside the caller's `train` frame (see tuning/search.py's module
docstring for why this must never touch validation/test).
"""
from __future__ import annotations
import pandas as pd
from sklearn.metrics import mean_pinball_loss
from measurement.families.model_based.estimators import fit_family
from lib.tuning.search import select_by_kfold

CANDIDATE_GRIDS = {
    "M2": [{"alpha": a} for a in (0.0003, 0.001, 0.01)],
    "M5": [{"alpha": a} for a in (0.01, 0.1, 1.0)],
    "M7": [{"n_estimators": n, "max_depth": d} for n, d in ((150, 2), (250, 3), (350, 4))],
}


def tune_family(family: str, train: pd.DataFrame, *, features: list[str], target: str,
                 quantile: float, country_col: str = "country", seed: int = 20260811,
                 k: int = 3) -> dict:
    """Return the best hyperparameter dict for `family`, or {} if it has no grid."""
    grid = CANDIDATE_GRIDS.get(family)
    if not grid:
        return {}
    n = len(train)
    if n < 10 * k:  # not enough rows for a meaningful k-fold search -- keep the default
        return {}

    def fit_score(candidate, fit_idx, hold_idx):
        fit_frame, hold_frame = train.iloc[fit_idx], train.iloc[hold_idx]
        try:
            _, pred, _ = fit_family(family, fit_frame, hold_frame, features=features, target=target,
                                     quantile=quantile, country_col=country_col, params=candidate)
        except ValueError:  # e.g. INSUFFICIENT_TRAINING_ROWS on a small/degenerate fold
            return float("nan")
        y_true = pd.to_numeric(hold_frame[target], errors="coerce").to_numpy()
        valid = ~pd.isna(y_true)
        if valid.sum() == 0:
            return float("nan")
        return mean_pinball_loss(y_true[valid], pred[valid], alpha=quantile)

    best, _, _ = select_by_kfold(grid, fit_score, n, k=k, seed=seed, maximize=False)
    return best
