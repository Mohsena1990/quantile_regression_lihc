"""Continuous cost-model evaluation metrics, shared by every script that scores a
predicted-cost column against observed_energy_cost_eur_reference at a given
quantile (originally lived in the now-retired scripts/compare_h_hc.py)."""
from __future__ import annotations
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_pinball_loss, mean_squared_error, r2_score


def cost_score(d: pd.DataFrame, q: float) -> dict[str, float]:
    valid = d.observed_energy_cost_eur_reference.notna() & d.predicted_cost.notna()
    y = d.observed_energy_cost_eur_reference[valid]
    p = d.predicted_cost[valid]
    mse = mean_squared_error(y, p)
    return {"mae": mean_absolute_error(y, p), "mse": mse, "rmse": mse ** 0.5,
            "mape": mean_absolute_percentage_error(y, p), "r2": r2_score(y, p),
            "pinball": mean_pinball_loss(y, p, alpha=q),
            "empirical_coverage_error": float((y <= p).mean() - q)}
