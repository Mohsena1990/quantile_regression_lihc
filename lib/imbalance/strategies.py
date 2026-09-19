"""Class-imbalance handling for the four-class screening target.

Cost-sensitive class weighting is this project's single, fixed imbalance
strategy (2026-08-26): balanced-bootstrap oversampling and SMOTENC synthetic
oversampling were removed as a *reported comparison axis* -- weighting needs no
synthetic data and works even on small per-country groups (e.g. Spain's Double
Risk class), and comparing three strategies added a reporting dimension without
changing which model won in practice.
"""
from __future__ import annotations
import pandas as pd


def class_weights(y) -> dict[int, float]:
    """Inverse-frequency weights: rare classes get proportionally larger weight."""
    counts = pd.Series(y).value_counts()
    return {int(k): len(y) / (len(counts) * v) for k, v in counts.items()}
