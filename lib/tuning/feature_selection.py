"""SHAP-CatBoost feature-importance ranking against the four-class screening target
(risk_class_code) -- "which of these features actually help identify the four risk
classes", not just "which correlate with cost".

Used by scripts/audit_shap_feature_effectiveness.py as the project's central
interpretation tool (see that script's module docstring). Previously also backed
select_features_shap/select_features_lassonet, which built the automatic per-country
A3/A4 feature-config variants -- both removed: picking a different feature subset
per country is incompatible with interpreting one factor's effect consistently
across countries, this project's stated goal (see lib/reporting/codes.py).
"""
from __future__ import annotations
import numpy as np, pandas as pd
from catboost import CatBoostClassifier, Pool


def _cat_mask(frame: pd.DataFrame, features: list[str]) -> list[int]:
    return [i for i, x in enumerate(features) if not pd.api.types.is_numeric_dtype(frame[x])]


def rank_features_shap(train: pd.DataFrame, features: list[str], *, target: str = "risk_class_code",
                        seed: int = 20260811) -> pd.Series:
    """Fit CatBoost on `features` -> `target`, return every feature's mean |SHAP|
    importance (pooled over samples and classes) as a Series sorted descending.

    On degenerate input (fewer than 2 target classes or under 10 rows) returns an
    unsorted, all-zero Series in the original feature order: a genuine ranking isn't
    possible.
    """
    tr = train.dropna(subset=[target]).copy()
    used = list(features)
    cat = _cat_mask(tr, used)
    x = tr[used].copy()
    for i, col in enumerate(used):
        if i in cat:
            x[col] = x[col].fillna("__MISSING__").astype(str)
        else:
            x[col] = pd.to_numeric(x[col], errors="coerce").fillna(x[col].median())
    y = tr[target].astype(int)
    if y.nunique() < 2 or len(y) < 10:
        return pd.Series(0.0, index=used)
    model = CatBoostClassifier(iterations=150, depth=6, loss_function="MultiClass", verbose=False, random_seed=seed)
    model.fit(x, y, cat_features=cat)
    shap = np.asarray(model.get_feature_importance(Pool(x, y, cat_features=cat), type="ShapValues"))
    importance = np.abs(shap[:, :, :-1]).mean(axis=(0, 1))  # drop the trailing bias column
    return pd.Series(importance, index=used).sort_values(ascending=False)
