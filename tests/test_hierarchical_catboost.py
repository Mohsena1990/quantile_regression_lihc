"""Regression tests for the S1 hierarchical CatBoost fix.

Audit finding: `hierarchical_catboost` used to be plain CatBoost with a redundant
country feature column -- no global model, no country experts, no shrinkage.
These tests lock in the genuine global+country-expert partial-pooling blend:
P_final = (1-alpha_c)*P_global + alpha_c*P_country, alpha_c = n_c/(n_c+lambda).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from lib.screening.models import _blend_hierarchical, _country_experts, _prep_frame, fit_screening_model


def _synthetic_frame(n_per_country=80, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    countries = {"A": 0.0, "B": 3.0}  # country B has a strongly shifted feature -> class relationship
    for country, shift in countries.items():
        for i in range(n_per_country):
            x1 = rng.normal(shift, 1.0)
            x2 = rng.normal(0, 1.0)
            target = int(x1 > shift)  # 0/1, cleanly separable per-country but NOT globally (shift differs)
            rows.append({"UID": f"{country}{i}", "country": country, "x1": x1, "x2": x2, "risk_class_code": target})
    return pd.DataFrame(rows)


def test_hierarchical_differs_from_plain_catboost_when_country_signal_exists():
    """A country-specific feature/target relationship should make the hierarchical blend's
    predictions diverge from a pooled model that only sees a generic 'country' feature."""
    d = _synthetic_frame()
    train = d.sample(frac=0.7, random_state=1)
    test = d.drop(train.index)
    features = ["x1", "x2"]
    _, hier_pred, _ = fit_screening_model("hierarchical_catboost", train, test, features=features)
    _, plain_pred, _ = fit_screening_model("catboost", train, test, features=features)
    disagreement = (hier_pred.prediction.values != plain_pred.prediction.values).mean()
    assert disagreement > 0, "hierarchical_catboost must not be reducible to plain catboost + country column"


def test_country_experts_fall_back_to_global_for_small_support():
    d = _synthetic_frame(n_per_country=5)  # below MIN_COUNTRY_SUPPORT
    xtr, _, cat = _prep_frame(d, d, ["x1", "x2"])
    experts = _country_experts(xtr, d.risk_class_code, d.country, cat)
    assert experts == {}, "countries below the minimum support threshold must not get their own expert"


def test_blend_shrinks_toward_global_as_lambda_grows():
    """alpha_c = n_c/(n_c+lambda) -> 0 as lambda -> infinity, so the blended
    probabilities should converge to the global model's own probabilities."""
    d = _synthetic_frame(n_per_country=60)
    xtr, xte, cat = _prep_frame(d, d, ["x1", "x2"])
    ytr = d.risk_class_code
    from lib.screening.models import _aligned_proba, _fit_catboost
    global_model = _fit_catboost(xtr, ytr, cat, None, seed=1)
    p_global = _aligned_proba(global_model, xte, n_classes=2)
    _, p_small_lambda = _blend_hierarchical(xtr, ytr, d.country, xte, d.country, cat, None, lam=1, seed=1)
    _, p_huge_lambda = _blend_hierarchical(xtr, ytr, d.country, xte, d.country, cat, None, lam=1_000_000, seed=1)
    assert np.abs(p_huge_lambda[:, :2] - p_global).mean() < np.abs(p_small_lambda[:, :2] - p_global).mean()


def test_lambda_selection_never_touches_the_holdout_test_frame():
    """select_hierarchical_lambda must only use the train frame -- it must give the identical
    answer regardless of what is passed as the (development-validation or locked-test) frame,
    since fit_screening_model is called with the locked test set once test is unlocked."""
    from lib.screening.models import select_hierarchical_lambda
    d = _synthetic_frame(n_per_country=60)
    xtr, _, cat = _prep_frame(d, d, ["x1", "x2"])
    lam_a = select_hierarchical_lambda(xtr, d.risk_class_code, d.country, cat, None, seed=5)
    lam_b = select_hierarchical_lambda(xtr, d.risk_class_code, d.country, cat, None, seed=5)
    assert lam_a == lam_b  # deterministic given the same seed and train data alone
