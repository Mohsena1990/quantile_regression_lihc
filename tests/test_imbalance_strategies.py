import numpy as np
import pandas as pd
import pytest
from lib.identification.imbalance import fit_class_scales, lr_class_weight, resample_positions
from lib.identification.protocol import fit_predict_proba


def _frame(n=240, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice(4, n, p=[0.1, 0.55, 0.08, 0.27])
    d = pd.DataFrame({"country": "X", "x1": rng.normal(size=n) + y, "x2": rng.normal(size=n), "y": y})
    d["li"], d["hc"] = d.y % 2, (d.y >= 2).astype(int)
    return d


def test_weights_balanced_sqrt_and_none():
    y = np.array([0] * 10 + [1] * 30 + [2] * 60)
    assert lr_class_weight("balanced", y) == "balanced" and lr_class_weight("none", y) is None
    w = lr_class_weight("sqrt", y)
    bal = {c: len(y) / (3 * n) for c, n in zip((0, 1, 2), (10, 30, 60))}
    assert all(abs(w[c] - np.sqrt(bal[c])) < 1e-12 for c in w) and w[0] > w[1] > w[2]


def test_resampling_targets_and_no_test_rows_involved():
    y = np.array([0] * 10 + [1] * 30 + [2] * 60 + [3] * 100)
    ros = np.bincount(y[resample_positions(y, "ros", 1)])
    assert (ros == 100).all()                                   # everything oversampled to the largest class
    rus = np.bincount(y[resample_positions(y, "rus", 1)])
    assert rus.tolist() == [10, 30, 45, 45]                     # classes above the median (45) capped at it; smaller kept
    assert resample_positions(y, "ros", 1).max() < len(y)       # positions come from the training fold only


def test_class_scales_lift_an_under_predicted_minority_class():
    rng = np.random.default_rng(3)
    y = rng.choice(2, 600, p=[0.85, 0.15])
    p1 = np.clip(0.15 + 0.25 * y + rng.normal(0, 0.08, 600), 0.02, 0.98)      # informative, but minority prob rarely > 0.5
    proba = np.column_stack([1 - p1, p1])
    s = fit_class_scales(proba, y)
    assert s[0] == 1.0 and s[1] > 1.5                                          # minority scaled up
    assert (proba * s).argmax(1).mean() > proba.argmax(1).mean()               # so it is predicted more often
    flat = np.column_stack([np.full(600, 0.5), np.full(600, 0.5)])
    assert fit_class_scales(flat, y)[1] >= 1.0                                 # never returns NaN / degenerate scales


@pytest.mark.parametrize("strategy", ["none", "balanced", "sqrt", "ros", "rus", "smote", "posthoc", "balanced_posthoc", "alpha_tuned", "bagging", "alpha=0.25"])
def test_every_strategy_returns_valid_probabilities_for_the_logistic_model(strategy):
    d = _frame()
    tr, te = d.iloc[:180], d.iloc[180:]
    p = fit_predict_proba("lr4", tr, te, ["x1", "x2"], imbalance=strategy)
    assert p.shape == (len(te), 4) and np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1, atol=1e-6)


def test_default_is_identical_to_explicit_balanced_and_bad_names_are_rejected():
    d = _frame()
    tr, te = d.iloc[:180], d.iloc[180:]
    np.testing.assert_allclose(fit_predict_proba("lr4", tr, te, ["x1", "x2"]), fit_predict_proba("lr4", tr, te, ["x1", "x2"], imbalance="balanced"))
    with pytest.raises(ValueError):
        fit_predict_proba("lr4", tr, te, ["x1", "x2"], imbalance="nonsense")
    with pytest.raises(ValueError, match="weight strategies only"):
        fit_predict_proba("axis2", tr, te, ["x1", "x2"], imbalance="ros")


def test_alpha_weights_interpolate_between_none_and_balanced():
    from lib.identification.imbalance import weight_vector
    y = np.array([0] * 10 + [1] * 30 + [2] * 60 + [3] * 100)
    assert weight_vector(y, 0.0) == [1.0, 1.0, 1.0, 1.0]
    full = weight_vector(y, 1.0)
    assert abs(full[0] - len(y) / (4 * 10)) < 1e-12
    half = weight_vector(y, 0.5)
    assert all(abs(h - np.sqrt(f)) < 1e-12 for h, f in zip(half, full))
