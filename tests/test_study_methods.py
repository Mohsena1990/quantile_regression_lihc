"""Study methods: per-household thresholds, partly asked survey modules, bill model, noisy-label weights, weighted models, tuning."""
import numpy as np
import pandas as pd
import pytest


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"country": np.where(np.arange(n) < n // 2, "A", "B"), "H3": rng.integers(1, 7, n),
                      "household_size": rng.integers(1, 6, n), "H1": rng.integers(1, 5, n), "H2": rng.integers(1, 9, n),
                      "insulation_count": rng.integers(0, 4, n), "hdd_hs1617": rng.normal(2500, 300, n),
                      "cdd_s2017": rng.normal(100, 20, n), "income_bracket": rng.integers(1, 11, n).astype(float)})
    d["cost"] = np.exp(6 + 0.25 * d.H3 + rng.normal(0, 0.4, n))
    d["income_eur_total"] = 3000 * d.income_bracket
    return d


@pytest.mark.parametrize("family", ["p80", "needqr", "needqr_noclim", "burden"])
def test_labels_store_their_threshold(family):
    from lib.identification.lines import make_labels
    b = make_labels(_toy(), family, 3, 0.8)
    assert ((b.cost > b.thr) == b.hc.astype(bool)).all() and (b.y == b.li + 2 * b.hc).all()


@pytest.mark.parametrize("family", ["p80", "burden", "needqr"])
def test_fold_threshold_uses_training_only(family):
    from lib.identification.lines import fold_threshold
    d = _toy()
    tr, te = d.iloc[:300], d.iloc[300:]
    t1 = fold_threshold(tr, te, family, 0.8)
    t2 = fold_threshold(tr, te.assign(cost=te.cost * 100), family, 0.8)  # the target's own bills never enter its threshold
    assert np.allclose(t1, t2) and np.isfinite(t1).all()


def test_unlabelled_and_cost_only():
    from lib.identification.lines import cost_only, unlabelled
    d = _toy()
    d.loc[:9, "cost"] = np.nan
    d.loc[10:14, "income_bracket"] = np.nan
    u = unlabelled(d, "pct", 3)
    assert len(u) == 10 and set(u.li) <= {0, 1}
    assert len(cost_only(d)) == 5


def test_partial_modules():
    from lib.features.partial_modules import partial_modules
    raw = pd.DataFrame({"H15A": [1, 4, np.nan], "H15B": [1, 4, np.nan], "H15C": [1, 4, np.nan], "H15D": [4, 1, np.nan],
                        "H15E": [1, 1, np.nan], "H15F": [1, 99, np.nan], "H15G": [4, 1, np.nan],
                        "G1A1": [np.nan, 1, 0], "G1A2": [np.nan, 1, 0], "G1A3": [np.nan, 0, 0], "G1A4": [np.nan, 0, 0],
                        "G1A5": [np.nan, 0, 1], "G3A": [np.nan, 1, 3], "G3B": [np.nan, 2, 3], "G3C": [np.nan, 1, 1],
                        "G3D": [np.nan, 99, 2], "G3E": [np.nan, 2, 2], "E1": [np.nan, 99, 20], "C5C": [6, 3, np.nan]})
    out = partial_modules(raw)
    assert out.env_attitude_index.iloc[0] == 4 and out.env_attitude_index.iloc[1] == 1   # E excluded, 99 dropped
    assert np.isnan(out.env_attitude_index.iloc[2])
    assert np.isnan(out.energy_saving_index.iloc[0]) and out.energy_saving_index.iloc[1] == 4 and out.energy_saving_index.iloc[2] == 1
    assert np.isnan(out.knows_kwh_price.iloc[0]) and out.knows_kwh_price.iloc[1] == 0 and out.knows_kwh_price.iloc[2] == 1
    assert np.isnan(out.bill_too_complicated.iloc[0]) and out.bill_too_complicated.iloc[1] == 3


def test_four_class_and_prior():
    from lib.identification.bill_model import four_class
    p = four_class(np.array([0, 1]), np.array([0.3, 0.9]))
    assert np.allclose(p, [[0.7, 0, 0.3, 0], [0, 0.1, 0, 0.9]])
    q = four_class(np.array([0]), np.array([0.3]), prior=np.array([0.6, 0.1, 0.2, 0.1]))
    assert q[0, 2] > q[0, 0]                                           # balanced decision: the rarer class needs less evidence


def test_bill_model_learns_cost():
    from lib.identification.bill_model import BillModel
    d = _toy(600)
    m = BillModel(d.iloc[:500], ["H3", "household_size", "country"], seed=1)
    te = d.iloc[500:]
    assert np.corrcoef(m.mu(te), np.log(te.cost))[0, 1] > 0.6
    p = m.p_above(te, np.full(len(te), te.cost.median()))
    assert ((p >= 0) & (p <= 1)).all()
    assert p[te.H3.to_numpy() == 6].mean() > p[te.H3.to_numpy() == 1].mean()


def test_borderline_weight():
    from lib.identification.robust import borderline_weight
    w = borderline_weight(np.array([100, 110, 200, 50]), np.array([100, 100, 100, 100]))
    assert w[0] == pytest.approx(0.5) and 0.5 < w[1] < 1 and w[2] == 1 and w[3] == 1


def _clf_toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.choice(["x", "y"], n)})
    d["y"] = (d.a > 0).astype(int) + 2 * (d.b > 0).astype(int)
    return d


@pytest.mark.parametrize("kind", ["catboost", "xgboost"])
def test_weights_change_the_model(kind):
    from lib.identification.models import fit
    d = _clf_toy()
    p0 = fit(kind, d, ["a", "b", "c"], 1).predict_proba(d)
    w = np.where(d.y == 3, 5.0, 1.0)
    p1 = fit(kind, d, ["a", "b", "c"], 1, weight=w).predict_proba(d)
    assert p1[:, 3].mean() > p0[:, 3].mean()


def test_confident_learning_finds_flipped_labels():
    from lib.identification.robust import confident_learning_flags
    d = _clf_toy(400, 2)
    flip = np.zeros(len(d), bool); flip[:20] = True
    d.loc[flip, "y"] = d.loc[flip, "y"] ^ 2                            # spending side flipped
    f = confident_learning_flags(d, ["a", "b", "c"], 3)
    assert f[flip].mean() > 3 * max(f[~flip].mean(), 0.01)
    assert confident_learning_flags(d, ["a", "b", "c"], 3, margin=0.0).sum() >= f.sum()


def test_tuning_runs():
    from lib.identification.tuning import tune_xgboost
    params, score = tune_xgboost(_clf_toy(), ["a", "b", "c"], 1, n_trials=2)
    assert {"max_depth", "learning_rate"} <= set(params) and 0 <= score <= 1


def test_ft_transformer_accepts_weights():
    from lib.identification.models import fit
    d = _clf_toy()
    p = fit("ft_transformer", d, ["a", "b", "c"], 1, weight=np.where(d.y == 3, 3.0, 1.0)).predict_proba(d)
    assert p.shape == (len(d), 4) and np.allclose(p.sum(1), 1, atol=1e-5)
