import numpy as np
import pandas as pd
import pytest
from lib.identification.labels import CLASS_NAMES, _need_threshold_oof, support_table, viability
from lib.identification.protocol import oof_proba, summarise
from lib.identification.factors import add_factors


def _synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"country": "X", "H3": rng.integers(1, 7, n).astype(float), "household_size": rng.integers(1, 6, n).astype(float),
                      "H1": rng.integers(1, 5, n).astype(float), "H2": rng.integers(1, 9, n).astype(float),
                      "hdd": rng.normal(2500, 300, n), "cdd": rng.normal(200, 50, n)})
    d["cost"] = np.exp(0.35 * d.H3 + 0.15 * d.household_size + rng.normal(0, 0.5, n))
    return d


def test_need_adjusted_threshold_leaves_about_one_minus_q_above_and_is_unpredictable_from_need():
    d = _synthetic()
    thr = _need_threshold_oof(d, 0.63, folds=5, seed=1)
    hc = (d.cost > thr).mean()
    assert abs(hc - 0.37) < 0.07
    # the label is (close to) independent of the need features by construction
    corr = np.corrcoef((d.cost > thr).astype(float), d.H3)[0, 1]
    assert abs(corr) < 0.15


def test_random_labels_are_not_learned_no_cv_leakage():
    rng = np.random.default_rng(987654)   # must differ from _synthetic's seed or labels correlate with features
    d = _synthetic(500, 1)
    d["li"] = rng.integers(0, 2, len(d)); d["hc"] = rng.integers(0, 2, len(d)); d["y"] = d.li + 2 * d.hc
    pr = oof_proba(d, ["H3", "household_size", "H2", "hdd"], "lr4", n_repeats=2)
    assert pr.shape == (2, 500, 4) and np.allclose(pr.sum(axis=2), 1)
    assert summarise(d.y.values, pr, n_boot=50)["macro_f1"] < 0.33      # chance is ~0.25; leakage would push it up


def test_viability_gate_flags_countries_with_a_thin_class():
    lab = pd.DataFrame({"country": ["A"] * 200 + ["B"] * 200, "y": [0] * 60 + [1] * 60 + [2] * 60 + [3] * 20 + [0] * 50 + [1] * 50 + [2] * 50 + [3] * 50})
    lab["li"] = lab.y % 2; lab["hc"] = (lab.y >= 2).astype(int)
    v = viability(support_table(lab)).set_index("country")
    assert not v.loc["A", "four_class_ok"] and v.loc["B", "four_class_ok"]
    assert list(CLASS_NAMES.values())[3] == "Double Risk"


def test_factor_recodes_keep_missing_as_missing():
    d = pd.DataFrame({"country": ["X"] * 4, "primary_heating_fuel": ["wood", "gas_lpg", None, "coal_coke"], "H2": [1, 5, np.nan, 3],
                      "respondent_age": [70, 40, np.nan, 66], "household_size": [4, 2, 3, np.nan], "hdd": [1., 2., 3., 4.],
                      "H3": [1., 2., 3., 4.]})
    f = add_factors(d)
    assert f.solid_fuel.tolist()[:2] == [1.0, 0.0] and np.isnan(f.solid_fuel[2])
    assert np.isnan(f.old_dwelling[2]) and f.old_dwelling[0] == 1.0 and f.old_dwelling[3] == 1.0 and f.old_dwelling[1] == 0.0
    assert np.isnan(f.large_hh[3]) and f.large_hh[0] == 1.0
