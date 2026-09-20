import json
import numpy as np
import pandas as pd
import pytest
from lib.identification.final_test import FinalTestLocked, assert_can_load, load_test_frame, record_evaluation
from lib.identification.labels import apply_labels


def _sel(**kw):
    base = {"test_unlock_authorized": True, "authorized_by": "someone", "test_accessed_at": None, "test_evaluated_at": None}
    return {**base, **kw}


def test_locked_by_default_and_needs_a_named_authoriser():
    with pytest.raises(FinalTestLocked, match="false"):
        assert_can_load(_sel(test_unlock_authorized=False))
    with pytest.raises(FinalTestLocked, match="authorized_by"):
        assert_can_load(_sel(authorized_by=None))
    assert_can_load(_sel())            # authorised, never accessed: allowed


def test_a_second_access_is_refused_even_if_the_first_never_finished(tmp_path, monkeypatch):
    path = tmp_path / "sel.json"
    path.write_text(json.dumps(_sel()))
    monkeypatch.setattr("lib.identification.final_test._load_frame", lambda parts: pd.DataFrame({"partition": list(parts)}))
    first = load_test_frame(path)                                   # records test_accessed_at before returning data
    assert list(first.partition) == ["test"] and json.loads(path.read_text())["test_accessed_at"]
    with pytest.raises(FinalTestLocked, match="already accessed"):
        load_test_frame(path)
    record_evaluation("abc", "deadbeef", path)
    assert json.loads(path.read_text())["result_sha256"] == "abc"


def _households(n, seed):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"country": "Bulgaria", "H3": rng.integers(1, 7, n).astype(float), "household_size": rng.integers(1, 6, n).astype(float),
                      "H1": rng.integers(1, 5, n).astype(float), "H2": rng.integers(1, 9, n).astype(float),
                      "hdd": rng.normal(2500, 300, n), "cdd": rng.normal(200, 50, n), "income_bracket": rng.integers(1, 11, n).astype(float)})
    d["income_percentile"] = (d.income_bracket - 0.5) / 10
    d["cost"] = np.exp(0.35 * d.H3 + 0.15 * d.household_size + rng.normal(0, 0.5, n))
    return d


def test_test_thresholds_depend_on_development_data_only():
    dev, test = _households(300, 1), _households(80, 2)
    a = apply_labels(dev, test, cost_kind="need_adjusted")
    test2 = test.assign(cost=test.cost * 50)                        # wildly different test costs...
    b = apply_labels(dev, test2, cost_kind="need_adjusted")
    np.testing.assert_allclose(a.threshold.to_numpy(), b.threshold.to_numpy())      # ...same thresholds
    assert (b.hc.mean() > a.hc.mean())                              # only the labels move
    q = apply_labels(dev, test, cost_kind="raw_q80")
    assert q.threshold.nunique() == 1
    assert abs(q.threshold.iloc[0] - dev.cost.quantile(0.8)) < 1e-9      # Van Hove's Q80 comes from development costs
