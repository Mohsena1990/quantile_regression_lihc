"""Version 6: extra-country regions, multilevel climate test, placebo shuffle, multilevel bill model, comfort labels."""
import numpy as np
import pandas as pd


def test_extra_region_mapping():
    from lib.geography.climate_regions import extra_region
    f = pd.DataFrame({"country": ["Germany", "Germany", "Norway", "Norway", "Serbia", "Ukraine", "France"],
                      "NUTS1": ["5", "16", "NO061", "NO011", np.nan, "11", np.nan],
                      "NUTS2": [np.nan, np.nan, np.nan, np.nan, "Vojvodina", "2", np.nan]})
    assert extra_region(f).tolist()[:6] == ["DEA", "DEG", "NO060", "NO011", "RS12", "UA11"]
    assert pd.isna(extra_region(f).iloc[6])


def _regions(n_reg=40, per=30, slope=1.0, seed=0):
    rng = np.random.default_rng(seed)
    reg = pd.DataFrame({"climate_region": [f"R{i}" for i in range(n_reg)], "country": np.where(np.arange(n_reg) < n_reg // 2, "A", "B"),
                        "hdd_hs1617": rng.uniform(1500, 4000, n_reg), "cdd_s2017": rng.uniform(0, 300, n_reg)})
    reg["u"] = rng.normal(0, 0.15, n_reg)
    d = reg.loc[np.repeat(np.arange(n_reg), per)].reset_index(drop=True)
    d["y"] = slope * np.log(d.hdd_hs1617) + d.u + rng.normal(0, 0.4, len(d))
    return d


def test_shuffle_climate_stays_within_country():
    from lib.identification.multilevel import shuffle_climate
    d = _regions()
    s = shuffle_climate(d, ["hdd_hs1617", "cdd_s2017"], 3)
    for c in ("A", "B"):
        assert sorted(d[d.country == c].hdd_hs1617.unique()) == sorted(s[s.country == c].hdd_hs1617.unique())
    assert (s.groupby("climate_region").hdd_hs1617.nunique() == 1).all()
    assert not np.allclose(d.hdd_hs1617, s.hdd_hs1617)


def test_mixed_model_recovers_climate_slope():
    from lib.identification.multilevel import add_region_terms, fit, s2_region
    d = add_region_terms(_regions(slope=1.0))
    m0, m1 = fit(d, "C(country)"), fit(d, "C(country) + log_hdd + cdd100")
    assert abs(m1.params["log_hdd"] - 1.0) < 0.3
    assert s2_region(m1) < s2_region(m0)                                # climate explains part of the region variance


def test_mixed_bill_model_uses_region_effect():
    from lib.identification.bill_mixed import MixedBillModel
    rng = np.random.default_rng(1)
    n = 900
    d = pd.DataFrame({"country": "A", "climate_region": rng.choice([f"R{i}" for i in range(15)], n), "H3": rng.integers(1, 7, n).astype(float)})
    eff = {f"R{i}": 0.5 * (i % 2) for i in range(15)}
    d["cost"] = np.exp(6 + 0.2 * d.H3 + d.climate_region.map(eff) + rng.normal(0, 0.2, n))
    m = MixedBillModel(d.iloc[:700], ["H3", "country", "region"], 1)
    te = d.iloc[700:]
    hi, lo = te[te.climate_region.map(eff) > 0], te[te.climate_region.map(eff) == 0]
    assert m.mu(hi).mean() - m.mu(lo).mean() > 0.3
    p = m.p_above(te, np.full(len(te), te.cost.median()))
    assert ((p >= 0) & (p <= 1)).all()


def test_comfort_labels():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from build_climate_labels import build_labels
    from unittest import mock
    d = pd.DataFrame({"C1A": [1, 4, 5, 99, np.nan], "C1B": [1, 2, 99, 1, 3], "income_bracket": [2, 2, 8, 1, np.nan],
                      "hdd_hs1617": np.nan})
    with mock.patch("build_climate_labels.hidden_underheating", side_effect=lambda f: f.assign(hidden_underheating=np.nan)):
        out = build_labels(d)
    assert out.cold_home.tolist()[:3] == [0, 1, 1] and out.cold_home.isna().tolist()[3:] == [True, True]
    assert out.hot_home.tolist()[:2] == [1, 0] and np.isnan(out.hot_home.iloc[2])
    assert out.winter_ep.tolist()[:3] == [0, 1, 0] and np.isnan(out.winter_ep.iloc[4])


def test_grouped_splits_hold_out_whole_regions():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from build_climate_power import grouped_splits
    f = pd.DataFrame({"climate_region": np.repeat([f"R{i}" for i in range(12)], 10)})
    splits = grouped_splits(f)
    assert len(splits) == 15
    for tr, te in splits:
        assert not set(f.climate_region.iloc[tr]) & set(f.climate_region.iloc[te])
    for r in range(3):
        te_all = np.concatenate([te for _, te in splits[5 * r:5 * r + 5]])
        assert sorted(te_all) == list(range(len(f)))                    # every household scored once per repeat


def test_power_injection_is_within_country():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from build_climate_power import labels_for
    from unittest import mock
    d = pd.DataFrame({"country": ["A"] * 4 + ["B"] * 4, "hdd_hs1617": [1000, 2000, 1000, 2000, 3000, 3000, 3000, 3000],
                      "cost": 100.0})
    with mock.patch("build_climate_power.make_labels", side_effect=lambda x, *a: x):
        out = labels_for(d, "pct", 3, 0.8, 1.0)
    assert np.isclose(out.cost[1] / out.cost[0], 2.0)                    # elasticity 1: twice the degree days, twice the bill
    assert np.allclose(out.cost[4:], 100.0)                             # a country with one climate is unchanged
    assert np.isclose(np.exp(np.log(out.cost[:4]).mean()), 100.0)       # centred within the country


def test_mixed_bill_model_uses_fisher_scoring():
    from lib.identification import bill_mixed
    assert bill_mixed.OPTIMIZER_COV == "fisher_scoring"
