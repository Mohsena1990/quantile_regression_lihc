"""Climate indices, climate regions, the three classifiers, nested selection tools, engineered features, hidden under-heating and
the reference-replica scores."""
import numpy as np
import pandas as pd
import pytest
from lib.climate.b2_indices import HDD_BASE, days_in_runs, longest_run, normal_indices, survey_indices
from lib.geography.climate_regions import climate_region
from lib.identification.required_heat import hidden_underheating
from lib.identification.selection import paired_delta, rank_candidates
from lib.identification.stages import B0_FEATURES


def test_run_helpers():
    assert longest_run([0, 1, 1, 0, 1, 1, 1, 0]) == 3
    assert longest_run([]) == 0
    assert days_in_runs([1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1], 3) == 7      # runs of 2 (dropped), 3 and 4
    assert days_in_runs([1, 1, 1], 3) == 3                              # run touching the end counts


def _om_daily(tmean=10.0):
    idx = pd.date_range("2016-10-01", "2017-12-31", freq="D")
    d = pd.DataFrame(index=idx)
    d["temperature_2m_mean"] = tmean
    d["temperature_2m_min"], d["temperature_2m_max"] = d.temperature_2m_mean - 5, d.temperature_2m_mean + 5
    d["apparent_temperature_mean"], d["apparent_temperature_min"] = d.temperature_2m_mean - 2, d.temperature_2m_min - 2
    for c, v in (("relative_humidity_2m_mean", 80), ("dew_point_2m_mean", 5), ("precipitation_sum", 2), ("snowfall_sum", 0.5),
                 ("wind_speed_10m_mean", 10), ("wind_gusts_10m_max", 30), ("shortwave_radiation_sum", 8),
                 ("sunshine_duration", 3 * 3600), ("cloud_cover_mean", 60)):
        d[c] = float(v)
    return d


def test_survey_indices_constant_climate():
    out = survey_indices(_om_daily())
    assert out["hdd_hs1617"] == pytest.approx((HDD_BASE - 10) * 212)    # 212 days Oct 2016 - Apr 2017
    assert out["cdd_s2017"] == 0 and out["heatwave_days"] == 0 and out["tropical_nights"] == 0
    assert out["frost_days_hs"] == 0 and out["cold_spell_max"] == 0
    assert out["precip_hs"] == pytest.approx(2 * 212) and out["sunshine_hs"] == pytest.approx(3)


def test_survey_indices_heatwave_and_cold_spell():
    d = _om_daily()
    d.loc["2017-01-05":"2017-01-14", ["temperature_2m_mean", "temperature_2m_min"]] = [-3, -8]
    d.loc["2017-07-01":"2017-07-04", "temperature_2m_max"] = 31            # 4-day heatwave
    d.loc["2017-08-01":"2017-08-02", "temperature_2m_max"] = 31            # too short
    out = survey_indices(d)
    assert out["cold_spell_max"] == 10 and out["frost_days_hs"] == 10 and out["heatwave_days"] == 4


def test_survey_indices_reject_gaps():
    with pytest.raises(ValueError):
        survey_indices(_om_daily().drop(pd.Timestamp("2017-01-15")))


def test_normal_indices_monthly():
    t = pd.Series(25.0, index=pd.date_range("1991-01-01", "2020-12-01", freq="MS"))
    out = normal_indices(t)
    assert out["hdd_normal"] == 0 and out["cdd_normal"] == pytest.approx(3 * 153)
    cold = normal_indices(pd.Series(8.0, index=t.index))
    assert cold["hdd_normal"] == pytest.approx(10 * (212 + 8 / 29))   # Oct-Apr: 212 days, 213 in the 8 leap seasons (Feb 1992 ... Feb 2020)


def test_climate_region_mapping():
    f = pd.DataFrame({"country": ["Hungary", "Hungary", "Poland", "Italy"], "NUTS2": ["HU10", "HU23", "PL21", "ITC1"],
                      "NUTS3": ["HU101", "HU231", np.nan, "ITC11"]})
    assert climate_region(f).tolist() == ["HU110", "HU231", "PL21", "ITC11"]


def _toy_country(n=240, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"H3": rng.integers(1, 7, n), "household_size": rng.integers(1, 6, n), "H1": rng.integers(1, 5, n),
                      "H2": rng.integers(1, 9, n), "hdd": rng.normal(2500, 100, n), "cdd": rng.normal(100, 10, n)})
    d["cost"] = np.exp(6 + 0.25 * d.H3 + 0.1 * d.household_size + rng.normal(0, 0.4, n))
    d["income_bracket"] = rng.integers(1, 11, n).astype(float)
    d["country"] = "A"
    return d


def test_three_models_and_attention():
    from lib.identification.models import fit
    d = _toy_classification(240, 4)
    d["cat"] = np.where(d.signal > 0.5, "a", "b")
    feats = ["base", "signal", "noise1", "cat"]
    for kind in ("catboost", "xgboost", "ft_transformer"):
        m = fit(kind, d.iloc[:180], feats, seed=1)
        p = m.predict_proba(d.iloc[180:])
        assert p.shape == (60, 4) and np.allclose(p.sum(1), 1, atol=1e-5)
        imp = m.importance(d.iloc[180:])
        assert list(imp.index) == feats and (imp >= 0).all()


def test_hidden_underheating_scale_free():
    """Only relative need matters: scaling HDD (or any common factor) leaves the flag unchanged."""
    d = _toy_country(seed=3).rename(columns={"hdd": "hdd_hs1617"})
    d["primary_heating_fuel"] = "gas"
    a = hidden_underheating(d)
    b = hidden_underheating(d.assign(hdd_hs1617=d.hdd_hs1617 * 3.7))
    assert a.hidden_underheating.tolist() == b.hidden_underheating.tolist()
    assert 0 < a.hidden_underheating.mean() < 0.5


def _toy_classification(n=300, seed=1):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({f: rng.normal(size=n) for f in ["signal", "noise1", "noise2", "noise3"]})
    d["base"] = rng.normal(size=n)
    y = (d.signal > 0).astype(int) + 2 * (d.base > 0).astype(int)
    d["y"] = y
    return d


def test_ranking_finds_the_informative_candidate():
    d = _toy_classification()
    ranking = rank_candidates(d, ["base"], ["noise1", "signal", "noise2", "noise3"], seed=5)
    assert ranking.index[0] == "signal"
    assert np.isclose(ranking[["weight_attention", "weight_shap_catboost", "weight_shap_xgboost"]].iloc[0].sum(), 1)


def test_paired_delta_zero_for_identical_predictions():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 120)
    p = rng.dirichlet(np.ones(4), size=(2, 120))
    out = paired_delta(y, p, p, n_boot=50)
    assert out["delta_f1_macro"] == 0 and out["delta_f1_macro_lo"] == 0 and out["p_improve_f1_macro"] == 0


def test_b0_is_the_core_profile():
    """The core profile (the predictor set of the reference study, Table 1): heating strategy is H9 (categorical), birth year is S4."""
    assert B0_FEATURES == ["income_bracket", "H3", "household_size", "H1", "H2", "birth_year", "heating_strategy"]


def test_one_se_rule():
    from lib.identification.selection import one_se_choice
    curve = {0: (0.40, 0.02), 2: (0.45, 0.02), 4: (0.47, 0.03), 8: (0.48, 0.03)}
    assert one_se_choice(curve) == 2            # best k=8: 0.48 - SE 0.03 = 0.45; the smallest k reaching 0.45 is 2


def test_placebo_climate_moves_every_region():
    from lib.identification.selection import placebo_climate
    regions = [f"R{i}" for i in range(6)]
    f = pd.DataFrame({"country": "A", "climate_region": np.repeat(regions, 3)})
    f["hdd"] = f.climate_region.str[1].astype(int) * 100.0
    p = placebo_climate(f, ["hdd"], seed=4)
    by_region = p.groupby("climate_region").hdd.agg(["nunique", "first"])
    assert (by_region["nunique"] == 1).all()                                     # one value per region
    assert sorted(by_region["first"]) == sorted(f.groupby("climate_region").hdd.first())   # same set of values
    assert (by_region["first"] != f.groupby("climate_region").hdd.first()).all()            # no region keeps its own


def test_engineered_features_codes():
    from lib.features.engineered import RAW_EXTRA, engineer_extra
    row = {c: 0 for c in RAW_EXTRA}
    row.update({"H4A": 2, "H4B": 3, "H4C": 1, "H4D": 1, "H4E": 1, "H4F": 2, "H4G": 1, "H4H": 99,     # 1 petrol + 2 diesel, 1 moto
                "H5A1": 1, "H5A3": 1, "H6A3": 70, "H6A4": 30, "H11A": 4, "H11B": 5, "H11C": 2, "H11D": 4, "H11E": 5,
                "H11F": 1, "H11G": 99, "H11H": 2, "H12A": 5, "H12B": 1, "H13A": 2, "H13C": 1, "H3": 3})
    f = pd.DataFrame([row]).assign(household_size=2, elderly_count=2, children_count=0)
    o = engineer_extra(f).iloc[0]
    assert o.car_count == 3 and o.motor_vehicle_count == 4 and o.insulation_count == 2
    assert o.gas_heating_share == 70 and o.solid_fuel_share == 30 and o.n_heating_sources == 2
    assert o.appliance_count == 5 and o.old_appliance_count == 2            # H11G don't know is not counted
    assert o.has_portable_heater == 1 and o.has_air_conditioning == 0 and np.isnan(o.has_electric_water_heater)
    assert o.efficient_bulb_level == 4 and o.incandescent_level == 0 and o.smart_meter == 1
    assert o.floor_area_per_person == 39 and o.elderly_alone == 1 and o.single_person == 0


def test_replica_scores_match_sklearn():
    from sklearn.metrics import f1_score
    from lib.identification.vanhove_replica import scores
    rng = np.random.default_rng(2)
    y = rng.integers(0, 4, 200)
    p = rng.dirichlet(np.ones(4), 200)
    s = scores(y, p)
    assert s["weighted_f1"] == pytest.approx(f1_score(y, p.argmax(1), average="weighted"))
    assert s["balanced_accuracy"] == pytest.approx(np.mean([s[f"tpr_{k}"] for k in ("no_risk", "income_risk", "expenditure_risk", "double_risk")]))


def test_paired_gate():
    from lib.identification.selection import paired_gate
    base = np.array([0.50, 0.52, 0.48, 0.51, 0.49, 0.50])
    assert paired_gate(4, {0: base, 4: base + 0.03}) == 4                       # consistent gain in every fold
    assert paired_gate(4, {0: base, 4: base + np.array([0.04, -0.03, 0.02, -0.02, 0.01, -0.01])}) == 0   # noise
    assert paired_gate(0, {0: base}) == 0
