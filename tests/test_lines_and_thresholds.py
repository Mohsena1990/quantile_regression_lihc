"""Lines and thresholds: the optimisation helpers, the four line families, climate x home features, the placebo, the stage design
and the guarantee that no predictor is built from the label."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_pareto_and_topsis():
    from build_threshold_moo import pareto, topsis
    df = pd.DataFrame({"a": [1, 2.5, 3, 1], "b": [3, 2.5, 1, 1], "c": [1, 1, 1, 1]})
    keep = pareto(df, ["a", "b", "c"])
    assert keep.tolist() == [True, True, True, False]                 # (1,1) is dominated by every other point
    s = topsis(df[keep], ["a", "b", "c"])
    assert s.argmax() == 1                                             # the balanced compromise wins with equal weights


def _toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"country": np.where(np.arange(n) < n // 2, "A", "B"), "H3": rng.integers(1, 7, n),
                      "household_size": rng.integers(1, 6, n), "H1": rng.integers(1, 5, n), "H2": rng.integers(1, 9, n),
                      "insulation_count": rng.integers(0, 4, n), "hdd_hs1617": rng.normal(2500, 300, n),
                      "cdd_s2017": rng.normal(100, 20, n), "income_bracket": rng.integers(1, 11, n).astype(float)})
    d["cost"] = np.exp(6 + 0.25 * d.H3 + rng.normal(0, 0.4, n))
    d["income_eur_total"] = 3000 * d.income_bracket
    return d


@pytest.mark.parametrize("family", ["p80", "needqr", "needqr_noclim", "burden"])
def test_line_families(family):
    from lib.identification.lines import make_labels
    lab = make_labels(_toy(), family, 3, 0.8)
    assert set(lab.y.unique()) <= {0, 1, 2, 3}
    assert (lab.li == (lab.income_bracket <= 3)).all()
    share = lab.hc.mean()
    assert 0.1 < share < 0.35                                          # about 20% above an 80th-percentile line


def test_country_percentile_is_per_country():
    from lib.identification.lines import make_labels
    d = _toy(); d.loc[d.country == "B", "cost"] *= 10                  # B spends 10x more: still 20% high in each country
    lab = make_labels(d, "p80", 3, 0.8)
    assert lab.groupby("country").hc.mean().between(0.15, 0.25).all()


def test_climate_features_vary_by_household_and_follow_region():
    from lib.features.indices import climate_features
    d = pd.DataFrame({"H3": [3, 3, 5], "H2": [2, 7, 2], "H1": [1, 4, 1], "household_size": [2, 2, 4],
                      "hdd_hs1617": [3000.0, 3000.0, 3000.0], "hdd_normal": [2500.0] * 3, "cdd_s2017": [100.0] * 3,
                      "cdd_normal": [50.0] * 3, "rh_hs": [80.0] * 3, "wind_hs": [10.0] * 3, "no_insulation": [1, 0, 1],
                      "solid_fuel_share": [100, 0, 50], "electric_heating_share": [0, 100, 0], "has_air_conditioning": [0, 1, 0],
                      "income_eur_total": [10000.0, 20000.0, 10000.0], "pop_density": [100.0] * 3, "S6": [1, 5, 3]})
    out = climate_features(d)
    assert out.winter_anomaly.iloc[0] == pytest.approx(1.2) and out.summer_anomaly.iloc[0] == 50
    assert out.hdd_x_old.tolist() == [3000, 0, 3000] and out.hdd_x_solid.tolist() == [3000, 0, 1500]
    assert out.cdd_x_ac.tolist() == [0, 100, 0] and out.wind_x_detached.tolist() == [10, 0, 10]
    assert out.heating_need_kwh.iloc[0] > out.heating_need_kwh.iloc[1]  # old detached > new block flat, same size
    hot = climate_features(d.assign(hdd_hs1617=6000.0))
    assert (hot.heating_need_kwh / out.heating_need_kwh).round(6).eq(2).all()   # need scales with the region's HDD


def test_placebo_recomputes_household_climate():
    from lib.identification.stages import placebo_frame
    from lib.climate.b2_indices import B2_CLIMATE
    n = 40
    regions = [f"R{i}" for i in range(4)]
    d = pd.DataFrame({"country": "A", "climate_region": np.repeat(regions, n // 4)})
    for j, c in enumerate(B2_CLIMATE):
        d[c] = d.climate_region.str[1].astype(int) * 100.0 + j + 1
    extra = {"H3": 3, "H2": 2, "H1": 1, "household_size": 2, "no_insulation": 1, "solid_fuel_share": 100, "electric_heating_share": 0,
             "has_air_conditioning": 1, "income_eur_total": 10000.0, "pop_density": 100.0, "S6": 3}
    for k, v in extra.items():
        d[k] = v
    p = placebo_frame(d, seed=3)
    assert (p.groupby("climate_region").hdd_hs1617.first() != d.groupby("climate_region").hdd_hs1617.first()).all()
    assert np.allclose(p.hdd_x_solid, p.hdd_hs1617)                    # interaction follows the permuted climate


def test_stage_design():
    from lib.identification.stages import B1_GROUPS, B2_CANDIDATES
    assert set(B1_GROUPS) == {"household_indices", "enable_top3", "partial_modules", "region"}
    assert set(B1_GROUPS["enable_top3"]) == {"S8", "floor_area_per_person", "H11B"} and B1_GROUPS["region"] == ["region"]
    assert "need_burden" not in B2_CANDIDATES and len(B2_CANDIDATES) == 30


def test_no_predictor_is_built_from_the_label():
    """The label uses the energy cost, its line and the income bracket; no predictor may carry the cost or the line."""
    from lib.identification.stages import UNIVERSE
    from lib.identification.pool import B1
    label_inputs = {"cost", "thr", "hc", "li", "y", "observed_energy_cost_eur_reference", "heating_cost_eur", "electricity_cost_eur",
                    "S9MONTH", "S9YEAR"}
    assert not label_inputs & set(UNIVERSE) and not label_inputs & set(B1)
