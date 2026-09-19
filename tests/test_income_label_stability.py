"""The low-income label of a household must not depend on which other households are present
(regression test for the v1 defect: N = max(bracket) over the rows passed in, which made Spain's
bracket 3 low-income in validation/test but not in train)."""
import numpy as np
import pandas as pd
from lib.labels.income_harmonization import (build_bracket_percentile_table, harmonized_income_percentile,
                                             legacy_subset_relative_percentile, low_income_flag)


def _cohort():
    # country A: brackets 1..10; heavy at the bottom (Hungary-like). country B: brackets 1..8 (Poland-like)
    a = np.repeat(np.arange(1, 11), [40, 30, 10, 6, 4, 3, 3, 2, 1, 1])
    b = np.repeat(np.arange(1, 9), [5, 15, 30, 10, 4, 2, 1, 1])
    return pd.DataFrame({"country": ["A"] * len(a) + ["B"] * len(b), "bracket": np.concatenate([a, b])})


def test_bracket_table_midpoints_are_weighted_cumulative_shares():
    c = _cohort()
    t = build_bracket_percentile_table(c.bracket, c.country)
    a = t[t.country == "A"].set_index("bracket")
    assert abs(a.loc[1, "mid_percentile"] - 0.5 * 40 / 100) < 1e-9          # first bracket: half its own share
    assert abs(a.loc[2, "mid_percentile"] - (40 + 15) / 100) < 1e-9
    w = pd.Series(np.where(c.bracket == 1, 3.0, 1.0), index=c.index)      # weights shift the CDF
    t_w = build_bracket_percentile_table(c.bracket, c.country, w)
    assert t_w[(t_w.country == "A") & (t_w.bracket == 2)].mid_percentile.iloc[0] > a.loc[2, "mid_percentile"]


def test_low_income_flag_is_invariant_to_the_rows_passed_in():
    c = _cohort()
    ref = build_bracket_percentile_table(c.bracket, c.country)
    full = low_income_flag(c.bracket, c.country, rule="percentile", reference=ref)
    subset = c[(c.country == "B") & (c.bracket <= 6)]                       # a "partition" missing B's top brackets
    assert (low_income_flag(subset.bracket, subset.country, rule="percentile", reference=ref) == full.loc[subset.index]).all()


def test_legacy_rule_depended_on_the_subset_the_defect_this_fixes():
    c = _cohort()
    b = c[c.country == "B"]
    small = b[b.bracket <= 6]                                               # max bracket 6 instead of 8
    v_full = legacy_subset_relative_percentile(b.bracket, b.country)
    v_small = legacy_subset_relative_percentile(small.bracket, small.country)
    assert not np.allclose(v_full.loc[small.index], v_small)                # same households, different percentile


def test_out_of_range_brackets_are_missing_not_low_income():
    c = _cohort()
    ref = build_bracket_percentile_table(c.bracket, c.country)
    p = harmonized_income_percentile(pd.Series([98.0, np.nan, 3.0]), pd.Series(["A", "A", "A"]), reference=ref)
    assert p.iloc[:2].isna().all() and p.iloc[2] == p.iloc[2]
    assert not low_income_flag(pd.Series([98.0]), pd.Series(["A"]), rule="percentile", reference=ref).iloc[0]


def test_unmapped_country_or_bracket_raises_instead_of_silently_becoming_not_low_income():
    import pytest
    c = _cohort()
    ref = build_bracket_percentile_table(c.bracket, c.country)
    with pytest.raises(ValueError, match="missing from the frozen"):
        harmonized_income_percentile(pd.Series([3.0]), pd.Series(["Atlantis"]), reference=ref)
    with pytest.raises(ValueError, match="missing from the frozen"):
        harmonized_income_percentile(pd.Series([9.0]), pd.Series(["B"]), reference=ref)   # B has no bracket 9-10 households


def test_headline_bracket3_rule_is_a_fixed_cutoff_independent_of_other_rows():
    b = pd.Series([1.0, 2.0, 3.0, 4.0, 10.0, np.nan, 98.0])
    country = pd.Series(["A"] * 7)
    flag = low_income_flag(b, country)                       # default rule = headline
    assert flag.tolist() == [True, True, True, False, False, False, False]
    assert flag.iloc[:3].all() and low_income_flag(b.iloc[:2], country.iloc[:2]).all()   # a subset gives the same labels


def test_dynamic_rules_move_with_household_size_and_energy_cost():
    from lib.labels.income_harmonization import equivalence_scale, income_proxy_frame
    c = _cohort().assign(country=lambda d: d.country.map({"A": "Bulgaria", "B": "Poland"}))   # needs a real Eurostat country
    ref = build_bracket_percentile_table(c.bracket, c.country)
    n = 4
    bracket = pd.Series([3.0] * n); country = pd.Series(["Bulgaria"] * n)
    adults = pd.Series([1, 2, 2, 2]); kids = pd.Series([0, 0, 2, 4])
    scale = equivalence_scale(adults, kids)
    assert scale.tolist() == [1.0, 1.5, 2.1, 2.7]
    cost = pd.Series([100.0, 100.0, 100.0, 10000.0])
    out = income_proxy_frame(bracket, country, scale, cost=cost)
    assert out.proxy_equiv_eur.is_monotonic_decreasing                 # same bracket, bigger household -> poorer per adult
    assert out.low_income_residual60.iloc[3] and not out.low_income_residual60.iloc[0]   # a huge energy bill pushes below the line
