"""Shared fixtures.

`low_income_flag` looks brackets up in the frozen cohort-wide table and (deliberately) raises for any
(country, bracket) it does not know. Several unit tests use toy countries "A"/"B"/"C" with brackets
1-10, so they get a synthetic equal-mass reference for those countries, merged with the real table
when it exists. Tests that pass `reference=` explicitly bypass this entirely.
"""
import pandas as pd
import pytest
import lib.labels.income_harmonization as ih


@pytest.fixture(autouse=True)
def toy_country_income_reference(monkeypatch):
    toy = pd.DataFrame([{"country": c, "bracket": k, "n": 1, "weighted_share": 0.1, "cum_share_below": (k - 1) / 10,
                         "mid_percentile": (k - 0.5) / 10} for c in "ABC" for k in range(1, 11)])
    parts = [toy]
    if ih.BRACKET_REFERENCE_PATH.exists():
        parts.append(pd.read_csv(ih.BRACKET_REFERENCE_PATH))
    merged = pd.concat(parts, ignore_index=True)
    monkeypatch.setattr(ih, "load_bracket_reference", lambda path=ih.BRACKET_REFERENCE_PATH: merged)
