"""Country-level economic context features (the A2' config: HC + country context).

Every value here is a single number per country (2018 Eurostat/EC reference data
already governed under artifacts/reference/ -- see scripts/prepare_country_references.py,
scripts/prepare_energy_prices.py). Because these are constants within a country,
they are POOLED-ONLY: inside a single country's own model they carry zero variance
and therefore zero information, so they are never added to a per-country feature
config (see scripts/run_percountry_feature_selection.py).
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]

CONTEXT_COLUMNS = ["ctx_electricity_price", "ctx_gas_price", "ctx_income_d5_eur",
                   "ctx_rent_burden_pct", "ctx_owner_occupied_pct"]


def country_context_table() -> pd.DataFrame:
    """One row per country, columns = CONTEXT_COLUMNS."""
    prices = pd.read_csv(ROOT / "artifacts/reference/historical_energy_prices.csv")
    price_by_fuel = prices.groupby(["country", "fuel"]).price_eur_per_kwh.mean().unstack("fuel")

    income = pd.read_csv(ROOT / "artifacts/reference/income_quantiles_2018.csv")
    d5 = income.loc[income["quantile"].eq("D5"), ["country", "cutoff_eur"]].set_index("country").cutoff_eur

    rent = pd.read_csv(ROOT / "artifacts/reference/eu_silc_rent_burden_2018.csv")
    rent_burden = rent.loc[rent.hhcomp.eq("TOTAL") & rent.rskpovth.eq("A_60") & rent.unit.eq("PC"),
                            ["country", "value"]].set_index("country").value

    tenure = pd.read_csv(ROOT / "artifacts/reference/eu_silc_tenure_distribution_2018.csv")
    owned = tenure.loc[tenure.hhcomp.eq("TOTAL") & tenure.rskpovth.eq("B_60") & tenure.unit.eq("PC")
                        & tenure.tenure.isin(["OWN_L", "OWN_NL"])].groupby("country").value.sum()

    out = pd.DataFrame({
        "ctx_electricity_price": price_by_fuel.get("electricity"),
        "ctx_gas_price": price_by_fuel.get("gas"),
        "ctx_income_d5_eur": d5,
        "ctx_rent_burden_pct": rent_burden,
        "ctx_owner_occupied_pct": owned,
    })
    return out.reset_index().rename(columns={"index": "country"})


def add_country_context(frame: pd.DataFrame, *, country_col: str = "country") -> pd.DataFrame:
    """Left-join CONTEXT_COLUMNS onto `frame` by country."""
    return frame.merge(country_context_table(), left_on=country_col, right_on="country", how="left",
                        suffixes=("", "_ctx"))
