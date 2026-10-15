"""Observed annual energy cost (EUR) for the version-5 cohort (four countries + Spain), built exactly as
scripts/build_expenditure_m0.py builds it for the four (lib/expenditure/reconstruct.py; Spain uses the euro), without touching
the four-country artefacts, the frozen split (Spain's households are already in it) or the M0 labels.

Output: artifacts/data/harmonised_observed_cost_extended.parquet
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.enable import EXTENDED_COUNTRY_CODES, HOUSEHOLD_SHEET, retained_cohort
from lib.expenditure.reconstruct import annual_electricity_cost, annual_heating_cost, combine_energy_cost

CURRENCY_PER_EUR_SPAIN = 1.0


def main() -> None:
    cols = ["UID", "Country", "NUTS1", "H7A1", "H7A2", "H7A88", "H7AA", "H8A", "H8B", "S9MONTH", "w", "H6A1"]
    d = retained_cohort(pd.read_excel(ROOT / "data/raw/ENABLE.EU_dataset_survey of households.xlsx", sheet_name=HOUSEHOLD_SHEET,
                                      usecols=cols), EXTENDED_COUNTRY_CODES)
    d["income_bracket"] = pd.to_numeric(d.S9MONTH, errors="coerce").where(lambda s: s.between(1, 10))
    heat = d.apply(lambda r: annual_heating_cost(r.H7A2, r.H7A1, r.H7AA, no_heating=(r.H7A88 == 1)), axis=1)
    elec = d.apply(lambda r: annual_electricity_cost(r.H8B, r.H8A), axis=1)
    d["heating_cost_local"], d["electricity_cost_local"] = [h[0] for h in heat], [e[0] for e in elec]
    both = [combine_energy_cost(h, e) for h, e in zip(d.heating_cost_local, d.electricity_cost_local)]
    d["observed_energy_cost_local"] = [b[0] for b in both]
    rates = pd.read_csv(ROOT / "artifacts/reference/currency_conversion.csv").set_index("country").local_currency_per_eur.to_dict()
    rates["Spain"] = CURRENCY_PER_EUR_SPAIN
    rate = d.country.map(rates)
    for c in ("heating_cost", "electricity_cost", "observed_energy_cost"):
        d[f"{c}_eur" if c != "observed_energy_cost" else "observed_energy_cost_eur_reference"] = pd.to_numeric(d[f"{c}_local"]) / rate
    out = d[["UID", "country", "income_bracket", "w", "heating_cost_eur", "electricity_cost_eur", "observed_energy_cost_eur_reference"]]
    out.to_parquet(ROOT / "artifacts/data/harmonised_observed_cost_extended.parquet", index=False)
    print(out.groupby("country").agg(n=("UID", "size"), cost=("observed_energy_cost_eur_reference", "count"),
                                     heat=("heating_cost_eur", "count"), elec=("electricity_cost_eur", "count"),
                                     income=("income_bracket", "count")).to_string())


if __name__ == "__main__":
    main()
