"""Annual energy cost in EUR for every ENABLE household, and the frozen train / validation / test split of the five study countries.

Cost (lib/expenditure/reconstruct.py): heating = the season total, else the monthly cost x months paid, else monthly cost x the
country's median months ("no heating system" = 0); electricity = the annual bill, else the monthly bill x 12; total = heating +
electricity. Converted to EUR with Eurostat ert_bil_eur_a 2017 annual averages (the fixed BGN rate; euro countries 1).
Income bracket = S9MONTH (1-10); for the six extra countries the yearly question S9YEAR fills a missing monthly answer.

The train / validation / test split of the five study countries (60 / 20 / 20 within each country, drawn 2026-08-11) is a fixed,
version-controlled input: config/household_split_v1.csv. This script only checks that every study household is in it.

Outputs: artifacts/reference/currency_conversion.csv; artifacts/data/energy_cost_study.parquet (five study countries),
artifacts/data/energy_cost_all.parquet (all eleven).
"""
from __future__ import annotations
import json, sys
from datetime import date
from pathlib import Path
from urllib.request import urlopen
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.enable import ALL_COUNTRY_CODES, HOUSEHOLD_SHEET, STUDY_COUNTRY_CODES, retained_cohort
from lib.expenditure.reconstruct import annual_electricity_cost, annual_heating_cost, combine_energy_cost
from scripts.prepare_country_references import BASE, parse_jsonstat

CURRENCY = {"Bulgaria": "BGN", "France": "EUR", "Germany": "EUR", "Hungary": "HUF", "Italy": "EUR", "Norway": "NOK", "Poland": "PLN",
            "Serbia": "RSD", "Spain": "EUR", "Ukraine": "UAH", "United Kingdom": "GBP"}
BGN_PER_EUR = 1.95583                       # fixed (currency board) rate
SPLIT = ROOT / "config/household_split_v1.csv"


def currency_reference() -> pd.DataFrame:
    url = f"{BASE}/ert_bil_eur_a?lang=EN&time=2017"
    raw = parse_jsonstat(json.load(urlopen(url, timeout=120)))
    avg = raw.loc[raw.statinfo.eq("AVG")].set_index("currency").value.to_dict()
    rows = []
    for country, cur in CURRENCY.items():
        rate, source = ((1.0, "Euro-area currency") if cur == "EUR" else (BGN_PER_EUR, "EU irrevocable/fixed EUR-BGN rate")
                        if cur == "BGN" else (avg[cur], "Eurostat ert_bil_eur_a annual average"))
        rows.append({"country": country, "currency": cur, "reference_year": 2017, "local_currency_per_eur": rate,
                     "conversion_equation": "EUR = local currency / local_currency_per_eur", "source": source, "source_url": url,
                     "retrieval_date": date.today().isoformat()})
    return pd.DataFrame(rows)


def check_split(d: pd.DataFrame) -> None:
    split = pd.read_csv(SPLIT, dtype={"UID": str})
    missing = set(d.UID.astype(str)) - set(split.UID)
    if missing:
        raise SystemExit(f"SPLIT_INCOMPLETE: {len(missing)} study households are absent from {SPLIT}")


def main() -> None:
    cols = ["UID", "Country", "NUTS1", "H7A1", "H7A2", "H7A88", "H7AA", "H8A", "H8B", "S9MONTH", "S9YEAR", "w"]
    d = retained_cohort(pd.read_excel(ROOT / "data/raw/ENABLE.EU_dataset_survey of households.xlsx", sheet_name=HOUSEHOLD_SHEET,
                                      usecols=cols), ALL_COUNTRY_CODES, require_nuts1=False)
    study = d.Country.isin(STUDY_COUNTRY_CODES)
    d["income_bracket"] = pd.to_numeric(d.S9MONTH, errors="coerce").where(lambda s: s.between(1, 10))
    yearly = pd.to_numeric(d.S9YEAR, errors="coerce").where(lambda s: s.between(1, 10))
    d.loc[~study, "income_bracket"] = d.loc[~study, "income_bracket"].fillna(yearly[~study])
    heat = d.apply(lambda r: annual_heating_cost(r.H7A2, r.H7A1, r.H7AA, no_heating=(r.H7A88 == 1)), axis=1)
    elec = d.apply(lambda r: annual_electricity_cost(r.H8B, r.H8A), axis=1)
    d["heating_cost_local"], d["electricity_cost_local"] = [h[0] for h in heat], [e[0] for e in elec]
    d["observed_energy_cost_local"] = [combine_energy_cost(h, e)[0] for h, e in zip(d.heating_cost_local, d.electricity_cost_local)]
    rates = currency_reference()
    (ROOT / "artifacts/reference").mkdir(parents=True, exist_ok=True)
    rates.to_csv(ROOT / "artifacts/reference/currency_conversion.csv", index=False)
    rate = d.country.map(rates.set_index("country").local_currency_per_eur)
    d["heating_cost_eur"] = pd.to_numeric(d.heating_cost_local) / rate
    d["electricity_cost_eur"] = pd.to_numeric(d.electricity_cost_local) / rate
    d["observed_energy_cost_eur_reference"] = pd.to_numeric(d.observed_energy_cost_local) / rate
    out = d[["UID", "country", "income_bracket", "w", "heating_cost_eur", "electricity_cost_eur", "observed_energy_cost_eur_reference"]]
    (ROOT / "artifacts/data").mkdir(parents=True, exist_ok=True)
    out.to_parquet(ROOT / "artifacts/data/energy_cost_all.parquet", index=False)
    out[study.to_numpy()].reset_index(drop=True).to_parquet(ROOT / "artifacts/data/energy_cost_study.parquet", index=False)
    check_split(d[study])
    print(out.groupby("country").agg(n=("UID", "size"), cost=("observed_energy_cost_eur_reference", "count"),
                                     income=("income_bracket", "count")).to_string())


if __name__ == "__main__":
    main()
