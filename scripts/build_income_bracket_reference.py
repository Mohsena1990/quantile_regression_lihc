"""Freeze the cohort-wide income-bracket -> percentile table, and audit the v1 defect it fixes.

`lib/labels/income_harmonization.py` v1 recomputed N = max(bracket) over whichever rows were
passed in, so the low-income cutoff moved with the partition (Spain's bracket 3 was low-income
in validation/test but not in train). v2 looks each bracket up in ONE survey-weighted empirical
table built here from the full retained cohort (brackets and weights only -- no cost or outcome
information), so the label of a household never depends on which other households are present.

Outputs:
  artifacts/reference/income_bracket_percentiles.csv   the frozen lookup used by every label
  outputs/tables/table_income_label_shift_audit.csv    low-income share per country x partition,
                                                       v1 (partition-relative) vs v2 (frozen)
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.enable import HOUSEHOLD_SHEET, retained_cohort
from lib.labels.income_harmonization import (BRACKET_REFERENCE_PATH, build_bracket_percentile_table,
                                             harmonized_income_percentile, legacy_subset_relative_percentile)

BOUNDARY = 0.30


def main() -> None:
    raw = pd.read_excel(ROOT / "data/raw/ENABLE.EU_dataset_survey of households.xlsx", sheet_name=HOUSEHOLD_SHEET,
                        usecols=["UID", "Country", "NUTS1", "S9MONTH", "w"])
    hh = retained_cohort(raw)
    table = build_bracket_percentile_table(hh["S9MONTH"], hh["country"], hh["w"])
    BRACKET_REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(BRACKET_REFERENCE_PATH, index=False)

    base = pd.read_parquet(ROOT / "artifacts/data/harmonised_observed_cost.parquet")
    import json
    split = json.loads((ROOT / "artifacts/splits/household_split_v1.json").read_text())["assignments"]
    part = {str(a["household_id"]): a["partition"] for a in split}
    d = base.assign(partition=base.UID.astype(str).map(part))
    d = d[d.observed_energy_cost_eur_reference.notna() & d.income_bracket.notna()]
    d["li_v2"] = harmonized_income_percentile(d.income_bracket, d.country) <= BOUNDARY
    # v1 as the pipeline applied it: per partition frame, N = max bracket within that frame
    v1 = []
    for _, g in d.groupby("partition"):
        v1.append((legacy_subset_relative_percentile(g.income_bracket, g.country) <= BOUNDARY).rename("li_v1"))
    d["li_v1"] = pd.concat(v1).reindex(d.index)
    audit = (d.groupby(["country", "partition"]).agg(n=("UID", "size"), low_income_v1=("li_v1", "mean"),
                                                      low_income_v2=("li_v2", "mean")).reset_index())
    spread = audit.groupby("country").agg(v1_spread=("low_income_v1", lambda s: s.max() - s.min()),
                                          v2_spread=("low_income_v2", lambda s: s.max() - s.min()))
    out = ROOT / "outputs/tables/table_income_label_shift_audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out, index=False)
    print(audit.round(3).to_string(index=False))
    print("\nMax within-country spread across partitions (lower = stabler labels):")
    print(spread.round(3).to_string())
    print(f"\nWrote {BRACKET_REFERENCE_PATH.relative_to(ROOT)} ({len(table)} bracket rows) and {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
