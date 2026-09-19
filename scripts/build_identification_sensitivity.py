"""Sensitivity of the headline result to the income-axis rule.

Headline = Van Hove's fixed rule (bracket <= 3). Here the SAME reduced grid (need_adjusted cost axis, tier T2,
local per-country models, income excluded and included) is rerun under the other rules:
  percentile   frozen survey-weighted empirical bracket percentile <= 0.30
  equiv60      DYNAMIC: household-size-adjusted income proxy below 60% of the Eurostat national median
  residual60   DYNAMIC: (income proxy - energy cost) per equivalent adult below that line
plus the headline itself, so every metric is comparable row by row. The dynamic rules use an EUR proxy built from an
ordinal bracket, so they are sensitivity analyses, never the headline (see lib/labels/income_harmonization.py).

Output: outputs/tables/table_income_rule_sensitivity.csv (all metrics with CIs) and
        table_income_rule_support.csv (class support and low-income share per rule)
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.identification.data import load_development_frame
from lib.labels.income_harmonization import INCOME_RULES
from scripts.build_identification_benchmark import run_grid

OUT = ROOT / "outputs/tables"


def main() -> None:
    dev = load_development_frame()
    bench, support = [], []
    for rule in INCOME_RULES:
        b, s, _ = run_grid(dev, cost_kinds=["need_adjusted"], income_states=["excluded", "included"], tiers_wanted=["T2"],
                           models=["prior", "lr4", "axis2"], scopes=("local",), income_rule=rule)
        bench.append(b); support.append(s)
    bench = pd.concat(bench, ignore_index=True)
    bench.to_csv(OUT / "table_income_rule_sensitivity.csv", index=False)
    support = pd.concat(support, ignore_index=True)
    lab_share = []
    from lib.identification.labels import add_labels
    for rule in INCOME_RULES:
        e = add_labels(dev, cost_kind="need_adjusted", income_rule=rule)
        lab_share.append(e.groupby("country").li.mean().rename("low_income_share").reset_index().assign(income_rule=rule))
    support.merge(pd.concat(lab_share), on=["country", "income_rule"]).to_csv(OUT / "table_income_rule_support.csv", index=False)
    real = bench[bench.model.ne("prior")].sort_values("f1_macro", ascending=False).groupby(["income_rule", "income_state", "country"], as_index=False).head(1)
    print(real.pivot_table(index=["income_state", "country"], columns="income_rule", values="f1_macro").round(3).to_string())
    print(real.pivot_table(index=["income_state", "country"], columns="income_rule", values="accuracy").round(3).to_string())


if __name__ == "__main__":
    main()
