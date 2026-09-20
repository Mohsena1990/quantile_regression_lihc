"""Sensitivity of the headline result to the cost-threshold quantile.

The need-based high-cost flag is "cost above the household's conditional q-quantile given physical need". q is a design
choice (the calibration search returns its lower boundary here, see lib/identification/labels.py), so the headline grid is
rerun for q in {0.50, 0.55, 0.63, 0.70, 0.80} (need_adjusted only; tier T2, income excluded and included, local models,
logistic and two-axis; income rule = bracket <= 3). q also sets how many households count as high cost, so the table
reports the share of households flagged and the smallest class.

Output: outputs/tables/table_cost_quantile_sensitivity.csv
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.identification.data import load_development_frame
from lib.identification.labels import add_labels
from scripts.build_identification_benchmark import run_grid

QS = (0.50, 0.55, 0.63, 0.70, 0.80)


def main() -> None:
    dev = load_development_frame()
    out = []
    for q in QS:
        b, sup, _ = run_grid(dev, cost_kinds=["need_adjusted"], income_states=["excluded", "included"], tiers_wanted=["T2"],
                             models=["prior", "lr4", "axis2"], scopes=("local",), q=q)
        lab = add_labels(dev, cost_kind="need_adjusted", q=q)
        share = lab.groupby("country").hc.mean().rename("high_cost_share")
        b = b.merge(share, left_on="country", right_index=True).merge(sup[["country", "min_class_n"]], on="country")
        out.append(b.assign(q=q))
    t = pd.concat(out, ignore_index=True)
    t.to_csv(ROOT / "outputs/tables/table_cost_quantile_sensitivity.csv", index=False)
    real = t[t.model != "prior"].sort_values("f1_macro", ascending=False).groupby(["q", "income_state", "country"], as_index=False).head(1)
    print(real.pivot_table(index=["income_state", "country"], columns="q", values="f1_macro").round(3).to_string())
    print(real.pivot_table(index="country", columns="q", values="high_cost_share").round(3).to_string())
    print(real.pivot_table(index="country", columns="q", values="min_class_n").to_string())


if __name__ == "__main__":
    main()
