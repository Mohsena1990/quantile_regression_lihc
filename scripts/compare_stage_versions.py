"""Did the selector fixes help? Compare feature-stage results of version 1 (commit a25daff; tables archived in
outputs/tables/archive_v1/) with the current run, and the Hansen + Quantile Regression line with and without S8.

Version 1 ranked candidates on one 80/20 split with equal weight on FT-Transformer attention and SHAP, and sized the set by
macro-F1. Version 2 averages over three splits, weights each importance measure by its rank stability, and sizes the set by
balanced accuracy (lib/identification/stages.py).

Outputs: outputs/tables/table_stage_versions.csv (balanced accuracy and B1 - B0 gain per cell, both versions),
table_stage_noS8.csv (Hansen + Quantile Regression with / without S8 as a candidate).
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "outputs/tables"


def gains(deltas: pd.DataFrame) -> pd.DataFrame:
    d = deltas[deltas.comparison == "B1 vs B0"]
    return d.set_index(["label", "country", "model"])[["delta_balanced_accuracy", "delta_balanced_accuracy_lo", "delta_balanced_accuracy_hi"]]


def main() -> None:
    old_m, new_m = pd.read_csv(TAB / "archive_v1/table_stages_metrics.csv"), pd.read_csv(TAB / "table_stages_metrics.csv")
    keys = ["label", "country", "model", "stage"]
    m = old_m[keys + ["balanced_accuracy"]].merge(new_m[keys + ["balanced_accuracy"]], on=keys, suffixes=("_v1", "_v2"))
    m = m[m.model != "chance"]
    m["change"] = m.balanced_accuracy_v2 - m.balanced_accuracy_v1
    g = gains(pd.read_csv(TAB / "archive_v1/table_stages_deltas.csv")).join(gains(pd.read_csv(TAB / "table_stages_deltas.csv")), lsuffix="_v1", rsuffix="_v2")
    m.to_csv(TAB / "table_stage_versions.csv", index=False)
    g.reset_index().to_csv(TAB / "table_stage_versions_b1_gain.csv", index=False)
    pd.set_option("display.width", 220)
    print("Mean balanced accuracy by stage and model, v1 -> v2:")
    print(m.groupby(["stage", "model"])[["balanced_accuracy_v1", "balanced_accuracy_v2", "change"]].mean().round(3).to_string())
    for v in ("v1", "v2"):
        print(f"{v}: B1 significantly better than B0 in {int((g[f'delta_balanced_accuracy_lo_{v}'] > 0).sum())} cells, "
              f"significantly worse in {int((g[f'delta_balanced_accuracy_hi_{v}'] < 0).sum())} of {len(g)}")
    p = TAB / "table_stages_metrics_noS8.csv"
    if p.exists():
        n = pd.read_csv(p)
        x = new_m[(new_m.label == "hansen_qr") & new_m.stage.isin(["B0", "B1"]) & (new_m.model != "chance")][keys + ["balanced_accuracy"]].merge(
            n[n.stage.isin(["B0", "B1"]) & (n.model != "chance")][keys + ["balanced_accuracy"]], on=keys, suffixes=("_with_S8", "_without_S8"))
        x.to_csv(TAB / "table_stage_noS8.csv", index=False)
        print("Hansen + Quantile Regression, with vs without S8 as a candidate (mean over countries):")
        print(x.groupby(["stage", "model"])[["balanced_accuracy_with_S8", "balanced_accuracy_without_S8"]].mean().round(3).to_string())


if __name__ == "__main__":
    sys.exit(main())
