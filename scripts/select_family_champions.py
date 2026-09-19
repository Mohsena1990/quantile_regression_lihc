"""Collapse the 7 raw M0/A0/A1 measurements into 4 measurement-family champions.

Two selection questions, kept separate per the project's measurement-vs-screening
distinction (see AUDIT_AND_REPAIR_REPORT.md and docs/methodology_sources.md):

1. Measurement-quality champion (per M2/M5/M7 family): which of the 2 raw configs
   (feature in {A0, A1}), each already calibrated at its own independently-searched
   quantile (lib/tuning/quantile_search.py), is best CALIBRATED, using the
   five-fold OOF absolute calibration error (outputs/tables/table_{a0,a1}_oof_calibration.csv,
   "Five-country overall" row) -- i.e. does the (1-q) nominal high-cost rate actually hold on
   held-out data. This never looks at screening performance, so a family cannot buy a
   better headline slot by being easier for a classifier to reproduce.
2. Given each family's single champion measurement, its per-classifier screening
   performance on validation (development stage) -- the condition table (4 screening-
   eligible families x 2 classifiers) plus each family's single best classifier
   (the "family champion" summary).

Requires: A1 OOF calibration (scripts/build_oof_predictions.py --feature A1) and the
full development screening matrix (scripts/run_screening_matrix.py).
"""
import json
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
FAMILIES = json.loads((ROOT / "config/pipeline.json").read_text())["families"]


def champion_for_family(family: str, oof: pd.DataFrame) -> dict:
    rows = oof[(oof.family == family) & (oof.country == "Five-country overall")]
    best = rows.sort_values("absolute_calibration_error").iloc[0]
    return {"family": family, "champion_experiment_id": best["experiment_id"],
            "feature_config": "A1" if best["experiment_id"].endswith("_a1") else "A0",
            "quantile": float(best["quantile"]), "absolute_calibration_error": float(best["absolute_calibration_error"]),
            "status": "AVAILABLE"}


def classifier_rows_for_measurement(measurement: str) -> pd.DataFrame:
    path = ROOT / f"outputs/tables/screening_{measurement}.csv"
    d = pd.read_csv(path)
    return d[d.status.eq("AVAILABLE")]


def main() -> None:
    a0_oof = pd.read_csv(ROOT / "outputs/tables/table_a0_oof_calibration.csv")
    a1_path = ROOT / "outputs/tables/table_a1_oof_calibration.csv"
    oof = pd.concat([a0_oof, pd.read_csv(a1_path)], ignore_index=True) if a1_path.exists() else a0_oof

    champions = [{"family": "M0", "champion_experiment_id": "m0_predictive", "feature_config": None,
                  "quantile": None, "absolute_calibration_error": None, "status": "AVAILABLE"}]
    for family in FAMILIES:
        champions.append(champion_for_family(family, oof))
    champion_table = pd.DataFrame(champions)
    champion_table.to_csv(ROOT / "outputs/tables/table_family_champions.csv", index=False)

    condition_rows, family_final = [], []
    for row in champion_table.itertuples():
        if not row.status == "AVAILABLE":
            continue
        per_classifier = classifier_rows_for_measurement(row.champion_experiment_id)
        for c in per_classifier.itertuples():
            condition_rows.append({"family": row.family, "champion_experiment_id": row.champion_experiment_id,
                                    "classifier": c.model, "macro_f1": c.macro_f1,
                                    "double_risk_f1": c.double_risk_f1, "double_risk_recall": c.double_risk_recall,
                                    "balanced_accuracy": c.balanced_accuracy, "log_loss": c.log_loss})
        best = per_classifier.sort_values("macro_f1", ascending=False).iloc[0]
        family_final.append({"family": row.family, "champion_experiment_id": row.champion_experiment_id,
                              "feature_config": row.feature_config, "quantile": row.quantile,
                              "best_classifier": best.model,
                              "macro_f1": best.macro_f1, "double_risk_f1": best.double_risk_f1,
                              "double_risk_recall": best.double_risk_recall,
                              "balanced_accuracy": best.balanced_accuracy, "log_loss": best.log_loss})
    condition_table = pd.DataFrame(condition_rows)
    condition_table.to_csv(ROOT / "outputs/tables/table_condition_summary.csv", index=False)
    final_table = pd.DataFrame(family_final).sort_values("macro_f1", ascending=False)
    final_table.to_csv(ROOT / "outputs/tables/table_family_champions_final.csv", index=False)

    print(f"Wrote {len(champion_table)} family champions -> "
          f"outputs/tables/table_family_champions.csv")
    print(f"Wrote {len(condition_table)}-condition table ({condition_table.family.nunique()} families x "
          f"2 classifiers) -> outputs/tables/table_condition_summary.csv")
    print(f"Wrote {len(final_table)} family-champion finals -> outputs/tables/table_family_champions_final.csv")
    if len(final_table):
        overall = final_table.iloc[0]
        print(f"Overall best screening measurement family: {overall.family} "
              f"({overall.champion_experiment_id}, {overall.best_classifier}, "
              f"macro_f1={overall.macro_f1:.3f})")


if __name__ == "__main__":
    main()
