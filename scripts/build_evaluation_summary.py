"""Consolidated evaluation-metrics table: for each R0-R3 regression method (at its
locked champion quantile), join full continuous cost-model quality (MAE, MAPE, MSE,
RMSE, R^2, Adjusted R^2, plus the quantile-specific pinball loss/empirical coverage
error, on the champion measurement's validation partition) with full downstream
four-class screening quality (Accuracy, Precision, Recall/Sensitivity, F1-score,
ROC-AUC, Log Loss, plus the project's Double-Risk F1/recall and balanced accuracy)
for each of the 2 classifiers -- one reference table covering both regression and
classification evaluation together.

Reuses lib.evaluation.cost_metrics.cost_score() (same metric definitions already used
for the H-vs-HC comparison, extended with mse/mape/r2) and
scripts/select_family_champions.py's already-computed
outputs/tables/table_condition_summary.csv (per champion family x classifier,
macro-F1/Double-Risk/balanced-accuracy/log-loss) for the metrics that
require nothing beyond the aggregate. Accuracy/Precision/Recall/ROC-AUC are computed
directly from the per-row prediction parquets scripts/run_screening.py already wrote
to artifacts/predictions/<champion>_<classifier>_development.parquet --
no re-fitting of the locked validation matrix.

R0 (M0, the Van Hove benchmark) has no continuous predicted-cost column -- it is a
per-country Q80 threshold benchmark, not a conditional cost model -- so its
regression-metric columns are left blank; only its classification metrics are filled.

Adjusted R^2 needs the fitted model's predictor count, which cost_score() (fit on
observed/predicted cost alone) cannot see -- computed here instead from R^2 plus the
champion's feature_config (A0 = config/pipeline.json's `features`;
A1 = A0 + `climate_features`, de-duplicated via stack_features since A0 can already
contain a climate column the recursive search picked up).

"Sensitivity" and "Recall" are the same statistic (macro-averaged here, across all
four risk classes) -- reported once as `recall_sensitivity_macro`, not duplicated.

Output:
  outputs/tables/table_evaluation_summary.csv
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.reporting.codes import regression_code, classifier_code
from lib.evaluation.cost_metrics import cost_score
from lib.features.household_features import stack_features

def n_predictors(feature_config: str, cfg: dict) -> int:
    a0 = list(cfg["features"])
    return len(a0) if feature_config == "A0" else len(stack_features(a0, cfg["climate_features"]))


def regression_metrics(champion_row: pd.Series, cfg: dict) -> dict[str, float | None]:
    p = ROOT / f"artifacts/predictions/{champion_row['champion_experiment_id']}.parquet"
    d = pd.read_parquet(p)
    keys = ("mae", "mse", "rmse", "mape", "r2", "adjusted_r2", "pinball", "empirical_coverage_error")
    if "predicted_cost" not in d.columns or "partition" not in d.columns:
        return dict.fromkeys(keys)
    d = d[d.partition.eq("validation")]
    quantile = champion_row["quantile"]
    scored = cost_score(d, float(quantile))
    n = int((d.observed_energy_cost_eur_reference.notna() & d.predicted_cost.notna()).sum())
    p_features = n_predictors(champion_row["feature_config"], cfg)
    denom = n - p_features - 1
    scored["adjusted_r2"] = float(1 - (1 - scored["r2"]) * (n - 1) / denom) if denom > 0 else None
    return scored


def classification_metrics(row) -> dict[str, float | None]:
    p = ROOT / f"artifacts/predictions/{row.champion_experiment_id}_{row.classifier}_development.parquet"
    keys = ("accuracy", "precision_macro", "recall_sensitivity_macro", "roc_auc_macro_ovr")
    if not p.exists():
        return dict.fromkeys(keys)
    d = pd.read_parquet(p)
    truth, pred = d.truth, d.prediction
    probs = d[[f"prob_{k}" for k in range(4)]].to_numpy()
    try:
        roc_auc = float(roc_auc_score(truth, probs, multi_class="ovr", average="macro", labels=[0, 1, 2, 3]))
    except ValueError:
        roc_auc = None
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "precision_macro": float(precision_score(truth, pred, average="macro", zero_division=0)),
        "recall_sensitivity_macro": float(recall_score(truth, pred, average="macro", zero_division=0)),
        "roc_auc_macro_ovr": roc_auc,
    }


def main() -> None:
    cfg = json.loads((ROOT / "config/pipeline.json").read_text())
    champions = pd.read_csv(ROOT / "outputs/tables/table_family_champions.csv")
    champions = champions[champions.status.eq("AVAILABLE")]
    condition = pd.read_csv(ROOT / "outputs/tables/table_condition_summary.csv")

    reg_metrics_by_family = {}
    for row in champions.itertuples():
        champion_row = champions.loc[champions.family.eq(row.family)].iloc[0]
        reg_metrics_by_family[row.family] = regression_metrics(champion_row, cfg)

    rows = []
    for row in condition.itertuples():
        reg = reg_metrics_by_family.get(row.family, {})
        clf = classification_metrics(row)
        rows.append({
            "regression_code": regression_code(row.family), "family": row.family,
            "champion_experiment_id": row.champion_experiment_id,
            "classifier_code": classifier_code(row.classifier), "classifier": row.classifier,
            "mae_eur": reg.get("mae"), "mape": reg.get("mape"), "mse": reg.get("mse"), "rmse_eur": reg.get("rmse"),
            "r2": reg.get("r2"), "adjusted_r2": reg.get("adjusted_r2"), "pinball_loss": reg.get("pinball"),
            "empirical_coverage_error": reg.get("empirical_coverage_error"),
            "accuracy": clf.get("accuracy"), "precision_macro": clf.get("precision_macro"),
            "recall_sensitivity_macro": clf.get("recall_sensitivity_macro"), "macro_f1": row.macro_f1,
            "roc_auc_macro_ovr": clf.get("roc_auc_macro_ovr"), "log_loss": row.log_loss,
            "double_risk_f1": row.double_risk_f1, "double_risk_recall": row.double_risk_recall,
            "balanced_accuracy": row.balanced_accuracy,
        })
    table = pd.DataFrame(rows).sort_values(["regression_code", "classifier_code"])
    table.to_csv(ROOT / "outputs/tables/table_evaluation_summary.csv", index=False)
    print(f"Wrote {len(table)} evaluation-summary rows ({table.regression_code.nunique()} regression codes x "
          f"{table.classifier_code.nunique()} classifiers) -> outputs/tables/table_evaluation_summary.csv")


if __name__ == "__main__":
    main()
