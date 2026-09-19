"""Scenario-simulation table: what does the four-class outcome look like under
each (config, regression method, classifier) choice, per country + Whole? A
robustness/what-if matrix, not new modeling -- every row is read off predictions
scripts/run_percountry_feature_selection.py already wrote
(artifacts/predictions/percountry/{am_id}_{scope}_{classifier}.parquet)
and its metrics table (outputs/tables/table_percountry_feature_selection.csv).

Lets the paper show, e.g., "choosing A0R3C1 instead of A0R1C0 moves Poland's
predicted Double-Risk share from X% to Y%" without any new fitting.

Output:
  outputs/tables/table_scenario_simulation.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.reporting.codes import regression_code, classifier_code, combo_id

RISK_ORDER = ["No Risk", "Income Risk", "Expenditure Risk", "Double Risk"]
PRED_DIR = ROOT / "artifacts/predictions/percountry"


def class_shares(codes: pd.Series) -> dict[str, float]:
    counts = codes.value_counts(normalize=True)
    return {f"share_{RISK_ORDER[k]}": float(counts.get(k, 0.0)) for k in range(4)}


def main() -> None:
    metrics = pd.read_csv(ROOT / "outputs/tables/table_percountry_feature_selection.csv")
    available = metrics[metrics.status.eq("AVAILABLE")]

    rows = []
    for row in available.itertuples():
        p = PRED_DIR / f"{row.am_id}_{row.country}_{row.classifier}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        r_code = regression_code(row.family)
        c_code = classifier_code(row.classifier)
        truth_shares = {f"truth_{k}": v for k, v in class_shares(d.truth).items()}
        pred_shares = {f"predicted_{k}": v for k, v in class_shares(d.prediction).items()}
        rows.append({
            "scope": row.country, "a_config": row.config, "regression_code": r_code, "family": row.family,
            "classifier_code": c_code, "classifier": row.classifier,
            "combo_id": combo_id(row.config, row.family, row.classifier), "n": len(d),
            "macro_f1": row.macro_f1, "double_risk_f1": row.double_risk_f1,
            "double_risk_recall": row.double_risk_recall, "balanced_accuracy": row.balanced_accuracy,
            "log_loss": row.log_loss, **truth_shares, **pred_shares,
        })

    table = pd.DataFrame(rows).sort_values(["scope", "regression_code", "a_config", "classifier_code"])
    table.to_csv(ROOT / "outputs/tables/table_scenario_simulation.csv", index=False)
    print(f"Wrote {len(table)} scenario-simulation rows ({table.scope.nunique()} scopes x "
          f"{table.combo_id.nunique()} combos) -> "
          f"outputs/tables/table_scenario_simulation.csv")


if __name__ == "__main__":
    main()
