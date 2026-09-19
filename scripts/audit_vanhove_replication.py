"""Van Hove et al. replication check.

Van Hove's own CatBoost classifier used exactly 7 predictors (their Table 1):
income bracket, floor area, household size, house detachment, house age,
respondent birth year, and heating strategy -- crucially WITH income included as a
predictor (their own stated design choice: "income is both used to label and to
classify the data"). They report accuracy above ~71%. This project's production
predictor set (A0, see scripts/build_a0_predictor_selection.py) deliberately
excludes income, so a raw accuracy comparison against their number is
apples-to-oranges. This script builds the closest available replica of their exact
7-predictor, income-included setup on this project's own cohort and labels
(constructed the same way: harmonized income percentile <=0.30, country-Q80 cost
threshold -- see lib/labels/income_harmonization.py) and reports accuracy on the
same footing, to separate "does their mechanism reproduce on our data" from "are
our models worse."

Outputs:
  outputs/tables/table_vanhove_replication.csv
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.enable import HOUSEHOLD_SHEET, retained_cohort
from lib.features.household_features import BASE_RAW_COLUMNS, RAW_COLUMNS as FEATURE_RAW_COLUMNS, engineer
from lib.labels.income_harmonization import harmonized_income_percentile
from lib.screening.models import fit_screening_model

COMP = [f"S1{g}c{i}" for g in ("A", "B") for i in (1, 2, 3)]
VANHOVE_BASE = ["harmonized_income_percentile", "H3", "household_size", "H1", "H2",
                "respondent_age", "primary_heating_fuel"]
CLASSIFIERS = ("catboost", "hierarchical_catboost")


def load_frame() -> pd.DataFrame:
    base = pd.read_parquet(ROOT / "artifacts/data/harmonised_observed_cost.parquet")
    cols = ["UID", "Country", "NUTS1"] + BASE_RAW_COLUMNS + FEATURE_RAW_COLUMNS + COMP
    hh = retained_cohort(pd.read_excel(ROOT / "data/raw/ENABLE.EU_dataset_survey of households.xlsx",
                                        sheet_name=HOUSEHOLD_SHEET, usecols=list(dict.fromkeys(cols))))
    hh["household_size"] = hh[COMP].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=len(COMP))
    hh = engineer(hh)
    d = base.merge(hh.drop(columns=["Country", "NUTS1", "country"]), on="UID", how="left", validate="one_to_one")
    d["harmonized_income_percentile"] = harmonized_income_percentile(d.income_bracket, d.country)
    m0 = pd.read_parquet(ROOT / "artifacts/predictions/m0_predictive.parquet")[["UID", "split", "risk_class_code"]]
    d = d.merge(m0, on="UID", how="inner", validate="one_to_one")
    return d.rename(columns={"split": "partition"})


def main() -> None:
    d = load_frame()
    d = d[d.risk_class_code.notna()].copy()
    d["risk_class_code"] = d.risk_class_code.astype(int)
    train = d[d.partition.eq("train")]
    val = d[d.partition.eq("validation")]

    rows = []
    for clf in CLASSIFIERS:
        _, pred, metrics = fit_screening_model(clf, train, val, features=VANHOVE_BASE)
        metrics["accuracy"] = float(accuracy_score(pred.truth, pred.prediction))
        rows.append({"classifier": clf, "predictors": ";".join(VANHOVE_BASE), "n_predictors": len(VANHOVE_BASE),
                     **metrics})
        print(f"{clf}: accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
              f"(Van Hove reports accuracy > ~0.71)")

    pd.DataFrame(rows).to_csv(ROOT / "outputs/tables/table_vanhove_replication.csv", index=False)
    print("Wrote outputs/tables/table_vanhove_replication.csv")


if __name__ == "__main__":
    main()
