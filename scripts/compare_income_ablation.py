"""Van-Hove-style income-included ablation: how much of any classification-quality
gap against Van Hove et al. (M0/R0's benchmark, whose own 7 predictors include
income bracket -- "income is both used to label and to classify the data", by the
paper's own account) is explained by that income-as-predictor shortcut, rather than
genuine signal in the other, non-income household features this project's M2/M5/M7
champions rely on?

This project deliberately excludes income from the predictor set everywhere else
(tests/test_governance.py enforces it) because a classifier that can see income
gets to trivially relearn the low_income<=boundary half of the four-class label,
which is not a meaningful test of "can risk be identified from other observables."
Van Hove's own design does the opposite on purpose. This script adds one explicit,
clearly-labelled arm that follows Van Hove's choice, so the two design philosophies
can be compared directly rather than only argued about.

The income predictor used here is `harmonized_income_percentile`
(lib/labels/income_harmonization.py) -- the same cross-country-comparable percentile
this project's low_income label itself now uses -- NOT the raw ENABLE bracket
integer, which carries the country-specific bracket-scale problem that module fixes
(see its docstring: a flat bracket cutoff produced a 22%-80% "low income" share
spread across countries that was a scale artifact, not a real income difference;
feeding that same raw, non-comparable integer to a classifier as "the Van Hove
income predictor" would reintroduce the same distortion one country at a time).

Deliberately lighter-weight than the production screening matrix (single classifier,
CatBoost/cost_sensitive, point estimates not bootstrapped CIs) -- same diagnostic-
weight tradeoff scripts/build_quantile_sensitivity.py already makes, appropriate for
a targeted methodological probe rather than a permanent frozen pipeline axis.

Outputs:
  outputs/tables/table_income_ablation.csv   (family x country x {income_excluded, income_included}: mae/r2/macro_f1)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.enable import HOUSEHOLD_SHEET, retained_cohort
from lib.features.household_features import BASE_RAW_COLUMNS, RAW_COLUMNS as FEATURE_RAW_COLUMNS, engineer
from lib.labels.income_harmonization import harmonized_income_percentile
from lib.reporting.codes import regression_code
from lib.screening.models import fit_screening_model
from measurement.families.model_based import fit_measurement_strategy

COUNTRIES = ("Bulgaria", "Hungary", "Italy", "Poland", "Spain")
COMP = [f"S1{g}c{i}" for g in ("A", "B") for i in (1, 2, 3)]


def load_household_frame(cfg: dict) -> pd.DataFrame:
    base = pd.read_parquet(ROOT / "artifacts/data/harmonised_observed_cost.parquet")
    cols = ["UID", "Country", "NUTS1"] + BASE_RAW_COLUMNS + FEATURE_RAW_COLUMNS + COMP
    hh = retained_cohort(pd.read_excel(ROOT / "data/raw/ENABLE.EU_dataset_survey of households.xlsx",
                                        sheet_name=HOUSEHOLD_SHEET, usecols=list(dict.fromkeys(cols))))
    hh["household_size"] = hh[COMP].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=len(COMP))
    hh = engineer(hh)
    d = base.merge(hh.drop(columns=["Country", "NUTS1", "country"]), on="UID", how="left", validate="one_to_one")
    split = json.loads((ROOT / cfg["split"]).read_text())["assignments"]
    part = {str(x["household_id"]): x["partition"] for x in split}
    d["partition"] = d.UID.astype(str).map(part)
    d["harmonized_income_percentile"] = harmonized_income_percentile(d.income_bracket, d.country)
    return d


def fit_and_score(family: str, train: pd.DataFrame, apply: pd.DataFrame, *, features: list[str],
                   quantile: float, target: str) -> tuple[pd.DataFrame, dict]:
    _, measured = fit_measurement_strategy(family, train, apply, features=features, target=target,
                                            quantile=quantile, tune=False)
    val = measured[measured.partition.eq("validation") & measured.measurement_eligible]
    train_m = measured[measured.partition.eq("train") & measured.measurement_eligible]
    valid = val.observed_energy_cost_eur_reference.notna() & val.predicted_cost.notna()
    cost = {"mae": float(mean_absolute_error(val.observed_energy_cost_eur_reference[valid], val.predicted_cost[valid])),
            "r2": float(r2_score(val.observed_energy_cost_eur_reference[valid], val.predicted_cost[valid]))}
    macro_f1 = None
    if len(val) >= 10 and train_m.risk_class_code.nunique() >= 2:
        _, _, metrics = fit_screening_model("catboost", train_m, val, features=features)
        macro_f1 = metrics["macro_f1"]
    return measured, {**cost, "macro_f1": macro_f1}


def main() -> None:
    cfg = json.loads((ROOT / "config/pipeline.json").read_text())
    h_features = list(cfg["features"])
    champions = pd.read_csv(ROOT / "outputs/tables/table_family_champions.csv")
    champions = champions[champions.status.eq("AVAILABLE") & champions.family.ne("M0")]

    d = load_household_frame(cfg)
    rows = []
    for champ in champions.itertuples():
        family, quantile = champ.family, float(champ.quantile)
        train = d[d.partition.eq("train")]
        for arm, features in (("income_excluded", h_features), ("income_included", h_features + ["harmonized_income_percentile"])):
            try:
                measured, scores = fit_and_score(family, train, d, features=features, quantile=quantile,
                                                  target=cfg["target"])
            except Exception as e:
                print(f"FIT_FAIL {family} {arm}: {type(e).__name__}: {e}")
                continue
            val = measured[measured.partition.eq("validation") & measured.measurement_eligible]
            for country in list(COUNTRIES) + ["Five-country overall"]:
                g = val if country == "Five-country overall" else val[val.country.eq(country)]
                if len(g) < 5:
                    continue
                valid = g.observed_energy_cost_eur_reference.notna() & g.predicted_cost.notna()
                rows.append({
                    "regression_code": regression_code(family), "family": family, "arm": arm, "country": country,
                    "n": int(valid.sum()),
                    "mae_eur": float(mean_absolute_error(g.observed_energy_cost_eur_reference[valid], g.predicted_cost[valid])),
                    "r2": float(r2_score(g.observed_energy_cost_eur_reference[valid], g.predicted_cost[valid])),
                    "macro_f1": scores["macro_f1"] if country == "Five-country overall" else None,
                })
        print(f"DONE {family}", flush=True)

    table = pd.DataFrame(rows).sort_values(["regression_code", "country", "arm"])
    table.to_csv(ROOT / "outputs/tables/table_income_ablation.csv", index=False)
    print(f"Wrote {len(table)} income-ablation rows -> outputs/tables/table_income_ablation.csv")


if __name__ == "__main__":
    main()
