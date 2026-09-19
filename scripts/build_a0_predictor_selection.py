"""Builds A0 -- this project's standard predictor set -- by recursive (forward)
feature addition, starting from Van Hove et al.'s 6 non-income base predictors
(floor area, household size, house detachment, house age, respondent age, heating
strategy -- their 7th, income bracket, is deliberately excluded here: this project
enforces income out of every production predictor set, see
tests/test_governance.py's FORBIDDEN_SCREENING_COLUMNS and
scripts/compare_income_ablation.py's docstring for why -- the income-included
replication of Van Hove's own design stays a separate, clearly-labelled comparison,
not the production config).

At each step, every remaining candidate feature (this project's full engineered
household set ONLY -- see HOUSEHOLD_CANDIDATE_POOL below) is tried; the one that
improves the AVERAGE of C0 (CatBoost) and C1 (Hierarchical CatBoost) validation
macro-F1 the most is added permanently. Stops when no candidate improves it by
more than TOLERANCE. Run once on the pooled ("Whole") five-country data -- one
shared A0 used everywhere, so cross-country comparisons of any one factor's
effect stay apples-to-apples (this is why A3/A4, the old per-country
auto-selected configs, were removed).

Climate and country-context columns are deliberately NOT candidates here: A0 is
household-only by design, A1 adds climate, A2 adds country context (see README's
"How A0 is built" / run_percountry_feature_selection.py's docstring). An earlier
version of this script let the search reach into climate/context too, which let
A0 pick up ctx_gas_price/frost_days/temperature_mean directly -- breaking every
downstream script that assumes A0-mode fits don't need a climate/context merge
(KeyError: those columns aren't in the frame outside A1/A2 mode).

The winning predictor list is written directly into config/pipeline.json's
`features` key -- this becomes A0. A1 = A0 + climate_features, A2 = A1 + country
context (Whole scope only), unchanged from before. `features_base` and every
H_BASE/HC_BASE ablation script are retired: this recursive search now measures
each feature's marginal value directly, superseding the old base/H/HC ablation.

Outputs:
  outputs/tables/table_a0_recursive_selection.csv  (one row per step, both classifiers' metrics)
  outputs/figures/a0_recursive_selection.png/.pdf
  config/pipeline.json                             (features key overwritten with the winning A0)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.enable import HOUSEHOLD_SHEET, retained_cohort
from lib.features.household_features import (
    BASE_RAW_COLUMNS, RAW_COLUMNS as FEATURE_RAW_COLUMNS, ENGINEERED_COLUMNS, engineer,
)
from lib.reporting.figures import save_figure
from lib.screening.models import fit_screening_model

COMP = [f"S1{g}c{i}" for g in ("A", "B") for i in (1, 2, 3)]
VANHOVE_BASE_NO_INCOME = ["H3", "household_size", "H1", "H2", "respondent_age", "primary_heating_fuel"]
CLASSIFIERS = ("catboost", "hierarchical_catboost")
TOLERANCE = 0.001
MAX_STEPS = 15
SURFACE, GRID, INK, INK_MUTED = "#fcfcfb", "#e1e0d9", "#0b0b0b", "#898781"
# A0's candidate pool: every engineered household column except the PV/solar/heat-pump-tech
# flags (near-zero signal in this cohort, see household_features.py's module docstring) and
# respondent gender (deliberately kept out of the predictor set -- reporting/policy-map
# segmentation only, see household_features.py's GENDER_COLUMN comment). Climate and country
# context are excluded entirely: those are A1/A2, not A0 (see this script's module docstring).
_EXCLUDE_FROM_A0_CANDIDATES = {"has_pv", "has_solar_water_heating", "has_heat_pump_tech",
                                "has_heat_pump_heating", "gender"}
HOUSEHOLD_CANDIDATE_POOL = [c for c in ENGINEERED_COLUMNS if c not in _EXCLUDE_FROM_A0_CANDIDATES]


def load_frame() -> pd.DataFrame:
    base = pd.read_parquet(ROOT / "artifacts/data/harmonised_observed_cost.parquet")
    cols = ["UID", "Country", "NUTS1"] + BASE_RAW_COLUMNS + FEATURE_RAW_COLUMNS + COMP
    hh = retained_cohort(pd.read_excel(ROOT / "data/raw/ENABLE.EU_dataset_survey of households.xlsx",
                                        sheet_name=HOUSEHOLD_SHEET, usecols=list(dict.fromkeys(cols))))
    hh["household_size"] = hh[COMP].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=len(COMP))
    hh = engineer(hh)
    d = base.merge(hh.drop(columns=["Country", "NUTS1", "country"]), on="UID", how="left", validate="one_to_one")
    m0 = pd.read_parquet(ROOT / "artifacts/predictions/m0_predictive.parquet")[["UID", "split", "risk_class_code"]]
    d = d.merge(m0, on="UID", how="inner", validate="one_to_one")
    return d.rename(columns={"split": "partition"})


def score(train: pd.DataFrame, val: pd.DataFrame, features: list[str]) -> dict:
    """Fit both C0 and C1, return each classifier's metrics plus the averaged macro_f1
    that drives selection."""
    out = {}
    for clf in CLASSIFIERS:
        _, pred, metrics = fit_screening_model(clf, train, val, features=features)
        metrics["accuracy"] = float(accuracy_score(pred.truth, pred.prediction))
        out[clf] = metrics
    out["avg_macro_f1"] = sum(out[c]["macro_f1"] for c in CLASSIFIERS) / len(CLASSIFIERS)
    out["avg_accuracy"] = sum(out[c]["accuracy"] for c in CLASSIFIERS) / len(CLASSIFIERS)
    return out


def flatten(step: int, feature_added: str | None, n_features: int, delta: float, m: dict) -> dict:
    row = {"step": step, "feature_added": feature_added, "n_features": n_features,
           "avg_macro_f1": m["avg_macro_f1"], "avg_accuracy": m["avg_accuracy"], "delta_avg_macro_f1": delta}
    for clf in CLASSIFIERS:
        row[f"{clf}_macro_f1"] = m[clf]["macro_f1"]
        row[f"{clf}_accuracy"] = m[clf]["accuracy"]
    return row


def main() -> None:
    cfg = json.loads((ROOT / "config/pipeline.json").read_text())
    d = load_frame()
    d = d[d.risk_class_code.notna()].copy()
    d["risk_class_code"] = d.risk_class_code.astype(int)
    train = d[d.partition.eq("train")]
    val = d[d.partition.eq("validation")]

    print("=== Base: Van Hove's 6 non-income predictors, both classifiers ===")
    base_metrics = score(train, val, VANHOVE_BASE_NO_INCOME)
    rows = [flatten(0, None, len(VANHOVE_BASE_NO_INCOME), 0.0, base_metrics)]
    print(f"avg_macro_f1={base_metrics['avg_macro_f1']:.4f} avg_accuracy={base_metrics['avg_accuracy']:.4f}  "
          f"(C0={base_metrics['catboost']['macro_f1']:.4f} C1={base_metrics['hierarchical_catboost']['macro_f1']:.4f})")

    print("=== Recursive feature addition (both C0 and C1 must both be scored; selection by their average) ===")
    candidates = sorted(set(HOUSEHOLD_CANDIDATE_POOL) - set(VANHOVE_BASE_NO_INCOME))
    selected = list(VANHOVE_BASE_NO_INCOME)
    best_f1 = base_metrics["avg_macro_f1"]
    for step in range(1, MAX_STEPS + 1):
        trial = {}
        for cand in candidates:
            try:
                trial[cand] = score(train, val, selected + [cand])
            except Exception as e:
                print(f"  SKIP {cand}: {type(e).__name__}: {e}")
        if not trial:
            break
        winner = max(trial, key=lambda c: trial[c]["avg_macro_f1"])
        win_f1 = trial[winner]["avg_macro_f1"]
        if win_f1 - best_f1 <= TOLERANCE:
            print(f"STOP at step {step}: best candidate {winner} improves avg macro_f1 by only "
                  f"{win_f1 - best_f1:+.4f} (<= {TOLERANCE})")
            break
        selected.append(winner)
        candidates.remove(winner)
        rows.append(flatten(step, winner, len(selected), win_f1 - best_f1, trial[winner]))
        print(f"step {step}: +{winner}  avg_macro_f1={win_f1:.4f} (+{win_f1 - best_f1:.4f})  "
              f"avg_accuracy={trial[winner]['avg_accuracy']:.4f}")
        best_f1 = win_f1

    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "outputs/tables/table_a0_recursive_selection.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(SURFACE); ax1.set_facecolor(SURFACE)
    ax1.plot(table.n_features, table.avg_macro_f1, marker="o", color="#2a78d6", label="Avg Macro-F1 (C0+C1)")
    ax1.set_xlabel(f"Number of predictors ({len(VANHOVE_BASE_NO_INCOME)} = Van Hove non-income base)", fontsize=9)
    ax1.set_ylabel("Average Macro-F1 (C0+C1)", color="#2a78d6", fontsize=9)
    ax2 = ax1.twinx()
    ax2.plot(table.n_features, table.avg_accuracy, marker="s", color="#eb6834", label="Avg Accuracy (C0+C1)")
    ax2.set_ylabel("Average Accuracy (C0+C1)", color="#eb6834", fontsize=9)
    ax1.axvline(len(VANHOVE_BASE_NO_INCOME), color=INK_MUTED, linestyle="--", linewidth=1)
    for _, row in table.iloc[1:].iterrows():
        ax1.annotate(row.feature_added, (row.n_features, row.avg_macro_f1), fontsize=6.5, rotation=30,
                      xytext=(2, 4), textcoords="offset points", color=INK_MUTED)
    ax1.set_title("A0 construction: recursive feature addition on the Van Hove non-income base\n"
                  "(dashed line = Van Hove base; each point is the single most-improving remaining feature, "
                  "averaged over C0+C1)", fontsize=10)
    ax1.grid(color=GRID, linewidth=0.8, zorder=0)
    fig.tight_layout()
    save_figure(fig, ROOT / "outputs/figures/a0_recursive_selection")
    plt.close(fig)

    cfg["features"] = selected
    cfg.pop("features_base", None)
    (ROOT / "config/pipeline.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"\nA0 = {len(selected)} predictors: {selected}")
    print("Wrote config/pipeline.json (features=A0, features_base removed), "
          "outputs/tables/table_a0_recursive_selection.csv, outputs/figures/a0_recursive_selection.png/.pdf")


if __name__ == "__main__":
    main()
