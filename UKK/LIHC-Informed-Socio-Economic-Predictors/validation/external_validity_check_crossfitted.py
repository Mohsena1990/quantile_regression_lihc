"""
Cross-fitted rerun of external_validity_check.py.

external_validity_check.py already answers the handling editor's
circularity objection (predicting a label from the variables that built
it) by testing HQRTM's flag against markers never used in construction.
But that script's labels are still fit in-sample: every household's
threshold used a model that included that household's own data point.

This script reruns the exact same reclassification and horse-race
analyses (imported, not duplicated, from external_validity_check.py) on
K-fold cross-fitted labels instead (see cross_fitted_labels.py) -- every
household's HQRTM/LIHC label now comes from a model that never saw that
household -- and prints the in-sample vs out-of-fold results side by side
so it's visible whether the findings survive removing that last
in-sample-fit advantage.
"""

from pathlib import Path

import pandas as pd

from cross_fitted_labels import cross_fit_hqrtm, cross_fit_traditional_lihc
from external_validity_check import (
    build_analysis_frame,
    reclassification_table,
    horse_race_regressions,
    MARKERS,
    MARKER_LABELS,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "outputs" / "validation" / "external_validity_results"

QUANTILES = [(0.60, "HQRTM (q=0.60, cross-fitted)"),
             (0.65, "HQRTM (q=0.65, cross-fitted)"),
             (0.70, "HQRTM (q=0.70, cross-fitted)")]

CARRY_COLS = ["Country", "equivalized_income", "household_size", "total_expenditure", "S7", "C5B", "C1A", "S8"]


def _attach_carry_cols(labels: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([source[CARRY_COLS].reset_index(drop=True), labels.reset_index(drop=True)], axis=1)


def run_all() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(REPO_ROOT / "df_hqrtm_65.csv", low_memory=False)

    print("=== Cross-fitting traditional LIHC labels (5-fold, stratified by Country) ===")
    lihc_labels = cross_fit_traditional_lihc(df, exp_quantile=0.80)
    lihc_df_cf = _attach_carry_cols(lihc_labels, df)

    reclass_tables, regression_tables = [], []

    for quantile, label in QUANTILES:
        print(f"\n=== Cross-fitting HQRTM labels, {label} ===")
        hqrtm_labels = cross_fit_hqrtm(df, quantile=quantile)
        hqrtm_df_cf = _attach_carry_cols(hqrtm_labels, df)

        frame = build_analysis_frame(hqrtm_df_cf, lihc_df_cf)
        print("Reclassification group sizes:")
        print(frame["group"].value_counts().to_string())

        reclass_tables.append(reclassification_table(frame, label))
        regression_tables.append(horse_race_regressions(frame, label))

    reclass_all = pd.concat(reclass_tables, ignore_index=True)
    regression_all = pd.concat(regression_tables, ignore_index=True)

    reclass_path = OUTPUT_DIR / "reclassification_marker_prevalence_crossfitted.csv"
    regression_path = OUTPUT_DIR / "horse_race_regressions_crossfitted.csv"
    reclass_all.to_csv(reclass_path, index=False)
    regression_all.to_csv(regression_path, index=False)
    print(f"\nSaved: {reclass_path}")
    print(f"Saved: {regression_path}")

    in_sample_path = OUTPUT_DIR / "horse_race_regressions.csv"
    if not in_sample_path.exists():
        print(f"\n(No in-sample comparison: {in_sample_path} not found -- run external_validity_check.py first.)")
        return

    in_sample = pd.read_csv(in_sample_path)
    in_sample_main = in_sample[in_sample["quantile_spec"] == "HQRTM (q=0.65)"]
    cf_main = regression_all[regression_all["quantile_spec"] == "HQRTM (q=0.65, cross-fitted)"]

    print("\n=== In-sample vs out-of-fold (cross-fitted) horse-race odds ratios, q=0.65 ===")
    for marker in MARKERS:
        print(f"\n{MARKER_LABELS[marker]}")
        for flag in ["hqrtm_high_cost", "lihc_high_cost"]:
            in_row = in_sample_main[(in_sample_main["marker"] == marker) & (in_sample_main["flag"] == flag)]
            cf_row = cf_main[(cf_main["marker"] == marker) & (cf_main["flag"] == flag)]
            if in_row.empty or cf_row.empty:
                continue
            in_row, cf_row = in_row.iloc[0], cf_row.iloc[0]
            print(
                f"  {flag:16s} in-sample OR={in_row['odds_ratio']:.2f} (p={in_row['p_value']:.4f})  "
                f"-> cross-fitted OR={cf_row['odds_ratio']:.2f} (p={cf_row['p_value']:.4f})"
            )


if __name__ == "__main__":
    run_all()
