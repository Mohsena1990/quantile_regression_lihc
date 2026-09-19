#!/usr/bin/env bash
# Runs the full quickstart pipeline (README.md "Running the pipeline") end to end,
# in order, from a clean checkout through the frozen validation-only selection and
# the per-country/B-family layer. Stops on the first failing step (`set -e`).
#
# Does NOT run the locked final-test stage (scripts/run_screening.py --stage
# final_test): that requires a deliberate, human-authorized edit of
# test_unlock_authorized in artifacts/manifests/final_selection_frozen.json and is
# intentionally excluded from any automated script -- see README.md "Reproducibility
# locks". Does NOT run `pip install -r requirement.txt`: install dependencies once,
# separately, before invoking this script.
#
# Usage: bash scripts/run_full_pipeline.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Banner + timestamp per step. Resume after a failure with:  START_AT=scripts/<name>.py bash scripts/run_full_pipeline.sh
# (every step before the first one whose command contains START_AT is skipped).
_reached=0
run() {
  if [ -n "${START_AT:-}" ] && [ "$_reached" = 0 ]; then
    case "$*" in *"$START_AT"*) _reached=1 ;; *) echo "[skip] $*"; return 0 ;; esac
  fi
  echo "[$(date +%T)] >>> $*"; "$@"
}
# Generated-output directories (git-ignored) do not exist on a fresh checkout.
mkdir -p artifacts/{data,predictions,manifests,frozen,metrics,models,identification,splits,reference,climate} outputs/tables outputs/figures

# Diagnostics + governed external reference data (country income/rent/tenure context,
# energy/heating-oil prices) that later steps read from outputs/tables/ and
# artifacts/reference/ -- prepare_country_references.py hits the live Eurostat API,
# so this needs network access.
run python scripts/audit_enable_workbook.py
run python scripts/validate_nuts.py
run python scripts/prepare_country_references.py
run python scripts/prepare_energy_prices.py
run python scripts/prepare_heating_oil_prices.py

# Data construction, M0 benchmark, immutable household split
run python scripts/build_expenditure_m0.py
# Frozen cohort-wide income-bracket -> percentile table (needs the cost parquet above for its audit half).
# The headline income rule (bracket <= 3) does not use it, so M0 labels above are unaffected; the percentile
# sensitivity and everything that calls harmonized_income_percentile does, so it must precede those stages.
run python scripts/build_income_bracket_reference.py

# ERA5-Land climate (skips download if the NetCDF files are already present; needs
# CDS API credentials in ~/.cdsapirc only on an actual cache miss)
run python scripts/acquire_era5_land.py --configuration annual_2017
run python scripts/fuse_era5_nuts1.py --configuration annual_2017

# Build A0, this project's standard predictor set: recursive feature addition
# starting from Van Hove et al.'s 6 non-income base predictors, evaluated with
# both C0 and C1 on the pooled cohort. Overwrites config/pipeline.json's `features`
# key -- every step below reads A0 from there. See the module docstring for why
# income and the old A3/A4 per-country auto-selected configs are excluded.
run python scripts/build_a0_predictor_selection.py
run python scripts/audit_vanhove_replication.py

# M2/M5/M7, feature configuration A0, then A1 (A0 + climate). Quantile is discovered
# per family by a continuous OOF calibration search (lib/tuning/quantile_search.py),
# so build_oof_predictions.py must run BEFORE run_model_measurements.py: it finds
# each family's q* and writes the OOF-calibrated train rows that the production fit
# then splices in (preserve_oof_train_rows).
run python scripts/build_oof_predictions.py --feature A0
run python scripts/run_model_measurements.py --feature A0
run python scripts/build_oof_predictions.py --feature A1
run python scripts/run_model_measurements.py --feature A1

# Screening: CatBoost, hierarchical CatBoost, cost-sensitive class weighting
# throughout (this project's single fixed imbalance strategy -- no more
# cost_sensitive/balanced_bootstrap/smote_nc comparison axis). All 7 M0/A0/A1
# measurements (14 development runs total); each family's own quantile is found by
# continuous OOF calibration search, not a fixed grid.
run python scripts/run_screening_matrix.py

# Collapse to 4 measurement-family champions, compare the 2 classifiers, freeze
# selection, then the single best-configuration confusion matrix (not one per
# measurement x stage -- see the module docstring).
run python scripts/select_family_champions.py
run python scripts/compare_income_ablation.py
run python scripts/build_classifier_comparison.py
run python scripts/build_confusion_matrices.py
run python scripts/freeze_validation_selection.py

# Quantile-sensitivity sweep for R1-R3 (needs the champion config just frozen above)
run python scripts/build_quantile_sensitivity.py

# Gates, manifests, figures/tables
run python scripts/check_milestone_gates.py
run python scripts/build_pipeline_status.py
run python scripts/build_progressive_outputs.py
run python scripts/build_four_class_figures.py
run python scripts/build_policy_maps.py
run python scripts/build_policy_maps_by_feature.py

# Per-country/whole feature-selection layer (A0-A2, fixed predictor set) and
# B-family (genuinely separate model per country), then the paper/appendix output split
run python scripts/run_percountry_feature_selection.py
run python scripts/build_b_measurements.py
run python scripts/build_b_confusion_matrices.py
run python scripts/build_b_policy_maps.py
run python scripts/build_paper_appendix_outputs.py

# Symbol legend (R0-R3/C0-C1/A0-A2) and paper-facing diagnostics -- see README.md "Symbol legend"
run python scripts/build_symbol_legend.py
run python scripts/build_regression_threshold_figures.py
run python scripts/build_threshold_scatter_income_states.py
run python scripts/audit_shap_feature_effectiveness.py
run python scripts/build_scenario_simulation.py
run python scripts/build_evaluation_summary.py
run python scripts/build_nightingale_metrics.py
run python scripts/build_swimlane_flowchart.py
run python scripts/build_data_flow_diagram.py

# Identification study: Van Hove-based four-class classification, predictor discovery/universality, cross-country
# factor comparison, managerial implications. Development partitions only; repeated CV with CIs. Needs
# artifacts/climate from the ERA5 steps above and the frozen income-bracket table from the data step. + two/three-factor cross-country interpretation (development
# partitions only; repeated CV with CIs). Needs artifacts/climate from the ERA5 steps above.
run python scripts/build_step_scatter_figures.py
run python scripts/replicate_vanhove_importance.py
run python scripts/build_predictor_discovery.py
run python scripts/build_identification_benchmark.py
run python scripts/build_identification_sensitivity.py
run python scripts/build_factor_comparison.py
run python scripts/build_hardship_target_check.py
run python scripts/build_managerial_implications.py

# Tests
run python -m pytest tests/ -q

echo "run_full_pipeline.sh: all steps complete."
