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

# Diagnostics + governed external reference data (country income/rent/tenure context,
# energy/heating-oil prices) that later steps read from outputs/tables/ and
# artifacts/reference/ -- prepare_country_references.py hits the live Eurostat API,
# so this needs network access.
python scripts/audit_enable_workbook.py
python scripts/validate_nuts.py
python scripts/prepare_country_references.py
python scripts/prepare_energy_prices.py
python scripts/prepare_heating_oil_prices.py

# Data construction, M0 benchmark, immutable household split
# Frozen cohort-wide income-bracket -> percentile table: every low-income label depends on it,
# so it must exist before ANY label is built (lib/labels/income_harmonization.py "Fix v2").
python scripts/build_income_bracket_reference.py
python scripts/build_expenditure_m0.py

# ERA5-Land climate (skips download if the NetCDF files are already present; needs
# CDS API credentials in ~/.cdsapirc only on an actual cache miss)
python scripts/acquire_era5_land.py --configuration annual_2017
python scripts/fuse_era5_nuts1.py --configuration annual_2017

# Build A0, this project's standard predictor set: recursive feature addition
# starting from Van Hove et al.'s 6 non-income base predictors, evaluated with
# both C0 and C1 on the pooled cohort. Overwrites config/pipeline.json's `features`
# key -- every step below reads A0 from there. See the module docstring for why
# income and the old A3/A4 per-country auto-selected configs are excluded.
python scripts/build_a0_predictor_selection.py
python scripts/audit_vanhove_replication.py

# M2/M5/M7, feature configuration A0, then A1 (A0 + climate). Quantile is discovered
# per family by a continuous OOF calibration search (lib/tuning/quantile_search.py),
# so build_oof_predictions.py must run BEFORE run_model_measurements.py: it finds
# each family's q* and writes the OOF-calibrated train rows that the production fit
# then splices in (preserve_oof_train_rows).
python scripts/build_oof_predictions.py --feature A0
python scripts/run_model_measurements.py --feature A0
python scripts/build_oof_predictions.py --feature A1
python scripts/run_model_measurements.py --feature A1
python scripts/compare_income_ablation.py

# Screening: CatBoost, hierarchical CatBoost, cost-sensitive class weighting
# throughout (this project's single fixed imbalance strategy -- no more
# cost_sensitive/balanced_bootstrap/smote_nc comparison axis). All 7 M0/A0/A1
# measurements (14 development runs total); each family's own quantile is found by
# continuous OOF calibration search, not a fixed grid.
python scripts/run_screening_matrix.py

# Collapse to 4 measurement-family champions, compare the 2 classifiers, freeze
# selection, then the single best-configuration confusion matrix (not one per
# measurement x stage -- see the module docstring).
python scripts/select_family_champions.py
python scripts/build_classifier_comparison.py
python scripts/build_confusion_matrices.py
python scripts/freeze_validation_selection.py

# Quantile-sensitivity sweep for R1-R3 (needs the champion config just frozen above)
python scripts/build_quantile_sensitivity.py

# Gates, manifests, figures/tables
python scripts/check_milestone_gates.py
python scripts/build_pipeline_status.py
python scripts/build_progressive_outputs.py
python scripts/build_four_class_figures.py
python scripts/build_policy_maps.py
python scripts/build_policy_maps_by_feature.py

# Per-country/whole feature-selection layer (A0-A2, fixed predictor set) and
# B-family (genuinely separate model per country), then the paper/appendix output split
python scripts/run_percountry_feature_selection.py
python scripts/build_b_measurements.py
python scripts/build_b_confusion_matrices.py
python scripts/build_b_policy_maps.py
python scripts/build_paper_appendix_outputs.py

# Symbol legend (R0-R3/C0-C1/A0-A2) and paper-facing diagnostics -- see README.md "Symbol legend"
python scripts/build_symbol_legend.py
python scripts/build_regression_threshold_figures.py
python scripts/build_threshold_scatter_income_states.py
python scripts/audit_shap_feature_effectiveness.py
python scripts/build_scenario_simulation.py
python scripts/build_evaluation_summary.py
python scripts/build_nightingale_metrics.py
python scripts/build_swimlane_flowchart.py
python scripts/build_data_flow_diagram.py

# Tests
# Identification-accuracy protocol + two/three-factor cross-country interpretation (development
# partitions only; repeated CV with CIs). Needs artifacts/climate from the ERA5 steps above.
python scripts/build_step_scatter_figures.py
python scripts/replicate_vanhove_importance.py
python scripts/build_predictor_discovery.py
python scripts/build_identification_benchmark.py
python scripts/build_identification_sensitivity.py
python scripts/build_factor_comparison.py
python scripts/build_hardship_target_check.py
python scripts/build_managerial_implications.py

python -m pytest tests/ -q

echo "run_full_pipeline.sh: all steps complete."
