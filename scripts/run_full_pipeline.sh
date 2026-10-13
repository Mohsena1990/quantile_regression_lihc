#!/usr/bin/env bash
# Runs the whole study end to end, in order, from a clean checkout (plus the raw ENABLE workbook at
# data/raw/, see README "Data"). Stops on the first failing step (`set -e`).
#
# Deliberately does NOT run scripts/run_final_test.py: the single evaluation on the locked test partition is a one-time,
# human-authorised step (config/final_selection.json; see README "Reproducibility locks"). This script only *freezes*
# (pre-registers) the model choices from development results.
# Does not run `pip install -r requirement.txt`: install dependencies once, separately.
#
# Usage:  bash scripts/run_full_pipeline.sh
#         START_AT=scripts/<name>.py bash scripts/run_full_pipeline.sh     # resume at that step
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Banner + timestamp per step; every step before the first one whose command contains START_AT is skipped.
_reached=0
run() {
  if [ -n "${START_AT:-}" ] && [ "$_reached" = 0 ]; then
    case "$*" in *"$START_AT"*) _reached=1 ;; *) echo "[skip] $*"; return 0 ;; esac
  fi
  echo "[$(date +%T)] >>> $*"; "$@"
}
# Generated-output directories (git-ignored) do not exist on a fresh checkout.
mkdir -p artifacts/{data,predictions,manifests,identification,splits,reference,climate} outputs/tables outputs/figures

# --- 1. Data, labels, climate -----------------------------------------------------------------------------------------
run python scripts/audit_enable_workbook.py           # dictionary / coverage audit, cohort tables
run python scripts/validate_nuts.py                   # every cohort NUTS1 region exists in GISCO (network for first download)
run python scripts/prepare_country_references.py      # Eurostat national income deciles (network)
run python scripts/build_expenditure_m0.py            # observed cost, immutable split, Van Hove M0 benchmark (network: exchange rates)
# Frozen cohort-wide bracket -> percentile table (needs the cost parquet above). The headline income rule (bracket <= 3) does
# not use it; the percentile sensitivity does.
run python scripts/build_income_bracket_reference.py
run python scripts/acquire_era5_land.py --configuration annual_2017   # skips the download when the NetCDF files exist
run python scripts/fuse_era5_nuts1.py --configuration annual_2017     # fuses to NUTS1, region-coverage QC gate
# Stage-B2 climate from the Open-Meteo reanalysis API: one point per NUTS3 (Poland NUTS2) region, survey heating season
# 2016/17, 1991-2020 normals. Responses are cached (artifacts/climate/openmeteo/http_cache), so reruns are offline.
# Must run before any script that reads the development frame's B2 columns.
run python scripts/acquire_openmeteo_b2.py
run python scripts/acquire_era5_normals.py            # 1991-2020 monthly normals (one CDS request)
run python scripts/fuse_openmeteo_b2.py

# --- 2. Predictor core and cost-threshold estimators -----------------------------------------------------------------
# A0: recursive forward search from Van Hove's six non-income predictors. Overwrites config/pipeline.json `features`.
run python scripts/build_a0_predictor_selection.py
run python scripts/audit_vanhove_replication.py       # Van Hove's exact seven-predictor design (income included), pooled
# M1/M2/M3 cost-threshold estimators: the quantile is found by an OOF calibration search, so OOF runs BEFORE the fit.
run python scripts/build_oof_predictions.py --feature A0
run python scripts/run_model_measurements.py --feature A0
run python scripts/build_oof_predictions.py --feature A1
run python scripts/run_model_measurements.py --feature A1
run python scripts/build_threshold_estimator_comparison.py
run python scripts/check_milestone_gates.py

# --- 3. Identification study (development partitions only: train + validation) ----------------------------------------
run python scripts/build_step_scatter_figures.py      # steps 1, 2, 2b, 4
run python scripts/replicate_vanhove_importance.py    # credibility check against Van Hove's Fig. 7
run python scripts/build_predictor_discovery.py       # SHAP + permutation + PDP, universal vs contextual, LOCO test
run python scripts/build_identification_benchmark.py  # classification grid, all metrics with CIs
run python scripts/build_identification_sensitivity.py    # income rules
run python scripts/build_identification_q_sensitivity.py  # cost-threshold quantile
run python scripts/build_imbalance_strategy_comparison.py           # class-imbalance strategies, round 1 (pre-declared rule)
run python scripts/build_imbalance_strategy_comparison.py --round 2  # round 2: three further strategies, fresh confirmation seed
run python scripts/build_factor_comparison.py
run python scripts/build_hardship_target_check.py
run python scripts/build_managerial_implications.py
# Van Hove (2022) replica on all 11 countries and the factor-by-factor gap decomposition
run python scripts/diagnose_vanhove_gap.py
# Feature stages B0 (Van Hove) -> B1 -> B2 (income always a predictor), four expenditure lines, attention + SHAP selection,
# CatBoost / XGBoost / FT-Transformer
run python scripts/build_expenditure_lines.py         # Hansen tipping points; Quantile Regression / Hansen lines; SHAP of the line
run python scripts/build_feature_stages.py --jobs 4   # attention + SHAP selection, three models; > 4 GPU workers crash CUDA under WSL
run python scripts/build_hidden_underheating.py       # sensitivity (b): modelled heating need, hidden under-heating
run python scripts/freeze_identification_selection.py     # pre-register model choices in config/final_selection.json

# --- 4. Tests ----------------------------------------------------------------------------------------------------------
run python -m pytest tests/ -q

echo "run_full_pipeline.sh: all steps complete."
