#!/usr/bin/env bash
# Runs the whole study end to end, in order, from a clean checkout plus the raw ENABLE workbook at
# data/raw/ENABLE.EU_dataset_survey of households.xlsx (see README "How to run it"). Stops at the first failing step.
# Network: Eurostat, GISCO, Open-Meteo (cached after the first run) and the Copernicus CDS (credentials in ~/.cdsapirc).
# Does not install dependencies: `pip install -r requirement.txt` once, separately.
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
mkdir -p artifacts/{data,identification,reference,climate} outputs/tables outputs/figures

# --- 1. Data -------------------------------------------------------------------------------------------------------------
run python scripts/prepare_country_references.py      # Eurostat national income deciles
run python scripts/build_costs.py                     # energy cost in EUR (eleven countries), checks the frozen split
run python scripts/acquire_nuts.py                    # GISCO NUTS 2016 polygons; every study region present
run python scripts/acquire_openmeteo_b2.py            # daily weather 2016/17 per region (cached)
run python scripts/acquire_openmeteo_b2.py --extra    # the same for the extra training countries
run python scripts/acquire_era5_normals.py            # 1991-2020 monthly normals (Copernicus CDS)
run python scripts/acquire_era5_normals.py --extra
run python scripts/fuse_openmeteo_b2.py               # climate measures per region, QC gate
run python scripts/fuse_openmeteo_b2.py --extra
run python scripts/acquire_regional_context.py        # Eurostat / Census regional context

# --- 2. Validation against the reference study -------------------------------------------------------------------------
run python scripts/validate_van_hove_replica.py       # the reference design run through this pipeline

# --- 3. Main study --------------------------------------------------------------------------------------------------------
run python scripts/build_threshold_moo.py             # thresholds of every line (multi-objective optimisation)
run python scripts/build_feature_stages.py --jobs 4   # stages, model sets, improvements, targeting, "not sure" (GPU; ~4 h)

# --- 4. More data, multilevel models, climate ----------------------------------------------------------------------------
run python scripts/build_pooled_extra.py --jobs 4     # extra training countries, multilevel bill model (GPU; ~1.5 h)
run python scripts/build_best_vs_main.py
run python scripts/build_climate_multilevel.py        # climate and the energy bill
run python scripts/build_climate_labels.py --jobs 4   # climate-sensitive outcomes
run python scripts/build_bill_climate.py --jobs 6     # climate inside the bill models
run python scripts/build_climate_power.py --jobs 6    # power, region known
run python scripts/build_climate_power.py --grouped --jobs 6   # power, unseen regions
run python scripts/build_hidden_underheating.py       # hidden under-heating
run python scripts/build_loco_validation.py --jobs 4  # leave one country out (GPU)

# --- 5. Tests -------------------------------------------------------------------------------------------------------------
run python -m pytest tests/ -q

echo "run_full_pipeline.sh: all steps complete."
