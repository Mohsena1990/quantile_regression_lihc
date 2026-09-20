# Running the pipeline

All commands run from the project root. Every script resolves paths relative to the repository root itself.

```bash
pip install -r requirement.txt
bash scripts/run_full_pipeline.sh                                  # everything below, in order (~1-1.5 h)
START_AT=scripts/<name>.py bash scripts/run_full_pipeline.sh       # resume at a step after a failure
python -m pytest tests/ -q
```

Prerequisites: the ENABLE.EU workbook at `data/raw/ENABLE.EU_dataset_survey of households.xlsx` (not committed; see the
README "Data"), network access for Eurostat / GISCO downloads, and Copernicus CDS credentials in `~/.cdsapirc` only if the ERA5
NetCDF files in `artifacts/climate/` are missing (they are reused otherwise).

## 1. Data, labels, climate

| Step | Script | Writes |
|---|---|---|
| dictionary and coverage audit, cohort tables | `audit_enable_workbook.py` | `table01..table04`, `table07..table10`, `artifacts/manifests/variable_dictionary.parquet` |
| NUTS1 geography check (every cohort region exists in GISCO) | `validate_nuts.py` | `table_nuts1_validation.*` |
| Eurostat national income deciles (EUR) for the dynamic income rules | `prepare_country_references.py` | `artifacts/reference/income_quantiles_2018.csv` |
| observed annual energy cost, immutable split, Van Hove M0 benchmark | `build_expenditure_m0.py` | `artifacts/data/harmonised_observed_cost.parquet`, `artifacts/splits/household_split_v1.json`, `table_m0_*` |
| frozen bracket -> percentile table (used by the percentile sensitivity) | `build_income_bracket_reference.py` | `artifacts/reference/income_bracket_percentiles.csv`, `table_income_label_shift_audit.csv` |
| ERA5-Land 2017 daily temperature | `acquire_era5_land.py` | `artifacts/climate/*.nc` (skipped when present) |
| fuse to NUTS1; QC requires every cohort region | `fuse_era5_nuts1.py` | `artifacts/climate/nuts1_annual_2017.parquet`, climate figures |

## 2. Predictor core and cost-threshold estimators

* `build_a0_predictor_selection.py` -- recursive forward search from Van Hove's six non-income predictors; each step adds the
  candidate that most improves the average validation macro-F1 of CatBoost and hierarchical CatBoost, until nothing improves it.
  **Overwrites `config/pipeline.json` `features`**; the study's tier T1 is this list plus climate.
* `audit_vanhove_replication.py` -- Van Hove's exact seven-predictor design, income included, pooled.
* `build_oof_predictions.py --feature A0|A1` then `run_model_measurements.py --feature A0|A1` -- the M2 / M5 / M7 cost-threshold
  estimators (linear quantile regression, spline GAM, gradient-boosted quantile regression). The quantile comes from an out-of-fold
  calibration search, so the OOF step runs first.
* `build_threshold_estimator_comparison.py` -- regression accuracy of the three estimators on validation data and how often they
  flag the same households (`table_threshold_estimators.csv`, `table_threshold_label_agreement.csv`).
* `check_milestone_gates.py` -- sequential gates: M0 labels, A0 models, ERA5 QC, A1 models (`artifacts/manifests/milestone_gates.json`).

## 3. Identification study (train + validation households only)

```bash
python scripts/build_step_scatter_figures.py          # steps 1, 2, 2b, 4 scatter figures
python scripts/replicate_vanhove_importance.py        # credibility check against Van Hove's Fig. 7
python scripts/build_predictor_discovery.py           # SHAP + permutation + PDP; universal vs contextual; leave-one-country-out
python scripts/build_identification_benchmark.py      # classification grid, every metric with bootstrap CIs
python scripts/build_identification_sensitivity.py    # income rule: bracket <= 3 | percentile | dynamic equivalised | dynamic residual
python scripts/build_identification_q_sensitivity.py  # cost-threshold quantile 0.50 ... 0.80
python scripts/build_factor_comparison.py             # factor effects, heterogeneity, 2/3-factor cells, accuracy contribution
python scripts/build_hardship_target_check.py         # policy-anchored target (self-reported hardship)
python scripts/build_managerial_implications.py       # missed / wrongly-included households by information tier
python scripts/freeze_identification_selection.py     # pre-register the model choices for the test evaluation
```

Tables: `outputs/tables/table_{identification_*,predictor_stability,universal_predictors,pdp,universality_test,income_rule_*,cost_quantile_sensitivity,factor_*,hardship_identification,managerial_*,vanhove_importance_replication*,threshold_*}.csv`.
Figures: `outputs/figures/{step*,discovery_*,factor_*,managerial_*,country_transfer_link,vanhove_replication_importance}.{png,pdf}`.

## 4. The single test evaluation (not part of the pipeline script)

The test partition is spent by looking at it, so this step is separate, human-authorised and one-shot.

1. `freeze_identification_selection.py` (step 3 above) writes `config/final_selection.json`: for every cost definition x income
   state x information tier x country, the model with the best development cross-validation macro-F1. **Commit this file first** --
   that commit is the pre-registration.
2. A human sets `test_unlock_authorized: true`, `authorized_by` and `authorization_note` in that file.
3. `python scripts/run_final_test.py --dry-run` rehearses the identical code on train -> validation and reads no test data.
4. `python scripts/run_final_test.py` loads the test households once (recording `test_accessed_at` before returning them), fits each
   selected model on all development households, scores it on the test households, repeats the managerial analysis, and records the
   result hash and commit. A second run is refused by `lib/identification/final_test.py`, even if the first crashed.

Outputs: `table_final_test_{classification,generalization,managerial}.csv`, `artifacts/manifests/final_test_result.json`.

## 5. Tests

```bash
python -m pytest tests/ -q
```
