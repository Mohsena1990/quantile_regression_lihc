# Running the pipeline

All commands are run from the project root (`quantile_regression_lihc/`). Every
script resolves paths relative to the repo root itself, so `cd` there first.

## 0. Setup

```bash
pip install -r requirement.txt
```

The ENABLE.EU household workbook must be present at
`data/raw/ENABLE.EU_dataset_survey of households.xlsx` (already present in this
checkout). ERA5-Land acquisition additionally requires valid Copernicus CDS API
credentials in `~/.cdsapirc`.

Library code lives under `lib/` (climate, evaluation, expenditure, features,
geography, imbalance, labels, reporting, screening, tuning); `measurement/` (the
M0/M2/M5/M7 registry and families) and `data/` (ENABLE ingestion; `data/raw/` for
the raw workbook, `data/external/` reserved for any future externally-sourced
microdata -- external reference data actually in use today lives under
`artifacts/reference/`, see the note in the main README) stay at the top level.
`scripts/` holds every entry point; nothing under `lib/` or `measurement/` is
runnable directly.

## 1. Data construction, M0, splits

```bash
python scripts/audit_enable_workbook.py                 # dictionary/coverage audit (diagnostic)
python scripts/validate_nuts.py                         # NUTS1 24/24 geography check
python scripts/prepare_country_references.py             # income/rent/tenure country context (--year 2018 default)
python scripts/prepare_energy_prices.py                 # 2018 Eurostat electricity/gas prices + fuel coverage table
python scripts/prepare_heating_oil_prices.py             # heating-oil price series
python scripts/build_income_bracket_reference.py        # FREEZE cohort-wide bracket->percentile table first (all low-income labels use it)
python scripts/build_expenditure_m0.py                  # observed cost, immutable split (household_split_v1.json), M0 benchmark
```

## 2. ERA5-Land climate

Fusion passes the strict 24/24 NUTS1 QC gate (see `AUDIT_AND_REPAIR_REPORT.md`
§9/§21 for the acquisition history):

```bash
python scripts/acquire_era5_land.py --configuration annual_2017   # skips download if files already present
python scripts/fuse_era5_nuts1.py --configuration annual_2017     # fuses + runs the 24/24 QC gate
```

## 3. Build A0, the standard predictor set

`scripts/build_a0_predictor_selection.py` runs a recursive (forward)
feature-addition search starting from Van Hove et al.'s 6 non-income predictors
(floor area, household size, dwelling detachment, dwelling age, respondent age,
heating strategy) -- their 7th, income bracket, is deliberately excluded (see the
main README's "Why income is excluded"). At each step it tries every remaining
candidate feature, fits **both** C0 and C1 with it added, and keeps whichever
single candidate improves their averaged validation macro-F1 the most, until
nothing improves it further. The winning list overwrites `config/pipeline.json`'s
`features` key -- this becomes A0, read by every step below.

```bash
python scripts/build_a0_predictor_selection.py
python scripts/audit_vanhove_replication.py   # sanity check: replicate Van Hove's own 7-predictor, income-included design
```

Only M2, M5, and M7 remain of the original M2-M8 seven (removed 2026-08-25): an
empirical correlation audit found the other four's predictions correlated
0.86-0.98 (Spearman) with this retained trio -- redundant, not independent
evidence -- given the shared ~0.15-0.23 R² ceiling on this data. See
`measurement/families/model_based/estimators.py`'s module docstring for the full
per-family rationale.

## 4. M2/M5/M7, feature configuration A0, then A1 (A0 + climate)

Quantile is discovered per family by a continuous 5-fold OOF calibration search
(`lib/tuning/quantile_search.py`, q in [0.50, 0.80]), not a fixed grid, so
`build_oof_predictions.py` must run *before* `run_model_measurements.py`: it finds
each family's q* and writes the OOF-calibrated train rows that the production fit
then splices in.

```bash
python scripts/build_oof_predictions.py --feature A0
python scripts/run_model_measurements.py --feature A0
python scripts/build_oof_predictions.py --feature A1
python scripts/run_model_measurements.py --feature A1
python scripts/compare_income_ablation.py       # Van-Hove-style income-included vs income-excluded ablation
```

Every M2/M5/M7 family additionally tunes a small internal hyperparameter grid via
k-fold pinball-loss search strictly inside the train partition
(`lib/tuning/regression_search.py`) before fitting -- never touching
validation/test.

There is no more separate H/HC axis or `*_BASE` feature-ablation config (removed
2026-08-26): A1 is what "HC" used to mean, and the recursive search that builds
A0 already measures each candidate feature's marginal value directly, superseding
the old enriched-vs-base ablation comparison.

## 5. Screening (CatBoost / Hierarchical CatBoost), development stage

Cost-sensitive class weighting is this project's single, fixed imbalance-handling
method (no more cost_sensitive/balanced_bootstrap/SMOTENC comparison axis --
weighting needs no synthetic data and works even on small per-country groups).
Run all 7 M0/A0/A1 measurements through both classifiers and assert exactly 14
runs:

```bash
python scripts/run_screening_matrix.py
```

Or a single measurement/architecture:
`python scripts/run_screening.py --measurement m2_a0 --model hierarchical_catboost --stage development`

Each classifier tunes its own hyperparameters (CatBoost iterations/depth, the
hierarchical shrinkage lambda) on an internal train-only holdout
(`lib/tuning/search.py`), and every fit is checked for a degenerate (near-constant)
predicted-class distribution before its results are written
(`lib/screening/models.py:_assert_not_degenerate`) -- a fit that collapses raises
rather than silently persisting garbage predictions.

## 6. Family champions, classifier comparison, gates, status, figures/tables

```bash
python scripts/select_family_champions.py     # per-family best A0/A1 by OOF calibration error, then that
                                               # champion's screening performance (4 families x 2 classifiers)
python scripts/build_classifier_comparison.py # table + 5-panel figure comparing the 2 classifiers
python scripts/build_confusion_matrices.py    # ONE confusion-matrix figure, for the single best configuration
python scripts/check_milestone_gates.py       # writes artifacts/manifests/milestone_gates.json
python scripts/build_pipeline_status.py       # writes artifacts/manifests/pipeline_status.json
python scripts/build_progressive_outputs.py   # writes table_measurement_progress.csv + high_cost_prevalence figure
python scripts/build_four_class_figures.py    # per-country risk-class distribution for every available measurement
                                               # step (M0, M2/M5/M7 x A0/A1) + a cross-step overview heatmap;
                                               # writes table_four_class_by_country.csv
python scripts/build_policy_maps.py           # NUTS1 Double-Risk choropleths, one per family
python scripts/build_policy_maps_by_feature.py  # NUTS1 Double-Risk choropleths segmented by household factor
```

All figures are written to `outputs/figures/` as both a 600dpi PNG and a vector PDF
(`<name>.png` + `<name>.pdf`); all tables to `outputs/tables/` as CSV (+ `.tex`/`.xlsx`
where applicable). See `lib/reporting/figures.py`.

## 7. Model selection freeze and final test (locked)

Validation-only selection over all 7 M0/A0/A1 measurements:

```bash
python scripts/freeze_validation_selection.py
```

Final test evaluation is mechanically locked until `artifacts/manifests/final_selection_frozen.json`
has both `choices_frozen: true` (set by the command above) **and**
`test_unlock_authorized: true` (never set automatically by any script — this must be a
deliberate, human-authorized edit). Only then:

```bash
python scripts/run_screening.py --measurement <name> --stage final_test
```

## 8. Per-country/whole feature selection and the B-family

```bash
python scripts/run_percountry_feature_selection.py  # A0-A2 x (5 countries + pooled Whole) x 3 families x 2 classifiers
python scripts/build_b_measurements.py               # B0 (=M0) + B2/B5/B7: stitch each country's own winning config
python scripts/build_b_confusion_matrices.py         # confusion-matrix figure for the single best B-family
python scripts/build_b_policy_maps.py                # NUTS1 choropleths for the B-family results
python scripts/build_paper_appendix_outputs.py       # headline table + winning-cell confusion matrix -> outputs/paper;
                                                       # everything else -> outputs/appendix
```

## 9. Symbol legend and paper-facing diagnostics

```bash
python scripts/build_symbol_legend.py
python scripts/build_regression_threshold_figures.py
python scripts/build_threshold_scatter_income_states.py  # income-excluded vs included, four-class-coloured scatter
python scripts/audit_shap_feature_effectiveness.py
python scripts/build_scenario_simulation.py
python scripts/build_evaluation_summary.py
python scripts/build_nightingale_metrics.py           # stacked polar-area charts: regression / classification / SHAP
python scripts/build_swimlane_flowchart.py
python scripts/build_data_flow_diagram.py
```

## 10. Identification study (development partitions only; test partition locked)

```bash
python scripts/build_step_scatter_figures.py       # steps 1, 2, 2b, 4 scatter figures
python scripts/replicate_vanhove_importance.py     # credibility check against Van Hove's Fig. 7 (pooled)
python scripts/build_predictor_discovery.py        # SHAP + permutation + PDP, universal vs contextual, leave-one-country-out test (~15 min)
python scripts/build_identification_benchmark.py   # classification grid, all metrics with CIs (~20 min)
python scripts/build_identification_sensitivity.py # headline vs percentile / dynamic income rules
python scripts/build_factor_comparison.py          # factor effects, heterogeneity, 2/3-factor cells, accuracy contribution
python scripts/build_hardship_target_check.py      # policy-anchored target (self-reported hardship)
python scripts/build_managerial_implications.py    # missed / wrongly-included households by information tier
```

Tables: `outputs/tables/table_{identification_*,predictor_stability,universal_predictors,pdp,universality_test,income_rule_*,factor_*,hardship_identification,managerial_*,vanhove_importance_replication*}.csv`.

## 11. Tests

```bash
python -m pytest tests/ -q
```
