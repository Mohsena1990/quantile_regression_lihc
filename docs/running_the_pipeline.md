# Running the pipeline

All commands run from the project root.

```bash
pip install -r requirement.txt
bash scripts/run_full_pipeline.sh                              # every step below, in order (about 7 hours with a GPU)
START_AT=scripts/<name>.py bash scripts/run_full_pipeline.sh   # resume at a step after a failure
python -m pytest tests/ -q
```

Prerequisites: the ENABLE.EU workbook at `data/raw/ENABLE.EU_dataset_survey of households.xlsx` (not shared); network access for
Eurostat, GISCO and the Open-Meteo API (responses are cached in `artifacts/climate/openmeteo/http_cache`, so reruns are offline);
Copernicus CDS credentials in `~/.cdsapirc` for the 1991-2020 normals (downloads are skipped when the NetCDF files exist).
The frozen train / validation / test split is a committed input: `config/household_split_v1.csv`.
Generated files go to `artifacts/` (intermediate) and `outputs/` (tables and figures); both are git-ignored.

## 1. Data

| Step | Script | Writes |
|---|---|---|
| Eurostat national income deciles (EUR) | `prepare_country_references.py` | `artifacts/reference/income_quantiles_2018.csv` |
| Annual energy cost in EUR, eleven countries; split coverage check | `build_costs.py` | `artifacts/data/energy_cost_study.parquet`, `energy_cost_all.parquet`, `artifacts/reference/currency_conversion.csv` |
| GISCO NUTS 2016 polygons; every study region present | `acquire_nuts.py` | `artifacts/reference/NUTS_RG_01M_2016_4326_LEVL_{1,2,3}.geojson` |
| Daily weather Oct 2016 - Dec 2017 per region (Open-Meteo) | `acquire_openmeteo_b2.py [--extra]` | `artifacts/climate/openmeteo/` |
| 1991-2020 monthly normals (ERA5-Land, CDS) | `acquire_era5_normals.py [--extra]` | `artifacts/climate/era5_normals/` |
| Climate measures per region, QC gate | `fuse_openmeteo_b2.py [--extra]` | `artifacts/climate/b2/region_climate[_extra].parquet`, `table_b2_climate_regions*.csv`, `b2_climate_maps` |
| Regional context (Eurostat / Census) | `acquire_regional_context.py` | `artifacts/reference/regional_context.csv` |

`--extra` = the six extra training countries.

## 2. Validation against the reference study

| Step | Script | Writes |
|---|---|---|
| The reference design (P80 line, core profile, CatBoost, one split) run through this pipeline, 20 random splits | `validate_van_hove_replica.py` | `table_validation_van_hove*.csv`, `validation_van_hove` |

## 3. Main study

| Step | Script | Writes |
|---|---|---|
| Thresholds of every line (multi-objective optimisation) | `build_threshold_moo.py` | `artifacts/identification/moo_thresholds.json`, `table_threshold_moo.csv`, `threshold_moo` |
| Stages B0-B2 + placebo, model sets, improvements, targeting, "not sure", SHAP maps | `build_feature_stages.py [--report-only]` | `artifacts/identification/main_results.pkl`, `table_main_*.csv`, `main_*` figures |

## 4. More data, multilevel models, climate

| Step | Script | Writes |
|---|---|---|
| Extra training countries; multilevel (GPBoost) bill model | `build_pooled_extra.py [--report-only] [--refit-bill-mixed]` | `pool_results.pkl`, `table_pool_*.csv`, `pool` |
| Best recipe vs the main study's recipe (paired) | `build_best_vs_main.py` | `table_best_vs_main.csv` |
| Climate and the energy bill (mixed model, placebo) | `build_climate_multilevel.py` | `table_climate_elasticity.csv`, `table_region_variance.csv`, `table_country_slopes.csv`, `table_region_effects.csv`, `climate_bill` |
| Climate-sensitive outcomes | `build_climate_labels.py` | `table_climate_labels_*.csv`, `climate_labels` |
| Climate inside the bill models | `build_bill_climate.py [--report-only]` | `bill_climate_results.pkl`, `table_bill_climate_*.csv` |
| Power: region known / unseen regions | `build_climate_power.py [--grouped] [--figure-only]` | `table_power_*.csv`, `power`, `power_grouped` |
| Hidden under-heating | `build_hidden_underheating.py` | `table_hidden_underheating*.csv`, `hidden_underheating` |
| Leave one country out | `build_loco_validation.py [--figure-only]` | `table_loco.csv`, `loco` |
