Energy-poverty identification with machine learning: predictors, cross-country comparison and targeting errors
================================================================================================================

Which observable household and dwelling factors identify energy-poor households, are those factors the same in
different countries, and what does an imperfect classifier mean for a support scheme that has to target them?

The study extends Van Hove, Dalla Longa & van der Zwaan (2022), "Identifying predictors for energy poverty in Europe
using machine learning", *Energy & Buildings* 264:112064, on the same ENABLE.EU household survey and the same four
"low income x high expenditure" classes (No Risk, Income Risk, Expenditure Risk, Double Risk), with CatBoost and SHAP.
It adds: a need-based cost threshold, an explicit universal-vs-contextual test, honest uncertainty, and a
targeting-error ("managerial implications") layer. Target venue: *Energy Policy*.

Countries in the analysis: **Bulgaria, Hungary, Italy, Poland** (17 NUTS1 regions). Spain was screened and removed
because it cannot carry every stage of the analysis -- see [`docs/country_screening.md`](docs/country_screening.md).

Study design
------------

| Decision | Choice |
|---|---|
| Cohort | 4,047 households in four countries; immutable 60/20/20 train/validation/test split by household hash (`artifacts/splits/household_split_v1.json`). **The test partition is never read by any analysis** (see "Reproducibility locks"). |
| Income axis | **Headline: Van Hove's fixed rule, income bracket <= 3.** Sensitivities: frozen survey-weighted percentile <= 0.30, and two *dynamic* rules -- household-size-adjusted income, and income net of energy cost per equivalent adult, each below 60% of the Eurostat national median. |
| Cost axis | **Headline: a need-based threshold** -- per-country regression of energy cost on floor area, household size, dwelling type/age and climate, out of fold, at q = 0.63. Van Hove's country-wide Q80 of raw cost is the benchmark and replication. |
| Income as a predictor | **Both states**: excluded (proxy targeting, where income cannot be observed) and included (Van Hove's design; income from a benefits or tax register). |
| Information tiers | T1 structural (dwelling, household, fuel, climate) -> T2 + socio-economic items -> T3 + welfare screen (S7 aid received, S8 income adequacy). |
| Importance | mean abs(SHAP) (CatBoost's built-in exact TreeSHAP), permutation importance and partial dependence, all on held-out CV folds (Van Hove found SHAP and permutation can disagree). |
| Evaluation | Repeated 5-fold CV over train + validation, household-bootstrap 95% CIs, a chance baseline, and **every metric**: accuracy, balanced accuracy, precision, recall, specificity, F1, ROC-AUC, PR-AUC, MCC, kappa, log-loss, Brier -- per class and per axis (`lib/identification/metrics.py`). |

Why the design differs from a naive reading of Van Hove -- three defects found and fixed, each with a regression test:

1. **The low-income cutoff moved with the data it was given.** The first harmonisation computed its bracket count from
   whichever rows it received, so the same household was low-income in one partition and not in another. Now a fixed
   integer rule (headline) or a frozen, cohort-wide lookup (`artifacts/reference/income_bracket_percentiles.csv`).
2. **"Don't know" codes were used as numbers** (dwelling age 99 for 6-15% of households). Now missing.
3. **A cost threshold conditioned on the classifier's own inputs makes the label independent of them** (pooled AUC 0.51),
   so nothing can be learned. The headline threshold conditions on physical need only.

Flow (one scatter figure per step)
----------------------------------

| Step | What | Script | Figure |
|---|---|---|---|
| 1 | Cost threshold: observed cost against the need-based threshold, with Q80 | `build_step_scatter_figures.py` | `step1_cost_threshold` |
| 2 | Income x cost quadrants (Van Hove Fig. 4 style); static vs dynamic income | `build_step_scatter_figures.py` | `step2_income_cost_quadrants`, `step2b_income_dynamic_vs_static` |
| 3 | Predictor discovery: SHAP + permutation + PDP, universal vs contextual, leave-one-country-out test | `build_predictor_discovery.py` | `discovery_*` |
| 4 | Four-class classification, all metrics, local / pooled / leave-one-country-out | `build_identification_benchmark.py`, `build_identification_sensitivity.py` | `step4_prediction_space` |
| 5 | Cross-country factor comparison (solid fuel, dwelling age, elderly, household size; climate) | `build_factor_comparison.py` | `factor_*`, `country_transfer_link` |
| 6 | **Managerial implications**: households missed / wrongly included per 1,000, by information tier | `build_managerial_implications.py` | `managerial_*` |
| 7 | **Single test evaluation** of the pre-registered models (one look, human-authorised) | `freeze_identification_selection.py`, `run_final_test.py` | `table_final_test_*` |
| check | Replication of Van Hove's importance ranking; hardship as a policy-anchored target | `replicate_vanhove_importance.py`, `build_hardship_target_check.py` | `vanhove_replication_importance` |
| check | Sensitivity to the cost-threshold quantile; agreement between three regression estimators of the threshold | `build_identification_q_sensitivity.py`, `build_threshold_estimator_comparison.py` | `table_cost_quantile_sensitivity`, `table_threshold_*` |

All scripts are in `scripts/`; tables go to `outputs/tables/`, figures to `outputs/figures/` (600 dpi PNG + vector PDF).

Cost-threshold estimators. Three regression families estimate a household's conditional high-cost threshold and are compared
on validation data (`build_threshold_estimator_comparison.py`): **M2** linear quantile regression, **M5** a spline-basis GAM and
**M7** gradient-boosted quantile regression, each at an out-of-fold calibrated quantile, plus Van Hove's **M0** country-wide Q80
as the benchmark. The recursive predictor search (`build_a0_predictor_selection.py`) fixes the household-feature core **A0** that
forms tier T1 together with climate. The study's own headline threshold is the simple per-country quantile regression on
physical need described above; the estimator comparison shows how much the flag changes if a different estimator is used.
(An earlier validation-split classifier-selection layer -- champions, per-country configuration comparison, stitched
per-country datasets, NUTS1 policy maps -- was retired on 2026-09-20 because repeated CV supersedes it; it is preserved at git tag
`pre-cleanup-20260919`.)

Results
-------

Regenerated by `bash scripts/run_full_pipeline.sh` on 2026-09-19. Development households only (train + validation); the
locked test partition was not read. Intervals are 95% household bootstrap over repeat-averaged out-of-fold predictions;
"chance" is a classifier drawing classes at random from the training prior.

**1. Label support** (need-based cost axis, income bracket <= 3). Every class has at least 30 households in all four countries.

| Country | Households | No Risk | Income Risk | Expenditure Risk | Double Risk |
|---|---|---|---|---|---|
| Bulgaria | 584 | 224 | 145 | 144 | 71 |
| Hungary | 646 | 103 | 306 | 78 | 159 |
| Italy | 382 | 124 | 110 | 99 | 49 |
| Poland | 384 | 36 | 205 | 31 | 112 |

**2. Four-class identification** (tier T2, best of the logistic / two-axis / CatBoost models per country; `table_identification_headline.csv`
carries all 200+ columns -- per-class precision, recall, specificity, F1, ROC-AUC, PR-AUC, and both binary axes -- with CIs).

| Country | Income | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) [95% CI] | ROC-AUC (macro) | MCC | Chance F1 |
|---|---|---|---|---|---|---|---|---|
| Bulgaria | excluded | 0.473 | 0.474 | 0.478 | 0.455 [0.419, 0.488] | 0.749 | 0.305 | 0.254 |
| Hungary | excluded | 0.390 | 0.370 | 0.397 | 0.372 [0.343, 0.406] | 0.694 | 0.162 | 0.234 |
| Italy | excluded | 0.439 | 0.422 | 0.423 | 0.419 [0.373, 0.457] | 0.696 | 0.242 | 0.261 |
| Poland | excluded | 0.418 | 0.353 | 0.387 | 0.352 [0.307, 0.394] | 0.702 | 0.160 | 0.253 |
| Bulgaria | included | 0.617 | 0.610 | 0.618 | 0.603 [0.567, 0.641] | 0.883 | 0.485 | 0.254 |
| Hungary | included | 0.545 | 0.537 | 0.538 | 0.535 [0.497, 0.567] | 0.833 | 0.344 | 0.234 |
| Italy | included | 0.592 | 0.565 | 0.567 | 0.564 [0.523, 0.603] | 0.861 | 0.446 | 0.261 |
| Poland | included | 0.578 | 0.547 | 0.550 | 0.544 [0.485, 0.605] | 0.839 | 0.340 | 0.253 |

More information helps without income (macro-F1, T1 -> T3: Bulgaria 0.44 -> 0.46, Hungary 0.34 -> 0.39, Italy 0.37 -> 0.46,
Poland 0.30 -> 0.38) and little with it (0.51-0.59 -> 0.54-0.60). A country's own model beats a pooled one and a model
trained on the other three countries (e.g. Poland 0.35 local, 0.32 pooled, 0.24 leave-one-country-out).

**3. Van Hove replication.** With their labels (country Q80, income bracket <= 3) and income as a predictor, the pipeline reaches
accuracy 0.65-0.72 and macro-F1 0.54-0.71 per country; the pooled seven-predictor design reaches 0.66-0.74 accuracy (paper: ~0.71).
Importance on held-out folds ranks income first by a wide margin (100 vs 9 for the runner-up) and puts floor area in the top three,
as they report; house detachment ranks above household size here, which differs from the paper (Spearman 0.6 against their order).

**4. Universal vs contextual predictors** (top-6 stability under both SHAP and permutation importance in every country).
Only **income** is strictly universal (income axis and four-class target, income included). Without income nothing is strictly
universal: household size and dishwasher age (an appliance-age wealth proxy) pass in 3 of 4 countries on the income axis;
the cost axis has no strictly universal predictor (heating control passes in 3 of 4 countries once income is included). A factors-only model built without the held-out country falls short of that
country's own full model by 0.01-0.14 macro-F1 (largest without income: Bulgaria 0.14, Poland 0.12).

**5. Same factor, different meaning by country** (adjusted logit marginal effects, percentage points; `table_factor_*`).
Solid-fuel heating moves the cost axis by -14.5 pp in Bulgaria but +8.6 pp in Hungary, and the effects differ across countries
(p = 0.012); it also differs on the income axis (p = 0.006; Italy -17.7 pp against +11.6 to +17.9 pp elsewhere). Elderly
respondents (p = 0.049) and large households (p = 0.037) differ on the income axis; dwelling age shows no significant
heterogeneity. The income axis is identifiable (AUC 0.81-0.84) and transfers (leave-one-country-out AUC 0.77-0.82); the cost axis
is barely identifiable (AUC 0.54-0.61, transfer 0.50-0.54).

**6. Managerial implications -- who a scheme misses and who it wrongly includes** (rule: assist households predicted Double Risk;
per 1,000 households screened; `table_managerial_*`).

| Country | Double Risk share | Information | Missed | Wrongly included | Recall | Precision |
|---|---|---|---|---|---|---|
| Bulgaria | 12% | T1 structural | 79 | 146 | 0.35 | 0.23 |
| | | T3 + welfare screen | 74 | 127 | 0.39 | 0.27 |
| | | T3 + income register | 56 | 108 | 0.54 | 0.38 |
| Hungary | 25% | T1 structural | 152 | 211 | 0.38 | 0.31 |
| | | T3 + welfare screen | 163 | 186 | 0.34 | 0.31 |
| | | T3 + income register | 128 | 209 | 0.48 | 0.36 |
| Italy | 13% | T1 structural | 79 | 166 | 0.39 | 0.23 |
| | | T3 + welfare screen | 86 | 142 | 0.33 | 0.23 |
| | | T3 + income register | 75 | 133 | 0.42 | 0.29 |
| Poland | 29% | T1 structural | 159 | 194 | 0.46 | 0.41 |
| | | T3 + welfare screen | 162 | 194 | 0.44 | 0.40 |
| | | T3 + income register | 137 | 211 | 0.53 | 0.42 |

Even with an income register, a scheme misses 46-59% of Double-Risk households and 58-71% of the households it assists are not
Double Risk. Survey-style information (T2, T3) barely changes the picture; only the register lowers errors, in every country and
for every weighting of a missed household against a wrongly included one (1x, 2x, 5x).

**7. Sensitivity to the income rule** (macro-F1 income excluded / included, tier T2, best of the logistic and two-axis models -- CatBoost is left out here, so the bracket <= 3 column sits slightly below section 2):

| Country | bracket <= 3 | percentile <= 0.30 | dynamic: equivalised | dynamic: residual after energy |
|---|---|---|---|---|
| Bulgaria | 0.45 / 0.57 | 0.46 / 0.54 | 0.33 / 0.47 | 0.38 / 0.50 |
| Hungary | 0.37 / 0.51 | 0.41 / 0.46 | 0.36 / 0.46 | 0.38 / 0.48 |
| Italy | 0.42 / 0.52 | 0.42 / 0.52 | 0.43 / 0.52 | 0.40 / 0.47 |
| Poland | 0.35 / 0.47 | 0.41 / 0.54 | 0.47 / 0.49 | 0.43 / 0.47 |

The dynamic rules label far fewer households low-income (Hungary 23-47% against 72%) and leave Italy's Double-Risk class under 30
households under the equivalised rule; conclusions about country ranking depend on the income rule, which is why the fixed rule
is the headline and the others are reported beside it.

**8. External validity.** Self-reported hardship (survey item S8, "difficult / very difficult on present income") is 52% among Income Risk and 57%
among Double Risk households against 23% for No Risk -- but 21% for Expenditure Risk. The income axis tracks hardship; the cost axis
(spending relative to physical need) does not. Hardship itself is identifiable (AUC 0.73-0.82 from socio-economic items, 0.79-0.88
with income) and is reported as a policy-anchored alternative target (`table_hardship_identification.csv`).

**9. Cost-threshold choices.** *Estimator:* linear quantile regression (M2), a GAM (M5) and gradient-boosted quantile regression (M7)
predict cost from physical need equally poorly (validation R-squared 0.13 / 0.17 / 0.15; MAE
519-582 EUR), and they flag largely the same households as high-cost (Cohen's kappa
0.62-0.72 between estimators, 0.64-0.68 against the headline
need-based flag; `table_threshold_*`). The label therefore does not hinge on the estimator. *Quantile:* the out-of-fold calibration
search returns its lower bound (about 0.5), so q = 0.63 is a stated design choice and is swept from 0.50 to 0.80
(`table_cost_quantile_sensitivity.csv`): Hungary and Poland are stable (macro-F1 0.34-0.37 and 0.32-0.36 without income), Bulgaria
and Italy do better at low q (Bulgaria 0.49 at 0.50 against 0.41 at 0.80; Italy 0.43 against 0.32) as the high-cost class thins
out, and Poland falls below the 30-household support gate at q >= 0.70. Country comparisons are therefore made at one stated q
with the sweep beside them.

**10. Test evaluation (single look, pre-registered).** The model for every country x cost definition x income state x tier cell was frozen
from development cross-validation and committed (`556c054`) before the test partition was touched; the owner's authorisation was
committed separately (`7dfe397`); the evaluation ran once on 2026-09-20 on the 810 test households, of which 477 have a valid cost and income bracket and are
evaluated (Bulgaria 145, Hungary 156, Italy 84, Poland 92). `table_final_test_*.csv` carry every metric with 95% bootstrap CIs.

| Country | Income | Test households | Accuracy | F1 (macro) [95% CI] | ROC-AUC (macro) | MCC | Chance F1 | Development CV F1 |
|---|---|---|---|---|---|---|---|---|
| Bulgaria | excluded | 145 | 0.524 | 0.497 [0.410, 0.576] | 0.780 | 0.357 | 0.228 | 0.455 |
| Hungary | excluded | 156 | 0.404 | 0.396 [0.319, 0.461] | 0.744 | 0.210 | 0.237 | 0.372 |
| Italy | excluded | 84 | 0.393 | 0.390 [0.270, 0.481] | 0.723 | 0.195 | 0.242 | 0.419 |
| Poland | excluded | 92 | 0.337 | **0.248 [0.167, 0.317]** | 0.612 | -0.009 | 0.174 | 0.352 |
| Bulgaria | included | 145 | 0.669 | 0.639 [0.556, 0.710] | 0.879 | 0.551 | 0.228 | 0.603 |
| Hungary | included | 156 | 0.590 | 0.583 [0.486, 0.660] | 0.864 | 0.417 | 0.237 | 0.535 |
| Italy | included | 84 | 0.488 | 0.474 [0.355, 0.566] | 0.825 | 0.314 | 0.242 | 0.564 |
| Poland | included | 92 | 0.587 | 0.542 [0.399, 0.665] | 0.843 | 0.331 | 0.174 | 0.544 |

Across all 24 need-based cells (three tiers x two income states x four countries) the development estimate lies inside the test 95% CI
in 20 and the mean test-minus-development macro-F1 is -0.008: the repeated-CV numbers did not overstate performance. With Van Hove's own
labels (country Q80, income included, tier T2) the test accuracy is 0.63-0.78 and macro-F1 0.57-0.72, in line with the paper's 60-74%
true-positive rates. **Poland without income is the exception**: on test it is not distinguishable from chance (MCC -0.01), and its test
partition holds only 13 No-Risk and 5 Expenditure-Risk households, so the Poland income-excluded estimate should not be leaned on.
Managerial errors on the test households (`table_final_test_managerial.csv`) keep the development conclusion: with an income register a scheme
still misses 36-82% of Double-Risk households and 53-81% of those it assists are not Double Risk; survey-style information does not
reduce the errors reliably (test intervals are wide -- 14 to 44 Double-Risk households per country).

Data
----

* **ENABLE.EU household survey (2018)** -- place the workbook at `data/raw/ENABLE.EU_dataset_survey of households.xlsx`.
  It is not redistributed here (git-ignored). Van Hove et al. cite the source as
  `http://www.enable-eu.com/wp-content/uploads/2019/10/enable_eu_dataset_households.zip`.
* Fetched by the pipeline (network needed): Eurostat income deciles, rent burden, tenure, electricity/gas prices and
  exchange rates; the European Commission Weekly Oil Bulletin; GISCO NUTS 2016 geometries.
* ERA5-Land 2017 daily temperature (Copernicus CDS; credentials in `~/.cdsapirc` only on a cache miss).
* External reference data is country- or region-level and lives under `artifacts/reference/`; household microdata under `data/raw/`.

Running it
----------

```bash
pip install -r requirement.txt
bash scripts/run_full_pipeline.sh        # every stage in order; stops at the first failure; ~1-1.5 hours end to end
python -m pytest tests/ -q
```

Step-by-step commands and what each writes: [`docs/running_the_pipeline.md`](docs/running_the_pipeline.md).
Sources for every methodological choice: [`docs/methodology_sources.md`](docs/methodology_sources.md).
The full audit trail (defects found, fixes, what still limits a claim): [`AUDIT_AND_REPAIR_REPORT.md`](AUDIT_AND_REPAIR_REPORT.md).

Repository layout
-----------------

| Path | Contents |
|---|---|
| `data/` | ENABLE workbook ingestion and cohort rules (`enable.py`; `RETAINED_COUNTRY_CODES` defines the four countries) |
| `lib/identification/` | Study core: `data` (development frame, tiers), `labels` (income and cost axes, support gate), `protocol` (repeated CV), `metrics`, `discovery` (SHAP/permutation/PDP, universality), `factors` |
| `lib/labels/` | Four-class ontology and the income rules (`bracket3`, `percentile`, `equiv60`, `residual60`) |
| `lib/features/`, `lib/climate/`, `lib/expenditure/`, `lib/geography/` | Household features, degree days, cost reconstruction and currency, NUTS validation |
| `lib/screening/`, `lib/tuning/`, `lib/imbalance/`, `lib/evaluation/`, `lib/reporting/` | Classifiers and class weights used by the A0 search, quantile/regression search for M2/M5/M7, cost metrics and split registry, figure helper |
| `measurement/` | M0 (Van Hove) and M2/M5/M7 cost-threshold estimators |
| `scripts/` | Every entry point (nothing under `lib/` or `measurement/` is runnable) |
| `config/pipeline.json` | Seed, split path, families, the A0 feature list (overwritten by the A0 search), climate features, quantile search bounds |
| `config/final_selection.json` | Pre-registered model choices and the test-access record (see "Reproducibility locks") |
| `tests/` | Label stability, metric definitions against scikit-learn, CV leakage guard, managerial arithmetic, test-access guard, governance, ERA5 gates |
| `docs/` | Pipeline commands, methodology sources, country screening |

Reproducibility locks
---------------------

* **Test partition -- one look.** Every analysis reads train + validation households only (`lib/identification/data.py` has no
  test path). The single evaluation goes through `lib/identification/final_test.py`, which refuses unless
  `config/final_selection.json` records `test_unlock_authorized: true` and a named `authorized_by`, and refuses again once
  `test_accessed_at` is set -- even if the first attempt crashed. The model choices in that file are frozen from development
  results and **committed before the test is touched**, so the commit history shows the pre-registration. `run_final_test.py
  --dry-run` rehearses the same code on train -> validation without reading test data.
* **Immutable split.** `scripts/build_expenditure_m0.py` refuses to regenerate the split registry if any household is
  missing from it; a new split must be a new versioned file.
* **Milestone gates.** `scripts/check_milestone_gates.py` closes four sequential gates (M0 labels, A0 models, ERA5 QC over
  every cohort NUTS1 region, A1 models).
* **Country gate.** `lib/identification/labels.py:viability` refuses per-country four-class claims for any class with fewer than
  30 households; Spain was removed by this rule (`docs/country_screening.md`).
* **Seeds.** `config/pipeline.json` `seed`; CatBoost and CV seeds are fixed in code.

Limitations
-----------

* Survey cross-section (2018), about 1,000 households per country, with income only as an ordinal bracket and cost
  self-reported (21-44% of heating costs missing in Italy and Poland, unrelated to income). Associations, not causal effects.
* The headline income rule is a fixed bracket cutoff, so low-income shares differ by country (Bulgaria 37%, Hungary 72%,
  Italy 42%, Poland 83%); the percentile and dynamic rules are reported as sensitivities. The dynamic rules rest on an
  EUR proxy built from an ordinal bracket.
* The cost axis (spending relative to physical need) is not validated against self-reported hardship; the income axis is.
* Accuracy without income is modest; the value of the analysis is the comparison across information tiers, the
  cross-country differences, and the size of the targeting errors -- not a single accuracy figure.
* Poland's Expenditure Risk class is small under the headline income rule (31 households in development data, 5 in the test partition); Poland without income is not distinguishable from chance on the test households.
* Test partitions hold 84-156 eligible households per country, so test intervals are wide (macro-F1 CIs of about +-0.08 to +-0.13).
