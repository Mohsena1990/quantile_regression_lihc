Finding energy-poor households with machine learning
=====================================================

**The question.** Which households are energy poor -- they have a low income *and* spend a lot on energy -- and can we find
them from information a government could collect? We study **Bulgaria, Hungary, Italy and Poland**, and from version 5 also **Spain**.

**The two studies we build on.**

* **Van Hove, Dalla Longa & van der Zwaan (2022)**, *Energy & Buildings* 264:112064 -- the same ENABLE.EU survey, four groups of
  households (low income x high spending), seven predictors, CatBoost.
* **Dalla Longa, Sweerts & van der Zwaan (2021)**, *Energy Policy* 156:112373 -- the same four groups for the Netherlands; spending
  as a *share of income*; area context (house value, ownership, building year, household size, population density); Models A / B / C
  (income only / other features only / both); a sensitivity check of both thresholds; and the finding that about 1,000
  training households are needed for stable results.

**What this version does (version 5).** It keeps the version-4 design (optimised thresholds, one pooled model read country by
country, Models A / B / C, stages B0 -> B1 -> B2 with placebo checks) and adds:

1. **A fifth country, Spain** -- every piece of data it needs could be downloaded. Now 2,278 labelled households.
2. **Predict the bill, then classify:** a model learns each household's energy bill in euros and compares it with the line.
3. **More households:** households whose energy cost is missing are added to training with an estimated spending side;
   households that gave a bill but no income help the bill model.
4. **The region itself** as one variable, instead of six regional statistics that only worked as region labels.
5. **Borderline and likely mislabelled households** count less in training.
6. **Averages of models** (ensembles) and a **tuned XGBoost** (Optuna).
7. **Questions asked in only some countries** (environmental attitudes, energy-saving behaviour, energy literacy).
8. **Targeting curves:** if a scheme helps the top k% of households, how many Double-Risk households does it reach?
9. **"Not sure":** the least certain households are flagged for a follow-up check.

Models throughout: **CatBoost**, **XGBoost** and **FT-Transformer**; variables chosen with **Attention Mechanisms as Embedded
Selectors + SHAP**.


1. The data
-----------

* **ENABLE.EU household survey (2018).** About 1,000 households per country (Spain 760). We use the development part (train +
  validation of a fixed 60/20/20 split; 3,845 households in five countries). The 20% test part was used once, earlier, and is not
  touched again. Place the workbook at `data/raw/ENABLE.EU_dataset_survey of households.xlsx` (not shared).
* **Labelled households** (income bracket and energy cost both reported): 2,278 -- Bulgaria 584, Hungary 646, Italy 382, Poland 384,
  Spain 282. A further **533** households gave their income but no energy cost (mostly Italy 187, Spain 138, Poland 125), and **567**
  gave an energy cost but no income (`table_v5_sample.csv`).
* **Why Spain.** Of the other survey countries, Spain was the one where every input could be downloaded: costs in euros, Eurostat
  income deciles, NUTS3 regions in the survey, Open-Meteo weather, ERA5-Land normals and Eurostat regional data. It was left out
  earlier because it has too few Double-Risk households for a *separate* Spanish model; in the pooled model it can take part.
* **Climate.** Daily weather for October 2016 - December 2017 (the survey asked about the winter of 2016/17) from the Open-Meteo
  historical weather API, one point per region (NUTS3; Poland NUTS2 -- the survey has no finer location there): 132 regions.
  30-year averages (1991-2020) from ERA5-Land monthly data (Copernicus; two requests, one for Spain incl. the Canary Islands).
* **Area context (Eurostat).** Population density and GDP per head (NUTS3), unemployment and poverty rates (NUTS2), home ownership
  (Census 2011) and old dwellings (Census 2021). In version 5 these are no longer B1 candidates (see Section 3); population density
  is still used in one climate x home feature.


2. Who counts as energy poor: the four lines and their thresholds
-----------------------------------------------------------------

Every household gets one of four labels:

| | Spending is **not** high | Spending **is** high |
|---|---|---|
| **Income is not low** | No Risk | Expenditure Risk |
| **Income is low** | Income Risk | **Double Risk** (energy poor) |

"Low income" means an income bracket (1-10) at or below a cut-off. "High spending" means a yearly energy cost above the household's
**expenditure line**. We compare four lines:

| Line | High spending means | Where it comes from |
|---|---|---|
| **Van Hove** | above the country's 80th percentile of yearly energy spending | Van Hove et al. (2022) |
| **Quantile Regression on need** | above what similar homes spend -- a linear quantile regression of spending on floor area, household size, dwelling type and age, insulation and local heating/cooling degree days | this study |
| **Quantile Regression on need, no climate** | the same, without the climate variables | a fair test of climate (see Section 3) |
| **Energy burden** | above the country's percentile of *spending as a share of income* (income in euros from the bracket and Eurostat deciles) | Dalla Longa et al. (2021) |

Every line is stored as **a threshold in euros for each household** (`lines_v5.py`), so a predicted bill can be compared with it.

**How the thresholds were chosen (multi-objective optimisation).** For each line we tried every income cut-off (bracket 2, 3 or 4)
with every expenditure level (60th to 90th percentile in steps of 5) and scored each option on three goals: **easy to identify**
(balanced accuracy of Van Hove's seven predictors), **meaningful** (the "hardship gap": among low-income households, how much more
often Double Risk households say they find it difficult to live on their income than Income Risk households), and **stable** (size
of the smallest group). Options with fewer than 15 households in any group in any country were dropped. From the options no other
option beats on all three goals (the Pareto front), the best balance was chosen by **TOPSIS** (equal weights). Figure
`threshold_moo_v5`, table `table_threshold_moo_v5.csv`.

| Line | Chosen threshold (five countries) | Identifiability | Hardship gap | Smallest group |
|---|---|---|---|---|
| Van Hove (benchmark, kept as published) | bracket <= 3, 80th percentile | -- | 0.025 | 199 |
| Quantile Regression on need | bracket <= 3, 60th conditional percentile (same as version 4) | 0.516 | 0.047 | 453 |
| Energy burden | bracket <= 3, 60th percentile (version 4: bracket <= 2, 70th) | 0.663 | 0.049 | 303 |

**Spain changes one thing:** with Spain included, Van Hove's own threshold no longer has 15 households in every group -- Spain has
only **5 Double-Risk households** under it -- and the only country-percentile option that passes is the 60th percentile, whose
hardship gap is negative (-0.010). We keep Van Hove's line as the benchmark, but its Spanish results rest on 5 households.


3. What the models are allowed to see
-------------------------------------

| Step | Included |
|---|---|
| **B0** | Van Hove's seven predictors, as in his Table 1 (income bracket, floor area, household size, house detachment, house age, birth year of the respondent, heating strategy = how heating is controlled) + country |
| **B1** | B0 + the best of four groups: **household indices** (dwelling efficiency, inefficient equipment, heating system type, household vulnerability, wealth as house-value stand-in, income per adult in euros); **ENABLE top 3** (difficulty living on income, floor area per person, dishwasher age); **partly asked questions** (environmental attitudes H15 -- not asked in Bulgaria; energy-saving behaviour G1/G3 -- not asked in Italy and Spain; "knows the price of 1 kWh" E1 -- Bulgaria only; "my energy bill is too complicated" C5C -- Hungary and Spain; missing answers are handled by the models); **region** (the NUTS2 region code as one variable) |
| **B2** | B1 + the **top 3** of 30 climate features: 20 regional (degree days of the survey year and the 30-year normal, how unusual the winter and summer were, cold, frost, cold spells, heatwaves, tropical nights, humidity, rain, snow, wind, gusts, sun, cloud) and 10 **climate x home** features that differ for every household (heating need = floor area x heat loss for the building's age and type x degree days; heating need per person; cold winter x old home / no insulation / solid-fuel / electric heating; hot summer x air conditioning; humidity x old home; wind x detached house; population density x urban area) |
| **B2p** | **placebo for B2**: each region's climate is swapped with another region of the same country (household climate x home features recomputed). Only gains *beyond* B2p count as a climate effect. |

**Why "region" replaces the regional statistics.** In version 4 the regional statistics (GDP, unemployment, poverty, density,
ownership, old dwellings) did exactly as well when their values were scrambled between regions (placebo B1p): they helped only by
telling regions apart. Saying "region" directly gets the same gain with one honest variable that any government has.


4. How the variables are chosen
-------------------------------

**Attention Mechanisms as Embedded Selectors + SHAP**, inside each training round only (the households used for scoring are never seen):

1. Three models learn from all candidates on three different 80/20 splits: an **FT-Transformer**, whose **attention** shows which
   variables it looks at, and **CatBoost** and **XGBoost**, whose **SHAP values** show how much each variable moves each prediction.
2. **Reliability weighting:** each importance measure gets a weight equal to how consistently it ranks the variables across the
   splits. Each variable's score is the weighted average rank.
3. **Two levels for B1:** first the groups (ranked by their variables' scores), then the variables inside the chosen groups. Each
   time we keep the smallest number whose accuracy is within one standard error of the best (the one-standard-error rule), and only
   if it beats the previous step in the same test folds.
4. **B2:** the 3 best-ranked climate features (as agreed). The number the one-standard-error rule would have kept is recorded too.


5. Models and testing
---------------------

* **One pooled model** for all five countries (with a country variable). Every score is computed **per country** on that country's
  held-out households, and for all five together.
* **Models A / B / C** (Dalla Longa et al.): **A** = income bracket only; **B** = the step's variables *without* income;
  **C** = the step's variables *with* income.
* **Classifiers:** CatBoost (depth 4, learning rate 0.05, balanced class weights), XGBoost (depth 4, learning rate 0.05, 300 trees,
  balanced weights), FT-Transformer (2 layers, 4 attention heads, early stopping).
* **Testing:** repeated 5-fold cross-validation (3 times), stratified by country x group; 95% ranges from 200 bootstrap resamples;
  differences between two models from a *paired* bootstrap (same households for both).

**The version-5 improvements** -- all built on the B1 variables chosen in each round, and all trained on training households only:

| Idea | Name in the tables | What it does |
|---|---|---|
| Predict the bill, then classify | `bill` | **XGBoost regression** of the yearly energy bill (log euros) on the B1 variables, trained on the training households plus those who gave a bill but no income. The prediction error is treated as normal, with its spread estimated per country from 5-fold out-of-fold errors, giving each household a **probability that its bill is above its own line**. The income side needs no model: income is a predictor, so "low income" is known. Before picking a group the probabilities are divided by each group's share (the same "balanced" decision the classifiers make). |
| More households | `augmented` | The 533 households with income but no energy cost join the training data. The bill model gives each one a probability p of being above its line; it enters twice, once as "above" with weight 0.5 x p and once as "not above" with weight 0.5 x (1 - p). Scores are still computed only on households with a reported bill. |
| Region as a variable | B1 group `region` | See Section 3. |
| Borderline / noisy labels | `robust` | Training weight 0.5 for a household exactly on its line, rising to 1 at 20% away. **Confident learning** (Northcutt et al. 2021) finds likely mislabelled households from out-of-fold predictions; they get weight 0.3. As published, the rule flagged about 38% of households -- a sign of weak models, not of 38% wrong labels -- so we also require the model to prefer the other spending side by at least 0.15. |
| Average of the models | `ensemble`, `ensemble_plus` | `ensemble` = average probability of CatBoost, XGBoost and FT-Transformer; `ensemble_plus` = average of CatBoost, tuned XGBoost, FT-Transformer and the bill model. |
| Tuned XGBoost | `tuned` | **Optuna** (tree-structured Parzen estimator, 25 trials) searches trees, depth, learning rate, subsampling, minimum child weight, L2 penalty and minimum split gain, judged by balanced accuracy on 3 folds of the training households. |
| Partly asked questions | B1 group `partial_modules` | See Section 3. |
| Targeting curves | `table_v5_targeting.csv` | Households ranked by their probability of Double Risk; for the top 5%, 10%, ... we count the share of all Double-Risk households reached, and the share of those helped who really are Double Risk. |
| "Not sure" | `table_v5_abstention.csv` | The households with the lowest top probability are flagged "not sure" (0-50%); scores are computed on the rest. |

**How to read the scores.**

| Score | Meaning |
|---|---|
| Recall (sensitivity), per class | Of the households that truly are in a group, the share the model finds |
| Precision, per class | Of the households the model puts in a group, the share that really belong there |
| Accuracy | Share of all households labelled correctly |
| Balanced accuracy | Average of the four recalls -- every group counts equally. **Our main score** (chance = 0.25) |
| Weighted F1 | Balance of precision and recall, averaged by group size (the score Van Hove reported) |
| ROC AUC | How well the model ranks households of a group above the others (0.5 = chance) |


6. Results
----------

**Balanced accuracy, all five countries together** (`table_v5_metrics.csv`). "sig." = paired 95% range above (+) or below (-) zero.

| Line | Model | A: income only | B: B0 / B1 / B2 without income | C: B0 | C: B1 | C: B2 | C: B2p placebo |
|---|---|---|---|---|---|---|---|
| Van Hove | CatBoost | 0.509 | 0.477 / 0.496 / 0.495 | 0.637 | 0.651 | 0.647 | 0.646 |
| | XGBoost | 0.504 | 0.462 / 0.495 / 0.513 | 0.604 | 0.648 | 0.654 | 0.670 |
| | FT-Transformer | 0.508 | 0.465 / 0.493 / 0.489 | 0.641 | 0.659 | 0.665 | 0.662 |
| | ensemble (3 models) | 0.514 | 0.475 / 0.520 / 0.521 | 0.638 | 0.678 | 0.676 | 0.679 |
| Quantile Regression on need | CatBoost | 0.511 | 0.385 / 0.432 / 0.431 | 0.513 | 0.565 | 0.567 | 0.566 |
| | XGBoost | 0.508 | 0.400 / 0.456 / 0.479 | 0.514 | 0.581 | 0.603 | 0.595 |
| | FT-Transformer | 0.502 | 0.387 / 0.420 / 0.423 | 0.523 | 0.551 | 0.557 | 0.563 |
| | ensemble (3 models) | 0.507 | 0.398 / 0.462 / 0.477 | 0.521 | 0.582 | 0.591 | 0.597 |
| Need line, no climate | CatBoost | 0.515 | 0.387 / 0.446 / 0.447 | 0.527 | 0.593 | 0.595 | 0.600 |
| | XGBoost | 0.513 | 0.414 / 0.474 / 0.497 | 0.541 | 0.613 | 0.631 | 0.629 |
| | FT-Transformer | 0.514 | 0.393 / 0.442 / 0.442 | 0.533 | 0.587 | 0.587 | 0.585 |
| | ensemble (3 models) | 0.512 | 0.413 / 0.476 / 0.490 | 0.546 | 0.619 | 0.622 | 0.627 |
| Energy burden | CatBoost | 0.587 | 0.475 / 0.480 / 0.494 | 0.665 | 0.667 | 0.675 | 0.668 |
| | XGBoost | 0.585 | 0.474 / 0.481 / 0.536 | 0.655 | 0.663 | 0.704 | 0.700 |
| | FT-Transformer | 0.590 | 0.492 / 0.494 / 0.491 | 0.667 | 0.670 | 0.673 | 0.668 |
| | ensemble (3 models) | 0.587 | 0.488 / 0.498 / 0.534 | 0.671 | 0.678 | 0.705 | 0.699 |

**The improvements, on the B1 variables** (balanced accuracy, all five countries; difference to the plain model in brackets;
`table_v5_deltas.csv`, figure `v5_variants`):

| Line | XGBoost (plain) | bill model | tuned XGBoost | robust XGBoost | augmented XGBoost | ensemble (3) | ensemble_plus |
|---|---|---|---|---|---|---|---|
| Van Hove | 0.648 | 0.665 (+0.017) | **0.669 (+0.021 sig.)** | 0.662 (+0.013 sig.) | 0.636 (-0.012 sig.) | 0.678 | **0.683** |
| Need line | 0.581 | **0.598 (+0.017 sig.)** | 0.573 (-0.008) | 0.579 (-0.001) | 0.590 (+0.009 sig.) | 0.582 | 0.596 |
| Need, no climate | 0.613 | 0.615 (+0.002) | 0.610 (-0.003) | 0.613 (0.000) | 0.611 (-0.002) | 0.619 | **0.625** |
| Energy burden | 0.663 | **0.688 (+0.025 sig.)** | 0.674 (+0.011) | 0.669 (+0.005) | 0.662 (-0.001) | 0.678 | 0.685 |

**What this means.**

* **Averaging helps most, and reliably.** `ensemble_plus` beats each of the three classifiers on every line (+0.012 to +0.046,
  significant in the pooled data in 11 of 12 comparisons and borderline in the 12th; better in 21 of 80 country comparisons,
  never worse). The plain three-model average adds +0.01 to +0.03 over CatBoost and the FT-Transformer, less over XGBoost.
  On the need and burden lines the bill model alone is as good as `ensemble_plus`.
* **Predicting the bill moves the model towards the spending side.** On the need line, recall of Expenditure Risk rises from
  0.51 to 0.72 and of **Double Risk from 0.52 to 0.74**, while recall of No Risk and Income Risk falls (0.63 -> 0.49, 0.67 -> 0.44)
  and Double-Risk precision falls slightly (0.48 -> 0.44). Balanced accuracy gains +0.017 to +0.025 on three lines. This is the
  expected gain "on the spending axis" -- bought with more false alarms, so it is a choice of operating point as much as a better model.
* **Tuning XGBoost pays off only on Van Hove's line** (+0.021); on the need lines the defaults are as good.
* **Down-weighting borderline households changes little** (on average -0.001; 6 better, 3 worse of 80 comparisons).
* **Adding the 533 households with an estimated bill does not help** (on average -0.006; 2 better, 11 worse of 80), and it hurts the
  FT-Transformer. The estimated bills are too uncertain (the bill model explains 32-39% of the variation in log cost) to add
  information. The second option -- households with a bill but no income -- is used inside the bill model, which does help.
* **The region is what B1 picks.** On both need lines, the region group is chosen in **100%** of training rounds (Van Hove 87%,
  burden 60%); B1 beats B0 significantly in 39 of 80 comparisons (+0.05 to +0.07 on the need lines). With the region in, the
  household indices are chosen less often than in version 4 (7-47% of rounds).
* **The partly asked questions are almost never chosen** (group 0-13% of rounds; no single variable selected): attitudes,
  energy-saving behaviour and literacy do not help find the four groups.
* **Climate still adds nothing beyond the region.** B2 beats B1 for XGBoost on the need and burden lines (+0.02 to +0.04), but **B2 against its placebo is
  +0.001 on average** (4 better, 2 worse of 60). The gain comes from the household-level "heating need" features, which mostly
  measure home size and type; they stay just as useful when the climate in them is swapped between regions.
* **Income is essential, Van Hove's six other predictors help his own kind of line.** Model B scores 0.39-0.54; income alone
  0.50-0.59; the other B0 predictors add +0.07 to +0.13 for Van Hove's and the burden line and 0.00 to +0.03 for the need lines.

**All scores, Quantile Regression on need, ensemble_plus** (`table_v5_report.csv` has every line, step, model and country):

| Scope | Accuracy | Balanced accuracy | Weighted F1 | ROC AUC | Recall: No / Income / Expenditure / Double Risk | Precision: No / Income / Expenditure / Double Risk |
|---|---|---|---|---|---|---|
| All five | 0.59 | 0.60 | 0.59 | 0.88 | 0.58 / 0.54 / 0.62 / 0.65 | 0.66 / 0.72 / 0.53 / 0.45 |
| Bulgaria | 0.61 | 0.62 | 0.62 | 0.89 | 0.63 / 0.52 / 0.65 / 0.67 | 0.71 / 0.73 / 0.56 / 0.46 |
| Hungary | 0.51 | 0.52 | 0.51 | 0.84 | 0.52 / 0.41 / 0.48 / 0.67 | 0.59 / 0.67 / 0.41 / 0.41 |
| Italy | 0.62 | 0.58 | 0.62 | 0.87 | 0.54 / 0.78 / 0.69 / 0.31 | 0.65 / 0.72 / 0.59 / 0.39 |
| Poland | 0.64 | 0.62 | 0.64 | 0.89 | 0.50 / 0.59 / 0.65 / 0.75 | 0.61 / 0.79 / 0.54 / 0.54 |
| Spain | 0.60 | 0.60 | 0.60 | 0.86 | 0.60 / 0.69 / 0.57 / 0.53 | 0.66 / 0.74 / 0.50 / 0.47 |

**Spain** scores in line with the other countries (balanced accuracy 0.60-0.66 with `ensemble_plus` on the need, burden and
Van Hove lines), but its Double-Risk group is small (22 households on the need line, 5 on Van Hove's).

**Targeting: how many households must a scheme help to reach 70% of Double Risk?** (`table_v5_targeting_rule.csv`, figure
`v5_targeting`; all five countries; the share of households that are Double Risk is 9% for Van Hove's line, 20% for the need line,
27% for the burden line)

| Line | Income only (A) | Van Hove B0 | B1 ensemble_plus | Of those helped, really Double Risk (ensemble_plus) |
|---|---|---|---|---|
| Van Hove | top 40% | top 25% | **top 25%** | 27% |
| Need line | top 40% | top 40% | **top 35%** | 43% |
| Need, no climate | top 40% | top 40% | **top 30%** | 48% |
| Energy burden | top 35% | top 30% | **top 30%** | 66% |

Helping the top 20% reaches 50% of Double-Risk households on the need line with `ensemble_plus` (bill model the same), against 36%
with income alone. Per country, reaching 70% takes the top 15% in Spain, 25% in Bulgaria, 30% in Italy, 45-50% in Hungary and Poland
(need line), because Double Risk is much more common there.

**"Not sure"** (`table_v5_abstention.csv`, figure `v5_abstention`; `ensemble_plus`, all five countries):

| Line | Flag 0% | Flag 10% | Flag 20% | Flag 30% | Flag 50% |
|---|---|---|---|---|---|
| Van Hove | 0.683 | 0.704 | 0.724 | 0.743 | 0.784 |
| Need line | 0.596 | 0.608 | 0.617 | 0.635 | 0.663 |
| Need, no climate | 0.625 | 0.638 | 0.651 | 0.662 | 0.696 |
| Energy burden | 0.685 | 0.704 | 0.728 | 0.747 | 0.794 |

(balanced accuracy on the households not flagged). Flagging the least certain 20% for a follow-up check raises balanced accuracy by
0.02-0.04. The flagged households come from all four groups in about equal shares, so a follow-up does not single out Double Risk.

**Best combination:** `ensemble_plus` on every line (0.60-0.68); for one single model, the bill model on the need and burden lines and
the tuned XGBoost on Van Hove's line.


7. Which variables matter, and where
------------------------------------

**Groups chosen in B1** (share of training rounds; `table_v5_selection_groups.csv`):

| Group | Van Hove | Need line | Need, no climate | Energy burden |
|---|---|---|---|---|
| Region (NUTS2) | 87% | 100% | 100% | 60% |
| Household indices | 47% | 33% | 7% | 27% |
| ENABLE top 3 | 13% | 13% | 0% | 13% |
| Partly asked questions | 13% | 7% | 0% | 0% |

Within the chosen groups the region is kept almost every time; next come the wealth index and the vulnerability index (27% of
rounds on the need line). `table_v5_selection_features.csv`. Attention again gets a small weight (7-15%).

**Top 3 climate features** (`table_v5_climate_top3.csv`): **heating need** (Van Hove and burden lines, every round), **heating need per
person** and **cold winter x electric heating** (need lines), **population density x urban area** and **winter solar radiation**
(burden). None of them beats the placebo.

**Maps.** For each line, `v5_effect_maps_<line>` shows, region by region, how much each chosen variable pushes households
**towards (red) or away from (blue) Double Risk** (average SHAP value, CatBoost, pooled model); `v5_value_maps` shows the
variables' regional averages; `v5_country_effects` summarises the effect per country (`table_v5_country_effects.csv`,
`table_v5_region_effects.csv`). On the need line the **region** has by far the largest effect (up to +-0.15), strongest in Poland,
where neighbouring regions push in opposite directions; the household variables move predictions by 0.01-0.03. The maps show the
mainland; the Canary Islands are left off.


8. Problems found and how they were handled
-------------------------------------------

* **Confident learning flagged 38% of households.** With models this weak, the published rule marks almost every uncertain
  household as "mislabelled". We added a margin rule (about 3% flagged on the need lines; 19% on Van Hove's and the burden line,
  where the models are more confident). Both shares are in `table_v5_variant_info.csv`.
* **Van Hove's threshold fails the group minimum once Spain is in** (5 Spanish Double-Risk households). Kept as the benchmark;
  Spanish Van Hove results should not be read on their own.
* **The threshold optimisation crashed when only one option was feasible** (the country-percentile family on five countries):
  fixed, a single option is now chosen by default.
* **The imputed households did not help.** The bill model explains only a third of the variation in energy cost, so their
  estimated spending side is close to a coin toss for many of them. Kept as a sensitivity check, not in the recommended model.
* From version 4: "heating need per euro of income" is not a climate candidate (it was income in disguise); the regional statistics
  worked only as region labels (now replaced by the region itself); attention remains a weak selector (7-15% of the weight);
  house value and home ownership are not in the survey (wealth index as stand-in); the per-country group minimum is 15, not 30.


9. Compared with Van Hove and Dalla Longa: what this work adds
--------------------------------------------------------------

* **Thresholds.** Optimising both thresholds confirms Van Hove's choice *for his kind of line* on four countries, but shows that line
  does not track hardship; need-based and burden lines do. Dalla Longa et al. moved the thresholds by +-20% and found stable
  accuracy; we show that accuracy alone is the wrong goal -- the threshold that is easiest to predict is not the one that means hardship.
* **Variables.** Under Van Hove's line, his seven predictors capture most of what can be learned. Under a need-based line, extra
  information adds 5-7 points of balanced accuracy, and most of it is **"which region is this"**. Area statistics (Dalla Longa et
  al.'s density and ownership, regional economy) give the same gain only because they identify the region; saying the region
  directly is simpler and honest. Household-level indices (wealth, vulnerability) add a little on top. Questions on attitudes,
  energy-saving behaviour and energy literacy add nothing.
* **Methods.** Averaging CatBoost, a tuned XGBoost, the FT-Transformer and a bill model gives the best scores on every line (+0.01
  to +0.05 over a single classifier). Predicting the bill in euros and comparing it with the line finds far more of the
  Double-Risk households (recall 0.74 against 0.52 on the need line) at the price of more false alarms.
* **For policy.** Ranking households by the model's probability of Double Risk, a scheme that helps the top 30-35% reaches 70% of
  Double-Risk households on the need lines, against 40% of households with income alone. Flagging the least certain 20% for a
  follow-up check raises accuracy on the rest by 2-4 points.
* **Pooling and Spain.** One pooled model (Dalla Longa et al.'s sample-size argument) lets Spain take part although it is too small
  for a model of its own; Spain scores like the other countries.
* **Climate.** Even measured per household, and with a line that leaves climate out, climate adds nothing beyond knowing the region.
* **Our pipeline matches Van Hove's.** Rebuilt their way on our four countries, we get their numbers (balanced accuracy 0.626 +- 0.028
  against their about 0.655 on one split); earlier differences came from the metric, single-split noise and country difficulty
  (`diagnose_vanhove_gap.py`).


10. Households that heat too little
-----------------------------------

A household that keeps its home cold to save money spends little, so no expenditure line can mark it as high spending. We estimate
each home's heating *need* (floor area x heat loss for its age and type x local heating degree days) and flag households that spend
less than half of what similar households spend for the same need (`build_hidden_underheating.py`): 5-14% of households and 6-20% of
low-income ones. They do not report more income difficulty than other low-income households, so part of this may be free fuel (own
wood) or incomplete bills.


11. Limits
----------

* One survey round from 2018; 2,278 labelled households in five countries. Results show links, not causes.
* Income is only known as a bracket (1-10); energy costs are self-reported; income in euros is an estimate.
* Climate and area variables have one value per region (132 regions), so they cannot separate climate from other regional
  differences; the placebo shows how much of a gain is region identity.
* Small groups in some countries (Poland's high-income households, Italy's lowest brackets, Spain's Double Risk) limit
  country-specific results.
* Section 10 (households that heat too little) is still the four-country analysis.


12. How to run it
-----------------

```bash
pip install -r requirement.txt
bash scripts/run_full_pipeline.sh        # every step in order; stops at the first error
python -m pytest tests/ -q               # 126 tests
```

The version-5 steps, in order:

```bash
python scripts/prepare_country_references.py # Eurostat income deciles (incl. Spain)
python scripts/build_extended_cost.py       # energy cost in euros for the five countries
python scripts/acquire_openmeteo_b2.py      # daily weather per region (cached; reruns work offline)
python scripts/acquire_era5_normals.py      # 1991-2020 monthly temperature (two Copernicus requests)
python scripts/fuse_openmeteo_b2.py         # climate measures per region
python scripts/acquire_regional_context.py  # Eurostat / Census area context
python scripts/build_threshold_moo.py --v5  # Section 2: threshold optimisation (five countries)
python scripts/build_feature_stages_v5.py   # Sections 5-7 (GPU; about 4 hours with 4 workers; --report-only rebuilds the tables)
python scripts/build_hidden_underheating.py # Section 10
```

Tables go to `outputs/tables/`, figures to `outputs/figures/`. Details: [`docs/running_the_pipeline.md`](docs/running_the_pipeline.md).
The record of every check and correction: [`AUDIT_AND_REPAIR_REPORT.md`](AUDIT_AND_REPAIR_REPORT.md).

| Folder | What is inside |
|---|---|
| `data/` | Reading the survey and choosing the countries (four, plus Spain for version 5) |
| `lib/identification/` | Lines and thresholds (`lines_v4.py`, `lines_v5.py`), the version-5 design (`stages_v5.py`), bill model (`bill_model.py`), noisy-label weights (`robust.py`), Optuna tuning (`tuning.py`), the version-4 design (`stages_v4.py`), shared selection tools (`stages.py`), the three models (`models3.py`, `ft_transformer.py`), Hansen's Threshold Regression (`hansen.py`), Van Hove rebuild (`vanhove_replica.py`), scores (`metrics.py`) |
| `lib/features/` | Survey variables (`survey_items.py`), built features (`engineered.py`), indices and climate x home features (`indices.py`), partly asked questions (`partial_modules.py`) |
| `lib/climate/` | Climate measures (`b2_indices.py`) |
| `scripts/` | Every runnable step |
| `tests/` | Automatic checks |


13. Earlier versions
--------------------

* **Version 4 / 4b** (commits `7464ceb`, `673b0fe`): the same pooled design on the four countries, with Dalla Longa's area
  variables and regional statistics as B1 groups and a placebo for them (B1p). Findings: B1 helped the need lines by +0.03 to
  +0.08; the regional statistics worked only as region labels (B1 - B1p = +0.003); climate added nothing beyond the placebo; one
  pooled model was as good as one model per country. Script `build_feature_stages_v4.py`; tables `table_v4_*`, figures `v4_*`.
* **Version 3** (commits `a25daff`, `dca549d`, `ad666d3`): one model per country; B1 chosen from all 82 survey variables; four lines
  including **Quantile Regression** at the 80th percentile, **Hansen's Threshold Regression** (Hansen 2000; the tipping point where
  hardship rises, 0.58 among homes with similar need, p = 0.042) and the integrated **Hansen + Quantile Regression** line. Findings:
  extra variables helped the Hansen + Quantile Regression line by up to 8 points and Van Hove's line not at all; climate added
  nothing beyond the region; attention was an unreliable selector. Scripts `build_expenditure_lines.py`, `build_feature_stages.py`;
  tables archived in `outputs/tables/archive_v1`, `archive_v2`.
* **The first identification study** (`AUDIT_AND_REPAIR_REPORT.md`, sections 1-27): income with and without the model, information
  levels, targeting errors, and one pre-registered test on the 20% test households (done once on 2026-09-20).


References
----------

* Van Hove, W., Dalla Longa, F., van der Zwaan, B. (2022). Identifying predictors for energy poverty in Europe using machine
  learning. *Energy & Buildings* 264, 112064. https://doi.org/10.1016/j.enbuild.2022.112064
* Dalla Longa, F., Sweerts, B., van der Zwaan, B. (2021). Exploring the complex origins of energy poverty in The Netherlands with
  machine learning. *Energy Policy* 156, 112373. https://doi.org/10.1016/j.enpol.2021.112373
* Hansen, B. E. (1996). Inference when a nuisance parameter is not identified under the null hypothesis. *Econometrica* 64, 413-430.
* Hansen, B. E. (2000). Sample splitting and threshold estimation. *Econometrica* 68, 575-603.
* Koenker, R., Bassett, G. (1978). Regression quantiles. *Econometrica* 46, 33-50.
* Hwang, C.-L., Yoon, K. (1981). *Multiple Attribute Decision Making: Methods and Applications* (TOPSIS). Springer.
* Gorishniy, Y., Rubachev, I., Khrulkov, V., Babenko, A. (2021). Revisiting deep learning models for tabular data. *NeurIPS*.
* Lundberg, S. M., Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
* Chen, T., Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.
* Prokhorenkova, L., et al. (2018). CatBoost: unbiased boosting with categorical features. *NeurIPS*.
* Akiba, T., Sano, S., Yanase, T., Ohta, T., Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *KDD*.
* Northcutt, C. G., Jiang, L., Chuang, I. L. (2021). Confident learning: estimating uncertainty in dataset labels. *Journal of
  Artificial Intelligence Research* 70, 1373-1411.
* Breiman, L., Friedman, J., Olshen, R., Stone, C. (1984). *Classification and Regression Trees*. Hastie, T., Tibshirani, R.,
  Friedman, J. (2009). *The Elements of Statistical Learning*, Sec. 7.10.
* Eurostat regional statistics: demo_r_d3dens, nama_10r_3gdp, lfst_r_lfu3rt, ilc_li41, cens_11htts_r2, cens_21dwop_r3.

**Climate data -- please cite when using it:**

* Zippenfenig, P. (2023). Open-Meteo.com Weather API [Computer software]. Zenodo. https://doi.org/10.5281/ZENODO.7970649
* Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J., Nicolas, J., Peubey, C., Radu, R., Rozum, I.,
  Schepers, D., Simmons, A., Soci, C., Dee, D., Thépaut, J-N. (2023). ERA5 hourly data on single levels from 1940 to present
  [Data set]. ECMWF. https://doi.org/10.24381/cds.adbb2d47
* Muñoz Sabater, J. (2019). ERA5-Land hourly data from 2001 to present [Data set]. ECMWF. https://doi.org/10.24381/CDS.E2161BAC
* Schimanke S., Ridal M., Le Moigne P., Berggren L., Undén P., Randriamampianina R., Andrea U., Bazile E., Bertelsen A.,
  Brousseau P., Dahlgren P., Edvinsson L., El Said A., Glinton M., Hopsch S., Isaksson L., Mladek R., Olsson E., Verrelle A.,
  Wang Z.Q. (2021). CERRA sub-daily regional reanalysis data for Europe on single levels from 1984 to present [Data set]. ECMWF.
  https://doi.org/10.24381/CDS.622A565A
