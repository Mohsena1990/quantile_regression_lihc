Finding energy-poor households with machine learning
=====================================================

**The question.** Which households are energy poor -- they have a low income *and* spend a lot on energy -- and can we find
them from information a government could collect? We study five countries of the ENABLE.EU household survey (2018):
**Bulgaria, Hungary, Italy, Poland and Spain**.

**The two studies that inspired this work.**

* **Van Hove, Dalla Longa & van der Zwaan (2022)**, *Energy & Buildings* 264:112064 -- the same ENABLE.EU survey; four groups of
  households (low income x high spending); a country percentile of energy spending as the "high spending" line; seven household
  predictors; CatBoost.
* **Dalla Longa, Sweerts & van der Zwaan (2021)**, *Energy Policy* 156:112373 -- the same four groups for the Netherlands; spending
  as a share of income (the energy burden); area context; a comparison of models with income only, without income, and with both;
  and the finding that about 1,000 training households are needed for stable results.

**How this study names its own parts.** The study starts from these ideas but has its own baseline and its own names; the
references appear where we explain what inspired a choice (here and in Section 9, *Validation against the reference studies*):

| This study | What it is | Inspired by |
|---|---|---|
| **P80 line** (baseline line) | low income = bracket <= 3; high spending = above the country's 80th percentile | Van Hove et al.'s definition |
| **B0 core profile** (baseline stage) | income bracket, floor area, household size, house detachment, house age, birth year of the respondent, heating strategy | Van Hove et al.'s seven predictors |
| **Energy-burden line** | high spending = above the country's percentile of spending as a share of income | Dalla Longa et al. |
| **Income only / Without income / Full model** | the same stage with the income bracket alone, without income, and with income | Dalla Longa et al.'s Models A / B / C |
| **Need line**, **need line without climate** | high spending = above what similar homes spend (quantile regression on physical need) | this study |

**What the study does.**

1. **Chooses both thresholds of every line** (who is "low income", who "spends a lot") by **multi-objective optimisation**.
2. Trains **one pooled model** for the five countries and **reads it country by country**.
3. Adds information in **stages**: B0 core profile -> B1 household and region information -> B2 climate, with **placebo checks**.
4. Compares **CatBoost**, **XGBoost** and **FT-Transformer**, choosing variables with **Attention Mechanisms as Embedded Selectors
   + SHAP**.
5. Tests improvements: **predict the bill, then classify**; **multilevel models**; **model averages**; **tuning**; extra
   households from **six other ENABLE countries**; weighting of uncertain labels.
6. Asks for policy: **targeting curves** and a **"not sure"** option.
7. Tests **climate** properly: on the four groups, on the **energy bill itself**, on **comfort outcomes**, with a **power analysis**.
8. Checks how far the results carry: **validation against the reference study** and **leave one country out**.


1. The data
-----------

* **ENABLE.EU household survey (2018).** About 1,000 households per country (Spain 760). We use the development part (train +
  validation of a fixed 60/20/20 split, `config/household_split_v1.csv`; 3,845 households). The 20% test part was used once, on
  2026-09-20, before the final design, and stays locked (`config/final_selection.json`). Place the workbook at
  `data/raw/ENABLE.EU_dataset_survey of households.xlsx` (not shared).
* **Labelled households** (income bracket and energy cost both reported): 2,278 -- Bulgaria 584, Hungary 646, Italy 382, Poland 384,
  Spain 282. A further 533 households gave their income but no energy cost, and 567 gave a cost but no income
  (`table_main_sample.csv`).
* **Energy cost** = yearly heating + electricity cost, converted to euros with Eurostat 2017 exchange rates (`build_costs.py`).
* **Climate.** Daily weather for October 2016 - December 2017 (the survey asked about the winter of 2016/17) from the Open-Meteo
  historical weather API, one point per region (NUTS3; Poland NUTS2 -- the survey has no finer location there): 132 regions.
  30-year averages (1991-2020) from ERA5-Land monthly data (Copernicus).
* **Area context (Eurostat).** Population density, GDP per head, unemployment, poverty rate, home ownership, old dwellings.
* **Six extra training countries** -- France, Germany, Norway, Serbia, Ukraine, the UK (6,458 households) -- used only to *train*
  pooled models (Section 8), never to score them.


2. Who counts as energy poor: four lines and their thresholds
-------------------------------------------------------------

Every household gets one of four labels:

| | Spending is **not** high | Spending **is** high |
|---|---|---|
| **Income is not low** | No Risk | Expenditure Risk |
| **Income is low** | Income Risk | **Double Risk** (energy poor) |

"Low income" means an income bracket (1-10) at or below a cut-off. "High spending" means a yearly energy cost above the household's
**expenditure line**, stored as a threshold in euros for each household (`lib/identification/lines.py`):

| Line | High spending means |
|---|---|
| **P80 line** (baseline) | above the country's 80th percentile of yearly energy spending |
| **Need line** | above what similar homes spend -- a linear quantile regression of spending on floor area, household size, dwelling type and age, insulation and local heating/cooling degree days |
| **Need line without climate** | the same, without the degree days -- the fair test of climate (Section 3) |
| **Energy-burden line** | above the country's percentile of spending as a share of income (income in euros from the bracket and Eurostat deciles) |

**How the thresholds were chosen (multi-objective optimisation).** For each line we tried every income cut-off (bracket 2, 3 or 4)
with every expenditure level (60th to 90th percentile in steps of 5) and scored each option on three goals: **easy to identify**
(balanced accuracy of the B0 core profile), **meaningful** (the "hardship gap": among low-income households, how much more often
Double Risk households say they find it difficult to live on their income than Income Risk households), and **stable** (size of the
smallest group). Options with fewer than 15 households in any group in any country were dropped. From the options no other option
beats on all three goals (the Pareto front), the best balance was chosen by **TOPSIS** (equal weights). Figure `threshold_moo`, table
`table_threshold_moo.csv`.

| Line | Threshold | Identifiability | Hardship gap | Smallest group |
|---|---|---|---|---|
| P80 line (baseline, fixed) | bracket <= 3, 80th percentile | -- (below the 15-per-group minimum in Spain) | 0.025 | 199 |
| Need line | bracket <= 3, 60th conditional percentile (chosen) | 0.516 | 0.047 | 453 |
| Energy-burden line | bracket <= 3, 60th percentile (chosen) | 0.663 | 0.049 | 303 |

The country-percentile family itself has only one option that passes the minimum, the 60th percentile, and its hardship gap is
negative (-0.010): a pure spending percentile does not single out households in hardship. We keep the P80 line as the baseline;
Spain has only **5 Double-Risk households** under it, so Spain's own P80 results should not be read alone.


3. What the models are allowed to see
-------------------------------------

| Stage | Included |
|---|---|
| **B0 core profile** | income bracket, floor area, household size, house detachment, house age, birth year of the respondent, heating strategy (how heating is controlled, H9) + country |
| **B1** | B0 + the best of four groups: **household indices** (dwelling efficiency, inefficient equipment, heating system type, household vulnerability, wealth as house-value stand-in, income per adult in euros); **top survey items** (difficulty living on income, floor area per person, dishwasher age); **partly asked questions** (environmental attitudes H15 -- not asked in Bulgaria; energy-saving behaviour G1/G3 -- not asked in Italy and Spain; "knows the price of 1 kWh" E1 -- Bulgaria only; "my energy bill is too complicated" C5C -- Hungary and Spain); **region** (the NUTS2 code as one variable) |
| **B2** | B1 + the **top 3** of 30 climate features: 20 regional (degree days of the survey year and the 30-year normal, how unusual the winter and summer were, cold, frost, cold spells, heatwaves, tropical nights, humidity, rain, snow, wind, gusts, sun, cloud) and 10 **climate x home** features that differ for every household (heating need = floor area x heat loss for the building's age and type x degree days; heating need per person; cold winter x old home / no insulation / solid-fuel / electric heating; hot summer x air conditioning; humidity x old home; wind x detached house; population density x urban area) |
| **B2p** | **placebo for B2**: each region's climate is swapped with another region of the same country. Only gains *beyond* B2p count as a climate effect. |

**Model sets:** **income only** (the income bracket alone), **without income** (the stage's variables without the income-derived
ones), **full model** (the stage's variables with income).

**Why "region" is a variable.** Regional statistics (GDP, unemployment, poverty, density, ownership, old dwellings) did exactly as
well when their values were scrambled between regions: they helped only by telling regions apart. Saying "region" directly gets the
same gain with one honest variable that any government has.


4. How the variables are chosen
-------------------------------

**Attention Mechanisms as Embedded Selectors + SHAP**, inside each training round only (the households used for scoring are never
seen; `lib/identification/selection.py`):

1. Three models learn from all candidates on three different 80/20 splits: an **FT-Transformer**, whose **attention** shows which
   variables it looks at, and **CatBoost** and **XGBoost**, whose **SHAP values** show how much each variable moves each prediction.
2. **Reliability weighting:** each importance measure gets a weight equal to how consistently it ranks the variables across the
   splits. Each variable's score is the weighted average rank.
3. **Two levels for B1:** first the groups, then the variables inside the chosen groups. Each time we keep the smallest number whose
   accuracy is within one standard error of the best (the one-standard-error rule), and only if it beats the previous step in the
   same test folds.
4. **B2:** the 3 best-ranked climate features; the number the one-standard-error rule would have kept is recorded too.


5. Models and testing
---------------------

* **One pooled model** for the five countries (with a country variable). Every score is computed per country on that country's
  held-out households, and for all five together.
* **Classifiers:** CatBoost (depth 4, learning rate 0.05, balanced class weights), XGBoost (depth 4, learning rate 0.05, 300 trees,
  balanced weights), FT-Transformer (2 layers, 4 attention heads, early stopping).
* **Testing:** repeated 5-fold cross-validation (3 times), stratified by country x group; 95% ranges from 200 bootstrap resamples;
  differences between two models from a *paired* bootstrap (same households for both).

**Improvements of the full B1 model** (all trained on training households only):

| Idea | Name in the tables | What it does |
|---|---|---|
| Predict the bill, then classify | `bill` | **XGBoost regression** of the yearly energy bill (log euros), trained on the training households plus those who gave a bill but no income. The prediction error is treated as normal (spread per country from out-of-fold errors), giving each household a **probability that its bill is above its own line**. Income is a predictor, so "low income" is known. Probabilities are divided by each group's share before picking a group (the classifiers' "balanced" decision). |
| More households | `augmented` | The 533 households with income but no cost join training: each enters twice, "above its line" with weight 0.5 x p and "not above" with weight 0.5 x (1 - p), p from the bill model. Scores use only households with a reported bill. |
| Uncertain labels | `robust` | Weight 0.5 for a household exactly on its line, rising to 1 at 20% away; **confident learning** (Northcutt et al. 2021) finds likely mislabelled households, weight 0.3 (with a margin rule, Section 10). |
| Model averages | `ensemble`, `ensemble_plus` | `ensemble` = CatBoost + XGBoost + FT-Transformer; `ensemble_plus` = CatBoost + tuned XGBoost + FT-Transformer + bill model. |
| Tuned XGBoost | `tuned` | **Optuna** (tree-structured Parzen estimator, 25 trials) on 3 folds of the training households. |
| Targeting curves | `table_main_targeting.csv` | Households ranked by their probability of Double Risk: the share of all Double-Risk households among the top 5%, 10%, ... |
| "Not sure" | `table_main_abstention.csv` | The households with the lowest top probability are flagged for a follow-up check; scores on the rest. |

**How to read the scores.**

| Score | Meaning |
|---|---|
| Recall (sensitivity), per group | Of the households that truly are in a group, the share the model finds |
| Precision, per group | Of the households the model puts in a group, the share that really belong there |
| Accuracy | Share of all households labelled correctly |
| Balanced accuracy | Average of the four recalls -- every group counts equally. **Our main score** (chance = 0.25) |
| Weighted F1 | Balance of precision and recall, averaged by group size |
| ROC AUC | How well the model ranks households of a group above the others (0.5 = chance) |


6. Results
----------

**Balanced accuracy, all five countries together** (`table_main_metrics.csv`, figure `main_model_sets`):

| Line | Model | Income only | Without income: B0 / B1 / B2 | Full: B0 core profile | Full: B1 | Full: B2 | Full: B2p placebo |
|---|---|---|---|---|---|---|---|
| P80 line | CatBoost | 0.509 | 0.477 / 0.496 / 0.495 | 0.637 | 0.651 | 0.647 | 0.646 |
| | XGBoost | 0.504 | 0.462 / 0.495 / 0.513 | 0.604 | 0.648 | 0.654 | 0.670 |
| | FT-Transformer | 0.508 | 0.465 / 0.493 / 0.489 | 0.641 | 0.659 | 0.665 | 0.662 |
| | ensemble (3 models) | 0.514 | 0.475 / 0.520 / 0.521 | 0.638 | 0.678 | 0.676 | 0.679 |
| Need line | CatBoost | 0.511 | 0.385 / 0.432 / 0.431 | 0.513 | 0.565 | 0.567 | 0.566 |
| | XGBoost | 0.508 | 0.400 / 0.456 / 0.479 | 0.514 | 0.581 | 0.603 | 0.595 |
| | FT-Transformer | 0.502 | 0.387 / 0.420 / 0.423 | 0.523 | 0.551 | 0.557 | 0.563 |
| | ensemble (3 models) | 0.507 | 0.398 / 0.462 / 0.477 | 0.521 | 0.582 | 0.591 | 0.597 |
| Need line, no climate | CatBoost | 0.515 | 0.387 / 0.446 / 0.447 | 0.527 | 0.593 | 0.595 | 0.600 |
| | XGBoost | 0.513 | 0.414 / 0.474 / 0.497 | 0.541 | 0.613 | 0.631 | 0.629 |
| | FT-Transformer | 0.514 | 0.393 / 0.442 / 0.442 | 0.533 | 0.587 | 0.587 | 0.585 |
| | ensemble (3 models) | 0.512 | 0.413 / 0.476 / 0.490 | 0.546 | 0.619 | 0.622 | 0.627 |
| Energy-burden line | CatBoost | 0.587 | 0.475 / 0.480 / 0.494 | 0.665 | 0.667 | 0.675 | 0.668 |
| | XGBoost | 0.585 | 0.474 / 0.481 / 0.536 | 0.655 | 0.663 | 0.704 | 0.700 |
| | FT-Transformer | 0.590 | 0.492 / 0.494 / 0.491 | 0.667 | 0.670 | 0.673 | 0.668 |
| | ensemble (3 models) | 0.587 | 0.488 / 0.498 / 0.534 | 0.671 | 0.678 | 0.705 | 0.699 |

**The improvements, on the full B1 model** (difference to plain XGBoost in brackets; "sig." = paired 95% range above or below zero;
`table_main_deltas.csv`, figure `main_variants`):

| Line | XGBoost | bill model | tuned XGBoost | robust XGBoost | augmented XGBoost | ensemble (3) | ensemble_plus |
|---|---|---|---|---|---|---|---|
| P80 line | 0.648 | 0.665 (+0.017) | **0.669 (+0.021 sig.)** | 0.662 (+0.013 sig.) | 0.636 (-0.012 sig.) | 0.678 | **0.683** |
| Need line | 0.581 | **0.598 (+0.017 sig.)** | 0.573 (-0.008) | 0.579 (-0.001) | 0.590 (+0.009 sig.) | 0.582 | 0.596 |
| Need, no climate | 0.613 | 0.615 (+0.002) | 0.610 (-0.003) | 0.613 (0.000) | 0.611 (-0.002) | 0.619 | **0.625** |
| Energy-burden line | 0.663 | **0.688 (+0.025 sig.)** | 0.674 (+0.011) | 0.669 (+0.005) | 0.662 (-0.001) | 0.678 | 0.685 |

**What this means.**

* **Income is essential.** Without income scores fall to 0.39-0.54; income alone reaches 0.50-0.59. The other six core-profile
  variables add +0.07 to +0.13 on the P80 and burden lines but 0.00 to +0.03 on the need lines, which already account for home size.
* **The region is what B1 adds.** On both need lines the region group is chosen in **100%** of training rounds; B1 beats B0
  significantly in 39 of 80 comparisons (+0.05 to +0.07 on the need lines). Household indices add a little; the partly asked
  questions (attitudes, energy-saving behaviour, literacy) are almost never chosen.
* **Averaging helps most, and reliably.** `ensemble_plus` beats each of the three classifiers on every line (+0.012 to +0.046;
  better in 21 of 80 country comparisons, never worse).
* **Predicting the bill moves the model towards the spending side.** On the need line, Double-Risk recall rises from 0.52 to 0.74
  and Expenditure-Risk recall from 0.51 to 0.72, while No-Risk and Income-Risk recall fall and Double-Risk precision falls slightly
  (0.48 -> 0.44): a different operating point as much as a better model.
* **Tuning pays off only on the P80 line** (+0.021); **down-weighting uncertain labels changes little** (-0.001 on average);
  **imputed households do not help** (-0.006; the bill model explains only a third of the variation in cost).
* **Climate adds nothing beyond the region in the four groups.** B2 beats B1 for XGBoost on the need and burden lines (+0.02 to
  +0.04), but **B2 against its placebo is +0.001 on average** (4 better, 2 worse of 60): the gain comes from the household "heating
  need" features, which mostly measure home size. Section 8 shows where climate *does* matter.

**All scores, need line, ensemble_plus** (`table_main_report.csv` has every line, stage, model and country):

| Scope | Accuracy | Balanced accuracy | Weighted F1 | ROC AUC | Recall: No / Income / Expenditure / Double Risk | Precision: No / Income / Expenditure / Double Risk |
|---|---|---|---|---|---|---|
| All five | 0.59 | 0.60 | 0.59 | 0.88 | 0.58 / 0.54 / 0.62 / 0.65 | 0.66 / 0.72 / 0.53 / 0.45 |
| Bulgaria | 0.61 | 0.62 | 0.62 | 0.89 | 0.63 / 0.52 / 0.65 / 0.67 | 0.71 / 0.73 / 0.56 / 0.46 |
| Hungary | 0.51 | 0.52 | 0.51 | 0.84 | 0.52 / 0.41 / 0.48 / 0.67 | 0.59 / 0.67 / 0.41 / 0.41 |
| Italy | 0.62 | 0.58 | 0.62 | 0.87 | 0.54 / 0.78 / 0.69 / 0.31 | 0.65 / 0.72 / 0.59 / 0.39 |
| Poland | 0.64 | 0.62 | 0.64 | 0.89 | 0.50 / 0.59 / 0.65 / 0.75 | 0.61 / 0.79 / 0.54 / 0.54 |
| Spain | 0.60 | 0.60 | 0.60 | 0.86 | 0.60 / 0.69 / 0.57 / 0.53 | 0.66 / 0.74 / 0.50 / 0.47 |

**Targeting: how many households must a scheme help to reach 70% of Double Risk?** (`table_main_targeting_rule.csv`, figure
`main_targeting`; Double Risk is 9% of households on the P80 line, 20% on the need line, 27% on the burden line)

| Line | Income only | B0 core profile | B1 ensemble_plus | Of those helped, really Double Risk |
|---|---|---|---|---|
| P80 line | top 40% | top 25% | **top 25%** | 27% |
| Need line | top 40% | top 40% | **top 35%** | 43% |
| Need, no climate | top 40% | top 40% | **top 30%** | 48% |
| Energy-burden line | top 35% | top 30% | **top 30%** | 66% |

**"Not sure"** (`table_main_abstention.csv`, figure `main_abstention`): flagging the least certain 20% of households for a follow-up
check raises balanced accuracy on the rest by 0.02-0.04 (need line 0.596 -> 0.617; P80 0.683 -> 0.724); the flagged households come
from all four groups in similar shares.


7. Which variables matter, and where
------------------------------------

**Groups chosen in B1** (share of training rounds; `table_main_selection_groups.csv`):

| Group | P80 line | Need line | Need, no climate | Energy-burden line |
|---|---|---|---|---|
| Region (NUTS2) | 87% | 100% | 100% | 60% |
| Household indices | 47% | 33% | 7% | 27% |
| Top survey items | 13% | 13% | 0% | 13% |
| Partly asked questions | 13% | 7% | 0% | 0% |

Within the chosen groups the region is kept almost every time; next come the wealth and vulnerability indices. Attention gets 7-15%
of the selection weight, SHAP the rest. **Top climate features** (`table_main_climate_top3.csv`): heating need, heating need per
person, cold winter x electric heating, population density x urban area, winter solar radiation -- none beats the placebo.

**Maps.** `main_effect_maps_<line>` shows, region by region, how much each chosen variable pushes households **towards (red) or away
from (blue) Double Risk** (average SHAP value, CatBoost, pooled model); `main_value_maps` the variables' regional averages;
`main_country_effects` the effect per country. On the need line the **region** has by far the largest effect (up to +-0.15),
strongest in Poland; the household variables move predictions by 0.01-0.03.


8. More households, a multilevel model and a fair test of climate
-----------------------------------------------------------------

All scores are still computed only on the five study countries.

* **A multilevel (mixed) model.** Households are grouped in regions, and each region's effect is estimated with *partial pooling*.
  Used twice: a **linear mixed model** (statsmodels MixedLM) for the climate test, and **GPBoost** (Sigrist 2022) -- tree boosting
  with a region random effect -- as a second bill model ("multilevel bill model").
* **Six extra training countries** (France, Germany, Norway, Serbia, Ukraine, the UK; regions decoded for Germany, Norway, Serbia and
  Ukraine; France and the UK have none).
* **Climate tested on the energy bill itself**: the log bill regressed on household characteristics, country (so only differences
  *within* a country count) and the region's winter and summer degree days, with a random effect per region; the share of the
  regional differences explained by climate; a **placebo** that shuffles climate between regions of the same country 200 times.
* **Outcomes where climate should matter**: the usual indoor temperature (asked in Hungary and Spain, and in Germany, France,
  Ukraine) -- **cold home** (winter 19 C or below), **hot home** (summer 24 C or above), each also combined with low income
  (**winter / summer energy poverty**) -- and **hidden under-heating** (all five countries).

**Climate drives what households pay, especially for heating** (`table_climate_elasticity.csv`, `table_region_variance.csv`,
figure `climate_bill`):

| Outcome | Countries | Households / regions | 10% colder winter changes the bill by | Share of regional differences explained by climate | Placebo |
|---|---|---|---|---|---|
| Energy bill | five | 2,845 / 131 | **+5.6%** (1.3 to 10.1) | 6% | p = 0.055 |
| Energy bill | + DE, NO, RS, UA | 5,524 / 194 | **+4.7%** (1.2 to 8.3) | 5% | **p = 0.025** |
| Heating bill | five | 2,904 / 130 | **+12.7%** (6.5 to 19.3) | **19%** | **p < 0.005** |
| Heating bill | + DE, NO, RS, UA | 5,723 / 193 | **+11.8%** (6.9 to 16.9) | **16%** | **p < 0.005** |

Of the regional differences in heating bills, climate explains 19%, the regional economy 25%, both together 29%; climate stays
significant next to the economy (p = 0.02). Country by country the effect is measurable in Italy and Spain, the countries with a
wide climate range inside their borders (`table_country_slopes.csv`).

**Climate-sensitive outcomes** (`table_climate_labels_glm.csv`, `table_climate_labels_ml.csv`, figure `climate_labels`):

| Outcome | Share (Hungary / Spain) | Climate adds (ROC AUC, CatBoost / XGBoost) | Beyond the placebo? |
|---|---|---|---|
| Cold home | 7% / 32% | +0.03 / +0.04 | no |
| Hot home | 44% / 34% | +0.05 / +0.06 | no |
| Winter energy poverty | 5% / 9% | +0.01 / +0.02 | partly (XGBoost +0.02) |
| **Summer energy poverty** | 30% / 9% | +0.02 / +0.02 | **yes (+0.01 both)** |
| **Hidden under-heating** (five countries) | 8-16% | +0.04 / +0.07 | **yes (CatBoost +0.02; regression p < 0.005)** |

Cold homes are more common in mild Spain (32%) than in cold Hungary (7%) and, within countries, less common in colder regions: homes
in cold places are built and heated for the cold. With 31 regions in two countries this cannot be separated from other regional
differences.

**More households and the multilevel bill model, on the four groups** (`table_pool_metrics.csv`, `table_best_vs_main.csv`, figure
`pool`; same households and folds as the main study; fixed B1 variables):

| Line | Main study's recipe (ensemble_plus, five countries) | + extra countries | + multilevel bill model and extra countries (**ensemble_mixed**) |
|---|---|---|---|
| P80 line | 0.703 | 0.702 | **0.713** (+0.010, not significant) |
| Need line | 0.599 | 0.599 | **0.616** (+0.017, significant) |
| Need, no climate | 0.633 | 0.623 | 0.637 (+0.005) |
| Energy-burden line | 0.711 | 0.709 | 0.710 (-0.001) |

Extra countries alone hardly help (+0.004 on average; better in 23, worse in 18 of 240 comparisons). The multilevel bill model needs
them: trained on five countries it is worse than the plain bill model (-0.03); with them it is as good, and inside the model average
gives the best need-line score. For comfort outcomes the extra countries hurt (up to -0.05 AUC).

**Further checks of the climate question.**

* **Climate inside the bill model** (`table_bill_climate_deltas.csv`): +0.01 to +0.02 over none, but the placebo climate gains as
  much (-0.007 to +0.002; 0 of 60 comparisons significant).
* **Power** (`table_power_bill.csv`, `table_power_detection*.csv`, figures `power`, `power_grouped`). The bill test, pooled, detects
  an elasticity of 0.67-0.85 for the heating bill (observed 1.2) and 0.51-0.62 for the total bill (observed 0.48-0.57); within
  single countries only Italy and Spain have enough climate range. For the four groups, bills were made artificially more
  climate-dependent: **with the region known, even an added elasticity of 2 is not detected** -- climate has one value per region,
  so the region variable (and the placebo) absorbs any regional effect. Scoring households in **regions the model never saw**
  detects an added elasticity of 1 in 6 of 9 cases and 2 in all; on the real data the bill model gains +0.012 to +0.020 over the
  placebo there (positive in 88-97% of resamples, not significant). **The real climate effect on the bill is about half of what
  this sample can detect in the four groups.**

**Bottom line.** Climate *does* matter -- for the size of the energy bill (+12% heating bill per 10% colder winter), for summer
energy poverty and for under-heating. For sorting households into the four expenditure-based groups its effect is real but small and
fully captured once the region is known.


9. Validation against the reference studies
-------------------------------------------

**Van Hove et al. (2022): does our pipeline reproduce their results when run their way?** (`validate_van_hove_replica.py`,
`table_validation_van_hove_summary.csv`, figure `validation_van_hove`). The reference design -- P80 line, core profile, CatBoost, one
stratified 20% test split, best validation iteration -- run on the five study countries over 20 random splits:

| Design (20 random splits) | Recall: No / Income / Expenditure / Double Risk | Weighted F1 | Balanced accuracy |
|---|---|---|---|
| **Van Hove et al. (2022), as published** (one split, 11 countries) | 0.60-0.74 per group | about 0.65 (validation), 0.63 (5-fold CV) | -- |
| Reference design, five countries pooled, their settings | 0.66 / 0.64 / 0.58 / 0.67 | 0.68 +- 0.03 | 0.64 +- 0.03 |
| the same with this study's CatBoost settings | 0.67 / 0.64 / 0.59 / 0.70 | 0.68 +- 0.02 | 0.65 +- 0.03 |
| the same with "don't know" codes as missing | 0.66 / 0.66 / 0.58 / 0.67 | 0.68 +- 0.02 | 0.64 +- 0.03 |
| one model per country, their settings | -- | 0.64-0.77 | 0.54 (Italy) - 0.75 (Poland) |

**Our pipeline reproduces the reference results.** Run their way, the pooled model's per-group recall (0.58-0.67) and weighted F1
(0.68) sit within the range they report. The spread over 20 random splits (+-0.03, up to +-0.12 for Spain alone) is as large as
most differences between single-split studies, and per-country models are weaker in Bulgaria and Italy -- the countries' difficulty,
not the pipeline, explains lower numbers there.

**Leave one country out** (`build_loco_validation.py`, `table_loco.csv`, figure `loco`). The locked test partition is not reused, so
the strictest remaining check is a country the model has never seen: trained on the other four study countries + the six extra
countries.

| Line | Balanced accuracy in the unseen country (ensemble_mixed) | Same country inside training (cross-validation) |
|---|---|---|
| P80 line | 0.52-0.70 | 0.66-0.78 |
| Energy-burden line | 0.56-0.69 | 0.64-0.80 |
| Need line | 0.51-0.53 | 0.56-0.67 |
| Need, no climate | 0.50-0.59 | 0.61-0.68 |

**Models do not carry over to a new country**, least of all for the need lines (no better than income alone): the income side
transfers (brackets are national deciles), the spending side does not -- prices, fuel mix and what counts as "high" are national.
In an unseen country the simple B0 core-profile CatBoost does as well as the ensemble.

**Compared with Van Hove et al. and Dalla Longa et al.: what this study adds.**

* **Thresholds.** Optimising both thresholds shows that a pure spending percentile does not track hardship (hardship gap close to
  zero for the P80 line, negative for the optimised percentile); the need and burden lines do. Dalla Longa et al. moved their
  thresholds by +-20% and found stable accuracy; we show that accuracy alone is the wrong goal -- the easiest threshold to predict is
  not the one that means hardship.
* **Variables.** Under the P80 line the core profile captures most of what can be learned, as in Van Hove et al. Under need lines,
  extra information adds 5-7 points, mostly "which region is this". Area statistics, central in Dalla Longa et al.'s neighbourhood
  study, help single households only by identifying the region; household-level indices add a little.
* **Income only / without income / full** behave as in Dalla Longa et al.: income is indispensable, and without it scores fall close
  to their single-household results.
* **Sample size.** Following Dalla Longa et al.'s finding that about 1,000 training households are needed, one pooled model lets
  Spain take part although it is too small for a model of its own. More households from other countries help only through the
  multilevel model, and do not make a model portable.
* **Methods and policy.** Model averages and the bill model improve on the B0 core-profile baseline by 0.02-0.10; a scheme helping the
  top 30-35% of households reaches 70% of Double-Risk households on the need lines (40% with income alone).
* **Climate**, untested in either reference study: a real driver of the bill, not of the four-group classification.


10. Problems found and how they were handled
--------------------------------------------

* **Confident learning flagged 38% of households** as mislabelled -- a sign of weak models, not of 38% wrong labels. A margin rule
  (the model must prefer the other spending side by 0.15) flags about 3% on the need lines and 19% on the P80 and burden lines.
* **The P80 line fails the group minimum in Spain** (5 Double-Risk households); kept as the baseline, Spain's own P80 results are
  not read alone.
* **The threshold optimisation crashed when only one option was feasible**; fixed.
* **Imputed households did not help**; kept as a sensitivity check.
* **The mixed model's default optimiser stopped at a degenerate fit** (region variance 0, infinite likelihood): it now starts with
  Powell's method and refuses non-finite fits. **GPBoost's default optimiser aborted** on one training set: Fisher scoring is used
  everywhere (the gain of the best recipe on the P80 line changed from +0.014, significant, to +0.010, not significant).
* **The frozen split could not be regenerated** by the split code in its final form (only 44% of households matched), and it lived
  in a git-ignored folder: it is now a committed input (`config/household_split_v1.csv`).
* "Heating need per euro of income" is not a climate feature (it was income in disguise); regional statistics worked only as region
  labels (replaced by the region itself); house value and home ownership are not in the survey (wealth index as stand-in).


11. Households that heat too little
-----------------------------------

A household that keeps its home cold to save money spends little, so no expenditure line can mark it as high spending. We estimate
each home's heating *need* (floor area x heat loss for its age and type x local heating degree days) and flag households that spend
less than half of what similar households spend for the same need (`build_hidden_underheating.py`, `table_hidden_underheating.csv`):
5-14% of households and 6-20% of low-income ones. Adding the low-income ones would enlarge Double Risk by 13-52% on the need line.
They do not report more income difficulty than other low-income households (Spain: less), so part of this may be free fuel (own
wood) or incomplete bills.


12. Limits
----------

* One survey round from 2018; 2,278 labelled households in five countries. Results show links, not causes.
* Income is only known as a bracket (1-10); energy costs are self-reported; income in euros is an estimate.
* Climate and area variables have one value per region (132 regions); the placebo shows how much of a gain is region identity.
* Small groups in some countries (Poland's high-income households, Italy's lowest brackets, Spain's Double Risk) limit
  country-specific results.
* The comfort outcomes rest on one question in two study countries (31 regions).
* The extra countries' costs use 2017 exchange rates; Ukraine's prices were subsidised and it has no Eurostat income deciles.
* The locked test partition was used once, before the final design; later results are cross-validated, with leave one country out
  as the independent check.


13. How to run it
-----------------

```bash
pip install -r requirement.txt
bash scripts/run_full_pipeline.sh        # every step in order; stops at the first error (about 7 hours with a GPU)
python -m pytest tests/ -q               # automatic checks
```

Step-by-step guide with every script and its outputs: [`docs/running_the_pipeline.md`](docs/running_the_pipeline.md). Draft paper:
[`docs/paper_draft.md`](docs/paper_draft.md). Record of every check and correction: [`AUDIT_AND_REPAIR_REPORT.md`](AUDIT_AND_REPAIR_REPORT.md).

| Folder | What is inside |
|---|---|
| `config/` | the frozen train / validation / test split; the record of the one test-set evaluation |
| `data/` | reading the ENABLE workbook; the study and extra countries |
| `lib/identification/` | lines and thresholds (`lines.py`), the study design (`stages.py`), variable selection (`selection.py`), the classifiers (`models.py`, `ft_transformer.py`), bill models (`bill_model.py`, `bill_mixed.py`), uncertain labels (`robust.py`), tuning (`tuning.py`), extra countries (`pool.py`), multilevel climate test (`multilevel.py`), heating need (`required_heat.py`), reference replica (`vanhove_replica.py`), scores (`metrics.py`), the data frames (`data.py`) |
| `lib/features/` | survey items, engineered features, household indices and climate x home features, partly asked modules, income in euros |
| `lib/climate/`, `lib/geography/` | climate measures per region; regions of every country |
| `lib/reporting/` | figure helper; the names used in every table and figure (`names.py`) |
| `scripts/` | every runnable step |
| `tests/` | automatic checks |


14. Earlier versions
--------------------

The project went through earlier designs; their code was removed in the clean-up of 2026-10-16 and remains in the git history:
the first identification study with information tiers, cost-threshold estimators and one pre-registered test (commits up to
`a25daff`); per-country models with Hansen threshold regression lines (`dca549d`, `ad666d3`); a four-country pooled design with area
statistics and their placebo (`7464ceb`, `673b0fe`); and the developments that led to this version (`6e8aad9`, `9ed3560`,
`a6ee86c`). Their outputs are archived locally in `legacy-archives/pre_cleanup_2026-10-16/`.


References
----------

* Van Hove, W., Dalla Longa, F., van der Zwaan, B. (2022). Identifying predictors for energy poverty in Europe using machine
  learning. *Energy & Buildings* 264, 112064. https://doi.org/10.1016/j.enbuild.2022.112064
* Dalla Longa, F., Sweerts, B., van der Zwaan, B. (2021). Exploring the complex origins of energy poverty in The Netherlands with
  machine learning. *Energy Policy* 156, 112373. https://doi.org/10.1016/j.enpol.2021.112373
* Koenker, R., Bassett, G. (1978). Regression quantiles. *Econometrica* 46, 33-50.
* Hwang, C.-L., Yoon, K. (1981). *Multiple Attribute Decision Making: Methods and Applications* (TOPSIS). Springer.
* Gorishniy, Y., Rubachev, I., Khrulkov, V., Babenko, A. (2021). Revisiting deep learning models for tabular data. *NeurIPS*.
* Lundberg, S. M., Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
* Chen, T., Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.
* Prokhorenkova, L., et al. (2018). CatBoost: unbiased boosting with categorical features. *NeurIPS*.
* Akiba, T., Sano, S., Yanase, T., Ohta, T., Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *KDD*.
* Northcutt, C. G., Jiang, L., Chuang, I. L. (2021). Confident learning: estimating uncertainty in dataset labels. *Journal of
  Artificial Intelligence Research* 70, 1373-1411.
* Sigrist, F. (2022). Gaussian process boosting. *Journal of Machine Learning Research* 23(232), 1-46.
* Seabold, S., Perktold, J. (2010). statsmodels: Econometric and statistical modeling with Python. *Proceedings of the 9th Python in
  Science Conference*.
* Breiman, L., Friedman, J., Olshen, R., Stone, C. (1984). *Classification and Regression Trees*. Hastie, T., Tibshirani, R.,
  Friedman, J. (2009). *The Elements of Statistical Learning*, Sec. 7.10.
* Eurostat: ilc_di01, ert_bil_eur_a, demo_r_d3dens, nama_10r_3gdp, lfst_r_lfu3rt, ilc_li41, cens_11htts_r2, cens_21dwop_r3; GISCO NUTS
  2016.

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
