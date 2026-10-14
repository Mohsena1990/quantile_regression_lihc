Finding energy-poor households with machine learning
=====================================================

**The question.** Which households are energy poor -- they have a low income *and* spend a lot on energy -- and can we find
them from information a government could collect? We study four countries: **Bulgaria, Hungary, Italy and Poland**.

**The two studies we build on.**

* **Van Hove, Dalla Longa & van der Zwaan (2022)**, *Energy & Buildings* 264:112064 -- the same ENABLE.EU survey, four groups of
  households (low income x high spending), seven predictors, CatBoost.
* **Dalla Longa, Sweerts & van der Zwaan (2021)**, *Energy Policy* 156:112373 -- the same four groups for the Netherlands; spending
  as a *share of income*; area context (house value, ownership, building year, household size, population density); Models A / B / C
  (income only / other features only / both); a sensitivity check of both thresholds; and the finding that about 1,000
  training households are needed for stable results.

**What this version does (version 4, with the placebo checks of version 4b).**

1. **Chooses the two thresholds** (low income, high spending) by **multi-objective optimisation** instead of fixing them.
2. Trains **one model on all four countries together** (about 2,000 households) and **reads the results country by country**.
3. Reports **Models A / B / C** at every step, as Dalla Longa et al. did.
4. Adds **Dalla Longa's area variables** (from Eurostat) and **household indices** built from the survey, chosen in two levels.
5. Builds **climate x home** features that differ for every household, and keeps the **top 3** climate features.
6. Shows **how each chosen variable acts in each region** on maps.
7. Compares **CatBoost**, **XGBoost** and **FT-Transformer** throughout, choosing variables with
   **Attention Mechanisms as Embedded Selectors + SHAP**.


1. The data
-----------

* **ENABLE.EU household survey (2018).** About 1,000 households per country; we use the 1,996 households of the development part
  (train + validation of a fixed 60/20/20 split) that reported both income bracket and energy costs. The 20% test part was used
  once, earlier, and is not touched again. Place the workbook at `data/raw/ENABLE.EU_dataset_survey of households.xlsx` (not shared).
* **Climate.** Daily weather for October 2016 - December 2017 (the survey asked about the winter of 2016/17) from the Open-Meteo
  historical weather API, one point per region (NUTS3; Poland NUTS2 -- the survey has no finer location there): 121 regions.
  30-year averages (1991-2020) from ERA5-Land monthly data (Copernicus).
* **Area context (Eurostat).** Population density and GDP per head (NUTS3), unemployment and poverty rates (NUTS2), share of
  households owning their home (Census 2011, NUTS2) and share of dwellings built before 1981 (Census 2021, NUTS3). The survey has
  no house value or home ownership question, and no consistent regional house-value statistic exists for these countries, so a
  **wealth index** built from the survey stands in for house value.


2. Who counts as energy poor: the four lines and their thresholds
-----------------------------------------------------------------

Every household gets one of four labels:

| | Spending is **not** high | Spending **is** high |
|---|---|---|
| **Income is not low** | No Risk | Expenditure Risk |
| **Income is low** | Income Risk | **Double Risk** (energy poor) |

"Low income" means an income bracket (1-10) at or below a cut-off. "High spending" means above an *expenditure line*. We compare
four lines:

| Line | High spending means | Where it comes from |
|---|---|---|
| **Van Hove** | above the country's 80th percentile of yearly energy spending | Van Hove et al. (2022) |
| **Quantile Regression on need** | above what similar homes spend -- a linear quantile regression of spending on floor area, household size, dwelling type and age, insulation and local heating/cooling degree days | this study |
| **Quantile Regression on need, no climate** | the same, without the climate variables | a fair test of climate (see Section 3) |
| **Energy burden** | above the country's percentile of *spending as a share of income* (income in euros from the bracket and Eurostat deciles) | Dalla Longa et al. (2021) |

**How the thresholds were chosen (multi-objective optimisation).** For each line we tried every income cut-off (bracket 2, 3 or 4)
with every expenditure level (60th to 90th percentile in steps of 5) and scored each option on three goals:

* **Easy to identify:** how well Van Hove's seven predictors find the four groups (balanced accuracy).
* **Meaningful:** among low-income households, how much more often Double Risk households say they find it difficult to live on
  their income than Income Risk households (the "hardship gap"). This goal stops the search from simply picking lines that measure
  home size.
* **Stable:** the size of the smallest group (Dalla Longa et al.'s minimum sample size).

Options with fewer than 15 households in any group in any country were dropped. From the options that no other option beats on all
three goals (the Pareto front), the best balance was chosen by **TOPSIS** (equal weights). Figure `threshold_moo`, table
`table_threshold_moo.csv`.

| Line | Chosen threshold | Identifiability | Hardship gap | Smallest group |
|---|---|---|---|---|
| Van Hove | bracket <= 3, 80th percentile -- **the optimisation picked Van Hove's own choice** | 0.638 | 0.001 | 194 |
| Quantile Regression on need | bracket <= 3, 60th conditional percentile | 0.514 | 0.055 | 369 |
| Energy burden | bracket <= 2, 70th percentile | 0.641 | 0.040 | 297 |

What this shows: the country-percentile line is easy to predict but **does not track hardship at all** (gap close to 0 for every
option); the need-based and burden lines do. A per-country minimum of 30 per group (used before) was impossible for almost any
option: Poland's sample is 83% low-income, and Italy has few households in the lowest brackets.


3. What the models are allowed to see
-------------------------------------

| Step | Included |
|---|---|
| **B0** | Van Hove's seven predictors, as in his Table 1 (income bracket, floor area, household size, house detachment, house age, birth year of the respondent, heating strategy = how heating is controlled) + country |
| **B1** | B0 + the best of four groups: **Dalla Longa group** (population density, home-ownership share, wealth index as house-value stand-in); **ENABLE top 3** (difficulty living on income, floor area per person, dishwasher age); **indices** (dwelling efficiency, inefficient equipment, heating system type, household vulnerability, income per adult in euros); **regional economy** (GDP per head, unemployment, poverty rate, share of old dwellings) |
| **B1p** | **placebo for B1**: the six variables that have one value per region (population density, ownership share, GDP, unemployment, poverty, old dwellings) are swapped with another region of the same country before B1 is chosen. Only gains *beyond* B1p come from what those variables measure rather than from "which region is this". |
| **B2** | B1 + the **top 3** of 30 climate features: 20 regional (degree days of the survey year and the 30-year normal, how unusual the winter and summer were, cold, frost, cold spells, heatwaves, tropical nights, humidity, rain, snow, wind, gusts, sun, cloud) and 10 **climate x home** features that differ for every household: heating need (floor area x heat loss for the building's age and type x degree days), heating need per person, cold winter x old home, cold winter x no insulation, cold winter x solid-fuel or electric heating, hot summer x air conditioning, humidity x old home, wind x detached house, population density x urban area. ("Heating need per euro of income" was dropped from the list: it divides by income, and a first run showed its strength came from income, not climate.) |
| **B2p** | **placebo for B2**: each region's climate is swapped with another region of the same country (household climate x home features recomputed). Only gains *beyond* B2p count as a climate effect. |

The **no-climate need line** is the fair test of climate: when the line already contains climate, climate predictors cannot add much.


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

* **One pooled model** for all four countries (with a country variable). Every score is then computed **per country** on that
  country's held-out households, and also for all four together. As a check, a separate model per country was fitted at B0.
* **Models A / B / C** (Dalla Longa et al.): **A** = income bracket only; **B** = the step's variables *without* income;
  **C** = the step's variables *with* income.
* **Classifiers:** CatBoost (depth 4, learning rate 0.05, balanced class weights), XGBoost (depth 4, learning rate 0.05, 300 trees,
  balanced weights), FT-Transformer (2 layers, 4 attention heads, early stopping).
* **Testing:** repeated 5-fold cross-validation (3 times), stratified by country x group; 95% ranges from 200 bootstrap resamples.

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

**Balanced accuracy, all four countries together** (`table_v4_metrics.csv`):

| Line | Model | A: income only | B: B0 / B1 / B2 without income | C: B0 | C: B1 | C: B1p placebo | C: B2 | C: B2p placebo |
|---|---|---|---|---|---|---|---|---|
| Van Hove | CatBoost | 0.530 | 0.452 / 0.469 / 0.473 | 0.638 | 0.648 | 0.652 | 0.649 | 0.646 |
| | XGBoost | 0.531 | 0.440 / 0.496 / 0.509 | 0.596 | 0.649 | 0.647 | 0.658 | 0.669 |
| | FT-Transformer | 0.509 | 0.459 / 0.461 / 0.470 | 0.628 | 0.642 | 0.649 | 0.637 | 0.648 |
| Quantile Regression on need | CatBoost | 0.512 | 0.377 / 0.419 / 0.435 | 0.515 | 0.546 | 0.548 | 0.550 | 0.561 |
| | XGBoost | 0.510 | 0.397 / 0.480 / 0.493 | 0.520 | **0.584** | 0.574 | **0.605** | 0.597 |
| | FT-Transformer | 0.504 | 0.389 / 0.414 / 0.413 | 0.517 | 0.548 | 0.544 | 0.551 | 0.557 |
| Need line, no climate | CatBoost | 0.503 | 0.390 / 0.433 / 0.443 | 0.518 | 0.578 | 0.569 | 0.585 | 0.588 |
| | XGBoost | 0.501 | 0.410 / 0.486 / 0.503 | 0.541 | **0.624** | 0.607 | **0.633** | 0.628 |
| | FT-Transformer | 0.508 | 0.399 / 0.434 / 0.440 | 0.531 | 0.581 | 0.571 | 0.579 | 0.579 |
| Energy burden | CatBoost | 0.558 | 0.465 / 0.480 / 0.487 | 0.645 | 0.652 | 0.655 | 0.656 | 0.653 |
| | XGBoost | 0.555 | 0.466 / 0.482 / 0.537 | 0.624 | 0.637 | 0.638 | 0.674 | 0.669 |
| | FT-Transformer | 0.550 | 0.468 / 0.472 / 0.487 | 0.640 | 0.635 | 0.636 | 0.647 | 0.652 |

**What this means.**

* **Income is essential.** Without income (Model B) scores fall to 0.38-0.54, close to Dalla Longa et al.'s finding for single
  households. Income alone (Model A) reaches 0.50-0.56.
* **Van Hove's six other predictors help only his own kind of line.** Over income alone they add 0.07-0.12 for Van Hove's and the
  burden line (up to +0.25 in Poland) but only 0.00-0.04 for the need-based lines -- those lines already account for home size.
* **B1 helps -- but the regional statistics only as region labels.** B1 beats B0 significantly in 26 of 60 comparisons (+0.03 to
  +0.06 on the need line, +0.05 to +0.08 on the need line without climate). But **B1p, with the regional statistics swapped between
  regions, does almost as well** (B1 - B1p = +0.003 on average; better in 6, worse in 3 of 60). So GDP, unemployment, poverty,
  density, ownership and old dwellings help only because they tell regions apart. When their values are scrambled, the selector
  turns to **household-level information** instead -- the wealth index (73% of rounds), income in euros (62%), the vulnerability
  index (50%) and the inefficient-equipment index (47%) -- and reaches the same accuracy. One exception: on the need line without
  climate, XGBoost does a little better with the real regional values (+0.017), probably because they carry some of the climate
  the line leaves out.
* **Climate adds nothing beyond the region.** B2 beats B1 in 7 of 60 comparisons, but **B2 against its placebo is -0.001 on average
  (better in 2, worse in 2)**, also on the need line without climate -- the fair test -- and with the household-level
  climate x home features.
* **Pooling costs nothing.** One model for all four countries scores the same as a separate model per country (significantly better
  in 0, worse in 2 of 60 comparisons), with the larger sample Dalla Longa et al. recommend.
* **Best combinations:** XGBoost with B1/B2 on the need-based lines (0.58-0.63); CatBoost on Van Hove's and the burden line (0.65-0.66).

**All scores, Quantile Regression on need, Model C, step B1** (`table_v4_report.csv` has every line, step, model and country):

| Scope | Model | Accuracy | Balanced accuracy | Weighted F1 | ROC AUC | Recall: No / Income / Expenditure / Double Risk | Precision: No / Income / Expenditure / Double Risk |
|---|---|---|---|---|---|---|---|
| All four | CatBoost | 0.55 | 0.55 | 0.55 | 0.86 | 0.57 / 0.55 / 0.55 / 0.52 | 0.62 / 0.66 / 0.50 / 0.41 |
| | XGBoost | 0.60 | 0.58 | 0.60 | 0.87 | 0.63 / 0.67 / 0.58 / 0.47 | 0.65 / 0.68 / 0.55 / 0.45 |
| | FT-Transformer | 0.55 | 0.55 | 0.55 | 0.85 | 0.58 / 0.55 / 0.56 / 0.51 | 0.63 / 0.65 / 0.51 / 0.40 |
| Bulgaria | XGBoost | 0.62 | 0.59 | 0.62 | 0.88 | 0.67 / 0.70 / 0.57 / 0.44 | 0.68 / 0.67 / 0.56 / 0.47 |
| Hungary | XGBoost | 0.57 | 0.55 | 0.58 | 0.85 | 0.60 / 0.64 / 0.49 / 0.48 | 0.63 / 0.67 / 0.46 / 0.46 |
| Italy | XGBoost | 0.62 | 0.58 | 0.61 | 0.87 | 0.60 / 0.78 / 0.62 / 0.31 | 0.63 / 0.72 / 0.59 / 0.38 |
| Poland | XGBoost | 0.58 | 0.58 | 0.58 | 0.85 | 0.51 / 0.61 / 0.65 / 0.53 | 0.61 / 0.67 / 0.55 / 0.47 |

Figures: `v4_models_abc` (Models A/B/C at every step), `v4_pooled_vs_country`, `threshold_moo`.


7. Which variables matter, and where
------------------------------------

**Groups chosen in B1** (share of training rounds; `table_v4_selection_groups.csv`):

| Group | Van Hove | Need line | Need, no climate | Energy burden |
|---|---|---|---|---|
| Dalla Longa (density, ownership, wealth) | 33% | 53% | 33% | 80% |
| ENABLE top 3 | 0% | 33% | 20% | 13% |
| Indices | 27% | 93% | 40% | 53% |
| Regional economy | 87% | 93% | 100% | 47% |

With the real regional values, the most chosen variables are **GDP per head** and **share of old dwellings** (60% of rounds each),
**income per adult in euros** (48%), the **wealth index** (43%) and the **poverty rate** (37%). With the regional values scrambled
(B1p), the household-level indices take their place at the same accuracy (Section 6). `table_v4_selection_features.csv`.

**Top 3 climate features** (`table_v4_climate_top3.csv`): **heating need** and **population density x urban area** (Van Hove and
burden lines), **heating need per person**, **winter solar radiation** and **hot summer x air conditioning** (need line), **cold
winter x electric heating** (need line without climate). None of them beats the placebo.

**Maps.** For each line, `v4_effect_maps_<line>` shows, region by region, how much each chosen variable pushes households
**towards (red) or away from (blue) Double Risk** (average SHAP value, CatBoost, pooled model); `v4_value_maps` shows the
variables' regional averages; `v4_country_effects` summarises the effect per country (`table_v4_country_effects.csv`,
`table_v4_region_effects.csv`). The same variable can act in opposite directions in different countries: on the need line, a
higher income in euros pushes towards Double Risk in Bulgaria and Hungary (+0.02) but away from it in Italy (-0.07), and winter
solar radiation pushes towards it in Hungary and Poland but away in Bulgaria and Italy. Van Hove et al. called such variables
"contextual predictors"; with one value per region, the regional and climate patterns on the maps describe *where* households are,
not a measured effect of the variable.


8. Problems found and how they were handled
-------------------------------------------

* **Fixed: the top "climate" feature was income in disguise.** "Heating need per euro of income" was chosen every time in the first
  run, and the placebo chose it too. It is no longer a climate candidate; climate still adds nothing.
* **Tested: regional statistics as region labels.** The B1p placebo shows their gain is region identity (Section 6).
* **Attention remains a weak selector** at this sample size: it gets 5-10% of the weight (SHAP from CatBoost and XGBoost the rest).
* **House value and home ownership are not in the survey.** The wealth index and the regional ownership share are stand-ins.
* **The per-country group minimum had to be lowered to 15** (from 30) to keep any threshold option feasible.


9. Compared with Van Hove and Dalla Longa: what this work adds
--------------------------------------------------------------

* **Thresholds.** Optimising both thresholds confirms Van Hove's choice *for his kind of line*, but shows that line does not track
  hardship; need-based and burden lines do. Dalla Longa et al. moved the thresholds by +-20% and found stable accuracy; we show that
  accuracy alone is the wrong goal -- the threshold that is easiest to predict is not the one that means hardship.
* **Variables.** Under Van Hove's line, his seven predictors capture most of what can be learned. Under a need-based line, extra
  information adds 3-8 points of balanced accuracy, but its source matters: **household-level indices** (wealth, income in euros,
  vulnerability, inefficient equipment) carry real information, while **area statistics** -- Dalla Longa et al.'s density and
  ownership, and regional economy -- only help by identifying the region. Dalla Longa et al. worked with neighbourhood averages,
  where such area variables are the unit of analysis; for single households, household-level information matters more, in line
  with their own single-household (WoON) results.
* **Pooling.** Following Dalla Longa et al.'s sample-size finding, one pooled model gives the same per-country accuracy as separate
  models and a larger, more stable training set.
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

* One survey round from 2018; 1,996 usable households. Results show links, not causes.
* Income is only known as a bracket (1-10); energy costs are self-reported; income in euros is an estimate.
* Climate and area variables have one value per region (121 regions), so they cannot separate climate from other regional
  differences; the placebo shows how much of a gain is region identity.
* Small groups in some countries (Poland's high-income households, Italy's lowest brackets) limit country-specific results.


12. How to run it
-----------------

```bash
pip install -r requirement.txt
bash scripts/run_full_pipeline.sh        # every step in order; stops at the first error
python -m pytest tests/ -q               # 108 tests
```

The version-4 steps, in order:

```bash
python scripts/acquire_openmeteo_b2.py      # daily weather per region (cached; reruns work offline)
python scripts/acquire_era5_normals.py      # 1991-2020 monthly temperature (one Copernicus request)
python scripts/fuse_openmeteo_b2.py         # climate measures per region
python scripts/acquire_regional_context.py  # Eurostat / Census area context
python scripts/build_threshold_moo.py       # Section 2: threshold optimisation
python scripts/build_feature_stages_v4.py   # Sections 5-7 (GPU; about 1.5 hours with 4 workers)
python scripts/build_hidden_underheating.py # Section 10
```

Tables go to `outputs/tables/`, figures to `outputs/figures/`. Details: [`docs/running_the_pipeline.md`](docs/running_the_pipeline.md).
The record of every check and correction: [`AUDIT_AND_REPAIR_REPORT.md`](AUDIT_AND_REPAIR_REPORT.md).

| Folder | What is inside |
|---|---|
| `data/` | Reading the survey and choosing the four countries |
| `lib/identification/` | Lines and thresholds (`lines_v4.py`), the version-4 design (`stages_v4.py`), shared selection tools (`stages.py`), the three models (`models3.py`, `ft_transformer.py`), Hansen's Threshold Regression (`hansen.py`), Van Hove rebuild (`vanhove_replica.py`), scores (`metrics.py`) |
| `lib/features/` | Survey variables (`survey_items.py`), built features (`engineered.py`), indices and climate x home features (`indices.py`) |
| `lib/climate/` | Climate measures (`b2_indices.py`) |
| `scripts/` | Every runnable step |
| `tests/` | Automatic checks |


13. Earlier versions
--------------------

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
