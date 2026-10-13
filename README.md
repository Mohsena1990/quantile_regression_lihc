Finding energy-poor households with machine learning
=====================================================

**The question.** Which households are energy poor -- they have a low income *and* spend a lot on energy -- and can we find
them from information a government could collect? We study four countries: **Bulgaria, Hungary, Italy and Poland**.

**Where we start.** Van Hove, Dalla Longa & van der Zwaan (2022), "Identifying predictors for energy poverty in Europe using
machine learning", *Energy & Buildings* 264:112064. We use the same survey and the same four groups of households, and we
rebuild their model exactly before we try to improve it.

**What is new here.**
1. A better *expenditure line* -- the line above which a household counts as spending "a lot" -- built with
   **Quantile Regression** and **Hansen's Threshold Regression**.
2. Many more possible predictors, chosen automatically with **Attention Mechanisms as Embedded Selectors + SHAP**.
3. Local climate for every region, from the **Open-Meteo** weather archive.
4. Three different machine-learning models -- **CatBoost**, **XGBoost** and **FT-Transformer** -- compared on the same data.


1. The data
-----------

* **ENABLE.EU household survey (2018).** About 1,000 households per country. We use the households that reported both their
  income bracket and their energy costs (train + validation part of a fixed 60/20/20 split; the 20% test part was already used
  once and is not touched again). Place the workbook at `data/raw/ENABLE.EU_dataset_survey of households.xlsx` (not shared here).
* **Climate.** Daily weather for October 2016 - December 2017 (the survey asked about the winter of 2016/17) from the Open-Meteo
  historical weather API, one point per region (NUTS3 regions; Poland only has NUTS2 regions in the survey): 121 regions.
  30-year averages (1991-2020) from ERA5-Land monthly data (Copernicus).
* **Prices and income references** from Eurostat.


2. How a household is labelled
------------------------------

Every household gets one of four labels:

| | Spending is **not** high | Spending **is** high |
|---|---|---|
| **Income is not low** | No Risk | Expenditure Risk |
| **Income is low** | Income Risk | **Double Risk** (energy poor) |

**Low income** = income bracket 1, 2 or 3 out of 10 (the same rule as Van Hove).

**High spending** = the household's yearly energy bill (heating + electricity, in euros) is above an *expenditure line*.
We compare four ways to draw that line:

| Line | How it is drawn | In plain words |
|---|---|---|
| **Van Hove (80th percentile)** | The 80th percentile of energy spending in the country. | The 20% of households that spend the most. One line for the whole country. |
| **Quantile Regression** | Linear quantile regression (Koenker & Bassett 1978) of spending on physical need, at the 80th percentile. | The 20% who spend the most *compared with homes like theirs*: same floor area, household size, dwelling type and age, insulation and local climate. A big house in a cold region gets a higher line. |
| **Hansen Threshold Regression** | Hansen's Threshold Regression (Hansen 2000) finds the spending level where hardship jumps. | We look for a "tipping point" in spending above which low-income households suddenly report much more difficulty living on their income. That point becomes the country line. |
| **Hansen + Quantile Regression** | Hansen's tipping point is measured as a position among homes with the same need; that position becomes the percentile of the Quantile Regression line. | The tipping point decides *how high* the line is; Quantile Regression decides *how it bends* for each type of home. |

**How Hansen's Threshold Regression works.** Take the low-income households of a country. For each one we know where its spending
sits (from 0 = lowest to 1 = highest) and whether it says it finds it "difficult" or "very difficult" to live on its income
(survey question S8). We try every possible cut point and keep the one that best splits the households into "below" and "above"
groups with different hardship. A bootstrap test (Hansen 1996) then asks: is this cut real, or would we see one this strong by
chance? A 95% confidence range shows where the true point could be. The survey has one round only, so we use the
cross-section form of the method (Hansen 2000); the "panel" form needs the same households observed several times.

**What we found.**

* Using spending as it is, there is **no real tipping point** in any country (bootstrap p = 0.39 to 0.81) -- hardship among
  low-income households is about the same at every spending level.
* Using spending *compared with similar homes*, the four countries together show a **tipping point at the 58th position**
  (p = 0.042; 95% range 0.11 to 0.68). Above it, hardship rises from 51% to 57%. No single country has a clear tipping point of
  its own, so all four use this shared value (a country's own value would only be used if it were significant *and* hardship were
  higher above it).
* So the **Hansen + Quantile Regression** line is set at the 58th conditional percentile. It is the only line that gives every
  country at least 30 households in every group, and it separates hardship best: low-income households above the line report more
  hardship than those below it in Hungary (55% vs 47%), Italy (61% vs 53%) and Poland (47% vs 40%); with Van Hove's line the
  opposite happens in Bulgaria and Poland.
* The plain **Hansen Threshold** line falls at the 50th percentile, but it is not statistically supported (p = 0.48).

Figures: `hansen_hardship_jump` (where hardship jumps), `hansen_lr_curves` (Hansen's test), `expenditure_lines` (the four lines
drawn over real households), `expenditure_line_shap` (what moves the line: mainly floor area, household size and climate).
Tables: `table_hansen_tipping_points.csv`, `table_expenditure_lines.csv`, `table_expenditure_line_agreement.csv`.


3. What the models are allowed to see
-------------------------------------

We add information in steps. Income is used in every step (as in Van Hove's study).

| Step | What is included |
|---|---|
| **B0** | Van Hove's seven predictors, exactly as in their Table 1: income bracket, floor area, household size, house detachment (detached house ... large block of flats), house age, birth year of the respondent, and heating strategy (how the home's heating is controlled: one setting, adjusted by hand, programmed thermostat, no control, don't know). |
| **B1** | B0 + the best of **all 82 other variables**: every survey question answered in all four countries (vehicles, insulation, heating sources, renewables, the age of 8 appliances, lighting, smart meters, household members by age and sex, education, job, gender, area type), the help-with-bills question (S7) and the income-difficulty question (S8), the main heating fuel, and 23 features we built from the survey (for example floor area per person, number of old appliances, share of heating from wood, elderly living alone). |
| **B2** | B1 + the best of **18 climate measures** for the household's region: heating and cooling degree days (winter 2016/17, summer 2017 and the 1991-2020 average), coldest temperatures, frost days, longest cold spell, heatwaves, tropical nights, humidity, rain, snow, wind, gusts, sunshine, solar radiation and cloud. |
| **B2p** | A **placebo** check for B2: the same climate numbers, but swapped between regions of the same country. If B2p does as well as B2, the gain does not come from climate -- it comes from the model recognising regions (prices, incomes and housing also differ between regions). |


4. How the best variables are chosen
------------------------------------

**Attention Mechanisms as Embedded Selectors + SHAP.** Inside each training round:

1. Three models are trained with *all* candidate variables:
   * an **FT-Transformer** -- a neural network that turns each variable into a "token" and uses **attention** to decide which
     tokens to look at; how much attention it pays to each variable is its importance (an *embedded* selector: the choice is
     learned inside the model);
   * **CatBoost** and **XGBoost** -- tree models whose **SHAP values** (Lundberg & Lee 2017) say how much each variable moved
     each prediction.
2. Each variable gets a score = the average of its attention rank and its SHAP rank. Variables are sorted by this score.
3. We add the best variables one by one and measure accuracy each time. We keep **the smallest number of variables whose result
   is within one standard error of the best result** (the "one-standard-error rule", Breiman et al. 1984; Hastie, Tibshirani &
   Friedman 2009). This avoids adding variables that only help by luck.

All of this happens on training households only; the households used to score the models are never seen by the selection.


5. The three models and how they are tested
-------------------------------------------

* **CatBoost** -- gradient-boosted decision trees (the library Van Hove used); depth 4, learning rate 0.05, balanced class weights.
* **XGBoost** -- gradient-boosted decision trees (Chen & Guestrin 2016); depth 4, learning rate 0.05, 300 trees, balanced weights.
* **FT-Transformer** -- Feature Tokenizer + Transformer (Gorishniy et al. 2021); 2 layers, 4 attention heads, early stopping.

**Testing.** One model per country. Repeated 5-fold cross-validation (3 times): the households are split into 5 parts, each part
is predicted by a model trained on the other 4, and this is repeated 3 times with different splits. Every number has a 95%
confidence range from 200 bootstrap resamples. "Chance" = guessing labels at random in the right proportions.

**How to read the scores.**

| Score | Meaning |
|---|---|
| Recall (sensitivity), per class | Of the households that truly are in a group, the share the model finds. Van Hove call it the "true positive rate". |
| Precision, per class | Of the households the model puts in a group, the share that really belong there. |
| Accuracy | Share of all households labelled correctly. |
| Balanced accuracy | Average of the four recalls -- every group counts equally, however small. **Our main score.** |
| Weighted F1 | F1 (balance of precision and recall) averaged over groups by their size -- the score Van Hove reported ("TotalF1"). |
| ROC AUC | How well the model ranks households of a group above the others (0.5 = chance, 1 = perfect). |


6. Results
----------

**Balanced accuracy, averaged over the four countries** (`table_stages_metrics.csv`; chance is about 0.25 everywhere):

| Line | Model | B0 | B1 | B2 | B2p (placebo) |
|---|---|---|---|---|---|
| Van Hove (80th percentile) | CatBoost | 0.654 | 0.654 | 0.690 | 0.692 |
| | XGBoost | 0.588 | 0.587 | 0.641 | 0.651 |
| | FT-Transformer | 0.624 | 0.621 | 0.664 | 0.647 |
| Quantile Regression | CatBoost | 0.526 | 0.534 | 0.577 | 0.558 |
| | XGBoost | 0.534 | 0.531 | 0.577 | 0.566 |
| | FT-Transformer | 0.518 | 0.520 | 0.559 | 0.535 |
| Hansen Threshold Regression | CatBoost | 0.655 | 0.660 | 0.679 | 0.674 |
| | XGBoost | 0.617 | 0.633 | 0.668 | 0.664 |
| | FT-Transformer | 0.640 | 0.646 | 0.651 | 0.656 |
| Hansen + Quantile Regression | CatBoost | 0.510 | **0.558** | 0.578 | 0.579 |
| | XGBoost | 0.508 | **0.543** | 0.582 | 0.572 |
| | FT-Transformer | 0.501 | **0.545** | 0.562 | 0.555 |

**What this means.**

* **Extra variables (B1) help only when the line is fair to different homes.** With the Hansen + Quantile Regression line, B1
  raises balanced accuracy by 0.035 to 0.05 on average for all three models (significant in Poland for all three models, in
  Bulgaria for CatBoost and FT-Transformer, in Italy for XGBoost). With Van Hove's line,
  B1 adds nothing: his line mostly measures how big the home is, and floor area is already in B0.
* **Climate does not add much beyond "which region is this".** B2 beats its placebo B2p clearly in only 5 of 48 comparisons; on average
  the difference is between -0.01 and +0.02. The extra accuracy of B2 is mostly the model recognising regions.
* **The models are close.** CatBoost and XGBoost do about equally well; the FT-Transformer is a little behind (it is a neural
  network and our samples are small: 380-650 households per country).
* **Lines with a higher score are not "better" by that fact.** Van Hove's and the plain Hansen line are easier to predict
  because they mostly measure home size. The Hansen + Quantile Regression line is harder to predict, but it is the one that
  tracks hardship.

**Variables chosen most often in B1** (share of training rounds, averaged over the four lines; `table_stages_selection.csv`):
difficulty living on income (S8, 34%), floor area per person (31%), dishwasher age (30%), share of heating from wood (30%),
electric water heater age (30%), portable electric heater age (29%), area type (29%), number of appliances, share of elderly
members, number of vehicles, number of old appliances. Under the Hansen + Quantile Regression line the selector keeps about 8
variables (median); under Van Hove's line about 1. In B2 the most chosen climate measures are winter cloud cover, humidity,
snowfall and the longest cold spell -- the measures that best tell regions apart.

**What each model relies on** among the B1 variables (Hansen + Quantile Regression line; `table_stages_selected_importance.csv`,
figure `stages_selected_by_model`): CatBoost -- share of district heating, dishwasher age, S8, air-conditioner age, main heating
fuel, smart meter; XGBoost -- job status, main heating fuel, floor area per person, S8, air-conditioner age, district heating;
FT-Transformer (attention) -- district heating, solid-fuel share, area type, job status, air-conditioner age, S8.

**All scores for the Hansen + Quantile Regression line, step B1** (`table_stages_report.csv` has every line, step and model):

| Country | Model | Accuracy | Balanced accuracy | Weighted F1 | ROC AUC | Recall: No / Income / Expenditure / Double Risk | Precision: No / Income / Expenditure / Double Risk |
|---|---|---|---|---|---|---|---|
| Bulgaria | CatBoost | 0.60 | 0.60 | 0.61 | 0.87 | 0.60 / 0.57 / 0.64 / 0.57 | 0.67 / 0.69 / 0.57 / 0.45 |
| | XGBoost | 0.57 | 0.55 | 0.57 | 0.86 | 0.63 / 0.64 / 0.54 / 0.41 | 0.62 / 0.64 / 0.54 / 0.41 |
| | FT-Transformer | 0.58 | 0.57 | 0.58 | 0.86 | 0.64 / 0.54 / 0.57 / 0.54 | 0.64 / 0.66 / 0.57 / 0.42 |
| Hungary | CatBoost | 0.55 | 0.55 | 0.55 | 0.84 | 0.62 / 0.55 / 0.52 / 0.52 | 0.60 / 0.63 / 0.53 / 0.44 |
| | XGBoost | 0.55 | 0.53 | 0.55 | 0.84 | 0.59 / 0.61 / 0.45 / 0.48 | 0.56 / 0.63 / 0.48 / 0.45 |
| | FT-Transformer | 0.54 | 0.56 | 0.54 | 0.84 | 0.59 / 0.48 / 0.58 / 0.58 | 0.62 / 0.62 / 0.55 / 0.43 |
| Italy | CatBoost | 0.57 | 0.56 | 0.58 | 0.86 | 0.55 / 0.60 / 0.61 / 0.46 | 0.58 / 0.71 / 0.59 / 0.34 |
| | XGBoost | 0.59 | 0.56 | 0.59 | 0.85 | 0.56 / 0.73 / 0.59 / 0.35 | 0.57 / 0.72 / 0.59 / 0.37 |
| | FT-Transformer | 0.54 | 0.53 | 0.54 | 0.85 | 0.46 / 0.62 / 0.58 / 0.44 | 0.52 / 0.72 / 0.53 / 0.34 |
| Poland | CatBoost | 0.55 | 0.53 | 0.55 | 0.83 | 0.44 / 0.56 / 0.55 / 0.57 | 0.50 / 0.66 / 0.49 / 0.47 |
| | XGBoost | 0.58 | 0.53 | 0.58 | 0.83 | 0.46 / 0.67 / 0.47 / 0.51 | 0.47 / 0.67 / 0.46 / 0.51 |
| | FT-Transformer | 0.54 | 0.52 | 0.54 | 0.81 | 0.54 / 0.58 / 0.47 / 0.50 | 0.51 / 0.63 / 0.48 / 0.46 |

Figures: `stages_performance` (all lines, steps and models), `stages_selection_frequency`, `stages_selection_curve` (how many
variables to keep), `stages_selected_by_model`, `stages_shap_beeswarm_<line>` (what pushes a household towards Double Risk).


7. Checking our work against Van Hove
-------------------------------------

Our first Van Hove-style numbers looked lower than theirs. To find out why, we rebuilt their method step by step on our four
countries (`diagnose_vanhove_gap.py`; table `table_vanhove_gap_summary.csv`, figure `vanhove_gap`): their way of reading income
and costs, their exchange rates, their CatBoost settings (up to 500 trees, the best round chosen on a validation set) and their
80/20 test split -- repeated over 20 random splits instead of 1.

| Setting (20 random splits each) | Balanced accuracy | Weighted F1 |
|---|---|---|
| Van Hove's paper (11 countries, one split) | about 0.655 (recall 0.60-0.74) | 0.63-0.65 |
| Their method, our four countries in one model | 0.626 (+- 0.028) | 0.666 |
| Same, with our earlier CatBoost settings | 0.633 | 0.676 |
| Same, with "don't know" answers set to missing | 0.626 | 0.673 |
| One model per country: Bulgaria / Hungary / Italy / Poland | 0.56 / 0.64 / 0.54 / 0.75 | 0.64 / 0.69 / 0.67 / 0.75 |
| Same, with tuned settings | 0.56 / 0.66 / 0.54 / 0.73 | 0.67 / 0.73 / 0.70 / 0.76 |

**Conclusion: nothing was wrong.** Rebuilt their way, our results match theirs. Our numbers looked lower because (1) we reported a
stricter score (macro-F1 is about 0.11 lower than their weighted F1 on the same predictions), (2) one random split can move the
result by 0.05-0.08, and (3) Bulgaria and Italy are harder to predict than Hungary and Poland. Tuning the model raised accuracy
a little but not balanced accuracy (it finds fewer of the small groups). Two small data errors found on the way are fixed: the
floor-area answer "don't know" had been read as "larger than 200 m2", and some households had size 0 (41 of 1,996 households).


8. Households that heat too little
----------------------------------

A household that keeps its home cold to save money spends little, so no expenditure line can mark it as high spending. We
estimate each home's heating *need* (floor area x heat loss for its age and type x local heating degree days) and flag
households that spend less than half of what similar households spend for the same need (`build_hidden_underheating.py`):
5-14% of households, and 6-20% of low-income ones. Adding them would make the Double Risk group 16-51% larger under the
Hansen + Quantile Regression line. They do not report more income difficulty than other low-income households, so part of this
may be free fuel (own wood) or incomplete bills.


9. Limits
---------

* One survey round from 2018; about 400-650 usable households per country. Results show links, not causes.
* Income is only known as a bracket (1-10). Energy costs are self-reported.
* Van Hove's line gives fewer than 30 Double Risk households in Italy and fewer than 30 Expenditure Risk households in Poland, so
  those results are uncertain (marked in the tables).
* The Hansen tipping point is only significant when the four countries are pooled, and its confidence range is wide.
* Climate is measured at one point per region, so it cannot separate climate from other regional differences.


10. How to run it
-----------------

```bash
pip install -r requirement.txt
bash scripts/run_full_pipeline.sh        # every step in order; stops at the first error
python -m pytest tests/ -q               # 98 tests
```

The steps used by this README, in order:

```bash
python scripts/acquire_openmeteo_b2.py     # daily weather per region (cached; reruns work offline)
python scripts/acquire_era5_normals.py     # 1991-2020 monthly temperature (one Copernicus request)
python scripts/fuse_openmeteo_b2.py        # 18 climate measures per region
python scripts/diagnose_vanhove_gap.py     # Section 7
python scripts/build_expenditure_lines.py  # Section 2: Hansen tipping points and the four lines
python scripts/build_feature_stages.py     # Sections 3-6 (uses the GPU; about 1 hour with 4 workers)
python scripts/build_hidden_underheating.py  # Section 8
```

Tables go to `outputs/tables/`, figures to `outputs/figures/`. Step-by-step details: [`docs/running_the_pipeline.md`](docs/running_the_pipeline.md).
The full record of every check and correction: [`AUDIT_AND_REPAIR_REPORT.md`](AUDIT_AND_REPAIR_REPORT.md).

| Folder | What is inside |
|---|---|
| `data/` | Reading the survey and choosing the four countries |
| `lib/identification/` | Labels, expenditure lines (`expenditure_lines.py`), Hansen's Threshold Regression (`hansen.py`), steps and feature selection (`stages.py`), the three models (`models3.py`, `ft_transformer.py`), Van Hove rebuild (`vanhove_replica.py`), scores (`metrics.py`) |
| `lib/features/` | Survey variables (`survey_items.py`) and built features (`engineered.py`) |
| `lib/climate/` | Climate measures (`b2_indices.py`) |
| `scripts/` | Every runnable step |
| `tests/` | Automatic checks |


11. Earlier work in this repository
-----------------------------------

Before these steps, the project ran a broader study (README history and `AUDIT_AND_REPAIR_REPORT.md`, sections 1-27): income with
and without the model, three information levels, a check of which predictors hold in every country, the size of targeting errors
for a support scheme, and one final test on the 20% test households (pre-registered, done once on 2026-09-20). Those results were
computed before the two data fixes of Section 7 (2% of households) and with the earlier expenditure lines; they are kept for
reference and have not been rerun.


References
----------

* Van Hove, W., Dalla Longa, F., van der Zwaan, B. (2022). Identifying predictors for energy poverty in Europe using machine
  learning. *Energy & Buildings* 264, 112064. https://doi.org/10.1016/j.enbuild.2022.112064
* Hansen, B. E. (1996). Inference when a nuisance parameter is not identified under the null hypothesis. *Econometrica* 64, 413-430.
* Hansen, B. E. (2000). Sample splitting and threshold estimation. *Econometrica* 68, 575-603.
* Koenker, R., Bassett, G. (1978). Regression quantiles. *Econometrica* 46, 33-50.
* Gorishniy, Y., Rubachev, I., Khrulkov, V., Babenko, A. (2021). Revisiting deep learning models for tabular data. *NeurIPS*.
* Lundberg, S. M., Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
* Chen, T., Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.
* Prokhorenkova, L., et al. (2018). CatBoost: unbiased boosting with categorical features. *NeurIPS*.
* Breiman, L., Friedman, J., Olshen, R., Stone, C. (1984). *Classification and Regression Trees*. Hastie, T., Tibshirani, R.,
  Friedman, J. (2009). *The Elements of Statistical Learning*, Sec. 7.10.

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
