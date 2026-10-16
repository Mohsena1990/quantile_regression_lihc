# Who is energy poor, and does climate matter? Identifying low-income / high-cost households with machine learning in five European countries

*Draft, 2026-10-16. Numbers come from `outputs/tables`; the README and sections
34-36 of `AUDIT_AND_REPAIR_REPORT.md` document every step.*

## Abstract

Targeting energy-poverty support requires identifying households that combine low income with high energy expenditure, using
information a government can collect. Building on Van Hove et al. (2022) and Dalla Longa et al. (2021), we study 2,278 households of
the ENABLE.EU survey (2018) in Bulgaria, Hungary, Italy, Poland and Spain, with 6,458 households from six further countries as extra
training data. We (i) choose the income and expenditure thresholds by multi-objective optimisation, balancing identifiability, a
hardship-validity criterion and group size; (ii) compare four expenditure lines, including need-based quantile-regression lines and
the energy burden; (iii) select predictors in nested stages with attention- and SHAP-based embedded selection; and (iv) test climate
with region-swap placebos, a multilevel model of the energy bill, climate-sensitive comfort outcomes and a power analysis.
An ensemble of CatBoost, XGBoost, an FT-Transformer and a multilevel (GPBoost) bill model reaches a balanced accuracy of 0.62-0.71
(chance 0.25), against 0.51-0.67 for the seven-variable core profile (B0). Income is indispensable; beyond it, the household's region and
household-level indices carry most of the remaining information. Climate is a clear driver of the *level* of energy bills -- a 10%
colder winter raises the heating bill by about 12% within a country, and climate explains about a fifth of the regional differences
-- but it adds nothing to the *identification* of the four groups beyond knowing the region, even when built into a bill model.
It does carry signal for summer energy poverty and hidden under-heating. A targeting scheme that helps the 30-35% of households
ranked highest reaches 70% of Double-Risk households on need-based lines, against 40% with income alone. Models do not transfer to a country absent from training (balanced accuracy 0.50-0.70, need lines no better than income alone): each country needs survey data of its own.

## 1. Introduction

Energy poverty -- the inability to secure adequate energy services at an affordable cost -- is commonly measured by the
Low-Income/High-Cost (LIHC) logic: a household is at risk when its income is low *and* its energy expenditure is high. Support
schemes must find these households from information that registers or short screening questions can provide. Van Hove, Dalla Longa
and van der Zwaan (2022) showed with the ENABLE.EU survey that gradient-boosted trees can sort households into the four LIHC groups
from seven predictors; Dalla Longa, Sweerts and van der Zwaan (2021) studied the Netherlands, used the energy burden (spending as a
share of income), compared models with income only, without income and with both, and found that about 1,000 training households are needed
for stable results.

Three questions remain open. First, both thresholds -- who is "low income", who "spends a lot" -- are usually fixed by convention,
although they decide what the model must learn. Second, how much information beyond the seven predictors of Van Hove et al. helps, and of what
kind? Third, climate is the most frequently proposed driver of energy need, yet its role in *identification* is untested: climate
varies only between regions, so any apparent gain can simply reflect "which region is this".

Our contributions: (1) a multi-objective choice of both thresholds with an explicit validity criterion (the hardship gap);
(2) need-based expenditure lines from quantile regression, compared with the country percentile and the energy burden;
(3) nested, placebo-controlled feature stages with attention + SHAP embedded selection and three model families; (4) a set of
improvements tailored to small survey samples (a bill model, partial pooling, ensembles, extra countries); (5) a layered test of
climate -- on the four groups, on the bill itself with a multilevel model, on comfort outcomes, and with a power analysis; and
(6) policy read-outs: targeting curves and a "not sure" option; and (7) validation against the reference design and in an unseen
country.

**Naming.** The study borrows two building blocks and gives them its own names. Its baseline expenditure line, the **P80 line**
(income bracket <= 3 and spending above the country's 80th percentile), is the definition of Van Hove et al. (2022); its baseline
stage, the **B0 core profile**, is their seven-predictor set; and the comparison of **income-only, without-income and full models**
follows Dalla Longa et al.'s (2021) Models A, B and C, as does the **energy-burden line**. Tables and figures use these names; the
comparison with the two studies themselves is reported separately (Section 4.6).

## 2. Data

**Survey.** ENABLE.EU household survey (2018), about 1,000 households per country, interviews November 2017 - February 2018. We use
the development part (train + validation; 3,845 households) of a fixed 60/20/20 split of five countries; the 20% test part was used
once, before this study's final design, and is not used again. 2,278 households report both an income bracket (1-10) and energy costs
(Bulgaria 584, Hungary 646, Italy 382, Poland 384, Spain 282). Annual energy expenditure = heating (season total, or monthly cost x
months) + electricity, converted to EUR with Eurostat 2017 rates.

**Extra training countries.** France, Germany, Norway, Serbia, Ukraine and the United Kingdom (6,458 households) are used only to
train pooled models, never to score them. Their region codes were decoded (German states, Norwegian NUTS3, Serbian NUTS2, Ukrainian
oblasts; France and the UK have none).

**Climate.** Daily reanalysis for October 2016 - December 2017 from the Open-Meteo historical weather API (Zippenfenig 2023; ERA5,
ERA5-Land, CERRA), one point per NUTS3 region (Poland NUTS2): 132 study regions + 63 extra-country regions; 1991-2020 normals from
ERA5-Land monthly means. Twenty regional measures (degree days of the survey season and the normals, anomalies, cold spells,
heatwaves, tropical nights, humidity, precipitation, snow, wind, radiation, cloud) and ten climate x home interactions (e.g. modelled
heating need = floor area x heat-loss coefficient x degree days).

**Area context.** Eurostat population density, GDP per head, unemployment, poverty rate, ownership and old-dwelling shares.

## 3. Methods

### 3.1 Labels
Four groups: No Risk, Income Risk (low income only), Expenditure Risk (high expenditure only), Double Risk (both). Low income =
bracket <= c. High expenditure = cost above a per-household line in EUR: the **P80 line**, our baseline (country 80th percentile of
cost with c = 3, the definition of Van Hove et al.), **Quantile
Regression on need** (linear conditional tau-quantile of cost given floor area, household size, dwelling type and age, insulation and
degree days; out-of-fold within country), the same **without climate** (the fair test for climate predictors), and the **energy
burden** (country tau-percentile of cost / income).

### 3.2 Thresholds by multi-objective optimisation
For c in {2,3,4} and tau in {0.60, ..., 0.90}, three objectives: identifiability (balanced accuracy of the B0 core profile),
validity (hardship gap: share reporting difficulty living on income among Double Risk minus Income Risk households) and stability
(smallest group); at least 15 households per group and country; Pareto front, then TOPSIS. Chosen: need line c = 3, tau = 0.60;
burden c = 3, tau = 0.60; the P80 baseline (c = 3, tau = 0.80) is kept whatever the optimisation chooses (its hardship gap is near zero).

### 3.3 Feature stages and selection
B0 = the core profile, the seven predictors of Van Hove et al. (income bracket, floor area, household size, house detachment,
house age, respondent's birth year, heating strategy) + country. B1 adds groups chosen in two levels -- household indices (dwelling efficiency, inefficient equipment, heating
system, vulnerability, wealth, income per adult equivalent), three top survey items, partly asked modules (environmental attitudes,
energy-saving behaviour, energy literacy) and the region (NUTS2). B2 adds the three best-ranked climate measures; B2p is the same with
climate swapped between regions of the same country (placebo). Ranking: reliability-weighted average rank of FT-Transformer [CLS]
attention, CatBoost TreeSHAP and XGBoost TreeSHAP over three splits ("Attention Mechanisms as Embedded Selectors + SHAP"); size: the
one-standard-error rule with a paired gate. All selection is nested in the training folds.

### 3.4 Models and validation
One pooled model for all countries, read per country. CatBoost, XGBoost (also tuned with Optuna) and an FT-Transformer; model sets
(income only), B (without income), C (with income). Improvements: a **bill model** (XGBoost regression of log cost; probability that
the bill exceeds the household's line; balanced decision), a **multilevel bill model** (GPBoost: boosted trees + region random
effect), noisy-label weights (borderline down-weighting, confident learning), imputation of missing costs, and ensembles. Repeated
5-fold cross-validation (x3) stratified by country x group; household bootstrap 95% intervals; paired bootstrap for differences.
External validity is assessed by leaving each study country out: the model is trained on the other four study countries and the six extra countries and scored on the unseen country (the locked test partition is not reused).

### 3.5 Climate tests
(a) B2 against B2p in the four-group models; (b) climate measures inside the bill models, against the same placebo;
(c) a **linear mixed model** of the log bill and the log heating bill with country fixed effects, household terms, log heating degree
days and cooling degree days, and a region random intercept; the climate elasticity, the share of between-region variance explained
by climate, its comparison with regional economy, and 200 within-country climate shuffles as placebo; (d) **comfort outcomes** from
the usual indoor temperature (Hungary, Spain): cold home (<= 19 C in winter), hot home (>= 24 C in summer), each also combined with
low income, and hidden under-heating (bill per modelled heating need below half the country x fuel median); logistic regression with
region-clustered errors and placebo, and cross-validated CatBoost / XGBoost; (e) **power**: minimum detectable elasticities, and a
simulation adding known climate effects to the bills. In the simulation, bills are multiplied by exp(delta x within-country-centred log HDD) for delta in {0, 0.25, 0.5, 1, 2}, the lines recomputed, and the climate-versus-placebo test rerun (i) with the region known and (ii) scoring households in regions held out of training (folds by region, region variable removed).

## 4. Results

### 4.1 Thresholds
For the spending-percentile family the P80 baseline is easy to predict but has a hardship gap near zero (and the only option meeting the
minimum group size on five countries, the 60th percentile, has a negative gap):
households it labels Double Risk report no more difficulty than low-income households that spend little. The need-based and burden
lines have positive gaps (0.05). Easy-to-predict thresholds are not the meaningful ones.

### 4.2 Identification performance
Balanced accuracy (all five countries; chance 0.25):

| Line | Income only | B0 core profile (full model) | Main study's best (ensemble_plus) | Best with extra countries (ensemble_mixed) |
|---|---|---|---|---|
| P80 line (baseline) | 0.509 | 0.637 | 0.683 | **0.713** (0.691-0.735) |
| Need line | 0.511 | 0.513 | 0.596 | **0.616** (0.599-0.633) |
| Need line, no climate | 0.515 | 0.527 | 0.625 | **0.637** (0.621-0.652) |
| Energy burden | 0.587 | 0.665 | 0.685 | **0.710** (0.694-0.728) |

Double-Risk recall of the best model is 0.63-0.71, precision 0.34 (P80 line, where Double Risk is 9% of households) to 0.71
(burden, 27%). Without income scores fall to 0.39-0.54, as in Dalla Longa et al.'s single-household results. Averaging
models is the most reliable gain (+0.01 to +0.05 over each single model); predicting the bill and comparing it with the line moves
the operating point towards the spending side (need line: Double-Risk recall 0.52 -> 0.74); tuning, noisy-label weights and imputed
households add little or nothing. Extra countries help only through the multilevel bill model, which needs the larger sample to
estimate region effects.

### 4.3 What drives identification
Income first; the other six core-profile predictors add 0.07-0.13 for the P80 and burden lines but almost nothing for need lines,
which already condition on home size. Beyond B0, the region is chosen in every training round on the need lines; regional statistics
(GDP, unemployment, poverty, density, ownership) gained only as region labels (placebo of the regional statistics: +0.003). Household indices (wealth,
vulnerability) add a little; attitudes, energy-saving behaviour and energy literacy add nothing. Attention received 7-15% of the
selection weight, SHAP the rest.

### 4.4 Climate
*Identification.* Climate never beat its placebo in the four-group models (B2 - B2p = +0.001 on average; 4 better, 2 worse of 60),
including the climate-free need line, and not inside the bill models either (climate +0.01 to +0.02 over none, but -0.007 to
+0.002 against the placebo; 0 of 60 comparisons significant).

*The bill.* Within countries, the heating bill rises with winter severity with an elasticity of 1.26 (95% CI 0.66-1.85; +12.7% per
10% colder winter); with the extra countries 1.17 (0.70-1.64). Climate explains 19% (16%) of the between-region variance of heating
bills, and no placebo shuffle came close (p < 0.005). For the total bill the elasticity is 0.57 (0.14-1.01), 6% of the regional
variance (placebo p = 0.055; 0.025 with the extra countries). Regional economy explains 25%, both together 29%, and climate remains
significant next to the economy (p = 0.02). Country by country, the effect is estimable only in Italy and Spain, the countries with a
wide internal climate range.

*Comfort outcomes.* Cold homes are more frequent in mild Spain (32%) than in cold Hungary (7%) and, within countries, less frequent in
colder regions (odds ratio 0.63 per 10% more degree days), consistent with adaptation of buildings and heating to local climate; with
31 regions this is not separable from other regional differences (placebo p = 0.46). Climate improves the prediction of every comfort
outcome (+0.01 to +0.07 AUC) but beats its placebo only for summer energy poverty (+0.01) and hidden under-heating (+0.02).

*Power.* Pooled over countries the bill test detects elasticities of 0.67-0.85 for the heating bill (observed 1.2) and 0.51-0.62 for
the total bill (observed 0.48-0.57, borderline). Within single countries other than Italy and Spain, only implausibly large effects
(bill +50% per 10% colder winter) would be detectable. For the four groups the simulation is decisive about the design: with the region known, even an added elasticity of 2 is not
detected, because a region-constant climate effect is absorbed by the region identifier (and by the placebo, which also identifies
regions). Scored in unseen regions, the test detects added elasticities of 1 in 6 of 9 model-line cases and of 2 in all; on the real
data the bill model gains +0.012 to +0.020 over the placebo (P80 and climate-free need line; positive in 88-97% of
bootstrap resamples, not significant) and the classifiers nothing. The real effect on the total bill (elasticity about 0.5) is thus
about half of what this sample can detect in the four-group classification.

### 4.5 Targeting and uncertainty
Ranking households by their probability of Double Risk, a scheme reaches 70% of Double-Risk households by helping the top 25% (Van
Hove's line), 30-35% (need lines) or 30% (burden), against 35-40% with income alone; the top 20% contains half of all Double-Risk
households on the need line (36% with income alone). Flagging the least certain 20% of households for a follow-up check raises
balanced accuracy on the rest by 0.02-0.04; flagged households come from all four groups in similar shares.

### 4.6 Validation against the reference studies
*Van Hove et al.* The reference design (P80 line, core profile, CatBoost with their settings, one stratified 20% test split)
run through this pipeline on the five study countries over 20 random splits gives per-group recall 0.58-0.67 and weighted F1
0.68 +- 0.03 (balanced accuracy 0.64 +- 0.03), against their published 0.60-0.74 per group and weighted F1 of about 0.63-0.65 on
one split of eleven countries: the pipeline reproduces their results. The spread over splits (+-0.03; +-0.12 for Spain alone) is as
large as most differences between single-split studies, and one model per country is markedly weaker in Bulgaria (0.56) and Italy
(0.54) than in Poland (0.75). *Dalla Longa et al.* Their comparison of income-only, without-income and full models carries over:
income alone reaches 0.50-0.59, without income 0.39-0.54, and the full models 0.51-0.67 at B0 and up to 0.71 with the best
recipe; their sample-size argument motivates the pooled model, which lets Spain take part; and their energy-burden line is, with
the need line, the definition that tracks hardship.

### 4.7 External validity: an unseen country
Trained without the country, balanced accuracy falls to 0.52-0.70 for the P80 line (0.66-0.78 when the country is in
training), 0.56-0.69 for the burden line (0.64-0.80), and 0.51-0.53 for the need line (0.56-0.67) -- the level of income alone. The
income axis transfers because brackets are national deciles; the expenditure axis does not, because prices, fuel mix and the meaning
of "high" spending are national. Out of country, the simple B0 core-profile CatBoost does as well as the ensemble. A model
built in one set of countries cannot replace a (small) survey in the country where it is used.

## 5. Discussion

**Thresholds are a substantive choice.** Accuracy alone would favour lines that track home size; a validity criterion (hardship gap)
changes the choice, and the P80 line, while easy to predict, does not single out households in hardship.

**What information to collect.** Income (from registers), basic dwelling data, the region and a few household indices give most of
the attainable accuracy. Area statistics do not add information beyond the region itself, contrary to their role in Dalla Longa et
al.'s neighbourhood-level study; at the household level, household information matters more.

**Climate matters for the bill, not for the label.** Climate shapes how much households pay -- strongly for heating -- but the LIHC
groups compare households within a country against a line that, for need-based definitions, already adjusts for climate. Once the
region is known, the within-country climate contrast carries no further information about who falls above the line. Climate is
therefore better used where it acts: in modelling required energy (the need line), in under-heating indicators and in summer energy
poverty, rather than as a predictor of the LIHC class.

**Small samples.** Partial pooling (multilevel models) and model averaging were the effective remedies; adding households from other
countries helps only when the model can separate country and region effects, and it does not make a model portable: the spending
side of energy poverty is national, so each country needs at least a small survey of its own.

**Testing region-level predictors.** A placebo that swaps a regional variable between regions controls for "region identity", but
it also removes any power to see a regional effect when the region is known. Region-held-out validation is the design that can
detect it; we recommend it for any study of area-level drivers.

## 6. Limitations

One cross-section (2018); self-reported costs and bracketed income; climate and area data at regional resolution (one point per
region); comfort outcomes from a single question in two study countries; the extra countries' prices differ (Ukraine heavily
subsidised); all post-design results are cross-validated, with the locked test partition used once before the final design.

## 7. Conclusions

Low-income/high-cost energy poverty can be identified from register-type information with balanced accuracy of about 0.62-0.71
within the countries a model is trained on, but not in a country it has never seen.
The choice of thresholds and expenditure line matters as much as the model. Climate is a real driver of energy bills, but adds no
information for identifying LIHC households beyond their region; its policy value lies in defining need and in summer and hidden
energy poverty.

## Tables and figures (outputs/)

| Item | File |
|---|---|
| Threshold optimisation | `figures/threshold_moo_v5`, `tables/table_threshold_moo_v5.csv` |
| Model sets by stage | `figures/main_model_sets`, `tables/table_main_metrics.csv` |
| Improvements | `figures/main_variants`, `tables/table_main_deltas.csv` |
| Extra countries, multilevel bill model | `figures/pool`, `tables/table_pool_metrics.csv`, `table_best_vs_main.csv` |
| Climate and the bill | `figures/climate_bill`, `tables/table_climate_elasticity.csv`, `table_region_variance.csv` |
| Climate-sensitive outcomes | `figures/climate_labels`, `tables/table_climate_labels_glm.csv` |
| Climate inside the bill model | `tables/table_bill_climate_deltas.csv` |
| Power | `figures/power`, `figures/power_grouped`, `tables/table_power_bill.csv`, `table_power_detection*.csv` |
| Validation against the reference study | `figures/validation_van_hove`, `tables/table_validation_van_hove_summary.csv` |
| Leave one country out | `figures/loco`, `tables/table_loco.csv` |
| Targeting, "not sure" | `figures/main_targeting`, `figures/main_abstention` |
| Regional effects (SHAP maps) | `figures/main_effect_maps_<line>`, `figures/main_country_effects` |

## References

Akiba et al. (2019) Optuna, KDD. -- Chen & Guestrin (2016) XGBoost, KDD. -- Dalla Longa, Sweerts & van der Zwaan (2021) Energy Policy
156:112373. -- Gorishniy et al. (2021) Revisiting deep learning models for tabular data, NeurIPS. -- Hersbach et al. (2023) ERA5.
-- Koenker & Bassett (1978) Regression quantiles, Econometrica 46. -- Lundberg & Lee (2017) SHAP, NeurIPS. -- Munoz Sabater (2019)
ERA5-Land. -- Northcutt, Jiang & Chuang (2021) Confident learning, JAIR 70. -- Prokhorenkova et al. (2018) CatBoost, NeurIPS. --
Seabold & Perktold (2010) statsmodels. -- Sigrist (2022) Gaussian process boosting, JMLR 23. -- Van Hove, Dalla Longa & van der Zwaan
(2022) Energy & Buildings 264:112064. -- Zippenfenig (2023) Open-Meteo, doi:10.5281/zenodo.7970649.
