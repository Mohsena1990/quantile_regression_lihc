# LIHC vs. HQRTM: Household-Specific Quantile-Regression Thresholds for Energy Poverty

This repository builds and validates **HQRTM** (Household Quantile-Regression
Threshold Method), an extension of the conventional **LIHC** (Low Income,
High Cost) energy-poverty indicator. Instead of flagging a household as
"high energy cost" when its expenditure exceeds a fixed country-level
percentile, HQRTM predicts each household's *expected* expenditure from a
quantile regression on its structural/demographic characteristics (plus
country fixed effects), and flags it as high-cost only if actual
expenditure exceeds that household-specific expectation. The repository
also trains CatBoost classifiers on both label variants for descriptive
profiling, and includes a validation suite that stress-tests HQRTM's
central assumptions (see [Validation & Robustness Suite](#validation--robustness-suite)).

**Headline result from that suite:** after fixing two real bugs in how
energy cost was measured (see [Data Construction](#data-construction-total_expenditure)),
neither HQRTM nor traditional LIHC shows a validity advantage against
independent hardship markers, and there's a specific, documented reason
why -- see [Central Finding](#central-finding-the-consumption-anomaly-problem).
The mirror-image flag -- households spending *less* than their structural
profile predicts, rather than more -- does show a strong, robust
relationship with independent hardship markers; see
[Promising Alternative](#promising-alternative-the-restriction-flag).

Comparison figures for all of the above: see
[Output Layout](#output-layout) and `outputs/figures/`.

## Country Coverage

The pipeline covers **7 of the 11 ENABLE.EU survey countries**: Bulgaria,
Germany, Hungary, Italy, Serbia, Spain, Ukraine. France, Norway, Poland,
and the UK are dropped in `preprocessing.py` (`EXCLUDED_COUNTRIES`)
because those four have **0% valid `SettlementSize` responses** -- the
question was never recorded for them, not scattered individual
non-response -- so they can't support any analysis that conditions on
local settlement type, and were judged not to have data of comparable
quality to the rest of the sample. All labeled datasets, the CatBoost
training grid, and the validation suite below operate on this 7-country,
6,529-household sample.

Within those 7 countries, the survey wasn't administered uniformly either
-- two more module-level (not scattered) gaps exist:

| Module | 0% present in | 100% present in |
|---|---|---|
| `C2`/`C3` (aircon use, heating strategy) and `C7A`-`C7F` (policy-intervention support) | Bulgaria, Italy, Serbia | Germany, Spain, Hungary, Ukraine |
| `G2A4`/`G2A5`/`G6A`-`G6F` (policy-priority attitudes) | Spain, Italy (`G2A5` also missing for Germany specifically) | Bulgaria, Hungary, Serbia, Ukraine |

`S7` (received public financial aid / social tariff for energy bills) is
the one exception: **100% present in all 7 countries** -- it's a
revealed-behavior item, not an attitude-survey one, which is also why
it's usable as an external-validity marker (see
[Validation & Robustness Suite](#validation--robustness-suite)) and as
country policy context (see below) without needing a further country cut.
Requiring every policy-related item above to be complete as well would
cut the sample to just Germany, Hungary, and Ukraine -- more internally
consistent, but it specifically removes Bulgaria and Serbia, which have
among the highest Double-risk rates in the sample. `C2`/`C3` were
therefore dropped from the modeling feature set instead (see
Configuration below) rather than cutting more countries.

## Data Construction: `total_expenditure`

`total_expenditure` is the outcome variable behind every LIHC and HQRTM
threshold in this repository. `preprocessing.py` originally built it from
two bugs, both now fixed:

1. **Units bug.** `H8A` (monthly electricity bill) and `H8B` (annual
   electricity bill) were summed together and the sum multiplied by 12
   regardless of which was present. That's only correct when H8A alone is
   present (8,823 households); it silently corrupted the other 1,272
   (11% of the original 11,265-household sample) -- mixing incompatible
   units when both were answered, or inflating an already-annual figure
   12x when only H8B was answered. Fixed by coalescing (prefer the
   already-annual H8B, fall back to H8A x 12) instead of summing.
2. **Scope bug.** `total_expenditure` was electricity-only -- heating
   fuel cost (`H7A1`/`H7A2`, answered by over half the sample) was never
   included, even though for any household not heating with electricity,
   heating is typically the dominant energy cost. Fixed by adding
   annualized heating cost (`H7A2` if available, else `H7A1 x` the number
   of months heating was actually paid for, `H7AA`) -- except for
   households whose main heating source is electricity or an electric
   heat pump (`H6A1`/`H6A10`), whose heating cost is already inside their
   electricity bill; adding H7 for those would double-count it.

This changed the numbers substantially: the HQRTM high-cost rate at
q=0.65 moved from ~15% to ~25% of households, and Double-risk prevalence
roughly doubled. It also reversed the paper's one positive
external-validity finding -- see the next section.

## Repository Layout

```
ENABLE.EU_dataset_survey of households.xlsx   Raw survey data (input)
preprocessed_data_clean.csv                   Cleaned, feature-engineered dataset (generated)
df_lihc.csv, df_hqrtm_{60,65,70}.csv          Labeled datasets, one per method/quantile (generated)

UKK/LIHC-Informed-Socio-Economic-Predictors/
├── preprocessing/
│   ├── preprocessing.py       Main runner: raw Excel -> cleaned + labeled CSVs
│   ├── risk_category.py       assign_traditional_lihc / assign_hqrtm / assign_paper_lihc
│   └── ...                    feature engineering, missing values, outliers, encoding
├── model/baseline1(CatBoost)/
│   ├── catboost_run_preprocessed.py   Main runner: trains CatBoost on all label x feature-block combos
│   ├── catboost_run_visualization.py  Performance/SHAP diagnostics for the trained models
│   └── ...
├── visualization/
│   ├── viz_style.py                        Shared palette + chart-style helpers (validated categorical/sequential/diverging colors)
│   └── validation_findings_comparison.py   The 5 cross-method comparison figures, see below
└── validation/                 Robustness/validity checks for the HQRTM method itself (see below)

outputs/                        All generated results, see Output Layout below
├── catboost/                   CatBoost training runs, tuning results, per-country metrics
├── validation/                 Validation-suite + country-policy-context result tables
└── figures/                    Standalone comparison plots (PNG + PDF)
```

## Output Layout

Everything generated by the pipeline lands under `outputs/` at the repo
root, mirroring which stage produced it:

- **`outputs/catboost/final_saved_models_catboost/`** -- from
  `catboost_run_preprocessed.py`, trained fully separately per country
  (see [Running the Pipeline](#running-the-pipeline)): one subfolder per
  `{Country}/{dataset}/{model}/k{k}` combo (tuning results, best params,
  confusion matrices), plus master summaries `all_results_summary.csv`
  (one row per combo, with a `country` column) and
  `all_results_by_country.csv` at the top level.
- **`outputs/validation/`** -- one subfolder per validation script
  (`external_validity_results/`, `homogeneity_check_results/`,
  `specification_stability_results/`, `hqrtm_specification_search/`,
  `restriction_flag_search/`), see
  [Validation & Robustness Suite](#validation--robustness-suite), plus
  `country_policy_context/`, see [Country Policy Context](#country-policy-context).
- **`outputs/figures/`** -- the 5 cross-method comparison figures from
  `validation_findings_comparison.py`, each as `.png` + `.pdf`:
  `method_comparison_odds_ratios`, `restriction_flag_quantile_sweep`,
  `hqrtm_specification_search_heatmap`,
  `specification_stability_range_by_country`,
  `country_policy_context_scatter`.

## Requirements

- **Python >= 3.10 in principle** -- several modules use `X | None` type
  hints (PEP 604). In practice every module that matters for actually
  running the pipeline (`risk_category.py`, `catboost_run_preprocessed.py`)
  now starts with
  `from __future__ import annotations`, which defers evaluation of those
  hints and makes them safe on Python 3.9 too. Verified working end-to-end
  on Python 3.9.13. `setup.py` still declares `>=3.10` since that was the
  original target; treat 3.9 as "works, but not the declared floor."
- Dependencies: see `requirement.txt` (mirrors `setup.py`'s `install_requires`).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirement.txt
# or: pip install -e .        (editable install via setup.py, same dependencies)
```

## Configuration

All hardcoded paths from earlier development machines have been replaced
with paths resolved relative to the repo root (`Path(__file__).resolve().parents[N]`),
so every script below runs unmodified as long as you invoke it **from the
repository root**. Nothing needs manual editing.

Note: the model directory name `baseline1(CatBoost)` contains parentheses
and a space -- quote it in shell commands, e.g. `"UKK/LIHC-Informed-Socio-Economic-Predictors/model/baseline1(CatBoost)/catboost_run_preprocessed.py"`.

`catboost_run_visualization.py` is the one exception: its active section
(from the `import os, json, pandas as pd, ...` line onward) reads
`selected_features.json` and `best_params.json` from a flat model
directory that the current `catboost_run_preprocessed.py` does not
produce in that shape (it writes `best_params.json` per-run, nested under
`outputs/catboost/final_saved_models_catboost/<dataset>/<model>/<k>/`, and never writes
`selected_features.json` at all). This is a real mismatch between two
pipeline versions, not a path issue -- fixing the path alone will not make
it run. Skip it for now; it isn't required for steps 1, 2, or 4 below.

## Feature Configuration

`catboost_run_preprocessed.py`'s feature blocks (`BASE_STRUCTURAL`,
`CONTEXT_FEATURES`, `SOCIOECONOMIC_AUX`) had three issues, now fixed:

- **`C2`/`C3` removed from `CONTEXT_FEATURES`.** Both were 0% present in
  Bulgaria, Italy, and Serbia (see [Country Coverage](#country-coverage)),
  so every B3/B4 CatBoost model was getting a placeholder "0"/"Missing"
  category for those three countries instead of real data. `SettlementSize`
  and `S6` stay -- both are 100% present in all 7 countries.
- **`S6` added to `CATEGORICAL_LIKE`.** `S6`'s codes (big city / suburb /
  town / village / farm / don't know) have no numeric or ordinal meaning,
  but `S6` was missing from this set, so CatBoost had been treating it as
  a plain continuous number (implying, for instance, that "don't know"=6
  is numerically "beyond" a farm=5).
- **Dead `"ownership"` entry removed from `CATEGORICAL_LIKE`.** No column
  named `ownership` exists anywhere in the pipeline; this referenced a
  feature that had apparently been dropped from the feature blocks at
  some point without being cleaned out of this set.

These changes affect the CatBoost training grid's B3/B4 feature blocks;
if `outputs/catboost/final_saved_models_catboost/` already has results
from before this fix, rerun step 2 below to regenerate them.

## Country Policy Context

`S7` (received public financial aid / social tariff to pay energy bills)
is 100% present in all 7 countries -- unlike the attitude-survey items
above, it's a revealed measure of policy reach, not something with a
module-level administration gap. Interpreting a country's Double-risk
rate without this context risks reading policy coverage as if it were
purely a deprivation signal.

```bash
python "UKK/LIHC-Informed-Socio-Economic-Predictors/validation/country_policy_context.py"
```

Reports, per country: Double-risk rate (both methods), the `S7`
aid-receipt rate with a 95% CI, and -- where the module was administered
(NaN otherwise, see [Country Coverage](#country-coverage)) -- the share
who spontaneously mentioned price regulation or market liberalization as
a policy priority (`G2A4`/`G2A5`), and the perceived success of low-income
energy support policy (`G6A`, 1=very successful..5=very unsuccessful).
One result worth noting when interpreting country-level rates: Ukraine's
energy-aid receipt rate (51.8%) is far higher than every other country's
(1-7%), a striking difference in policy reach that has nothing to do with
the quantile-regression/percentile methodology and everything to do with
that country's existing utility-subsidy programs.

## Running the Pipeline

Run from the repository root, in this order.

**1. Preprocessing -- raw survey data to labeled datasets**
```bash
python "UKK/LIHC-Informed-Socio-Economic-Predictors/preprocessing/preprocessing.py"
```
Produces `preprocessed_data_clean.csv`, `df_lihc.csv`, `df_hqrtm_60.csv`,
`df_hqrtm_65.csv`, `df_hqrtm_70.csv` at the repo root.

**2. CatBoost training**
```bash
python "UKK/LIHC-Informed-Socio-Economic-Predictors/model/baseline1(CatBoost)/catboost_run_preprocessed.py"
```
Trains CatBoost **fully separately per country** (`PER_COUNTRY_MODE = True`):
each of the 7 countries gets its own train/test split, its own
class-stratified K-fold CV (`class_stratified_k_fold_split` -- plain
stratified K-fold on the risk label, since grouping by country no longer
makes sense once the data is already filtered to one), and its own tuned
CatBoost model, for every {label variant} x {feature block} x {k}
combination. This replaced an earlier pooled-training design specifically
to get a model whose feature importances and thresholds aren't averaged
across countries with different policy contexts (see
[Country Policy Context](#country-policy-context)). The real cost: ~7x
the combos of pooled training, and some `{country, dataset, k}`
combinations don't have enough "Double risk" households to support a
given k-fold split at all -- those are detected up front and skipped
(logged, not crashed) rather than silently failing partway through a
multi-day run. `n_iter` (CatBoost random-search tuning trials) is set to
**8, not 20**, specifically to keep the full 7-country grid inside a
few days of wall-clock time -- a deliberate accuracy/runtime tradeoff, not
an oversight (measured on this machine: the full grid at `n_iter=8`
extrapolates to roughly 2-2.5 days sequential across all 7 countries).
Results save under `outputs/catboost/final_saved_models_catboost/<Country>/<dataset>/<model>/k<k>/`,
with master summaries `all_results_summary.csv` (one row per combo, with
a `country` column) and `all_results_by_country.csv` at the top level.

**3. Model diagnostics and SHAP** -- currently broken independent of path, see
[Configuration](#configuration); skip until reconciled with step 2's actual output layout.
```bash
python "UKK/LIHC-Informed-Socio-Economic-Predictors/model/baseline1(CatBoost)/catboost_run_visualization.py"
```

**4. Visualization suite** -- run once the
[Validation & Robustness Suite](#validation--robustness-suite) below has
produced its result CSVs; this reads from `outputs/validation/`, not
from the CatBoost training output.
```bash
python "UKK/LIHC-Informed-Socio-Economic-Predictors/visualization/validation_findings_comparison.py"
```
Produces the 5 cross-method comparison figures described in
[Output Layout](#output-layout), each as `.png` + `.pdf` in
`outputs/figures/`, styled via the shared `viz_style.py` module (a
validated categorical palette with a fixed hue order, applied
consistently so "Traditional LIHC" / "HQRTM" / "Restriction flag" always
get the same color across every figure). These replaced an earlier set of
per-dataset diagnostic plots (feature analysis, training fit diagnostics,
a dynamic pipeline-story HTML) that showed results one at a time; the
current suite is built specifically to let the validation findings be
compared side by side -- e.g. the headline `method_comparison_odds_ratios`
figure puts all three methods' odds ratios against all four hardship
markers on one forest plot.

## Validation & Robustness Suite

Added to test HQRTM's core assumptions directly, rather than asserting
them (this is what an Energy Economics desk-reject specifically flagged:
predicting a label from the variables used to construct it demonstrates
internal coherence, not measurement validity). Run from the repo root;
each is self-contained and reads the CSVs produced by step 1 above.

```bash
python "UKK/LIHC-Informed-Socio-Economic-Predictors/validation/homogeneity_assumption_check.py"
python "UKK/LIHC-Informed-Socio-Economic-Predictors/validation/specification_stability.py"
python "UKK/LIHC-Informed-Socio-Economic-Predictors/validation/external_validity_check.py"
python "UKK/LIHC-Informed-Socio-Economic-Predictors/validation/external_validity_check_crossfitted.py"
python "UKK/LIHC-Informed-Socio-Economic-Predictors/validation/hqrtm_specification_search.py"
python "UKK/LIHC-Informed-Socio-Economic-Predictors/validation/restriction_flag_search.py"
python "UKK/LIHC-Informed-Socio-Economic-Predictors/validation/country_policy_context.py"
```
(the last two are needed for the visualization suite's
`restriction_flag_quantile_sweep` and `country_policy_context_scatter`
figures -- see [Running the Pipeline](#running-the-pipeline) step 4.)

- **`homogeneity_assumption_check.py`** -- tests whether HQRTM's pooled
  quantile regression (one slope + a per-country intercept) is adequate,
  via a residual leftover-structure test, a refit with local-condition
  controls (SettlementSize etc.), and country-varying slopes estimated by
  bootstrap + empirical-Bayes shrinkage (`partial_pooling_qr.py`) rather
  than excluding numerically unstable countries.
- **`specification_stability.py`** -- reports the "Double risk" prevalence
  as a *range* across quantile x feature-set x pooling-method
  specifications instead of a single point estimate.
- **`external_validity_check.py`** -- tests HQRTM's "high cost" flag
  against four survey items never used to build the label (energy-bill
  aid received, subjective bill burden, winter indoor temperature,
  subjective income difficulty), via reclassification tables and
  horse-race logistic regressions against the traditional-LIHC flag.
- **`external_validity_check_crossfitted.py`** -- reruns the above on
  K-fold cross-fitted labels (`cross_fitted_labels.py`) so no household's
  threshold was estimated from a model that saw that household, and
  prints the in-sample-vs-cross-fitted comparison directly.
- **`hqrtm_specification_search.py`** -- sweeps HQRTM's quantile
  (0.50-0.80) and its previously-unexplained `margin_scale` buffer
  (0.0-0.20) -- 28 combinations x 4 markers -- to check whether any
  untried specification shows a genuine external-validity advantage
  before concluding the method doesn't have one. See
  [Central Finding](#central-finding-the-consumption-anomaly-problem).
- **`restriction_flag_search.py`** -- tests the mirror-image flag (actual
  expenditure *below* a conditional quantile of similar households, i.e.
  restricting/rationing rather than overspending) across quantiles
  0.10-0.40 and two feature sets, against the same four markers. See
  [Promising Alternative](#promising-alternative-the-restriction-flag).

All six write their tables to a `*_results/` (or `hqrtm_specification_search/`,
`restriction_flag_search/`) subfolder under `outputs/validation/`.

## Central Finding: The Consumption-Anomaly Problem

Running the validation suite above -- after the `total_expenditure` fix
-- produced a result that changed the paper's central claim:

**Neither HQRTM nor traditional LIHC's "high cost" flag reliably tracks
independent hardship markers** (S7: received energy-bill aid; C5B:
self-reported bill burden; C1A: winter indoor temperature; S8: subjective
income difficulty -- none used anywhere in constructing either label).
Before the `total_expenditure` fix, HQRTM showed a significant,
correctly-signed advantage over traditional LIHC on `income_difficulty`
(OR=1.35-1.45 depending on quantile, p<0.02, robust to cross-fitting).
After the fix, that result reversed: HQRTM shows no significant
relationship with any of the four markers, and a **specification search
across 28 quantile x margin combinations (112 tests total) found only one
significant, correctly-signed result -- fewer than pure chance at p<0.05
would produce.** Meanwhile `income_difficulty` specifically shows a
**consistent, often-significant *negative* association** with HQRTM's
flag across essentially the entire grid: households HQRTM calls
"high-cost" are *less* likely to report income difficulty, not more.

**Why**, and why this isn't fixable by re-tuning HQRTM: both methods
define "high cost" as an anomaly in *actual* expenditure relative to a
reference built from *actual* expenditure (a flat country percentile for
LIHC, a structural-characteristics-conditional quantile for HQRTM).
Neither has any information about whether a household achieved adequate
warmth or is going without. This has two effects that both push in the
same direction:

- **Comfortable households can overconsume** (higher setpoints, more
  appliances, no need to ration) and get flagged "high cost" with no
  hardship involved.
- **Deprived households often *underconsume relative to their structural
  need*** (rationing, heating one room, self-disconnecting), which pulls
  their actual spending toward or below what similar households typically
  spend -- so they are systematically *missed*, not flagged.

This is the standard critique of consumption-based (vs. needs-based /
engineering-modeled) fuel-poverty indicators in the literature: measures
built from what a household actually spent cannot distinguish deprivation
from thrift, or comfort from need. The original Hills (2012) LIHC and the
UK's current LILEE indicator avoid this specific problem by using
*modeled* required cost (an engineering estimate of what the dwelling
*should* cost to heat adequately) instead of actual spending -- at the
price of needing dwelling-engineering data (insulation values, heating
system efficiency) that this survey, like most household expenditure
surveys, doesn't collect. That data gap is *why* the actual-expenditure
family of LIHC variants (van Hove et al. 2022, and this repository's
implementation) exists; refining the reference point within that family,
as HQRTM does, cannot fix a limitation that comes from what the reference
point is built from in the first place.

**Implication for how the two methods here should be described:**
neither `assign_traditional_lihc` nor `assign_hqrtm` should be described
as measuring energy "need" -- both measure a statistical norm of observed
spending, and the validation suite's job was exactly to check whether
that norm tracks real hardship. It doesn't, robustly, and there's a
documented reason why.

## Promising Alternative: The Restriction Flag

If overconsumption relative to structural need doesn't track hardship,
does *under*-consumption -- households rationing, heating one room, or
self-disconnecting, which spend *less* than their structural profile
would predict? `restriction_flag_search.py` tests exactly this, using the
identical pooled quantile-regression machinery as HQRTM, just evaluated
at low quantiles (0.10-0.40) instead of high ones. The result is a large
step up from anything in the Central Finding above:

| Marker | Result |
|---|---|
| `energy_aid` (S7) | OR 1.58-2.05, **p<0.0001 at every quantile tested (0.10-0.40), both feature sets** -- the strongest, most robust result anywhere in this validation suite. |
| `cold_home` (C1A) | Significant and correctly signed at q=0.10 (OR=1.59, p=0.0007) -- the most mechanistically direct marker, strongest at the most extreme cutoff. |
| `income_difficulty` (S8) | Positive at q=0.20-0.35 (OR 1.12-1.18, several p<0.05) -- a **sign flip in the right direction** from HQRTM's high-cost flag, which was negatively associated with this exact marker. |
| `bill_burden` (C5B) | No clear signal either way. |

The `energy_aid` and `cold_home` associations survive in a combined model
that also includes HQRTM's and traditional LIHC's high-cost flags --
this is an independent signal, not something either existing method
already captures under a different name.

Adding `SettlementSize` to the quantile regression made essentially no
difference to any of these results (nearly identical odds ratios and
p-values with or without it): **the improvement here comes from flipping
which side of the distribution is flagged, not from the feature set.**

One real tradeoff: the best quantile isn't the same for every marker --
`energy_aid` peaks around q=0.15-0.25, while `cold_home` only shows up at
the more extreme q=0.10. Choosing a single quantile for a paper means
picking a point on that tradeoff deliberately, not by default.

```bash
python "UKK/LIHC-Informed-Socio-Economic-Predictors/validation/restriction_flag_search.py"
```

## Known Limitations

- `catboost_run_visualization.py`'s active section, see [Configuration](#configuration).
- `cross_fitted_labels.py`'s K-fold split (5 folds, stratified by Country)
  can occasionally put every occurrence of a rare `main_heating_source`
  category into one fold, leaving another fold's fit data with no
  coefficient for it. Those households (a handful per run, e.g. 3 out of
  6,529 in a typical run) are dropped from that fold's predictions rather
  than guessed at, and `external_validity_check.py`'s
  `build_analysis_frame` explicitly excludes any household without a
  valid flag under both methods rather than silently treating a missing
  prediction as "not high cost."
