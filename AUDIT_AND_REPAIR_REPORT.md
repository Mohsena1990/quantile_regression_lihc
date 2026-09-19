# Audit and Repair Report

Date: 2026-08-11
Branch: `redesign/measurement-redesign-20260811`
Auditor: Claude (Sonnet 5), acting as senior scientific software engineer / statistician / energy-poverty methodologist / reproducibility auditor.

## 1. Executive summary

The repository implements a genuinely sophisticated, mostly-correct energy-poverty
measurement pipeline (M0 Van Hove benchmark, M1 historical-LIHC-aligned required-cost
engine, M2–M8 model-based measurement families, three screening architectures, two
imbalance strategies, an immutable split registry, and a staged execution-gate system).
The core scientific architecture described in the brief is faithfully represented in
code, and most of the "specifically investigate" bugs listed in the brief (H6A3
gas/LPG mispricing, pooled-Europe M0 thresholds, silent zero-price fill, fabricated
BREDEM coefficients, Kelvin/Celsius mistakes, forced M1_ADAPTED unblocking, income
leaking into primary screening, validation/test resampling) were checked and found
**absent** — the team that built this already got the hard scientific judgment calls
right in most places.

However, the audit found and fixed **3 CRITICAL** defects that would have silently
undermined the project's core validity claims if left unfixed, plus **1 additional
CRITICAL** split-stability defect found during the fix pass itself:

1. **S1 "hierarchical" CatBoost was not hierarchical** — it was plain CatBoost with a
   redundant country column, mislabeled. Fixed with a genuine global+country-expert
   shrinkage blend; all 44 affected artifacts regenerated.
2. **ERA5 fusion could never succeed** due to a QC check applied to the wrong grain
   (raw ocean-inclusive grid instead of the NUTS1 land aggregates), independent of a
   second bug where the gate-closure key didn't exist. Both fixed; fusion now runs
   correctly and reveals a **real, non-fabricatable** data gap (Canary Islands outside
   the ERA5 request's bounding box) as the genuine remaining blocker.
3. **The train/validation/test split was not actually UID-stable** — it used a
   positional permutation that would silently reshuffle up to 54% of households across
   partitions if the upstream cohort ever changed by even one row. This is the
   mechanism the entire leakage-prevention architecture depends on. Fixed with a
   per-UID hash and an immutability guard; the already-frozen split file was verified
   byte-identical before and after the fix (no household was moved).

7 HIGH-severity issues were fixed (screening governance test coverage, split registry
test coverage, a real train-row leakage-*reintroduction* path in
`run_model_measurements.py`, two divergent split-registry artifacts, and stale/
misleading M1 documentation). 10 MEDIUM and 4 LOW issues were fixed. 5 issues are
documented but deliberately not code-fixed (git tracking, a full content-hash
reproducibility redesign, an unwired-but-tested governance stub, a cohort-exclusion
documentation gap, and a low-risk hardcoded currency literal) — each with an explicit
rationale in `outputs/tables/audit_scientific_issues.csv`.

**Tests: 39 → 62 passing** (23 new regression tests added, one per confirmed fix
category; zero regressions).

## 2. Repository state when audit started

- Branch `redesign/measurement-redesign-20260811`, 39/39 tests passing.
- `git status` showed almost the entire pipeline codebase as untracked (`??`); only
  `README.md`, a nested README, `measurement/registry.py`, and `requirement.txt` were
  tracked/modified. See ISS-024.
- `artifacts/manifests/pipeline_status.json`: M0 AVAILABLE, 21/21 H measurements
  AVAILABLE, ERA5 not fused, 132 screening runs available, M1 required-cost
  IN_PROGRESS, M1 four-class BLOCKED_AHC_HOUSING_COST.
- Both ERA5-Land daily-mean and daily-minimum NetCDF files already existed on disk
  (118MB / 105MB, real gridded data), but `active_request.json` still claimed the
  minimum was not received, and fusion had never successfully run.
- Full snapshot: `artifacts/manifests/claude_audit_before.json`.

## 3. Scientific architecture understood

Five layers kept intentionally distinct: (A) data construction, (B) energy-cost/
energy-poverty measurement (M0/M1 benchmarks, M2–M8 models), (C) four-class household
labelling, (D) screening/classification (S1/S2/S3 × 2 imbalance strategies), (E)
validation/comparison. Confirmed in code: the 5-country/24-NUTS1 cohort, the 44-core/
58-sensitivity measurement registry, the fixed four-class ontology, the H/HC feature
parity design, the OOF-calibration approach, and the gated pipeline (M0 → H models →
ERA5 → HC models → M1 required-cost → screening).

## 4. Tests before repair

`python -m pytest tests/ -q` → **39 passed**, 0 failed (5 unrelated statsmodels
convergence warnings from the M8 mixed-model adapter).

## 5. Issues discovered

Full machine-readable list with severity, evidence, and fix status:
`outputs/tables/audit_scientific_issues.csv` (27 issues: 4 CRITICAL, 7 HIGH, 11
MEDIUM/MEDIUM-HIGH, 5 LOW/LOW-MEDIUM). Six parallel subagent audits covered: data
construction, the M0–M8 registry, M1's LIHC engine, the ERA5/NUTS1 pipeline, the
screening/imbalance layer, and split/cohort/gate/reproducibility governance; every
finding reported by a subagent was independently verified by direct code reading
and/or execution before being acted on.

## 6. Critical issues fixed

**ISS-001 — fake "hierarchical" CatBoost.** `screening/models.py`'s
`hierarchical_catboost` path was CatBoost with one extra (redundant) country feature
column fit as a single pooled model — no global model, no per-country experts, no
shrinkage. Replaced with a genuine partial-pooling blend:
`P_final = (1-alpha_c)*P_global + alpha_c*P_country`, `alpha_c = n_c/(n_c+lambda)`.
Countries below a minimum support threshold (20 rows, ≥2 classes) fall back to the
global model. `lambda` is selected via an internal train-only holdout split — never
via the caller's validation/test frame, so it is safe whether called in `development`
or `final_test` mode. Verified on real data: hierarchical predictions now disagree
with plain CatBoost on ~30% of validation rows. All 22×2=44 hierarchical_catboost
development-stage prediction artifacts regenerated; the 132-row governance table
rebuilt from the corrected per-measurement tables. 4 new tests in
`tests/test_hierarchical_catboost.py`.

**ISS-002/ISS-003 — ERA5 fusion structurally could not close.** Two independent bugs:
(a) `scripts/fuse_era5_nuts1.py` required the *entire raw* pan-European grid to be
finite before any NUTS1 subsetting, but ERA5-Land is land-only and ~53% of the raw
grid is legitimately NaN (sea mask) — this aborted fusion unconditionally, before it
could ever reach the actual 24 target regions; (b) `scripts/check_milestone_gates.py`
checked a QC key (`all_day_counts_valid`) that the fusion script never wrote, so even
a hypothetically successful fusion could never close Gate 3. Fixed both: the
finiteness check now applies only to the NUTS1-region daily aggregates that actually
feed HDD/CDD (using xarray's default skip-NaN spatial mean), and the fusion script now
writes one authoritative `qc['qc_all_clear']` boolean that the gate script reads. A
newly-exposed (previously unreachable) numpy-bool JSON-serialization bug in the same
script was also fixed. 5 new tests in `tests/test_era5_fusion.py`.

With both bugs fixed, **fusion was actually run against the real downloaded data**
(not simulated): it now completes real QC and correctly identifies that **23 of 24**
NUTS1 regions fuse successfully — see §9 for the genuine remaining blocker this
uncovered.

**ISS-020 — the "immutable" split was not actually UID-stable.**
`scripts/build_expenditure_m0.py`'s `deterministic_splits` used
`np.random.default_rng(seed).permutation(len(g))` — a *positional* permutation over
each country group. Empirically verified: dropping one household from a 1000-row
country group and re-running the identical seeded permutation reassigns 54% of the
*remaining* households to different partitions. Every leakage-prevention guarantee in
the entire pipeline (OOF calibration, common cohorts, the locked-test mechanism)
assumes the split is genuinely immutable at the UID level — this bug meant any future
upstream data fix would have silently invalidated all of it. Fixed with a per-UID
SHA-256 hash (order/membership-independent) and a `build_split_registry()` wrapper
that **refuses to silently regenerate an existing frozen split** (raises
`SPLIT_REGISTRY_IMMUTABLE` if the current cohort's UIDs aren't already a subset of the
frozen file). Verified the already-frozen `household_split_v1.json` is **byte-identical
before and after** this fix (md5 unchanged, 2884/961/962 train/val/test) — no household
was moved; this is a forward-looking fix plus a stability guard, not a data change.
7 new tests across `tests/test_split_registry.py`.

## 7. Statistical/leakage issues fixed

**ISS-006 — train-row leakage-reintroduction path.** `scripts/run_model_measurements.py`
always computes in-sample train predictions and in-sample residual-quantile threshold
offsets (correct for validation/test, but leaky for train); `build_h_oof_predictions.py`
corrects train rows afterward with proper 5-fold OOF values. Nothing stopped a plain
rerun of `run_model_measurements.py` from silently reverting the OOF-corrected train
rows back to the leaky in-sample state. Added `preserve_oof_train_rows()`: if an
existing artifact's train rows already carry a complete `oof_fold` marker, they are
kept verbatim and only validation/test rows are refreshed. 4 new tests in
`tests/test_no_threshold_leakage.py`. (Current on-disk artifacts were already
leakage-free — this closes the *regression* path, not an active bug in today's data.)

**ISS-004 — no automated leakage/governance tests for screening.** Added tests
asserting the screening feature whitelist excludes every target-derived/income column,
the 132-run governance table has exactly the expected model/imbalance/measurement
cardinality, and no stale duplicate-naming artifacts can silently reappear.

## 8. Data-processing issues fixed

- ISS-025: `9999` (a real "Refusal" sentinel for variable `S4`) added to both missing-
  code sentinel sets for defensive completeness (currently inert — no consumed field
  needs it, verified against the actual ENABLE dictionary sheets).
- ISS-026: the fuel-price diagnostic table's H6A3 row now carries an explicit caveat
  that it reflects pure-natural-gas price availability only, not resolved composite
  coverage (the pricing *engine* itself was already verified correctly blocking H6A3).
- ISS-008: fixed a mutable-list-aliasing bug (`features=cfg['features']` →
  `list(cfg['features'])`) that would have silently leaked climate features into H-only
  runs under a future refactor.
- A pre-existing import-ordering bug in `scripts/run_screening.py` (the `sys.path`
  fix ran *after* a local-package import, so `python scripts/run_screening.py` failed
  outright) was fixed — discovered while regenerating the hierarchical artifacts.

## 9. ERA5/NUTS issues fixed

Both blocking code bugs (§6) fixed and fusion actually run. Result: **23/24 NUTS1
regions fuse successfully with real, non-fabricated ERA5-Land data** (verified QC:
correct Kelvin→Celsius conversion, no duplicate region/date rows, `mean ≥ minimum`,
degree-days ≥ 0, correct 5-country NUTS1 prefixes). The missing region is **ES7
(Canarias)** — its bounding box (≈27.6–29.4°N, −18.2 to −13.4°E) falls entirely south
of the original CDS request's area `[72, -25, 34, 45]`. This is a genuine data
coverage gap, not a code defect, and it is consequential: ES7 has **44 real ENABLE
households** in the retained cohort. Per the spec's rule against fabricating ERA5
values and against submitting new external-service requests without authorization,
**no data was invented and no new CDS request was submitted** — this is flagged as the
primary remaining blocker (§21). `artifacts/climate/era5_land_annual_2017.active_request.json`
and `artifacts/climate/nuts1_annual_2017.qc.json` now accurately reflect this state.

## 10. M0 findings

Verified correct on direct code inspection and against `outputs/tables/table_m0_thresholds.csv`
/ `table_m0_class_distribution.csv`: country-specific (not pooled-Europe) Q80 on
absolute EUR expenditure (not a ratio), `low_income = income_bracket ≤ 3`, and the
exact four-class ontology (0 No Risk / 1 Income Risk / 2 Expenditure Risk / 3 Double
Risk) via `(high_cost, low_income)`. Train-only and train+validation-only threshold
scopes correctly used for the predictive-evaluation partitions. No changes required.

## 11. M2–M8 H findings

All 21 H experiments present and correct. M2 (linear quantile regression) and M7
(gradient-boosted quantile regression) are genuine native quantile models; M3
(Gamma GLM), M4 (Tweedie), M5 (approximate spline-basis GAM), M6 (gradient-boosted
mean regression) correctly use a per-country residual-quantile offset added to a
shared point prediction. M8 was mislabeled "Gamma/Lognormal" when only a Lognormal
mixed-effects model is implemented (ISS-007, fixed by correcting the label — no Gamma
GLMM was fabricated). H/HC feature parity confirmed (HC = H + climate features only).

## 12. HC status/findings

Still `BLOCKED_ERA5_FUSION` — correctly reported (no HC artifacts fabricated or
prematurely marked available). Blocked on the ES7 climate gap described in §9, not on
any remaining code defect.

## 13. M1 required-cost status

`IN_PROGRESS`, genuinely implemented (not a stub): BREDEM 2012 lighting/appliance/
cooking/hot-water/space-heating equations are real and running in production via
`scripts/build_m1_required_cost.py`; price-coverage bucketing (COMPLETE/HIGH/PARTIAL/
INSUFFICIENT) is correct and only COMPLETE-coverage households enter the headline
required-cost figure. `M1_EXACT` remains `BLOCKED_BY_DATA` and `M1_ADAPTED` remains
`BLOCKED_AHC_HOUSING_COST` — confirmed no income-bracket proxy is used anywhere to
bypass either block. The main gap found was **documentation, not science**: the
engineering source-audit CSV falsely claimed the BREDEM engine was still blocked/
pending (ISS-009, now corrected), and two adapted coefficients (heating-system
efficiency ranges, the 0.85 insulation multiplier) were undocumented in the project's
own governance trail (ISS-010, now flagged honestly as `SENSITIVITY_ASSUMPTION` — no
fabricated citation was added).

## 14. Fuel-price coverage

Confirmed correct: no silent zero-price fill anywhere (`PRICE_UNAVAILABLE` raised
instead); H6A3 (gas/LPG composite) is never priced as pure natural gas — it is
unconditionally blocked as `UNSUPPORTED_HEATING_FUEL` before pricing, verified by
direct execution and by `tests/test_bredem2012.py`; prices are dated to 2017/2018, not
current-day. One diagnostic-table mislabeling fixed (ISS-026).

## 15. Screening-model findings

S1 (hierarchical CatBoost) was fundamentally broken and is now fixed (§6). S2 (plain
CatBoost) and its imbalance/feature handling were correct. S3 (ARM-Net) architecture
was verified genuine (field embeddings, adaptive attention gate, exponential
interaction neurons, disclosed alpha=1 simplification) but essentially untested;
added tests that would catch a future silent substitution with a generic MLP
(ISS-014). The 132 governed screening runs were regenerated with the corrected
hierarchical implementation; 132 stale/orphaned pre-fix artifacts (from an earlier
naming convention) were archived, not deleted (ISS-005).

## 16. Imbalance/tuning findings

Exactly 2 imbalance strategies are implemented (`cost_sensitive` → class weighting,
`balanced_bootstrap` → oversampling), both strictly train-only, no SMOTE — confirmed
correct. True hyperparameter tuning (Optuna or similar) is **not implemented**
anywhere (`tuning/` is an empty stub); what exists is validation-only model/imbalance-
strategy *selection* with the correct Macro-F1 → Double-Risk-F1 → Double-Risk-recall →
balanced-accuracy → log-loss tie-breaker hierarchy, which is genuinely
validation-only and matches spec. This gap is documented, not fixed, given the scope
of implementing a full tuning framework under time constraints.

## 17. Common-cohort findings

`method_native_cohort` and `common_comparison_cohort` are genuinely separate code
paths (not conflated); currently identical in value (all M0/M2–M8 families share the
same base eligibility criterion) so the intersection logic is correct but not yet
exercised on a case where it matters. Cohorts are versioned by pipeline stage
(`v1_m0_h` → `v2_m0_h_hc`), not silently overwritten across versions, though a same-
stage rerun would still overwrite in place (documented, not fixed — MEDIUM, low risk
given no consumer currently re-reads the versioned files).

## 18. Reproducibility/cache findings

Hard-coded absolute paths: none found anywhere in source. All hashing found in the
pipeline is a provenance record, never a staleness-invalidation precondition — this is
a real architectural gap (ISS-023) documented as a recommended future direction rather
than redesigned under time pressure. `requirement.txt` dependencies were incomplete
(missing torch, xarray, geopandas, cdsapi, shapely, pyproj, netCDF4) and are now
fixed (ISS-013/ISS-018).

## 19. Figures/tables status

No figure/table generation logic was found to read from stale cached results beyond
the specific stale artifacts already identified and fixed above (screening
consolidated tables, fuel-price coverage table). `outputs/figures/` and
`outputs/tables/` remain the sole publication output locations, consistent with spec.

## 20. Tests after repair

`python -m pytest tests/ -q` → **62 passed**, 0 failed (same pre-existing statsmodels
warnings only). 23 new tests added across 5 new files
(`test_split_registry.py` additions, `test_era5_fusion.py`, `test_hierarchical_catboost.py`,
`test_no_threshold_leakage.py`, `test_governance.py`/`test_armnet.py` additions).

## 21. Remaining genuine blockers

1. **ERA5 Canary Islands (ES7) coverage gap** — 44 households cannot get HC climate
   features until a supplementary CDS request with an expanded/second bounding box
   (covering ≈27–30°N, −19 to −13°E) is submitted and fused. This was **not**
   submitted by Claude (external, quota-consuming, not a duplicate but a genuinely new
   request — flagged per the "do not submit without confirming" operating rule).
2. **M1_EXACT / M1_ADAPTED remain scientifically blocked** by the absence of
   continuous AHC income and defensible household housing costs in ENABLE — correctly
   left blocked, not worked around.
3. **True hyperparameter tuning (Optuna)** is not implemented; only model/imbalance
   selection exists. Implementing a full tuning framework was out of scope for a
   bug-fix pass.
4. **No content-hash-based staleness detection** exists for the pipeline's artifacts
   (ISS-023) — a real architectural gap, documented rather than redesigned.
5. **Nearly the entire codebase is untracked in git** (ISS-024) — recommend the user
   commit the working tree as the first next step so future audits have a real change
   history.

## 22. Recommended next executable command

```bash
python -m pytest tests/ -q   # confirm 62/62 still passing after review
git add -A && git commit -m "..."   # only if you want this audit's fixes committed
```

Then, to make further pipeline progress: either (a) submit a supplementary ERA5-Land
CDS request covering the Canary Islands bounding box and re-run
`python scripts/fuse_era5_nuts1.py --configuration annual_2017`, or (b) if a
Canarias-inclusive climate request is not feasible, make an explicit, documented
decision (not a silent one) about whether HC proceeds on 23/24 regions as a named
sensitivity while ES7 households remain H-only.

## 23. Post-audit addendum (2026-08-12)

Two follow-up items requested after the main audit:

- **ISS-028 (MEDIUM, FIXED):** `scripts/build_progressive_outputs.py` used a glob
  pattern that also matched the newly-regenerated per-architecture screening
  prediction files (not just the 22 core measurement parquets), crashing with a
  `KeyError` once those files existed. Fixed with an explicit stem list; see
  `outputs/tables/audit_scientific_issues.csv`.
- **Figure output format:** added `reporting/figures.py:save_figure()`, now used by
  every figure-producing script, saving each figure as both a 600dpi PNG and a vector
  PDF (`outputs/figures/<name>.png` + `.pdf`). 3 new tests in `tests/test_figures.py`.
  Tests: 62 → **65 passing**. See `docs/running_the_pipeline.md` for the full CLI
  command reference.


## 24. Post-audit addendum (2026-09-19): identification-accuracy audit

Two label/data defects and one design flaw explain most of the weak per-country F1 (details and
numbers in README "Identification-accuracy audit (2026-09-19)"):

- **`lib/labels/income_harmonization.py` v1**: cutoff N = max(bracket) over the rows passed in, so it
  changed by partition (Spain bracket 3: low-income in validation/test, not in train; train share 5.0% vs 17.7% test,
  1 Double-Risk household). Equal-mass-bracket assumption also gave Hungary 72% / Spain 10% low-income.
  Fixed with a frozen survey-weighted empirical table; regression tests in
  `tests/test_income_label_stability.py`. Unmapped (country, bracket) now raises instead of silently
  becoming "not low income".
- **H2 == 99 / H1 == 9 sentinels** passed to models as numbers (`MISSING_CODES` in `data/enable.py`
  was defined but never applied). Fixed in `engineer()`.
- **Conditional-threshold labels are independent of the features that define them** (pooled AUC 0.51),
  capping any classifier at chance on the expenditure axis. Identification labels now condition the
  threshold on physical need only.

Additional findings that limit what may be claimed: heating-cost non-response (21-44% in
Italy/Poland/Spain) is unrelated to income, so it costs sample size but does not bias the income mix;
the recorded-cost axis relative to need is **not** externally validated (hardship 0.65-0.72 among
low-income vs 0.24-0.27 otherwise, no difference by cost position), while the income axis is.
The test partition was not read by any of this work; `test_unlock_authorized` is unchanged.


## 25. Cleanup and full rerun (2026-09-19)

Scope: remove Spain from the analysis, remove dead code, rerun the whole pipeline from a cleared state, refresh the
documentation. The prior state is preserved at git tag `pre-cleanup-20260919` (almost all code was untracked before it).

**Spain.** Removed from the cohort by one constant (`data/enable.py: RETAINED_COUNTRY_CODES`, now four countries; 4,047
households, 17 NUTS1 regions). Evidence and gates in `docs/country_screening.md`. Consequences handled in code: a shared
`RETAINED_COUNTRY_NAMES` replaced eight hardcoded country lists; the ERA5 QC gates no longer hardcode 24 regions
(`fuse_era5_nuts1.py` coverage and daily-row checks, `check_milestone_gates.py`, and the mirrored gate test); the Canary
Islands download (Spain only) was dropped; Spain entries removed from the Eurostat/oil-bulletin/currency country maps.

**Dead code removed.** `data/cohorts.py` (only its own unit test used it), `scripts/make_lihc_requirements_xlsx.py` (read a CSV
no script writes; `audit_enable_workbook.py` already publishes the xlsx), two obsolete tests, ARM-Net citation (model removed
earlier), unused imports/variables/constants found with pyflakes (12 files). `scripts/run_screening.py` looked unwired but is
the engine `run_screening_matrix.py` calls via subprocess, so it stays.

**Defects a clean run exposed (all fixed):**
1. `build_income_bracket_reference.py` read a parquet that `build_expenditure_m0.py` creates later -> reordered.
2. `compare_income_ablation.py` read `table_family_champions.csv` before `select_family_champions.py` wrote it -> reordered.
3. `build_expenditure_m0.py` assumed `artifacts/data/` existed -> creates it; the pipeline script now creates every output directory.
4. `fuse_era5_nuts1.py` daily-row check hardcoded `days * 24 regions` -> `days * len(cohort regions)`.
5. `build_threshold_scatter_income_states.py` used `champ.quantile` on a Series (returns the method, not the column; the pandas
   shadowing bug noted in earlier sessions) -> bracket access.
6. My own cleanup script pasted `build_nightingale_metrics.py` twice (an index of `None` turned a slice into the whole file);
   pyflakes reported the duplicated `from __future__` line and the file was restored from the tag and edited correctly.
`run_full_pipeline.sh` gained per-step banners with timestamps and a `START_AT=scripts/<name>.py` resume option.

**What ran.** The pipeline was executed in three segments (first to A1 measurements, then to the policy maps, then to the end)
because of defects 2 and 5; every stage ran with its final code and the last stage is the test suite: 67 passed, 0 skipped.
The stale generated artifacts (old predictions, manifests, frozen selections, ~100 MB) were moved out of the repository before the
run so nothing from the previous state could leak into the new outputs; the immutable household split, ERA5 NetCDF cache and
reference data were kept. `final_selection_frozen.json` shows `choices_frozen: true`, `test_unlock_authorized: false`.

**Changed by the rerun.** The recursive predictor search now picks `has_solid_fuel_heating` instead of `ac_age` as the seventh A0
predictor, so every A0-based number moved slightly; all figures in the README results section come from the regenerated tables.

**Still limiting a claim.** No uninterrupted single-shot run of the final script was made (segments, above). The cost axis is not
validated against self-reported hardship. Dynamic income rules use an EUR proxy from an ordinal bracket. The test partition has not
been used and is not unlocked.
