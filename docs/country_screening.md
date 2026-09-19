# Country screening: why Spain is not in the analysis

Rule (decided 2026-09-19): a country that does not carry every stage of the analysis is removed from the analysis.
`data/enable.py: RETAINED_COUNTRY_CODES` is therefore Bulgaria, Hungary, Italy and Poland. Spain was screened with the
five-country pipeline at git tag `pre-cleanup-20260919` (`git checkout pre-cleanup-20260919` reproduces these numbers).

Development households (train + validation) with a valid cost and income bracket. Income axis = bracket <= 3 (Van Hove's
fixed rule), cost axis = need-based threshold, tier T2, income excluded; macro-F1 from repeated 5-fold CV (best of the
logistic models) with its bootstrap lower bound, against the chance floor (classes drawn at random from the training prior).

| Country | Households used | No Risk | Income Risk | Expenditure Risk | Double Risk | Gate 1: smallest class >= 30 | Gate 2: macro-F1 lower CI vs chance | Gate 3: factor cells estimable |
|---|---|---|---|---|---|---|---|---|
| Bulgaria | 584 | 224 | 145 | 144 | 71 | pass | 0.413 vs 0.254 | 8/8 |
| Hungary | 646 | 103 | 306 | 78 | 159 | pass | 0.343 vs 0.234 | 8/8 |
| Italy | 382 | 124 | 110 | 99 | 49 | pass | 0.392 vs 0.261 | 8/8 |
| Poland | 384 | 36 | 205 | 31 | 112 | pass | 0.336 vs 0.253 | 8/8 |
| Spain | 282 | 135 | 43 | 83 | 21 | **FAIL (21)** | 0.283 vs 0.263 (marginal) | 6/8 **FAIL** |

* Gate 1 -- every one of the four classes needs at least 30 households for a per-country four-class result to mean anything.
* Gate 2 -- the lower 95% bootstrap bound of macro-F1 must clear the chance floor by a visible margin (> 0.05).
* Gate 3 -- every factor comparison (4 factors x 2 axes) needs at least 15 households in both the exposed and the unexposed arm.

Spain has too few Double-Risk households for a four-class result, too few solid-fuel households (4) for the fuel comparison,
and only a marginal lead over chance. The other four countries pass all three gates. `lib/identification/labels.py:viability`
keeps applying gate 1 to whatever cohort it is given, so a future country change fails loudly instead of silently.
