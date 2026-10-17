# Draft: four-group identification — figures, Results 4.2 and Discussion 5.3 (manuscript v6 → v7)

Figures: `scripts/build_four_group_figures.py` → `outputs/figures/four_groups_space`, `four_groups_prediction`,
`four_groups_confusion`; counts in `outputs/tables/table_four_groups_confusion.csv`. Out-of-fold Ensemble+ probabilities
from `artifacts/identification/main_results.pkl`; balanced accuracies reproduce Table 2 (0.683 / 0.596 / 0.625 / 0.685).

---

## Where they go

| New figure | Place | Replaces / moves |
|---|---|---|
| `four_groups_space` | Section 4.1 or 3.2 (after Eq. 2): what the four groups look like | — |
| `four_groups_confusion` | Section 4.2 | Table A.8 can stay in the appendix |
| `four_groups_prediction` | Section 4.2 | Fig. B.4 stays in the appendix |

To keep the main text length, move Fig. 9 (power) to the appendix and keep Table A.3 summarising it in 4.7, or move
Fig. 8 (comfort outcomes; Table A.10 already summarises it).

---

## 3.2 (one sentence after Eq. 2)

Fig. X1 shows the four groups for every labelled household: income bracket against annual expenditure relative to the
household's line, for the P80 baseline and the need line.

**Caption, Fig. X1 — `four_groups_space`.** *The four LIHC groups. Each labelled household by income bracket (jittered;
vertical line: low-income cut-off, bracket ≤ 3) and annual energy expenditure relative to its expenditure line (log scale;
horizontal line: expenditure = line), for the P80 baseline (top) and the need line (bottom), by country. No Risk (grey),
Income Risk (yellow), Expenditure Risk (blue), Double Risk (red). Under the need line the line follows each household's
dwelling, household size and degree days, so high spending in large or old dwellings is no longer counted as high cost.*

## 4.2 Identifying the four groups (replaces the current 4.2)

Income is indispensable: income alone reaches a balanced accuracy of 0.50–0.59, and without income scores fall to
0.39–0.54 (Table 2; all models in Table A.5). The core profile adds 0.07–0.13 on the P80 and burden lines but little on
the need lines, which already condition on home size. The best recipe, Ensemble+ multilevel, reaches 0.62–0.71, with
Double-Risk recall of 0.63–0.71. Averaging models is the most reliable gain, and the bill model shifts the operating point
towards the spending side (need-line Double-Risk recall 0.52 to 0.74).

The confusion matrices show where the errors lie (Fig. X2). Because the income bracket is a predictor and the low-income
condition is a bracket cut-off, the models never confuse the income side: across all four lines and three repetitions, no
household is predicted on the wrong side of the income cut-off (0 of 27,336 predictions). All errors are on the expenditure
axis—No Risk against Expenditure Risk, and Income Risk against Double Risk. Income alone therefore fixes half of the
problem, which is why it already reaches a balanced accuracy of about 0.5, and everything above 0.5 measures how well the
models recognise high spending. On the need line, 42% of No-Risk households are predicted as Expenditure Risk and 46% of
Income-Risk households as Double Risk; Double Risk itself is recognised in 65% of cases (72–77% on the P80 and burden
lines).

Fig. X3 shows why. The predicted probability of high expenditure separates Expenditure Risk from No Risk in Bulgaria,
Italy and Spain (medians 0.56–0.57 against 0.45–0.48), and Double Risk from Income Risk in Poland (0.59 against 0.47). In
Hungary, the hardest country (balanced accuracy 0.53), the four groups have almost the same distribution around 0.5: the
predictors carry little information about which Hungarian households spend more than their line. Among low-income
households in Italy, both groups lie mostly below 0.5, so Double Risk is under-predicted there (recall 0.31, Table A.8).
Beyond the core profile, the region carries most of the remaining information … *(continue with the current text from
"it is selected in every training round" to the end of the paragraph.)*

**Caption, Fig. X2 — `four_groups_confusion`.** *Confusion matrices of Ensemble+ for the four lines (all five countries;
out-of-fold predictions summed over three repetitions of five-fold cross-validation). Cells: share of each true group
predicted as each group; the diagonal is the per-group recall. No household is predicted on the wrong side of the income
cut-off: all errors lie on the expenditure axis.*

**Caption, Fig. X3 — `four_groups_prediction`.** *Separation on the expenditure axis (need line, Ensemble+). Predicted
probability of high expenditure (Expenditure Risk + Double Risk; out of fold, averaged over three repetitions) by true
group, for households above (left pair) and below (right pair) the income cut-off, by country. Bars: median; lines:
interquartile range; dashed: 0.5. BA: country balanced accuracy.*

## 5.3 What machine learning can and cannot identify (new; before "Climate does not improve screening")

Machine learning identifies the energy poor far better than income alone—the best ensembles reach 0.62–0.71 balanced
accuracy, and helping the top 30–35% of households reaches 70% of the Double-Risk group against 40% with income alone—but
the gain comes entirely from the expenditure side. The income side is observed; what the models must infer is whether a
household spends more than its line, from its dwelling, household and region. That is where the four-group classification
fails: in Hungary the predictors hardly separate high from normal spenders, and everywhere about two in five low-income
households without high spending are flagged as Double Risk.

Two consequences follow. First, a more valid line is harder to predict. The P80 line is easy to predict partly because high
spending relative to a country-wide threshold tracks floor area and household size, which the core profile contains; the
need line removes exactly that information, and its balanced accuracy falls from 0.71 to 0.62. Predictability is therefore
not a criterion for choosing an EP definition on its own, and studies that report only predictive accuracy (van Hove et al.,
2022; Dalla Longa et al., 2021) favour lines that partly measure dwelling size. Our results for the P80 line reproduce van
Hove et al.'s per-group recall (Table A.4), so the difference in levels comes from the line, not from the pipeline. Second,
data that describe spending need directly—metered consumption, energy performance certificates, heating-system data—are
where better identification will come from, not more socio-economic or attitudinal variables, which were almost never
selected. For managers, this means that a model can reliably rank who is low-income and in a high-need dwelling, but the
spending condition should be confirmed with bills before support is withdrawn or granted—exactly the role of the "not
sure" follow-up stage in Fig. 10.

## Introduction: contribution list (add as the first item)

(i) It identifies the four LIHC groups with machine-learning ensembles in five countries, shows that all errors lie on the
expenditure axis and that the most valid line is the hardest to predict, and turns the models into a targeting protocol.
*(Renumber the others.)*
