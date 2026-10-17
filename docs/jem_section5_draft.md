# Draft: restructured Section 5 (Discussion) and Section 6 monitoring set — JEM manuscript v5 → v6

Numbers come from `outputs/tables/` (scripts named in brackets). New analyses since v5:
`scripts/build_climate_measure.py` → `table_weather_drift.csv`, `table_label_climate_regions.csv`,
figures `weather_drift`, `label_reclassification`.

---

## 5. Discussion

### 5.1 Climate shapes the bill

Within countries, a 10% colder winter is associated with a heating bill about 12% higher (elasticity 1.26), and climate
explains about a fifth of the between-region variance of heating bills; regional economy explains 25%, both together 29%,
and climate remains significant next to the economy. The estimate agrees with the established dependence of space-heating
expenditure on degree days (Meier and Rehdanz, 2010) and with within-country evidence on temperature and household energy
expenditure (Mashhoodi, 2020). Because it is identified from regions of the same country, national prices, fuel mix and
welfare systems cannot drive it. It is, however, a cross-sectional elasticity: households in colder regions have adapted
their dwellings and heating, as the lower frequency of cold homes in colder regions suggests, so the response of bills to
*future* warming within a region may be smaller than the contrast between regions implies.

### 5.2 Climate shapes the indicator

Climate also changes who is counted. Adding degree days to the need line changes the group of 6% of households and the
Double-Risk status of 71 households, 15% of the group (Table 5, Fig. X1a). Where climate varies within a country the
reclassification follows it: in Italy, Spain and Hungary, regions with colder winters than their country lose households
from high expenditure once the line rises with degree days (household-weighted correlation between relative log HDD and
net movement into high expenditure −0.68, −0.35 and −0.83; Fig. X1b), whereas in Bulgaria and Poland the movements do not
follow climate and mainly reflect the different fit of the two quantile regressions.

How stable an indicator is under changing winters depends on how its line is set (Fig. X2). When bills are rescaled to a
uniformly 10% milder or colder winter with the estimated elasticity, a spending line fixed in euros moves 3–4% of
households between groups, and so does the climate-free need line, whose threshold cannot respond (6%). A spending
percentile recomputed each year is almost immune to a uniform change (0.4–0.9%) but not to its regional pattern: removing
the regional pattern of the 2016/17 winter moves 1.5% of households overall and 4% in Italy. The need line with degree days
behaves best where its degree-day term is well identified—0.3–2.9% in Italy and Bulgaria across the three scenarios, and
3–6% in Spain, which has a wide climate range but only ten sampled regions—but overreacts in Hungary and Poland (21% and
13% of households change group under a milder winter), where regional log HDD has
a standard deviation of only 0.04 and the degree-day coefficient of a country-specific quantile regression is extrapolated
far beyond the range it was estimated on.

Two lessons follow for indicator design. First, climate belongs in the definition of energy need because, without it, the
high-cost condition partly measures the weather of the survey year and the region. Second, the degree-day term should not
be estimated afresh in each small national sample: it should be anchored in a pooled or physically based elasticity
(for example, the multilevel elasticity estimated here, or degree-day scaling of modelled heating need) so that the line
moves with the climate by the amount bills actually do. Validity also depends on the income side: under a country-relative
low-income cut-off the hardship gap of the P80 line triples and exceeds that of the need lines (Table 1), so the validity
of an expenditure line is a property of the line together with the definition of low income, a point the literature on
indicator divergence has not made explicit (Thomson and Snell, 2013; Bouzarovski et al., 2021).

### 5.3 Climate does not improve screening — and why

Once the region is known, climate adds no detectable information about who is LIHC: real climate beats its placebo by
+0.001 balanced accuracy, and 4 significant gains in 72 comparisons are what chance produces. Three mechanisms explain why a
variable can matter for the bill but not for the label: the label is relative to a line that, for need-based definitions,
already rises with degree days; climate has one value per region, so a region identifier already carries any
region-constant effect; and within-country climate contrasts are narrow in three of the five countries. The power
simulation makes this a statement about the design rather than about climate: with the region known, even an added
elasticity of 2 is detected in only 2 of 9 cases, whereas in regions held out of training an added elasticity of 1 is
detected in 6 of 9. The real total-bill effect (about 0.5) is roughly half of what this sample can detect.

### 5.4 Monitoring energy poverty in a changing climate

The outcomes that climate does predict beyond its placebo—summer energy poverty and hidden under-heating—are those that
LIHC labels cannot see. Summer discomfort does not appear in heating bills: 44% of Hungarian and 34% of Spanish households
report a usual summer indoor temperature of 24 °C or more, and 13–30% of Hungarian households combine a hot home with low
income, depending on how low income is defined. Bills already rise with cooling degree days (+6.4% per 100 CDD with the
extra countries). As heatwaves become more frequent, low-income households may respond with air conditioning that raises
both bills and emissions, a form of maladaptation (Barnett and O'Neill, 2010) that passive cooling, shading and ventilation
retrofits avoid. Hidden under-heaters—6–20% of low-income households—spend far below their modelled heating need; for them
renovation buys comfort rather than energy savings (the prebound effect; Sunikka-Blank and Galvin, 2012), so programme
estimates of carbon savings that include them are too high. Cold homes are more frequent in mild Spain than in cold Hungary
and, within countries, less frequent in colder regions, so renovation programmes that concentrate on the coldest regions
can miss the households in greatest thermal discomfort.

These findings carry a distributive dimension. Which households receive compensation from carbon pricing of heating
fuels, and which regions receive renovation funds, depends on the indicator—its climate term, its income definition and its
blind spots for under-heating and summer heat. Energy-justice scholarship has argued that the distribution of energy
burdens is spatially produced and must be measured as such (Jenkins et al., 2016; Bouzarovski and Simcock, 2017); our
results show concretely how much the measure itself moves with these choices.

### 5.5 Lessons for data-driven identification of energy poverty

Three lessons carry beyond climate. (i) Any region-level predictor—climate, prices, regional economy—must be tested
against a placebo that keeps region identity, scored in held-out regions, and accompanied by a power simulation; without
the placebo, the apparent +0.02 to +0.04 gain from climate in XGBoost would have been reported as a climate effect.
(ii) Predictability and validity must be assessed together: lines that are easy to predict can be easy because they track
home size. (iii) Models do not transfer across countries on the expenditure axis, so each country needs its own calibration
sample; the predictors transfer, the line does not.

*(Moves to the appendix or is cut: the detail of 5.3 "What information identifies" and 5.4 "Small samples, pooling and
portability" of v5, except the sentences used in (iii).)*

---

## 6.x A minimum climate-sensitive energy-poverty monitoring set (new table; replaces part of Table 7)

| Indicator | Definition | Data | Update | Validity check to report |
|---|---|---|---|---|
| Need-based LIHC (Double Risk) | Low income and spending above a need line conditioned on dwelling, household and degree days | Survey or registers (income, floor area, dwelling, household size) + reanalysis HDD/CDD | Annual | Hardship gap under a common and a country-relative income cut-off; degree-day term anchored in a pooled or physical elasticity, not re-estimated in small samples |
| Summer energy poverty | Usual summer indoor temperature ≥ 24 °C and low income | Indoor-temperature survey question; regional CDD, heatwave days, tropical nights | Annual (after summer) | Report under both income definitions; check against regional heat indicators |
| Hidden under-heating | Spending per unit of modelled heating need below half the median of country × heating fuel | Bills + modelled need (floor area × heat-loss coefficient × HDD) | Annual | Share of low-income households; cross-check with reported cold homes; exclude free-fuel households where known |
| Regional winter and summer anomaly | Region's HDD and CDD relative to its 1991–2020 normal | Reanalysis services (C3S ERA5/ERA5-Land, Open-Meteo) | Monthly / seasonal, within weeks | Used to scale emergency support (≈ +12% heating bills per 10% colder winter) and to flag indicator drift |
| Reclassification audit | Share of households whose group changes when the climate term or the income cut-off changes | The indicator's own inputs | With every methodological change | Report alongside the headline rate (here: 6% of households, 15% of Double Risk) |

---

## Figures

- **Fig. X1 — `outputs/figures/label_reclassification`** (new 4.4.3; Table 5 can move to the appendix).
  Caption: *Households reclassified when degree days enter the need line. (a) Groups under the need line without degree
  days (rows) and with degree days (columns); off-diagonal cells are reclassified households (141 of 2,278, 6.2%).
  (b) Net movement into high expenditure (households entering minus leaving, % of the region's households) against the
  region's log HDD relative to its country, Italian and Spanish regions with at least 10 labelled households; marker size
  proportional to households; lines: household-weighted least squares; r: household-weighted correlation.*
- **Fig. X2 — `outputs/figures/weather_drift`** (new, Section 5.2 or 4.4.3).
  Caption: *Indicator drift under changing winters. Heating bills are rescaled with the within-country heating-bill
  elasticity (1.26) to (a) a uniformly 10% milder winter, (b) a 10% colder winter and (c) the 2016/17 winter without its
  regional pattern (each region's anomaly against its 1991–2020 normal, relative to the country mean), and households are
  relabelled under four rules: the 2016/17 P80 threshold kept in euros, the P80 threshold recomputed, the climate-free need
  line and the need line with degree days (fitted coefficients kept, degree days updated). Bars: share of households whose
  group changes, five countries; markers: single countries.*

### Methods text for the drift analysis (add to 3.6 as test (f), or to 4.4)

Heating bills are rescaled as heating′ = heating × (HDD′/HDD)^β with β = 1.26, the within-country heating-bill
elasticity (Table 4); electricity spending and the income axis are unchanged. Normals are used only for their regional
pattern within a country, because they are computed from monthly means of a different reanalysis product and their level is
not comparable with the survey-year degree days. The need lines keep their fitted coefficients (the same five out-of-fold
fits as in the main analysis), and only the degree-day input changes. The simulation is illustrative: it applies a
cross-sectional elasticity to a change over time, and assumes no behavioural response.

### Limitations to add

The drift simulation transfers a cross-sectional elasticity to changes over time; adaptation and behaviour may make the
real response smaller. The 2016/17 anomalies use normals computed from monthly means, so only their within-country pattern
is used.

### References to add (verify before use)

- Jenkins, K., McCauley, D., Heffron, R., Stephan, H., Rehner, R., 2016. Energy justice: A conceptual review. Energy
  Research & Social Science 11, 174–182. doi:10.1016/j.erss.2015.10.004
- Bouzarovski, S., Simcock, N., 2017. Spatializing energy justice. Energy Policy 107, 640–648.
  doi:10.1016/j.enpol.2017.03.064
