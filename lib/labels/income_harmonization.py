"""Income-axis rules for the four-class energy-poverty label.

HEADLINE (decision 2026-09-19): Van Hove et al.'s fixed rule -- a household is low-income if its
ENABLE monthly income bracket is <= 3 (`rule="bracket3"`). It is a fixed integer cutoff, hence
identical in every partition/subset (the partition-dependence defect described below cannot occur),
and it keeps the project directly comparable with the paper it extends. Its known cost: brackets are
nominal-income bands, not equal-mass deciles, so the low-income share is 22% (Spain) to 82% (Poland).
SENSITIVITIES (reported next to the headline, never instead of it):
  rule="percentile"  frozen survey-weighted empirical bracket percentile <= 0.30 ("Fix v2" below)
  rule="equiv60"     DYNAMIC: household-size-adjusted income proxy below 60% of the country median
  rule="residual60"  DYNAMIC: (income proxy - energy cost) per equivalent adult below 60% of the
                     country median (the official LIHC residual-income idea)
The dynamic rules use an EUR proxy built from the ordinal bracket (see `income_proxy_frame`), so they
are sensitivity analyses only.

Earlier note -- cross-country-comparable low-income classification.

ENABLE's S9MONTH income-bracket question uses country-specific, undocumented-in-EUR
ordinal brackets (Poland has 10 distinct categories in the retained cohort; Bulgaria/
Hungary/Italy/Spain have 12 -- see table09_lihc_income_bracket_availability_audit.csv).
Applying a single raw ordinal cutoff (`income_bracket <= 3`) uniformly across countries
produced a "low income" share ranging from 22.2% (Spain) to 80.2% (Poland) on this
cohort -- a 4x spread that is a bracket-scale artifact, not a real income difference,
and specifically not usable for a cross-country comparison of LIHC prevalence or
factor effects (this project's stated purpose).

Fix v1 (2026-08): treat bracket k of N (N = that country's own number of distinct
bracket categories, read from the data) as the [(k-1)/N, k/N) population slice.

Why v1 was wrong (found 2026-09-19): (a) it assumes brackets are equal-mass, but they
are fixed nominal-income bands -- on this cohort 72% of Hungarian households fall in
brackets 1-3 versus 22% of Spanish ones, so "bottom 3 of 10" is not a common percentile;
(b) N was recomputed as `max(bracket)` over *whatever rows were passed in*, so the cutoff
moved with the partition: Spain's bracket 3 was low-income in validation/test but not in
train (train max = 8, validation max = 7 -> different N), leaving 1 Double-Risk
household in Spain's training set and a train/test label shift in Poland and Italy too.

Fix v2 (current): each bracket's percentile is its survey-weighted **empirical**
mid-cumulative share in that country's full retained cohort -- a frozen population
constant (`artifacts/reference/income_bracket_percentiles.csv`, built once by
`scripts/build_income_bracket_reference.py`) that is identical for every partition and
every subset of rows. It uses brackets/weights only (no cost or outcome information).
A household is low-income if its bracket's mid-percentile is at or below the boundary
(default 0.30). Bracket granularity still limits how close to 30% a country can get
(e.g. Spain ~22%, Italy ~42%); that residual is reported, not hidden.

Caveat, stated explicitly rather than silently assumed away: ENABLE's S9MONTH is a
*monthly*, non-equivalized household income bracket, while the Eurostat reference is
*annual*, equivalized (per-adult-equivalent) disposable income. The percentile-rank
mapping (what `low_income_flag` actually uses) is robust to this -- a monotonic
bracket-position-to-percentile assignment doesn't require unit agreement across the
two scales. The EUR proxy this module also produces (`harmonized_income_eur`)
inherits that unit mismatch and should be read as an ordinal proxy for relative
income standing, not a literal annual income figure.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUANTILES_PATH = ROOT / "artifacts/reference/income_quantiles_2018.csv"
_DECILE_FRACTIONS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])  # D1..D9


BRACKET_REFERENCE_PATH = ROOT / "artifacts/reference/income_bracket_percentiles.csv"
_BRACKET_MIN, _BRACKET_MAX = 1, 10  # valid ENABLE S9MONTH brackets; 98/99 are refusals / don't-know


def build_bracket_percentile_table(bracket: pd.Series, country: pd.Series,
                                   weight: pd.Series | None = None) -> pd.DataFrame:
    """Survey-weighted empirical mid-cumulative percentile of each income bracket, per country.

    Only rows with a valid bracket (1-10) contribute. `bracket`/`country`/`weight` must be
    aligned; `weight=None` means equal weights. Returns columns
    country, bracket, n, weighted_share, cum_share_below, mid_percentile.
    """
    b = pd.to_numeric(bracket, errors="coerce")
    w = pd.Series(1.0, index=b.index) if weight is None else pd.to_numeric(weight, errors="coerce").fillna(1.0)
    frame = pd.DataFrame({"country": country, "bracket": b, "w": w})
    frame = frame[frame.bracket.between(_BRACKET_MIN, _BRACKET_MAX)]
    rows = []
    for c, g in frame.groupby("country"):
        by = g.groupby("bracket").agg(n=("w", "size"), w=("w", "sum"))
        share = by.w / by.w.sum()
        cum_below = share.cumsum() - share
        for k in by.index:
            rows.append({"country": c, "bracket": int(k), "n": int(by.n[k]), "weighted_share": float(share[k]),
                         "cum_share_below": float(cum_below[k]), "mid_percentile": float(cum_below[k] + share[k] / 2)})
    return pd.DataFrame(rows)


_REFERENCE_CACHE: dict[str, pd.DataFrame] = {}


def load_bracket_reference(path: Path | str = BRACKET_REFERENCE_PATH) -> pd.DataFrame:
    key = str(path)
    if key not in _REFERENCE_CACHE:
        if not Path(path).exists():
            raise FileNotFoundError(f"{path} not found -- run scripts/build_income_bracket_reference.py first "
                                    "(it freezes the cohort-wide bracket->percentile table every partition shares)")
        _REFERENCE_CACHE[key] = pd.read_csv(path)
    return _REFERENCE_CACHE[key]


def harmonized_income_percentile(income_bracket: pd.Series, country: pd.Series, *,
                                 reference: pd.DataFrame | None = None) -> pd.Series:
    """Country-normalised income percentile in (0, 1): each household's bracket looked up in
    the frozen survey-weighted empirical table (see module docstring, "Fix v2"). Depends only
    on (country, bracket) -- never on which other rows are passed in. Brackets outside 1-10
    (refusals, NaN) -> NaN."""
    ref = load_bracket_reference() if reference is None else reference
    lookup = {(r.country, int(r.bracket)): r.mid_percentile for r in ref.itertuples()}
    bracket = pd.to_numeric(income_bracket, errors="coerce")
    valid = bracket.between(_BRACKET_MIN, _BRACKET_MAX)
    out = pd.Series([lookup.get((c, int(b)), np.nan) if ok else np.nan for c, b, ok in zip(country, bracket, valid)],
                    index=income_bracket.index, dtype=float)
    unmapped = valid & out.isna()
    if unmapped.any():  # never let an unknown country/bracket silently become "not low income"
        bad = sorted({(c, int(b)) for c, b in zip(country[unmapped], bracket[unmapped])})
        raise ValueError(f"(country, bracket) pairs missing from the frozen income-bracket table: {bad[:8]} -- "
                         "rebuild it with scripts/build_income_bracket_reference.py")
    return out


def legacy_subset_relative_percentile(income_bracket: pd.Series, country: pd.Series) -> pd.Series:
    """The v1 rule, kept ONLY so `scripts/build_income_bracket_reference.py` can demonstrate the
    partition-dependence defect. Do not use for labels: N = max bracket among the rows passed in."""
    bracket = pd.to_numeric(income_bracket, errors="coerce")
    n_brackets = bracket.groupby(country).transform("max")
    return (bracket - 0.5) / n_brackets


def _country_decile_table(quantiles_path: Path | str = DEFAULT_QUANTILES_PATH) -> dict[str, np.ndarray]:
    q = pd.read_csv(quantiles_path)
    q["rank"] = q["quantile"].str.removeprefix("D").astype(int)
    q = q.sort_values("rank")
    out = {}
    for country, g in q.groupby("country"):
        if not np.array_equal(g["rank"].to_numpy(), np.arange(1, 10)):
            raise ValueError(f"{country}: expected deciles D1..D9, got {sorted(g['rank'])}")
        out[country] = g["cutoff_eur"].to_numpy(dtype=float)
    return out


def harmonized_income_eur(percentile: pd.Series, country: pd.Series,
                           quantiles_path: Path | str = DEFAULT_QUANTILES_PATH) -> pd.Series:
    """Map a harmonized percentile onto each country's Eurostat D1-D9 EUR scale via linear
    interpolation (flat-extrapolated below D1 / above D9 -- see module docstring's unit-
    mismatch caveat: this is an ordinal proxy, not a literal annual income figure)."""
    deciles = _country_decile_table(quantiles_path)
    out = pd.Series(np.nan, index=percentile.index, dtype=float)
    for c, cutoffs in deciles.items():
        mask = (country == c).to_numpy()
        if mask.any():
            out.loc[mask] = np.interp(percentile.loc[mask], _DECILE_FRACTIONS, cutoffs)
    return out


HEADLINE_INCOME_RULE = "bracket3"
HEADLINE_BRACKET_CUTOFF = 3
INCOME_RULES = ("bracket3", "percentile", "equiv60", "residual60")
POVERTY_LINE_SHARE = 0.60   # EU at-risk-of-poverty convention: 60% of the national median


def equivalence_scale(adults: pd.Series, children: pd.Series) -> pd.Series:
    """OECD-modified equivalence scale: 1 for the first adult, 0.5 per further adult, 0.3 per child.
    NaN where the composition is missing. `adults` is floored at 1."""
    a = pd.to_numeric(adults, errors="coerce").clip(lower=1)
    c = pd.to_numeric(children, errors="coerce")
    return 1.0 + 0.5 * (a - 1.0) + 0.3 * c


def income_proxy_frame(income_bracket: pd.Series, country: pd.Series, scale: pd.Series, *,
                       cost: pd.Series | None = None, poverty_share: float = POVERTY_LINE_SHARE,
                       quantiles_path: Path | str = DEFAULT_QUANTILES_PATH) -> pd.DataFrame:
    """EUR income proxy and the two DYNAMIC low-income flags.

    proxy chain: bracket k -> national decile midpoint (k - 0.5) / 10 (Van Hove: respondents placed
    themselves in their own country's income deciles, so bracket k IS decile k and the sample skew is
    non-representative sampling) -> Eurostat national decile EUR (annual, equivalised,
    `harmonized_income_eur`) -> household total = proxy x that country's mean scale (a
    self-calibrating assumption that the bracket ranks TOTAL household income) -> equivalised again with
    the household's own scale. Households with a large scale therefore look poorer than the same
    bracket in a small household.

    The poverty line is `poverty_share` x the NATIONAL median equivalised income (Eurostat D5), NOT the
    sample median: these samples are skewed toward low income (Van Hove Fig. A2), so a sample-median line
    would sit too low and label almost nobody poor.

    Returns proxy_total_eur, proxy_equiv_eur, low_income_equiv60 and low_income_residual60 (energy cost
    subtracted from the household total before equivalising; NaN without `cost`). Unit mismatch and the
    ordinal source are why these are sensitivity analyses only.
    """
    b = pd.to_numeric(income_bracket, errors="coerce")
    pct = ((b - 0.5) / 10.0).where(b.between(_BRACKET_MIN, _BRACKET_MAX))
    eur = harmonized_income_eur(pct.dropna(), country.loc[pct.notna()], quantiles_path).reindex(pct.index)
    scale = pd.to_numeric(scale, errors="coerce")
    ok = eur.notna() & scale.notna()
    mean_scale = scale.where(ok).groupby(country).transform("mean")
    total = eur * mean_scale
    equiv = total / scale
    national_median = country.map({c: float(v[4]) for c, v in _country_decile_table(quantiles_path).items()})   # D5
    line = poverty_share * national_median
    out = pd.DataFrame({"proxy_total_eur": total, "proxy_equiv_eur": equiv}, index=pct.index)
    out["low_income_equiv60"] = (equiv < line).where(ok & line.notna())
    if cost is not None:
        c = pd.to_numeric(cost, errors="coerce")
        out["low_income_residual60"] = (((total - c) / scale) < line).where(ok & line.notna() & c.notna())
    else:
        out["low_income_residual60"] = np.nan
    return out


def low_income_flag(income_bracket: pd.Series, country: pd.Series, *, rule: str = HEADLINE_INCOME_RULE,
                    bracket_cutoff: int = HEADLINE_BRACKET_CUTOFF, percentile_boundary: float = 0.30,
                    reference: pd.DataFrame | None = None) -> pd.Series:
    """Low-income flag for the two BRACKET-ONLY rules (the dynamic rules need composition/cost -- use
    `income_proxy_frame`).

      rule="bracket3"    (headline) True iff the bracket is in 1..`bracket_cutoff` (default 3, Van Hove).
                         Depends on nothing but the household's own bracket. Refusals/NaN -> False;
                         callers gate on bracket availability (measurement_eligible).
      rule="percentile"  True iff the household's frozen weighted-empirical bracket percentile is at or
                         below `percentile_boundary` (default 0.30). Raises for an unmapped
                         (country, bracket) rather than silently returning "not low income".
    """
    if rule == "bracket3":
        b = pd.to_numeric(income_bracket, errors="coerce")
        return b.between(1, bracket_cutoff)
    if rule == "percentile":
        return harmonized_income_percentile(income_bracket, country, reference=reference) <= percentile_boundary
    raise ValueError(f"low_income_flag handles bracket-only rules {('bracket3', 'percentile')}; got {rule!r} "
                     "(dynamic rules: income_proxy_frame)")
