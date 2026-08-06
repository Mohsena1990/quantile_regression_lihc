"""
Partial-pooling (empirical-Bayes shrinkage) quantile regression across
countries.

Context
-------
homogeneity_assumption_check.py shows two things can't both be true at
once: a single pooled slope (intercept-only-by-country) is wrong for most
countries' floor_area cost coefficient, but fitting each country fully
separately is unstable for smaller/noisier countries -- the UK's
floor_area slope collapses to a numerically degenerate boundary solution
(~0 with a non-trivial standard error) rather than a real estimate.

Dropping the unstable country is the easy way out, but throws away every
household in it. Partial pooling -- shrinking each country's coefficient
toward the pooled estimate in proportion to how imprecisely it's
estimated -- is the standard remedy for exactly this tension
(DerSimonian-Laird / James-Stein shrinkage, the same machinery behind
random-effects meta-analysis). A country with a noisy or degenerate
estimate (like the UK) gets pulled hard toward the pooled prior instead
of being excluded; a country with a precise, genuinely different slope
(like Germany) keeps most of its own estimate.

Design choice: main_heating_source is a ~10-level categorical feature.
Re-estimating 10 dummy coefficients per country from a few hundred to
~1,500 rows per bootstrap draw is not identifiable/stable, so its effect
is estimated once, pooled across all countries, and held fixed. Only the
continuous block (floor_area, house_age, dwelling_type, insulation_count,
household_size) plus the country intercept is partially pooled. This is
a modeling choice made explicit here, not a silent simplification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

CONTINUOUS_QR_FEATURES = [
    "floor_area",
    "house_age",
    "dwelling_type",
    "insulation_count",
    "household_size",
]


def extract_heating_contribution(res, heating_col: str = "main_heating_source") -> dict:
    """Pull the fitted per-category effect of a categorical term out of a
    statsmodels formula fit, as a {category_value: coefficient} dict (the
    dropped reference category maps implicitly to 0)."""
    mapping = {}
    prefix = f"{heating_col}[T."
    for name, value in res.params.items():
        if name.startswith(prefix):
            category = name[len(prefix):-1]
            mapping[category] = value
    return mapping


@dataclass
class PartialPoolingResult:
    shrunk_coefs: pd.DataFrame       # index=Country, columns=[Intercept]+CONTINUOUS_QR_FEATURES
    raw_coefs: pd.DataFrame          # per-country bootstrap-mean coefficients (unshrunk)
    prior_coefs: pd.Series           # pooled prior mean, per coefficient
    shrinkage_weights: pd.DataFrame  # index=Country, columns=coefficients; 1=trust country data, 0=fully pooled
    bootstrap_se: pd.DataFrame       # index=Country, columns=coefficients
    heating_contribution: dict
    n_boot: int
    quantile: float


def _bootstrap_country_fit(
    data: pd.DataFrame,
    formula: str,
    quantile: float,
    n_boot: int,
    seed: int,
) -> pd.DataFrame | None:
    rng = np.random.default_rng(seed)
    n = len(data)
    draws = []
    for _ in range(n_boot):
        sample = data.iloc[rng.integers(0, n, size=n)]
        try:
            res = smf.quantreg(formula, data=sample).fit(q=quantile, max_iter=2000, disp=False)
            draws.append(res.params)
        except Exception:
            continue
    if len(draws) < max(20, n_boot // 4):
        return None
    return pd.DataFrame(draws)


def fit_partial_pooling(
    df: pd.DataFrame,
    country_col: str = "Country",
    exp_col: str = "total_expenditure",
    quantile: float = 0.65,
    min_rows: int = 200,
    n_boot: int = 200,
    heating_col: str = "main_heating_source",
) -> PartialPoolingResult:
    model_cols = [exp_col, heating_col] + CONTINUOUS_QR_FEATURES + [country_col]
    fit_data = df[model_cols].replace([np.inf, -np.inf], np.nan).dropna()

    pooled_formula = (
        f"{exp_col} ~ " + " + ".join(CONTINUOUS_QR_FEATURES) + f" + C({heating_col}) + C({country_col})"
    )
    pooled_res = smf.quantreg(pooled_formula, data=fit_data).fit(q=quantile, max_iter=5000, disp=False)
    heating_contribution = extract_heating_contribution(pooled_res, heating_col)

    fit_data = fit_data.copy()
    fit_data["y_adj"] = fit_data[exp_col] - fit_data[heating_col].map(heating_contribution).fillna(0.0)

    prior_formula = "y_adj ~ " + " + ".join(CONTINUOUS_QR_FEATURES) + f" + C({country_col})"
    prior_res = smf.quantreg(prior_formula, data=fit_data).fit(q=quantile, max_iter=5000, disp=False)

    coef_names = ["Intercept"] + CONTINUOUS_QR_FEATURES
    countries = sorted(fit_data[country_col].unique())

    prior_by_country = {}
    for country in countries:
        intercept = prior_res.params.get("Intercept", 0.0)
        dummy_name = f"C({country_col})[T.{country}]"
        if dummy_name in prior_res.params.index:
            intercept += prior_res.params[dummy_name]
        row = {"Intercept": intercept}
        for feat in CONTINUOUS_QR_FEATURES:
            row[feat] = prior_res.params.get(feat, 0.0)
        prior_by_country[country] = row
    prior_by_country_df = pd.DataFrame(prior_by_country).T[coef_names]

    global_prior = pd.Series(
        {**{"Intercept": prior_res.params.get("Intercept", 0.0)},
         **{feat: prior_res.params.get(feat, 0.0) for feat in CONTINUOUS_QR_FEATURES}}
    )

    raw_rows, se_rows = {}, {}
    per_country_formula = "y_adj ~ " + " + ".join(CONTINUOUS_QR_FEATURES)

    for i, country in enumerate(countries):
        group = fit_data[fit_data[country_col] == country]
        if len(group) < min_rows:
            continue
        draws = _bootstrap_country_fit(group, per_country_formula, quantile, n_boot, seed=1000 + i)
        if draws is None:
            continue
        for coef in coef_names:
            if coef not in draws.columns:
                draws[coef] = np.nan

        raw_rows[country] = draws[coef_names].mean()
        se_rows[country] = draws[coef_names].var(ddof=1).pow(0.5)

    raw_coefs = pd.DataFrame(raw_rows).T[coef_names]
    bootstrap_se = pd.DataFrame(se_rows).T[coef_names]

    # DerSimonian-Laird between-country variance tau^2 per coefficient, then
    # the shrinkage weight and shrunk estimate for every country (needs all
    # countries' raw estimates + bootstrap SEs first, hence the second pass).
    tau2 = {}
    for coef in coef_names:
        between_var = raw_coefs[coef].var(ddof=1) if raw_coefs[coef].notna().sum() > 1 else 0.0
        mean_within_var = (bootstrap_se[coef] ** 2).mean()
        tau2[coef] = max(0.0, between_var - mean_within_var)

    weight_rows, shrunk_rows = {}, {}
    for country in raw_coefs.index:
        weights, shrunk = {}, {}
        for coef in coef_names:
            v_c = bootstrap_se.loc[country, coef] ** 2
            t2 = tau2[coef]
            w = t2 / (t2 + v_c) if (t2 + v_c) > 0 else 0.0
            weights[coef] = w
            prior_mean = prior_by_country_df.loc[country, coef]
            shrunk[coef] = w * raw_coefs.loc[country, coef] + (1 - w) * prior_mean
        weight_rows[country] = weights
        shrunk_rows[country] = shrunk

    shrinkage_weights = pd.DataFrame(weight_rows).T[coef_names]
    shrunk_coefs = pd.DataFrame(shrunk_rows).T[coef_names]

    return PartialPoolingResult(
        shrunk_coefs=shrunk_coefs,
        raw_coefs=raw_coefs,
        prior_coefs=global_prior,
        shrinkage_weights=shrinkage_weights,
        bootstrap_se=bootstrap_se,
        heating_contribution=heating_contribution,
        n_boot=n_boot,
        quantile=quantile,
    )


def predict_expected_expenditure(
    df: pd.DataFrame,
    result: PartialPoolingResult,
    country_col: str = "Country",
    heating_col: str = "main_heating_source",
) -> pd.Series:
    """Predict expected expenditure for every household using its
    country's shrunk continuous-block coefficients plus the pooled
    heating-source effect."""
    expected = pd.Series(index=df.index, dtype=float)

    heating_effect = df[heating_col].map(result.heating_contribution).fillna(0.0)

    for country in result.shrunk_coefs.index:
        mask = df[country_col] == country
        if not mask.any():
            continue
        coefs = result.shrunk_coefs.loc[country]
        x = df.loc[mask, CONTINUOUS_QR_FEATURES]
        pred = coefs["Intercept"] + x.mul(coefs[CONTINUOUS_QR_FEATURES]).sum(axis=1)
        expected.loc[mask] = pred + heating_effect.loc[mask]

    return expected
