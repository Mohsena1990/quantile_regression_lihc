"""
Risk-category labeling functions for the LIHC / HQRTM pipeline.

Three labeling schemes are provided, all producing a four-class
`risk_category` column (No risk / Income risk / Expenditure risk /
Double risk) from a low-income flag crossed with a high-energy-cost flag:

- assign_traditional_lihc: fixed country-specific expenditure quantile as
  the high-cost threshold (the conventional LIHC approach).
- assign_hqrtm: household-specific expected expenditure from a pooled
  quantile regression on structural/demographic features (+ country fixed
  effects) as the high-cost threshold.
- assign_paper_lihc: the van Hove, Dalla Longa & van der Zwaan (2022)
  variant, using an income-bracket cut and an 80th-percentile expenditure
  cut.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def assign_traditional_lihc(
    df: pd.DataFrame,
    income_col: str = "equivalized_income",
    exp_col: str = "total_expenditure",
    country_col: str = "Country",
    income_rule: str = "country_median_60",
    exp_quantile: float = 0.80,
    income_bracket_col: str = "income_bracket",
    fit_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Traditional LIHC-style categorization.

    If fit_df is provided, thresholds are fit on fit_df and applied to df.
    """
    out = df.copy()
    fit_source = out if fit_df is None else fit_df.copy()

    required_cols = [country_col, exp_col]
    if income_rule == "country_median_60":
        required_cols.append(income_col)
    elif income_rule == "bracket_lt4":
        required_cols.append(income_bracket_col)
    else:
        raise ValueError("income_rule must be 'country_median_60' or 'bracket_lt4'")

    missing_cols = [c for c in required_cols if c not in out.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")
    missing_cols_fit = [c for c in required_cols if c not in fit_source.columns]
    if missing_cols_fit:
        raise KeyError(f"Missing required columns in fit_df: {missing_cols_fit}")

    # ---------- low-income side ----------
    out["national_median_income"] = np.nan
    out["lihc_income_threshold"] = np.nan

    if income_rule == "country_median_60":
        medians = fit_source.groupby(country_col)[income_col].median()
        fallback_income_median = fit_source[income_col].median()
        out["national_median_income"] = out[country_col].map(medians).fillna(fallback_income_median)
        out["lihc_income_threshold"] = 0.60 * out["national_median_income"]
        out["low_income"] = out[income_col] < out["lihc_income_threshold"]
    else:
        out["low_income"] = out[income_bracket_col] < 4

    # ---------- expenditure side ----------
    exp_thresholds = fit_source.groupby(country_col)[exp_col].quantile(exp_quantile)
    fallback_exp_threshold = fit_source[exp_col].quantile(exp_quantile)
    out["exp_threshold"] = out[country_col].map(exp_thresholds).fillna(fallback_exp_threshold)
    out["high_exp_flag"] = out[exp_col] > out["exp_threshold"]

    # ---------- four classes ----------
    out["risk_category"] = np.select(
        [
            out["low_income"] & out["high_exp_flag"],
            out["low_income"] & ~out["high_exp_flag"],
            ~out["low_income"] & out["high_exp_flag"],
        ],
        [
            "Double risk",
            "Income risk",
            "Expenditure risk",
        ],
        default="No risk",
    )

    return out


def assign_hqrtm(
    df: pd.DataFrame,
    qr_features: list,
    income_col: str = "equivalized_income",
    exp_col: str = "total_expenditure",
    country_col: str = "Country",
    income_rule: str = "country_median_60",
    quantile: float = 0.65,
    min_rows: int = 100,
    income_bracket_col: str = "income_bracket",
    add_country_effects: bool = True,
    margin_scale: float = 0.10,
    fit_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    HQRTM categorization with pooled quantile regression.

    Household-specific expected energy expenditure is estimated with a
    pooled quantile regression of exp_col on qr_features (+ country fixed
    effects if add_country_effects), fit at the given quantile. A household
    is flagged high-cost if its actual expenditure exceeds the predicted
    quantile plus a margin (margin_scale * std of fitted expenditure).

    If fit_df is provided, the quantile model and margin are fit on fit_df
    and then applied to df.
    """
    if not qr_features:
        raise ValueError("qr_features must be a non-empty list")

    if quantile not in [0.60, 0.65, 0.70]:
        raise ValueError("For this study, quantile should be one of [0.60, 0.65, 0.70]")

    out = df.copy()
    fit_source = out if fit_df is None else fit_df.copy()

    required_cols = [country_col, exp_col] + qr_features
    if income_rule == "country_median_60":
        required_cols.append(income_col)
    elif income_rule == "bracket_lt4":
        required_cols.append(income_bracket_col)
    else:
        raise ValueError("income_rule must be 'country_median_60' or 'bracket_lt4'")

    missing_cols = [c for c in required_cols if c not in out.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")
    missing_cols_fit = [c for c in required_cols if c not in fit_source.columns]
    if missing_cols_fit:
        raise KeyError(f"Missing required columns in fit_df: {missing_cols_fit}")

    # ---------- low-income side ----------
    out["national_median_income"] = np.nan
    out["lihc_income_threshold"] = np.nan

    if income_rule == "country_median_60":
        medians = fit_source.groupby(country_col)[income_col].median()
        fallback_income_median = fit_source[income_col].median()
        out["national_median_income"] = out[country_col].map(medians).fillna(fallback_income_median)
        out["lihc_income_threshold"] = 0.60 * out["national_median_income"]
        out["low_income"] = out[income_col] < out["lihc_income_threshold"]
    else:
        out["low_income"] = out[income_bracket_col] < 4

    # ---------- conditional expenditure side ----------
    out["expected_exp"] = np.nan
    out["relative_exp"] = np.nan
    out["high_exp_flag"] = False
    out["qr_valid_flag"] = False

    model_cols = [exp_col] + qr_features
    if add_country_effects:
        model_cols.append(country_col)

    pooled_fit = fit_source[model_cols].copy().replace([np.inf, -np.inf], np.nan)
    valid_fit = pooled_fit.dropna()

    if len(valid_fit) < min_rows:
        raise ValueError(
            f"Too few valid rows for pooled quantile regression: {len(valid_fit)} < {min_rows}"
        )

    rhs_terms = qr_features.copy()
    if add_country_effects:
        rhs_terms.append(f"C({country_col})")
    formula = f"{exp_col} ~ " + " + ".join(rhs_terms)

    try:
        model = smf.quantreg(formula, data=valid_fit)
        res = model.fit(q=quantile, max_iter=5000, disp=False)

        pred_source = out[model_cols].copy().replace([np.inf, -np.inf], np.nan)
        pred_subset = qr_features + ([country_col] if add_country_effects else [])
        valid_pred = pred_source.dropna(subset=pred_subset)

        if len(valid_pred) == 0:
            raise ValueError("No rows available for prediction after filtering qr features.")

        expected = pd.Series(index=out.index, dtype=float)
        expected.loc[valid_pred.index] = res.predict(valid_pred)

        out["expected_exp"] = expected
        out["relative_exp"] = out[exp_col] - out["expected_exp"]

        margin = np.nanstd(valid_fit[exp_col]) * margin_scale
        out["high_exp_flag"] = (out[exp_col] > (out["expected_exp"] + margin)).fillna(False)
        out.loc[valid_pred.index, "qr_valid_flag"] = True

        print(
            f"[Pooled HQRTM] q={quantile:.2f}, "
            f"pseudo_R2={getattr(res, 'prsquared', np.nan):.3f}, "
            f"high_exp={out['high_exp_flag'].mean():.2%}, "
            f"n_fit={len(valid_fit)}"
        )

        print("\nHigh expenditure rate by country:")
        print(
            out.groupby(country_col)["high_exp_flag"]
               .mean()
               .sort_index()
               .round(4)
        )

    except Exception as e:
        raise RuntimeError(f"Pooled quantile regression failed: {e}")

    out["risk_category"] = np.select(
        [
            out["low_income"] & out["high_exp_flag"],
            out["low_income"] & ~out["high_exp_flag"],
            ~out["low_income"] & out["high_exp_flag"],
        ],
        [
            "Double risk",
            "Income risk",
            "Expenditure risk",
        ],
        default="No risk",
    )

    return out


def assign_paper_lihc(
    df: pd.DataFrame,
    income_bracket_col: str = "income_bracket",
    exp_col: str = "total_expenditure",
    country_col: str = "Country",
    exp_quantile: float = 0.80,
    fit_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Paper-style LIHC categorization based on:
      - low income: income bracket < 4
      - high expenditure: above country-specific 80th percentile
        of annual energy expenditure

    This matches the framework used in:
    van Hove, Dalla Longa, van der Zwaan (2022),
    where the income threshold is set between deciles 3 and 4,
    and the expenditure threshold is the 80th percentile within country.

    Parameters
    ----------
    df : pd.DataFrame
        Data to classify.
    income_bracket_col : str
        Survey income bracket / decile column (1-10).
    exp_col : str
        Annual energy expenditure column.
    country_col : str
        Country identifier column.
    exp_quantile : float
        Country-specific expenditure quantile. Default is 0.80.
    fit_df : pd.DataFrame | None
        Optional reference dataset on which thresholds are fitted and then
        applied to df.

    Returns
    -------
    pd.DataFrame
        Copy of df with:
        - low_income
        - exp_threshold
        - high_exp_flag
        - risk_category
    """
    out = df.copy()
    fit_source = out if fit_df is None else fit_df.copy()

    required_cols = [country_col, income_bracket_col, exp_col]

    missing_cols = [c for c in required_cols if c not in out.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    missing_cols_fit = [c for c in required_cols if c not in fit_source.columns]
    if missing_cols_fit:
        raise KeyError(f"Missing required columns in fit_df: {missing_cols_fit}")

    # low income: threshold between brackets 3 and 4
    out["low_income"] = out[income_bracket_col] < 4

    # high expenditure: country-specific expenditure quantile
    exp_thresholds = fit_source.groupby(country_col)[exp_col].quantile(exp_quantile)
    out["exp_threshold"] = out[country_col].map(exp_thresholds)

    if out["exp_threshold"].isna().any():
        missing_countries = sorted(
            out.loc[out["exp_threshold"].isna(), country_col].dropna().unique().tolist()
        )
        raise ValueError(
            f"Missing expenditure thresholds for countries: {missing_countries}"
        )

    out["high_exp_flag"] = out[exp_col] > out["exp_threshold"]

    # four classes
    out["risk_category"] = np.select(
        [
            out["low_income"] & out["high_exp_flag"],
            out["low_income"] & ~out["high_exp_flag"],
            ~out["low_income"] & out["high_exp_flag"],
        ],
        [
            "Double risk",
            "Income risk",
            "Expenditure risk",
        ],
        default="No risk",
    )

    return out
