"""
Cross-fitted (out-of-fold) HQRTM and traditional-LIHC labels.

Motivation
----------
external_validity_check.py tests whether HQRTM's "high energy cost" flag
correlates with independent hardship markers (S7/C5B/C1A/S8) better than
traditional LIHC's flag does -- which answers the handling editor's
circularity objection about predicting a label from the variables used to
build it. But the flags themselves are still fit and applied to the same
households: every household's threshold was estimated using a model that
included that very household's own data point.

This module removes that last in-sample-fit advantage by K-fold
cross-fitting: for each fold, the quantile regression (HQRTM) or the
country-specific expenditure quantile (traditional LIHC) is fit only on
the OTHER folds, then applied to the held-out fold. Every household's
label is therefore built from a model that never saw that household.
Folds are stratified by Country so every fold keeps each country's
within-country structure for the model that includes country fixed
effects / country-specific thresholds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import StratifiedKFold

QR_FEATURES = [
    "floor_area",
    "house_age",
    "dwelling_type",
    "insulation_count",
    "main_heating_source",
    "household_size",
]
EXP_COL = "total_expenditure"
INCOME_COL = "equivalized_income"
COUNTRY_COL = "Country"
MARGIN_SCALE = 0.10
N_FOLDS = 5
RANDOM_STATE = 42


def _low_income_flag(df: pd.DataFrame, fit_df: pd.DataFrame) -> pd.Series:
    medians = fit_df.groupby(COUNTRY_COL)[INCOME_COL].median()
    fallback = fit_df[INCOME_COL].median()
    national_median = df[COUNTRY_COL].map(medians).fillna(fallback)
    return df[INCOME_COL] < 0.60 * national_median


def cross_fit_hqrtm(
    df: pd.DataFrame,
    quantile: float = 0.65,
    n_folds: int = N_FOLDS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Out-of-fold HQRTM labels: for each fold, the pooled quantile
    regression (QR_FEATURES + Country FE) and the country-median income
    threshold are both fit on the other folds only."""
    out = pd.DataFrame(index=df.index)
    out["low_income"] = pd.array([pd.NA] * len(df), dtype="boolean")
    out["high_exp_flag"] = pd.array([pd.NA] * len(df), dtype="boolean")
    out["expected_exp"] = np.nan

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    formula = f"{EXP_COL} ~ " + " + ".join(QR_FEATURES + [f"C({COUNTRY_COL})"])

    for fold_i, (fit_idx, held_idx) in enumerate(skf.split(df, df[COUNTRY_COL])):
        fit_df = df.iloc[fit_idx]
        held_df = df.iloc[held_idx]

        model_cols = [EXP_COL] + QR_FEATURES + [COUNTRY_COL]
        fit_data = fit_df[model_cols].replace([np.inf, -np.inf], np.nan).dropna()
        res = smf.quantreg(formula, data=fit_data).fit(q=quantile, max_iter=5000, disp=False)

        pred_data = held_df[QR_FEATURES + [COUNTRY_COL]].replace([np.inf, -np.inf], np.nan).dropna()
        # A held-out fold can contain a main_heating_source category that
        # never appeared in this fold's fit data (more likely with a
        # smaller sample / rarer categories) -- the fitted model has no
        # coefficient for it and can't predict for those rows. Drop them
        # from this fold rather than crash or silently guess; they surface
        # as "no valid prediction" like any other missing-feature case.
        seen_categories = set(fit_data["main_heating_source"].unique())
        unseen_mask = ~pred_data["main_heating_source"].isin(seen_categories)
        if unseen_mask.any():
            print(f"  [HQRTM fold {fold_i}] dropping {unseen_mask.sum()} rows with an unseen "
                  f"main_heating_source category: {sorted(set(pred_data.loc[unseen_mask, 'main_heating_source']))}")
            pred_data = pred_data[~unseen_mask]
        expected = res.predict(pred_data)
        margin = np.nanstd(fit_data[EXP_COL]) * MARGIN_SCALE

        out.loc[pred_data.index, "expected_exp"] = expected
        out.loc[pred_data.index, "high_exp_flag"] = (
            held_df.loc[pred_data.index, EXP_COL] > (expected + margin)
        ).astype("boolean")
        out.loc[held_df.index, "low_income"] = _low_income_flag(held_df, fit_df).astype("boolean")

        print(f"  [HQRTM fold {fold_i}] fit n={len(fit_data)}, held n={len(held_df)}, "
              f"pseudo_R2={res.prsquared:.4f}")

    out["risk_category"] = np.select(
        [
            out["low_income"].fillna(False) & out["high_exp_flag"].fillna(False),
            out["low_income"].fillna(False) & ~out["high_exp_flag"].fillna(False),
            ~out["low_income"].fillna(False) & out["high_exp_flag"].fillna(False),
        ],
        ["Double risk", "Income risk", "Expenditure risk"],
        default="No risk",
    )
    out.loc[out["high_exp_flag"].isna() | out["low_income"].isna(), "risk_category"] = np.nan
    return out


def cross_fit_traditional_lihc(
    df: pd.DataFrame,
    exp_quantile: float = 0.80,
    n_folds: int = N_FOLDS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Out-of-fold traditional-LIHC labels: the country-specific
    expenditure quantile and income-median thresholds are both fit on the
    other folds only."""
    out = pd.DataFrame(index=df.index)
    out["low_income"] = pd.array([pd.NA] * len(df), dtype="boolean")
    out["high_exp_flag"] = pd.array([pd.NA] * len(df), dtype="boolean")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for fold_i, (fit_idx, held_idx) in enumerate(skf.split(df, df[COUNTRY_COL])):
        fit_df = df.iloc[fit_idx]
        held_df = df.iloc[held_idx]

        exp_thresholds = fit_df.groupby(COUNTRY_COL)[EXP_COL].quantile(exp_quantile)
        fallback = fit_df[EXP_COL].quantile(exp_quantile)
        threshold = held_df[COUNTRY_COL].map(exp_thresholds).fillna(fallback)

        out.loc[held_df.index, "high_exp_flag"] = (held_df[EXP_COL] > threshold).astype("boolean")
        out.loc[held_df.index, "low_income"] = _low_income_flag(held_df, fit_df).astype("boolean")

        print(f"  [LIHC fold {fold_i}] fit n={len(fit_df)}, held n={len(held_df)}")

    out["risk_category"] = np.select(
        [
            out["low_income"].fillna(False) & out["high_exp_flag"].fillna(False),
            out["low_income"].fillna(False) & ~out["high_exp_flag"].fillna(False),
            ~out["low_income"].fillna(False) & out["high_exp_flag"].fillna(False),
        ],
        ["Double risk", "Income risk", "Expenditure risk"],
        default="No risk",
    )
    out.loc[out["high_exp_flag"].isna() | out["low_income"].isna(), "risk_category"] = np.nan
    return out
