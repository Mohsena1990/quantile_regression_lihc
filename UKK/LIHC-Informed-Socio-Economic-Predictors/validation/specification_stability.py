"""
Specification-stability band for the HQRTM "high energy cost" flag.

Motivation
----------
homogeneity_assumption_check.py shows the "Double risk" classification is
sensitive to reasonable, defensible specification choices: adding
SettlementSize changes the classification for some households, and the
pooled-vs-partial-pooling slope treatment changes country-level rates too.
Rather than presenting one point estimate as if it were exact, this script
reports the classification across every combination of:

  - quantile:      0.60 / 0.65 / 0.70 (the three specs already used in the
                    paper)
  - feature set:   baseline QR_FEATURES vs. +SettlementSize
  - slope pooling: fully pooled (single slope + country intercept) vs.
                    partial pooling (per-country slope shrunk toward the
                    pooled estimate)

and reports the resulting range of "Double risk" prevalence as an explicit
stability band, instead of a single number, both overall and per country.

A household with no valid prediction under a given spec (e.g. missing
SettlementSize, currently ~42% of the sample -- see the preprocessing fix
noted in preprocessing.py) is excluded from that spec's denominator, not
silently counted as "not high cost". Every rate below is reported next to
its own valid-N so a smaller-coverage spec is never mistaken for a
like-for-like comparison.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from homogeneity_assumption_check import (
    clean_local_conditions,
    fit_pooled_qr,
    EXP_COL,
    COUNTRY_COL,
    MARGIN_SCALE,
    QR_FEATURES,
)
from partial_pooling_qr import fit_partial_pooling, predict_expected_expenditure

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "outputs" / "validation" / "specification_stability_results"

QUANTILES = [0.60, 0.65, 0.70]
FEATURE_SETS = {
    "baseline": QR_FEATURES,
    "+SettlementSize": QR_FEATURES + ["settlement_size_clean"],
}
N_BOOT = 80  # lighter than homogeneity_assumption_check.py's headline run (200) -- a stability
             # sweep needs the right ballpark, not maximum precision, across 3 quantiles x 2 feature sets.


def double_risk_rate(high_exp_flag: pd.Series, low_income: pd.Series) -> tuple:
    """Double-risk rate among households with a valid high_exp_flag
    prediction, plus that valid count -- households with no prediction
    (missing a required feature) are excluded from the denominator rather
    than silently treated as low-cost."""
    valid = high_exp_flag.notna()
    n_valid = int(valid.sum())
    if n_valid == 0:
        return np.nan, 0
    double_risk = high_exp_flag[valid].astype(bool) & low_income[valid].astype(bool)
    return float(double_risk.mean()), n_valid


def per_country_double_risk_rates(flag: pd.Series, low_income: pd.Series, country: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"Country": country, "flag": flag, "low_income": low_income})
    rows = []
    for c, g in frame.groupby("Country"):
        rate, n_valid = double_risk_rate(g["flag"], g["low_income"])
        rows.append({"Country": c, "rate": rate, "n_valid": n_valid, "n_total": len(g)})
    return pd.DataFrame(rows)


def run_all() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(REPO_ROOT / "df_hqrtm_65.csv", low_memory=False)
    df = clean_local_conditions(raw)
    low_income = df["low_income"].astype(bool)
    n_total = len(df)

    overall_rows = []
    country_rows = []

    def record(quantile, feature_set, slope_pooling, flag):
        rate, n_valid = double_risk_rate(flag, low_income)
        overall_rows.append(
            {
                "quantile": quantile,
                "feature_set": feature_set,
                "slope_pooling": slope_pooling,
                "n_valid": n_valid,
                "n_total": n_total,
                "double_risk_rate_overall": rate,
            }
        )
        per_country = per_country_double_risk_rates(flag, low_income, df[COUNTRY_COL])
        per_country["quantile"] = quantile
        per_country["feature_set"] = feature_set
        per_country["slope_pooling"] = slope_pooling
        country_rows.append(per_country)

    for quantile in QUANTILES:
        for feature_label, features in FEATURE_SETS.items():
            fitted = fit_pooled_qr(df, features, f"pooled q={quantile} {feature_label}", quantile=quantile)
            record(quantile, feature_label, "fully_pooled", fitted["high_exp_flag"])

        # Partial pooling only applies to the continuous QR block (see
        # partial_pooling_qr.py), so it's evaluated once per quantile,
        # independent of the SettlementSize feature-set toggle above.
        pooling = fit_partial_pooling(df, quantile=quantile, n_boot=N_BOOT, min_rows=200)
        expected_partial = predict_expected_expenditure(df, pooling)
        margin = np.nanstd(df[EXP_COL]) * MARGIN_SCALE
        flag_partial = (df[EXP_COL] > (expected_partial + margin)).where(expected_partial.notna())
        record(quantile, "baseline", "partial_pooling", flag_partial)

    overall = pd.DataFrame(overall_rows)
    overall.to_csv(OUTPUT_DIR / "specification_stability_overall.csv", index=False)

    country = pd.concat(country_rows, ignore_index=True)
    country.to_csv(OUTPUT_DIR / "specification_stability_by_country_raw.csv", index=False)

    print("=== Overall Double-risk prevalence across specifications ===")
    display = overall.copy()
    display["double_risk_rate_overall"] = (display["double_risk_rate_overall"] * 100).round(2)
    print(display.to_string(index=False))

    full_coverage = overall[overall["n_valid"] == overall["n_total"]]
    band_low = full_coverage["double_risk_rate_overall"].min() * 100
    band_high = full_coverage["double_risk_rate_overall"].max() * 100
    print(
        f"\nStability band across the {len(full_coverage)} full-coverage specifications "
        f"(fully_pooled and partial_pooling baseline, all quantiles): "
        f"{band_low:.2f}% - {band_high:.2f}% (range = {band_high - band_low:.2f} pp)."
    )

    reduced_coverage = overall[overall["n_valid"] != overall["n_total"]]
    if len(reduced_coverage):
        rc_low = reduced_coverage["double_risk_rate_overall"].min() * 100
        rc_high = reduced_coverage["double_risk_rate_overall"].max() * 100
        print(
            f"+SettlementSize specs (reduced coverage, n_valid={reduced_coverage['n_valid'].iloc[0]}/{n_total} "
            f"households -- SettlementSize is 'No answer' for the rest, see the preprocessing.py fix): "
            f"{rc_low:.2f}% - {rc_high:.2f}%. Kept separate from the headline band above because it is "
            f"not a like-for-like comparison until that fix is re-run through the full pipeline."
        )

    print("\n=== Per-country Double-risk stability band, full-coverage specs only (widest first) ===")
    country_full = country[country["n_valid"] == country["n_total"]]
    country_band = (
        country_full.groupby("Country")["rate"]
        .agg(min="min", max="max", n_specs="count")
        .assign(range_pp=lambda d: (d["max"] - d["min"]) * 100)
        .sort_values("range_pp", ascending=False)
    )
    country_band[["min", "max"]] = (country_band[["min", "max"]] * 100).round(2)
    country_band["range_pp"] = country_band["range_pp"].round(2)
    country_band.to_csv(OUTPUT_DIR / "specification_stability_by_country.csv")
    print(country_band.to_string())


if __name__ == "__main__":
    run_all()
