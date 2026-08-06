"""
Homogeneity-assumption check for the HQRTM household-specific threshold.

Motivation
----------
The Energy Economics handling editor's first objection: HQRTM's "expected
energy expenditure" is a pooled quantile regression of total_expenditure on
QR_FEATURES (floor_area, house_age, dwelling_type, insulation_count,
main_heating_source, household_size) with `Country` entered only as an
intercept shift. That assumes two structurally similar households in
different countries -- once you subtract a national average -- should cost
the same to run, even though climate, urban/rural infrastructure, and local
fuel mix can differ substantially within that "similar characteristics"
group. This script tests that assumption three ways, instead of asserting
it:

(A) Residual leftover-structure test: does the *current* pooled model's
    residual (`relative_exp`, already computed in df_hqrtm_65.csv) still
    correlate with local-condition variables that were left out
    (SettlementSize, air-conditioner use as a climate proxy, whole-home vs
    partial heating strategy) after the existing Country fixed effect is
    already controlled for? If yes, the country intercept isn't fully
    absorbing local heterogeneity.

(B) Refit with those local-condition variables added directly to the
    quantile regression, and compare pseudo-R^2 and how many households get
    reclassified relative to the original spec.

(C) Country-varying slopes via partial pooling: instead of choosing between
    a single pooled slope (assumes homogeneity) and a fully separate
    per-country slope (unstable for smaller/noisier countries -- a
    single-fit floor_area slope can collapse to a numerically degenerate
    boundary solution for a country with too little regressor variation),
    shrink each country's bootstrap-estimated slope toward the pooled
    estimate in proportion to how precisely it's estimated
    (DerSimonian-Laird / James-Stein shrinkage). No country is dropped for
    instability -- see partial_pooling_qr.py.

(D) Reclassification impact: how much does the "high energy cost" flag
    change, overall and per country, when the pooled model is replaced by
    the partial-pooling model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from partial_pooling_qr import fit_partial_pooling, predict_expected_expenditure

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "outputs" / "validation" / "homogeneity_check_results"

QR_FEATURES = [
    "floor_area",
    "house_age",
    "dwelling_type",
    "insulation_count",
    "main_heating_source",
    "household_size",
]
EXP_COL = "total_expenditure"
COUNTRY_COL = "Country"
QUANTILE = 0.65
MARGIN_SCALE = 0.10
MIN_ROWS = 100
PER_COUNTRY_MIN_ROWS = 200
N_BOOT = 200


def clean_local_conditions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["settlement_size_clean"] = out["SettlementSize"].where(
        out["SettlementSize"] != "No answer"
    )

    c2 = pd.to_numeric(out["C2"], errors="coerce")
    out["aircon"] = np.select([c2 == 1.0, c2 == 2.0], [1.0, 0.0], default=np.nan)

    out["uniform_heating"] = np.select(
        [out["C3"] == "All rooms heated", out["C3"] == "Partial heating"],
        [1.0, 0.0],
        default=np.nan,
    )
    return out


def fit_pooled_qr(df: pd.DataFrame, features: list, label: str, quantile: float = QUANTILE):
    model_cols = [EXP_COL] + features + [COUNTRY_COL]
    fit_data = df[model_cols].replace([np.inf, -np.inf], np.nan).dropna()

    if len(fit_data) < MIN_ROWS:
        raise ValueError(f"[{label}] too few rows to fit: {len(fit_data)}")

    formula = f"{EXP_COL} ~ " + " + ".join(features + [f"C({COUNTRY_COL})"])
    model = smf.quantreg(formula, data=fit_data)
    res = model.fit(q=quantile, max_iter=5000, disp=False)

    pred_cols = features + [COUNTRY_COL]
    pred_data = df[pred_cols].replace([np.inf, -np.inf], np.nan).dropna()
    expected = pd.Series(index=df.index, dtype=float)
    expected.loc[pred_data.index] = res.predict(pred_data)

    relative_exp = df[EXP_COL] - expected
    margin = np.nanstd(fit_data[EXP_COL]) * MARGIN_SCALE
    # NB: `x > nan` evaluates to False in numpy/pandas, NOT NaN -- so
    # comparing against a missing `expected` would silently mark a household
    # "not high cost" instead of "undetermined". `.where(expected.notna())`
    # makes the missing-prediction rows explicitly NaN so callers can't
    # mistake missing data for a genuine low-cost classification.
    high_exp_flag = (df[EXP_COL] > (expected + margin)).where(expected.notna()).astype("boolean")

    print(
        f"[{label}] n_fit={len(fit_data)}, n_pred={len(pred_data)}, "
        f"pseudo_R2={res.prsquared:.4f}, high_exp_rate={high_exp_flag.mean():.2%} "
        f"(of {high_exp_flag.notna().sum()} households with a valid prediction)"
    )

    return {
        "label": label,
        "res": res,
        "n_fit": len(fit_data),
        "n_pred": len(pred_data),
        "pseudo_r2": res.prsquared,
        "expected_exp": expected,
        "relative_exp": relative_exp,
        "high_exp_flag": high_exp_flag,
    }


def residual_leftover_test(df: pd.DataFrame, relative_exp: pd.Series, candidate: str, label: str) -> dict:
    frame = df[[COUNTRY_COL, candidate]].copy()
    frame["relative_exp"] = relative_exp
    frame = frame.dropna()

    if frame[candidate].nunique() < 2 or len(frame) < 200:
        print(f"  [{label}] skipped (n={len(frame)}, unique={frame[candidate].nunique()})")
        return {"candidate": label, "n": len(frame), "f_pvalue": np.nan, "delta_r2": np.nan}

    restricted = smf.ols(f"relative_exp ~ C({COUNTRY_COL})", data=frame).fit()
    unrestricted = smf.ols(f"relative_exp ~ {candidate} + C({COUNTRY_COL})", data=frame).fit()

    anova = sm.stats.anova_lm(restricted, unrestricted)
    f_pvalue = anova["Pr(>F)"].iloc[-1]
    delta_r2 = unrestricted.rsquared - restricted.rsquared

    print(
        f"  [{label}] n={len(frame)}  R2 restricted={restricted.rsquared:.4f}  "
        f"R2 +{candidate}={unrestricted.rsquared:.4f}  delta_R2={delta_r2:.4f}  "
        f"F-test p={f_pvalue:.4g}"
    )
    return {"candidate": label, "n": len(frame), "f_pvalue": f_pvalue, "delta_r2": delta_r2}


def reclassification_summary(base_flag: pd.Series, alt_flag: pd.Series, country: pd.Series, label: str) -> pd.DataFrame:
    frame = pd.DataFrame({"Country": country, "base": base_flag, "alt": alt_flag}).dropna()
    rows = []
    for c, g in frame.groupby("Country"):
        rows.append(
            {
                "spec": label,
                "Country": c,
                "n": len(g),
                "base_high_cost_rate": g["base"].mean(),
                "alt_high_cost_rate": g["alt"].mean(),
                "newly_flagged": int((g["alt"] & ~g["base"]).sum()),
                "newly_unflagged": int((~g["alt"] & g["base"]).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("Country")


def run_all() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(REPO_ROOT / "df_hqrtm_65.csv", low_memory=False)
    df = clean_local_conditions(raw)

    print("=== Baseline pooled QR (current spec: QR_FEATURES + Country intercept) ===")
    baseline = fit_pooled_qr(df, QR_FEATURES, "baseline")

    print(
        "\nSanity check vs already-computed df_hqrtm_65.csv relative_exp "
        f"(should match closely): corr={baseline['relative_exp'].corr(df['relative_exp']):.4f}"
    )

    print("\n=== (A) Residual leftover-structure test ===")
    print("Does the baseline model's residual still correlate with left-out local-condition")
    print("variables, beyond the Country fixed effect already in the model?\n")
    residual_results = [
        residual_leftover_test(df, baseline["relative_exp"], "settlement_size_clean", "SettlementSize"),
        residual_leftover_test(df, baseline["relative_exp"], "aircon", "Air conditioner use (C2)"),
        residual_leftover_test(df, baseline["relative_exp"], "uniform_heating", "Whole-home vs partial heating (C3)"),
    ]
    pd.DataFrame(residual_results).to_csv(OUTPUT_DIR / "residual_leftover_tests.csv", index=False)

    print("\n=== (B) Refit with local-condition variables added to the QR ===")
    extended_settlement = fit_pooled_qr(
        df, QR_FEATURES + ["settlement_size_clean"], "extended: +SettlementSize"
    )
    extended_settlement_aircon = fit_pooled_qr(
        df, QR_FEATURES + ["settlement_size_clean", "aircon"], "extended: +SettlementSize+aircon"
    )

    for ext in [extended_settlement, extended_settlement_aircon]:
        # Restrict to households where BOTH models produced a real prediction.
        # A household missing the extended feature (e.g. SettlementSize =
        # "No answer") has no extended-model prediction at all -- it must be
        # excluded here, not silently counted as "reclassified to low cost".
        valid = baseline["high_exp_flag"].notna() & ext["high_exp_flag"].notna()
        base_flag = baseline["high_exp_flag"][valid]
        ext_flag = ext["high_exp_flag"][valid]
        n_excluded = (~valid).sum()
        agree = (base_flag == ext_flag).mean()
        newly_flagged = ((ext_flag) & (~base_flag)).sum()
        newly_unflagged = ((~ext_flag) & (base_flag)).sum()
        print(
            f"  {ext['label']}: pseudo_R2 {baseline['pseudo_r2']:.4f} -> {ext['pseudo_r2']:.4f}  "
            f"| compared on {valid.sum()} households with a valid prediction under both specs "
            f"({n_excluded} excluded, missing the extended feature)  "
            f"| high_exp_flag agreement={agree:.2%}  "
            f"newly flagged={newly_flagged}  newly unflagged={newly_unflagged}"
        )

    print("\n  Re-running residual leftover test on the EXTENDED (+SettlementSize) model's")
    print("  residual, to check whether adding SettlementSize actually absorbed its own signal:")
    residual_leftover_test(
        df, extended_settlement["relative_exp"], "settlement_size_clean", "SettlementSize (post-extension)"
    )

    print("\n=== (C) Country-varying slopes via partial pooling (every country retained) ===")
    print(
        f"Bootstrapping {N_BOOT} resamples per country to get a stable (not single-fit-fragile) "
        f"estimate of each country's continuous-block coefficients, then shrinking each country's "
        f"estimate toward the pooled prior in proportion to its imprecision (DerSimonian-Laird). "
        f"No country is dropped for instability; an unreliable country's shrinkage weight will "
        f"simply be closer to the pooled prior."
    )
    pooling = fit_partial_pooling(df, quantile=QUANTILE, n_boot=N_BOOT, min_rows=PER_COUNTRY_MIN_ROWS)

    comparison = pd.DataFrame(
        {
            "raw_bootstrap_mean": pooling.raw_coefs["floor_area"],
            "bootstrap_se": pooling.bootstrap_se["floor_area"],
            "shrinkage_weight": pooling.shrinkage_weights["floor_area"],
            "shrunk_estimate": pooling.shrunk_coefs["floor_area"],
        }
    ).sort_values("shrunk_estimate")
    comparison["pooled_prior"] = pooling.prior_coefs["floor_area"]
    comparison.to_csv(OUTPUT_DIR / "partial_pooling_floor_area.csv")
    print(comparison.to_string())

    most_shrunk = comparison["shrinkage_weight"].idxmin()
    row = comparison.loc[most_shrunk]
    print(
        f"\nMost-shrunk country: Country {most_shrunk}, floor_area raw bootstrap mean={row['raw_bootstrap_mean']:.3f} "
        f"(SE={row['bootstrap_se']:.3f}) -> shrinkage weight={row['shrinkage_weight']:.3f} "
        f"-> shrunk estimate={row['shrunk_estimate']:.3f}. Every country is retained regardless of "
        f"how imprecisely its slope is estimated; imprecision shows up as a low shrinkage weight, not exclusion."
    )

    print(
        f"\nPooled prior (continuous-block, country-FE-only) floor_area coefficient: "
        f"{pooling.prior_coefs['floor_area']:.3f}"
    )

    print("\n=== (D) Reclassification impact of partial pooling vs the pooled baseline ===")
    expected_partial = predict_expected_expenditure(df, pooling)
    margin = np.nanstd(df[EXP_COL]) * MARGIN_SCALE
    high_exp_flag_partial = (df[EXP_COL] > (expected_partial + margin)).where(expected_partial.notna())

    reclass = reclassification_summary(
        baseline["high_exp_flag"], high_exp_flag_partial, df[COUNTRY_COL], "partial_pooling_vs_pooled"
    )
    reclass.to_csv(OUTPUT_DIR / "reclassification_partial_pooling.csv", index=False)
    print(reclass.to_string(index=False))



if __name__ == "__main__":
    run_all()
